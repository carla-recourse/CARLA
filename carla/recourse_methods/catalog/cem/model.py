# Copyright (C) 2018, IBM Corp
#                     Chun-Chen Tu <timtu@umich.edu>
#                     PaiShun Ting <paishun@umich.edu>
#                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
#
# Modifications copyright (C) 2021, University of Tübingen
#                      Johan van den Heuvel
#                      Sascha Bielawski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from carla.models.api import MLModel
from carla.recourse_methods.autoencoder import Autoencoder, train_autoencoder

from ...api import RecourseMethod
from ...processing import check_counterfactuals
from ...processing.counterfactuals import merge_default_parameters


class CEM(RecourseMethod):
    """
    Implementation of CEM from Dhurandhar et.al. [1]_.
    CEM needs an variational autoencoder to generate counterfactual examples.
    By setting the train_ae key to True in hyperparams, a tensorflow VAE will be trained.

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    encode_normalize_order_factuals:
        Uses encoder and scaler from black-box-model to preprocess data as needed.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "kappa": float
            Hyperparameter for CEM
        * "init_learning_rate": float
            Learning rate for CEM optimization
        * "binary_search_steps": int
            Hyperparameter for CEM
        * "max_iterations": int
            Number of maximum iterations to find a counterfactual example.
        * "initial_const": int
            Hyperparameter for CEM
        * "beta": float
            Hyperparameter for CEM
        * "gamma": float
            Hyperparameter for CEM
        * "mode": {"PN", "PP"}
            Mode for CEM
        * "num_classes": int.
            Currently only binary classifier are supported
        * "data_name": str
            Identificates the loaded or saved autoencoder model
        * "ae_params:" dict
            Initialisation and training parameter for the autoencoder model

            + "hidden_layer": list
                Number of neurons and layer of autoencoder.
            + "train_ae": bool
                If True, an autoencoder will be trained and saved
            + "epochs": int
                Number of training epochs for autoencoder

    .. [1] Amit Dhurandhar, Pin-Yu Chen, Ronny Luss, Chun-Chen Tu, Paishun Ting, Karthikeyan Shanmugam,and Payel Das.
            2018. Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives.
            In Advances in Neural Information Processing Systems344(NeurIPS).
    """

    _DEFAULT_HYPERPARAMS = {
        "data_name": None,
        "batch_size": 1,
        "kappa": 0.1,
        "init_learning_rate": 0.01,
        "binary_search_steps": 9,
        "max_iterations": 100,
        "initial_const": 10,
        "beta": 0.9,
        "gamma": 0.0,
        "mode": "PN",
        "num_classes": 2,
        "ae_params": {
            "hidden_layer": None,
            "train_ae": True,
            "epochs": 5,
        },
    }

    def __init__(self, sess, mlmodel: MLModel, hyperparams):
        self.sess = sess  # Tensorflow session
        self._hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        self._kappa = self._hyperparams["kappa"]
        self._mode = self._hyperparams["mode"]

        batch_size = self._hyperparams["batch_size"]
        num_classes = self._hyperparams["num_classes"]
        beta = self._hyperparams["beta"]
        gamma = self._hyperparams["gamma"]

        super().__init__(mlmodel)
        shape_batch = (batch_size, len(mlmodel.feature_input_order))

        self._AE = self._load_ae(self._hyperparams, mlmodel)

        self._initialize_tf_variables(batch_size, num_classes, shape_batch)

        """Fast Iterative Soft Thresholding"""
        """--------------------------------"""
        # Commented by us:
        # BEGIN: conditions to compute the ell1 regularization
        # this should be the function S_beta(z) in the paper
        if self._mode not in ["PP", "PN"]:
            raise ValueError("Mode not known, please use either PP or PN")

        zt = tf.divide(self._global_step, self._global_step + tf.cast(3, tf.float32))

        self._assign_adv = self._compute_adv(self._orig, self._adv_s, beta)
        self._assign_adv_s = self._compute_adv_s(
            zt, self._orig, self._adv, self._assign_adv
        )

        self._adv_updater = tf.assign(self._adv, self._assign_adv)
        self._adv_updater_s = tf.assign(self._adv_s, self._assign_adv_s)

        # deviation delta
        delta = self._orig - self._adv
        delta_s = self._orig - self._adv_s

        # distance to the input data
        L1_dist, L2_dist = self._compute_l_norm(delta)
        _, L2_dist_s = self._compute_l_norm(delta_s)

        self._ToEnforceLabel_Score = self._get_label_score(delta, self._adv)
        ToEnforceLabel_Score_s = self._get_label_score(delta_s, self._adv_s)

        # composite distance loss
        self._L2_L1_dist = L2_dist + tf.multiply(L1_dist, beta)

        self._target_lab_score = self._compute_target_lab_score(
            self._target_label, self._ToEnforceLabel_Score
        )
        target_lab_score_s = self._compute_target_lab_score(
            self._target_label, ToEnforceLabel_Score_s
        )

        self._max_nontarget_lab_score = self._compute_non_target_lab_score(
            self._target_label, self._ToEnforceLabel_Score
        )
        max_nontarget_lab_score_s = self._compute_non_target_lab_score(
            self._target_label, ToEnforceLabel_Score_s
        )

        # sum up the losses
        self._Loss_L1Dist = tf.reduce_sum(L1_dist)
        (
            self._Loss_L2Dist,
            self._Loss_Attack,
            self._Loss_AE_Dist,
            Loss_ToOptimize,
        ) = self._compute_losses(
            delta,
            self._adv,
            L2_dist,
            self._max_nontarget_lab_score,
            self._target_lab_score,
            gamma,
        )
        (_, _, _, Loss_ToOptimize_s,) = self._compute_losses(
            delta_s,
            self._adv_s,
            L2_dist_s,
            max_nontarget_lab_score_s,
            target_lab_score_s,
            gamma,
        )
        self._Loss_Overall = Loss_ToOptimize + tf.multiply(beta, self._Loss_L1Dist)

        learning_rate = tf.train.polynomial_decay(
            self._hyperparams["init_learning_rate"],
            self._global_step,
            self._hyperparams["max_iterations"],
            0,
            power=0.5,
        )

        self._train = self._optimization(Loss_ToOptimize_s, self._adv_s, learning_rate)
        start_vars = set(x.name for x in tf.global_variables())
        new_vars = [x for x in tf.global_variables() if x.name not in start_vars]

        # these are the variables to initialize when we run
        self._setup = self._set_setup()

        self._init = tf.variables_initializer(
            var_list=[self._global_step] + [self._adv_s] + [self._adv] + new_vars
        )

    def _optimization(self, Loss_ToOptimize_s, adv_s, learning_rate):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        return optimizer.minimize(
            Loss_ToOptimize_s,
            var_list=[adv_s],
            global_step=self._global_step,
        )

    def _compute_losses(
        self, delta, adv, l2_dist, max_nontarget_lab_score, target_lab_score, gamma
    ):
        Loss_Attack = self._compute_attack_loss(
            max_nontarget_lab_score, target_lab_score, self._mode
        )
        loss_L2Dist = tf.reduce_sum(l2_dist)
        loss_Attack = tf.reduce_sum(self._const * Loss_Attack)
        loss_AE_Dist = self._compute_AE_dist(adv, delta, gamma)
        loss_ToOptimize = loss_Attack + loss_L2Dist + loss_AE_Dist
        return loss_L2Dist, loss_Attack, loss_AE_Dist, loss_ToOptimize

    def _compute_AE_dist(self, adv, delta, gamma):
        delta_input_ae = delta if self._mode == "PP" else adv
        return self._compute_AE_lost(delta_input_ae, gamma)

    def _set_setup(self):
        setup = []
        setup.append(self._orig.assign(self._assign_orig))
        setup.append(self._target_label.assign(self._assign_target_label))
        setup.append(self._const.assign(self._assign_const))
        setup.append(self._adv.assign(self._assign_adv))
        setup.append(self._adv_s.assign(self._assign_adv_s))
        return setup

    def _get_label_score(self, delta, adv):
        enforce_input = delta if self._mode == "PP" else adv
        return self._mlmodel.raw_model(enforce_input)

    def _compute_l_norm(self, delta):
        return tf.reduce_sum(tf.abs(delta), [1]), tf.reduce_sum(tf.square(delta), [1])

    def _load_ae(self, hyperparams, mlmodel):
        ae_params = hyperparams["ae_params"]
        ae = Autoencoder(
            data_name=hyperparams["data_name"],
            layers=[len(mlmodel.feature_input_order)] + ae_params["hidden_layer"],
        )
        if ae_params["train_ae"]:
            return train_autoencoder(
                ae,
                self._mlmodel.data,
                self._mlmodel.scaler,
                self._mlmodel.encoder,
                self._mlmodel.feature_input_order,
                epochs=ae_params["epochs"],
                save=True,
            )
        else:
            try:
                return ae.load(input_shape=len(mlmodel.feature_input_order))
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Loading of Autoencoder failed. {}".format(str(exc))
                )

    def _initialize_tf_variables(self, batch_size, num_classes, shape_batch):
        # these are variables to be more efficient in sending data to tf
        self._orig = tf.Variable(np.zeros(shape_batch), dtype=tf.float32)
        self._adv = tf.Variable(np.zeros(shape_batch), dtype=tf.float32)
        self._adv_s = tf.Variable(np.zeros(shape_batch), dtype=tf.float32)
        self._target_label = tf.Variable(
            np.zeros((batch_size, num_classes)), dtype=tf.float32
        )
        self._const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self._global_step = tf.Variable(0.0, trainable=False)
        # and here's what we use to assign them
        self._assign_orig = tf.placeholder(tf.float32, shape_batch)
        self._assign_adv = tf.placeholder(tf.float32, shape_batch)
        self._assign_adv_s = tf.placeholder(tf.float32, shape_batch)
        self._assign_target_label = tf.placeholder(
            tf.float32, (batch_size, num_classes)
        )
        self._assign_const = tf.placeholder(tf.float32, [batch_size])

    def _compute_target_lab_score(
        self, target_label, label_score: tf.Tensor
    ) -> tf.Tensor:
        return tf.reduce_sum(target_label * label_score, 1)

    def _compute_non_target_lab_score(
        self, target_label, label_score: tf.Tensor
    ) -> tf.Tensor:
        return tf.reduce_max(
            (1 - target_label) * label_score - (target_label * 10000),
            1,
        )

    def _compute_AE_lost(self, delta: tf.Tensor, gamma) -> tf.Tensor:
        return gamma * tf.square(tf.norm(self._AE(delta) - delta))

    def _compute_attack_loss(
        self, nontarget_label_score: tf.Tensor, target_label_score: tf.Tensor, mode: str
    ) -> tf.Tensor:
        sign = 1 if mode == "PP" else -1
        return tf.maximum(
            0.0,
            (sign * nontarget_label_score) - (sign * target_label_score) + self._kappa,
        )

    def _compute_with_mode(self, assign_adv_s, orig) -> tf.Tensor:
        # x^CF.s := assigned adv s
        # cond greater      -- x^CF.s - x^F > 0
        # cond less equal   -- x^CF.s - x^F =< 0
        cond_greater, cond_less_equal, _ = self._get_conditions(assign_adv_s, orig)
        if self._mode == "PP":
            assign_adv_s = tf.multiply(cond_less_equal, assign_adv_s) + tf.multiply(
                cond_greater, orig
            )
        elif self._mode == "PN":
            assign_adv_s = tf.multiply(cond_greater, assign_adv_s) + tf.multiply(
                cond_less_equal, orig
            )
        return assign_adv_s

    def _compute_adv_s(self, zt, orig, adv, assign_adv) -> tf.Tensor:
        assign_adv_s = assign_adv + tf.multiply(zt, assign_adv - adv)
        return self._compute_with_mode(assign_adv_s, orig)

    def _compute_adv(self, orig, adv_s, beta: float) -> tf.Tensor:
        cond_greater, cond_less_equal, cond_less = self._get_conditions(
            adv_s, orig, beta
        )
        # lower -- min(x^CF - beta, 0.5)
        # upper -- max(x^CF + beta, -0.5)
        upper = tf.minimum(tf.subtract(adv_s, beta), tf.cast(0.5, tf.float32))
        lower = tf.maximum(tf.add(adv_s, beta), tf.cast(-0.5, tf.float32))
        assign_adv = (
            tf.multiply(cond_greater, upper)
            + tf.multiply(cond_less_equal, orig)
            + tf.multiply(cond_less, lower)
        )

        return self._compute_with_mode(assign_adv, orig)

    def _get_conditions(
        self, adv: Union[tf.Tensor, tf.Variable], orig, beta: float = 0.0
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # (adv - orig) > beta
        cond_greater = tf.cast(
            tf.greater(tf.subtract(adv, orig), beta),
            tf.float32,
        )
        # (_adv - orig) <= beta
        cond_less_equal = tf.cast(
            tf.less_equal(tf.abs(tf.subtract(adv, orig)), beta),
            tf.float32,
        )
        # (_adv - orig) < beta
        cond_less = tf.cast(
            tf.less(tf.subtract(adv, orig), tf.negative(beta)),
            tf.float32,
        )
        return cond_greater, cond_less_equal, cond_less

    def _attack(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        def compare(x, y) -> bool:
            """
            Compare predictions with target labels and return whether PP or PN conditions hold.

            Parameters
            ----------
            x
                Predicted class probabilities or labels
            y
                Target or predicted labels

            Returns
            -------
            Bool whether PP or PN conditions hold.
            """
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self._mode == "PP":
                    x[y] -= self._kappa  # type:ignore
                elif self._mode == "PN":
                    x[y] += self._kappa  # type:ignore
                x = np.argmax(x)  # type:ignore
            if self._mode == "PP":
                return x == y
            else:
                return x != y

        batch_size = self._hyperparams["batch_size"]

        # set the lower and upper bounds accordingly
        Const_LB = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self._hyperparams["initial_const"]
        Const_UB = np.ones(batch_size) * 1e10

        # the best l2, score, and image attack
        overall_best_dist = [1e10] * batch_size
        overall_best_attack = np.array([np.zeros(X[0].shape)] * batch_size)

        for _ in range(self._hyperparams["binary_search_steps"]):
            # completely reset adam's internal state.
            self.sess.run(self._init)
            input_batch = X[:batch_size]
            label_batch = Y[:batch_size]

            current_step_best_dist = [1e10] * batch_size
            current_step_best_score = [-1] * batch_size

            # set the variables so that we don't have to send them over again
            self.sess.run(
                self._setup,
                {
                    self._assign_orig: input_batch,
                    self._assign_target_label: label_batch,
                    self._assign_const: CONST,
                    self._assign_adv: input_batch,
                    self._assign_adv_s: input_batch,
                },
            )

            for i in range(self._hyperparams["max_iterations"]):
                # perform the attack
                self.sess.run([self._train])
                self.sess.run([self._adv_updater, self._adv_updater_s])

                Loss_Overall, Loss_L2_L1_dist, OutputScore, adv = self.sess.run(
                    [
                        self._Loss_Overall,
                        self._L2_L1_dist,
                        self._ToEnforceLabel_Score,
                        self._adv,
                    ]
                )

                for batch_idx, (the_dist, the_score, the_adv) in enumerate(
                    zip(Loss_L2_L1_dist, OutputScore, adv)
                ):
                    if the_dist < current_step_best_dist[batch_idx] and compare(
                        the_score, np.argmax(label_batch[batch_idx])
                    ):
                        current_step_best_dist[batch_idx] = the_dist
                        current_step_best_score[batch_idx] = np.argmax(the_score)
                    if the_dist < overall_best_dist[batch_idx] and compare(
                        the_score, np.argmax(label_batch[batch_idx])
                    ):
                        overall_best_dist[batch_idx] = the_dist
                        overall_best_attack[batch_idx] = the_adv

            # adjust the constant as needed
            for batch_idx in range(batch_size):
                if (
                    compare(
                        current_step_best_score[batch_idx],
                        np.argmax(label_batch[batch_idx]),
                    )
                    and current_step_best_score[batch_idx] != -1
                ):
                    # success, divide const by two
                    Const_UB[batch_idx] = min(Const_UB[batch_idx], CONST[batch_idx])
                    if Const_UB[batch_idx] < 1e9:
                        CONST[batch_idx] = (
                            Const_LB[batch_idx] + Const_UB[batch_idx]
                        ) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    Const_LB[batch_idx] = max(Const_LB[batch_idx], CONST[batch_idx])
                    if Const_UB[batch_idx] < 1e9:
                        CONST[batch_idx] = (
                            Const_LB[batch_idx] + Const_UB[batch_idx]
                        ) / 2
                    else:
                        CONST[batch_idx] *= 10

        # return the best solution found
        overall_best_attack = np.array(overall_best_attack[0])
        overall_best_attack = overall_best_attack.reshape(
            (1,) + overall_best_attack.shape
        )
        return overall_best_attack

    def _counterfactual_search(self, instance: np.ndarray) -> np.ndarray:
        def model_prediction(model, inputs: np.ndarray) -> Tuple[np.ndarray, int, str]:
            prob = model.predict(inputs)
            predicted_class = np.argmax(prob)
            prob_str = np.array2string(prob).replace("\n", "")
            return prob, predicted_class, prob_str

        def generate_data(instance, target_label) -> Tuple[np.ndarray, np.ndarray]:
            inputs, target_vec = [], []
            inputs.append(instance)
            # 2: since we only look at binary classification
            target_vec.append(np.eye(2)[target_label])
            inputs = np.array(inputs)
            target_vec = np.array(target_vec)
            return inputs, target_vec

        orig_prob, orig_class, orig_prob_str = model_prediction(
            self._mlmodel.raw_model, np.expand_dims(instance, axis=0)
        )
        orig_sample, target = generate_data(instance, target_label=orig_class)
        # start the search
        counterfactual = self._attack(orig_sample, target)
        return counterfactual.reshape(-1)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a certain number of counterfactuals per factual example.


        Parameters
        ----------
        factuals : pd.DataFrame
            DataFrame containing all samples for which we want to generate counterfactual examples.
            All instances should belong to the same class.

        Returns
        -------

        """
        # normalize and one-hot-encoding
        df_enc_norm_fact = self.encode_normalize_order_factuals(factuals)
        df_enc_norm_fact = df_enc_norm_fact.reset_index(drop=True)

        # find counterfactuals
        df_cfs = df_enc_norm_fact.apply(
            lambda x: self._counterfactual_search(x), axis=1, raw=True
        )
        df_cfs = check_counterfactuals(self._mlmodel, df_cfs)
        return df_cfs
