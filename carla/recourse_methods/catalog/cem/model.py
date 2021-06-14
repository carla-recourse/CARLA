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


class CEM(RecourseMethod):
    def __init__(self, sess, mlmodel: MLModel, hyperparams):
        self.sess = sess
        self.hyperparams = hyperparams

        self.data = mlmodel.data
        self.kappa = hyperparams["kappa"]
        self.mode = hyperparams["mode"]

        batch_size = hyperparams["batch_size"]
        num_classes = hyperparams["num_classes"]
        beta = hyperparams["beta"]

        super().__init__(mlmodel)
        shape_batch = (batch_size, len(mlmodel.feature_input_order))

        self.AE = self.__load_ae(hyperparams, mlmodel)

        self.__initialize_tf_variables(batch_size, num_classes, shape_batch)

        """Fast Iterative Soft Thresholding"""
        """--------------------------------"""
        # Commented by us:
        # BEGIN: conditions to compute the ell1 regularization
        # this should be the function S_beta(z) in the paper
        if self.mode not in ["PP", "PN"]:
            raise ValueError("Mode not known, please use either PP or PN")

        zt = tf.divide(self.global_step, self.global_step + tf.cast(3, tf.float32))

        self.assign_adv_img = self.__compute_adv_img(beta)
        self.assign_adv_img_s = self.__compute_adv_img_s(zt)

        self.adv_updater = tf.assign(self.adv_img, self.assign_adv_img)
        self.adv_updater_s = tf.assign(self.adv_img_s, self.assign_adv_img_s)

        delta_img = self.orig_img - self.adv_img
        delta_img_s = self.orig_img - self.adv_img_s

        # distance to the input data
        L1_dist, L2_dist = self.__compute_l_norm(delta_img)
        _, L2_dist_s = self.__compute_l_norm(delta_img_s)

        self.ImgToEnforceLabel_Score = self.__get_label_score(delta_img, self.adv_img)
        ImgToEnforceLabel_Score_s = self.__get_label_score(delta_img_s, self.adv_img_s)

        # composite distance loss
        self.EN_dist = L2_dist + tf.multiply(L1_dist, beta)

        self.target_lab_score = self.__compute_target_lab_score(
            self.ImgToEnforceLabel_Score
        )
        target_lab_score_s = self.__compute_target_lab_score(ImgToEnforceLabel_Score_s)

        self.max_nontarget_lab_score = self.__compute_non_target_lab_score(
            self.ImgToEnforceLabel_Score
        )
        max_nontarget_lab_score_s = self.__compute_non_target_lab_score(
            ImgToEnforceLabel_Score_s
        )

        # sum up the losses
        self.Loss_L1Dist = tf.reduce_sum(L1_dist)
        (
            self.Loss_L2Dist,
            self.Loss_Attack,
            self.Loss_AE_Dist,
            Loss_ToOptimize,
        ) = self.__compute_losses(
            delta_img,
            self.adv_img,
            L2_dist,
            self.max_nontarget_lab_score,
            self.target_lab_score,
        )
        (_, _, _, Loss_ToOptimize_s,) = self.__compute_losses(
            delta_img_s,
            self.adv_img_s,
            L2_dist_s,
            max_nontarget_lab_score_s,
            target_lab_score_s,
        )
        self.Loss_Overall = Loss_ToOptimize + tf.multiply(beta, self.Loss_L1Dist)

        learning_rate = tf.train.polynomial_decay(
            hyperparams["init_learning_rate"],
            self.global_step,
            hyperparams["max_iterations"],
            0,
            power=0.5,
        )

        self.train = self.__optimization(Loss_ToOptimize_s, learning_rate)
        start_vars = set(x.name for x in tf.global_variables())
        new_vars = [x for x in tf.global_variables() if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = self.__set_setup()

        self.init = tf.variables_initializer(
            var_list=[self.global_step] + [self.adv_img_s] + [self.adv_img] + new_vars
        )

    def __optimization(self, Loss_ToOptimize_s, learning_rate):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        return optimizer.minimize(
            Loss_ToOptimize_s,
            var_list=[self.adv_img_s],
            global_step=self.global_step,
        )

    def __compute_losses(
        self, delta, adv, l2_dist, max_nontarget_lab_score, target_lab_score
    ):
        Loss_Attack = self.__compute_attack_loss(
            max_nontarget_lab_score, target_lab_score, self.mode
        )
        loss_L2Dist = tf.reduce_sum(l2_dist)
        loss_Attack = tf.reduce_sum(self.const * Loss_Attack)
        loss_AE_Dist = self.__compute_AE_dist(delta, adv)
        loss_ToOptimize = loss_Attack + loss_L2Dist + loss_AE_Dist

        return loss_L2Dist, loss_Attack, loss_AE_Dist, loss_ToOptimize

    def __compute_AE_dist(self, delta, adv):
        delta_input_ae = delta if self.mode == "PP" else adv
        return self.__compute_AE_lost(delta_input_ae)

    def __set_setup(self):
        setup = []
        setup.append(self.orig_img.assign(self.assign_orig_img))
        setup.append(self.target_lab.assign(self.assign_target_lab))
        setup.append(self.const.assign(self.assign_const))
        setup.append(self.adv_img.assign(self.assign_adv_img))
        setup.append(self.adv_img_s.assign(self.assign_adv_img_s))

        return setup

    def __get_label_score(self, delta, adv):
        enforce_input = delta if self.mode == "PP" else adv
        return self._mlmodel.raw_model(enforce_input)

    def __compute_l_norm(self, delta):
        return tf.reduce_sum(tf.abs(delta), [1]), tf.reduce_sum(tf.square(delta), [1])

    def __load_ae(self, hyperparams, mlmodel):
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

    def __initialize_tf_variables(self, batch_size, num_classes, shape_batch):
        # these are variables to be more efficient in sending data to tf
        self.orig_img = tf.Variable(np.zeros(shape_batch), dtype=tf.float32)
        self.adv_img = tf.Variable(np.zeros(shape_batch), dtype=tf.float32)
        self.adv_img_s = tf.Variable(np.zeros(shape_batch), dtype=tf.float32)
        self.target_lab = tf.Variable(
            np.zeros((batch_size, num_classes)), dtype=tf.float32
        )
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self.global_step = tf.Variable(0.0, trainable=False)
        # and here's what we use to assign them
        self.assign_orig_img = tf.placeholder(tf.float32, shape_batch)
        self.assign_adv_img = tf.placeholder(tf.float32, shape_batch)
        self.assign_adv_img_s = tf.placeholder(tf.float32, shape_batch)
        self.assign_target_lab = tf.placeholder(tf.float32, (batch_size, num_classes))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])

    def __compute_target_lab_score(self, label_score: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(self.target_lab * label_score, 1)

    def __compute_non_target_lab_score(self, label_score: tf.Tensor) -> tf.Tensor:
        return tf.reduce_max(
            (1 - self.target_lab) * label_score - (self.target_lab * 10000),
            1,
        )

    def __compute_AE_lost(self, delta_img: tf.Tensor) -> tf.Tensor:
        return self.hyperparams["gamma"] * tf.square(
            tf.norm(self.AE(delta_img) - delta_img)
        )

    def __compute_attack_loss(
        self, nontarget_lab_score: tf.Tensor, target_lab_score: tf.Tensor, mode: str
    ) -> tf.Tensor:
        sign = 1 if mode == "PP" else -1

        return tf.maximum(
            0.0, (sign * nontarget_lab_score) - (sign * target_lab_score) + self.kappa
        )

    def __compute_img_with_mode(self, assign_adv_img_s) -> tf.Tensor:
        # x^CF.s := assigned adv img s
        # cond 6 -- x^CF.s - x^F > 0
        # cond 7 -- x^CF.s - x^F =< 0
        cond6, cond7, _ = self.__get_conditions(assign_adv_img_s)
        if self.mode == "PP":
            assign_adv_img_s = tf.multiply(cond7, assign_adv_img_s) + tf.multiply(
                cond6, self.orig_img
            )
        elif self.mode == "PN":
            assign_adv_img_s = tf.multiply(cond6, assign_adv_img_s) + tf.multiply(
                cond7, self.orig_img
            )
        return assign_adv_img_s

    def __compute_adv_img_s(self, zt) -> tf.Tensor:
        assign_adv_img_s = self.assign_adv_img + tf.multiply(
            zt, self.assign_adv_img - self.adv_img
        )
        return self.__compute_img_with_mode(assign_adv_img_s)

    def __compute_adv_img(self, beta: float) -> tf.Tensor:
        cond1, cond2, cond3 = self.__get_conditions(self.adv_img_s, beta)
        # lower -- min(x^CF - beta, 0.5)
        # upper -- max(x^CF + beta, -0.5)
        upper = tf.minimum(tf.subtract(self.adv_img_s, beta), tf.cast(0.5, tf.float32))
        lower = tf.maximum(tf.add(self.adv_img_s, beta), tf.cast(-0.5, tf.float32))
        assign_adv_img = (
            tf.multiply(cond1, upper)
            + tf.multiply(cond2, self.orig_img)
            + tf.multiply(cond3, lower)
        )

        return self.__compute_img_with_mode(assign_adv_img)

    def __get_conditions(
        self, adv_img: Union[tf.Tensor, tf.Variable], beta: float = 0.0
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        cond_greater = tf.cast(
            tf.greater(tf.subtract(adv_img, self.orig_img), beta),
            tf.float32,
        )
        cond_less_equal = tf.cast(
            tf.less_equal(tf.abs(tf.subtract(adv_img, self.orig_img)), beta),
            tf.float32,
        )
        cond_less = tf.cast(
            tf.less(tf.subtract(adv_img, self.orig_img), tf.negative(beta)),
            tf.float32,
        )

        return cond_greater, cond_less_equal, cond_less

    def attack(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
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
                if self.mode == "PP":
                    x[y] -= self.kappa  # type:ignore
                elif self.mode == "PN":
                    x[y] += self.kappa  # type:ignore
                x = np.argmax(x)  # type:ignore
            if self.mode == "PP":
                return x == y
            else:
                return x != y

        batch_size = self.hyperparams["batch_size"]

        # set the lower and upper bounds accordingly
        Const_LB = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.hyperparams["initial_const"]
        Const_UB = np.ones(batch_size) * 1e10

        # the best l2, score, and image attack
        overall_best_dist = [1e10] * batch_size
        overall_best_attack = np.array([np.zeros(X[0].shape)] * batch_size)

        for binary_search_steps_idx in range(self.hyperparams["binary_search_steps"]):
            # completely reset adam's internal state.
            self.sess.run(self.init)
            img_batch = X[:batch_size]
            label_batch = Y[:batch_size]

            current_step_best_dist = [1e10] * batch_size
            current_step_best_score = [-1] * batch_size

            # set the variables so that we don't have to send them over again
            self.sess.run(
                self.setup,
                {
                    self.assign_orig_img: img_batch,
                    self.assign_target_lab: label_batch,
                    self.assign_const: CONST,
                    self.assign_adv_img: img_batch,
                    self.assign_adv_img_s: img_batch,
                },
            )

            for iteration in range(self.hyperparams["max_iterations"]):
                # perform the attack
                self.sess.run([self.train])
                self.sess.run([self.adv_updater, self.adv_updater_s])

                Loss_Overall, Loss_EN, OutputScore, adv_img = self.sess.run(
                    [
                        self.Loss_Overall,
                        self.EN_dist,
                        self.ImgToEnforceLabel_Score,
                        self.adv_img,
                    ]
                )

                for batch_idx, (the_dist, the_score, the_adv_img) in enumerate(
                    zip(Loss_EN, OutputScore, adv_img)
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
                        overall_best_attack[batch_idx] = the_adv_img

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

    def counterfactual_search(
        self, instance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        orig_prob, orig_class, orig_prob_str = self.model_prediction(
            self._mlmodel.raw_model, np.expand_dims(instance, axis=0)
        )

        target_label = orig_class
        orig_sample, target = self.generate_data(instance, target_label)

        # start the search
        counterfactual = self.attack(orig_sample, target)

        adv_prob, adv_class, adv_prob_str = self.model_prediction(
            self._mlmodel.raw_model, counterfactual
        )
        delta_prob, delta_class, delta_prob_str = self.model_prediction(
            self._mlmodel.raw_model, orig_sample - counterfactual
        )

        INFO = "[kappa:{}, Orig class:{}, Adv class:{}, Delta class: {}, Orig prob:{}, Adv prob:{}, Delta prob:{}".format(
            self.kappa,
            orig_class,
            adv_class,
            delta_class,
            orig_prob_str,
            adv_prob_str,
            delta_prob_str,
        )
        print(INFO)

        if np.argmax(
            self._mlmodel.raw_model.predict(instance.reshape(1, -1))
        ) != np.argmax(self._mlmodel.raw_model.predict(counterfactual.reshape(1, -1))):
            counterfactual = counterfactual
        else:
            counterfactual = counterfactual
            counterfactual[:] = np.nan

        return instance, counterfactual.reshape(-1)

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
        # drop targets
        instances = factuals.copy()
        instances = instances.reset_index(drop=True)

        # normalize and one-hot-encoding
        instances = self.encode_normalize_order_factuals(instances)

        counterfactuals = []

        for i, row in instances.iterrows():
            _, counterfactual = self.counterfactual_search(instances.values[i, :])
            counterfactuals.append(counterfactual)

        counterfactuals_df = check_counterfactuals(self._mlmodel, counterfactuals)

        return counterfactuals_df

    @staticmethod
    def generate_data(
        instance: np.ndarray, target_label: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        inputs = []
        target_vec = []

        inputs.append(instance)
        target_vec.append(
            np.eye(2)[target_label]
        )  # 2: since we only look at binary classification

        inputs = np.array(inputs)
        target_vec = np.array(target_vec)

        return inputs, target_vec

    @staticmethod
    def model_prediction(model, inputs: np.ndarray) -> Tuple[np.ndarray, int, str]:
        prob = model.predict(inputs)
        predicted_class = np.argmax(prob)
        prob_str = np.array2string(prob).replace("\n", "")
        return prob, predicted_class, prob_str
