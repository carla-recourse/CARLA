import timeit

import numpy as np
import pandas as pd
import tensorflow as tf

from ....models.catalog.catalog import MLModelCatalog
from ...api import RecourseMethod


# TODO helper function in utils?
def generate_data(instance, target_label):
    inputs = []
    target_vec = []

    inputs.append(instance)
    target_vec.append(
        np.eye(2)[target_label]
    )  # 2: since we only look at binary classification

    inputs = np.array(inputs)
    target_vec = np.array(target_vec)

    return inputs, target_vec


def model_prediction(model, inputs):
    prob = model.model.predict(inputs)
    predicted_class = np.argmax(prob)
    prob_str = np.array2string(prob).replace("\n", "")
    return prob, predicted_class, prob_str


# TODO helper function in measures?
def success_rate_and_indices(counterfactuals_df):
    """
    Used to indicate which counterfactuals should be dropped (due to lack of success indicated by NaN).
    Also computes percent of successfully found counterfactuals
    :param counterfactuals_df: pd df, where NaNs indicate 'no counterfactual found' [df should contain no object values)
    :return: success_rate, indices
    """

    # Success rate & drop not successful counterfactuals & process remainder
    success_rate = (counterfactuals_df.dropna().shape[0]) / counterfactuals_df.shape[0]
    counterfactual_indices = np.where(
        # np.any(np.isnan(counterfactuals_df.values) == True, axis=1) == False
        not np.any(np.isnan(counterfactuals_df.values), axis=1)
    )[0]

    return success_rate, counterfactual_indices


class CEM(RecourseMethod):
    def __init__(
        self,
        sess,
        catalog_model: MLModelCatalog,
        model,
        mode,
        AE,
        batch_size,
        kappa,
        init_learning_rate,
        binary_search_steps,
        max_iterations,
        initial_const,
        beta,
        gamma,
        num_classes=2,
    ):

        # TODO refactor names from img to more general
        dimension = len(catalog_model.feature_input_order)
        shape = (batch_size, dimension)

        self.catalog_model = catalog_model
        self.model = model
        self.data = catalog_model.data

        self.sess = sess
        self.INIT_LEARNING_RATE = init_learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.kappa = kappa
        self.init_const = initial_const
        self.batch_size = batch_size
        self.AE = AE
        self.mode = mode
        self.beta = beta
        self.gamma = gamma

        # these are variables to be more efficient in sending data to tf
        self.orig_img = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.adv_img = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.adv_img_s = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.target_lab = tf.Variable(
            np.zeros((batch_size, num_classes)), dtype=tf.float32
        )
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self.global_step = tf.Variable(0.0, trainable=False)

        # and here's what we use to assign them
        self.assign_orig_img = tf.placeholder(tf.float32, shape)
        self.assign_adv_img = tf.placeholder(tf.float32, shape)
        self.assign_adv_img_s = tf.placeholder(tf.float32, shape)
        self.assign_target_lab = tf.placeholder(tf.float32, (batch_size, num_classes))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])

        """Fast Iterative Soft Thresholding"""
        """--------------------------------"""
        # Commented by us:
        # BEGIN: conditions to compute the ell1 regularization
        # this should be the function S_beta(z) in the paper

        self.zt = tf.divide(self.global_step, self.global_step + tf.cast(3, tf.float32))

        # cond 1 -- x^CF - x^F > beta
        # cond 2 -- |x^CF - x^F| =< beta
        # cond 3 -- x^CF - x^F < -beta

        cond1 = tf.cast(
            tf.greater(tf.subtract(self.adv_img_s, self.orig_img), self.beta),
            tf.float32,
        )
        cond2 = tf.cast(
            tf.less_equal(
                tf.abs(tf.subtract(self.adv_img_s, self.orig_img)), self.beta
            ),
            tf.float32,
        )
        cond3 = tf.cast(
            tf.less(tf.subtract(self.adv_img_s, self.orig_img), tf.negative(self.beta)),
            tf.float32,
        )

        # lower -- min(x^CF - beta, 0.5)
        # upper -- max(x^CF + beta, -0.5)

        upper = tf.minimum(
            tf.subtract(self.adv_img_s, self.beta), tf.cast(0.5, tf.float32)
        )
        lower = tf.maximum(tf.add(self.adv_img_s, self.beta), tf.cast(-0.5, tf.float32))

        self.assign_adv_img = (
            tf.multiply(cond1, upper)
            + tf.multiply(cond2, self.orig_img)
            + tf.multiply(cond3, lower)
        )

        # x^CF. := assigned adv img
        # cond 4 -- x^CF. - x^F > 0
        # cond 5 -- x^CF. - x^F =< 0

        cond4 = tf.cast(
            tf.greater(tf.subtract(self.assign_adv_img, self.orig_img), 0), tf.float32
        )
        cond5 = tf.cast(
            tf.less_equal(tf.subtract(self.assign_adv_img, self.orig_img), 0),
            tf.float32,
        )

        if self.mode == "PP":
            self.assign_adv_img = tf.multiply(cond5, self.assign_adv_img) + tf.multiply(
                cond4, self.orig_img
            )
        elif self.mode == "PN":
            self.assign_adv_img = tf.multiply(cond4, self.assign_adv_img) + tf.multiply(
                cond5, self.orig_img
            )

        self.assign_adv_img_s = self.assign_adv_img + tf.multiply(
            self.zt, self.assign_adv_img - self.adv_img
        )

        # x^CF.s := assigned adv img s
        # cond 6 -- x^CF.s - x^F > 0
        # cond 7 -- x^CF.s - x^F =< 0

        cond6 = tf.cast(
            tf.greater(tf.subtract(self.assign_adv_img_s, self.orig_img), 0), tf.float32
        )
        cond7 = tf.cast(
            tf.less_equal(tf.subtract(self.assign_adv_img_s, self.orig_img), 0),
            tf.float32,
        )

        if self.mode == "PP":
            self.assign_adv_img_s = tf.multiply(
                cond7, self.assign_adv_img_s
            ) + tf.multiply(cond6, self.orig_img)
        elif self.mode == "PN":
            self.assign_adv_img_s = tf.multiply(
                cond6, self.assign_adv_img_s
            ) + tf.multiply(cond7, self.orig_img)

        self.adv_updater = tf.assign(self.adv_img, self.assign_adv_img)
        self.adv_updater_s = tf.assign(self.adv_img_s, self.assign_adv_img_s)

        # END: conditions to compute the ell1 regularization
        """--------------------------------"""

        # delta_img := delta^k+1
        # delta_ims_s := y^k+1 (slack variable to account for momentum acceleration)

        # prediction BEFORE-SOFTMAX of the model
        self.delta_img = self.orig_img - self.adv_img
        self.delta_img_s = self.orig_img - self.adv_img_s

        # TODO check model.predict gets correct input
        if self.mode == "PP":
            self.ImgToEnforceLabel_Score = model.predict(self.delta_img)
            self.ImgToEnforceLabel_Score_s = model.predict(self.delta_img_s)
        elif self.mode == "PN":
            self.ImgToEnforceLabel_Score = model.predict(self.adv_img)
            self.ImgToEnforceLabel_Score_s = model.predict(self.adv_img_s)

        # distance to the input data
        """ # use this way in combination with pictures and convolutions
        self.L2_dist = tf.reduce_sum(tf.square(self.delta_img), [1, 2, 3])
        self.L2_dist_s = tf.reduce_sum(tf.square(self.delta_img_s), [1, 2, 3])
        self.L1_dist = tf.reduce_sum(tf.abs(self.delta_img), [1, 2, 3])
        self.L1_dist_s = tf.reduce_sum(tf.abs(self.delta_img_s), [1, 2, 3])
        """

        self.L2_dist = tf.reduce_sum(tf.square(self.delta_img), [1])
        self.L2_dist_s = tf.reduce_sum(tf.square(self.delta_img_s), [1])

        self.L1_dist = tf.reduce_sum(tf.abs(self.delta_img), [1])
        self.L1_dist_s = tf.reduce_sum(tf.abs(self.delta_img_s), [1])

        # composite distance loss
        self.EN_dist = self.L2_dist + tf.multiply(self.L1_dist, self.beta)
        self.EN_dist_s = self.L2_dist_s + tf.multiply(self.L1_dist_s, self.beta)

        # compute the probability of the label class versus the maximum other
        self.target_lab_score = tf.reduce_sum(
            self.target_lab * self.ImgToEnforceLabel_Score, 1
        )
        target_lab_score_s = tf.reduce_sum(
            self.target_lab * self.ImgToEnforceLabel_Score_s, 1
        )

        self.max_nontarget_lab_score = tf.reduce_max(
            (1 - self.target_lab) * self.ImgToEnforceLabel_Score
            - (self.target_lab * 10000),
            1,
        )
        max_nontarget_lab_score_s = tf.reduce_max(
            (1 - self.target_lab) * self.ImgToEnforceLabel_Score_s
            - (self.target_lab * 10000),
            1,
        )

        if self.mode == "PP":
            Loss_Attack = tf.maximum(
                0.0, self.max_nontarget_lab_score - self.target_lab_score + self.kappa
            )
            Loss_Attack_s = tf.maximum(
                0.0, max_nontarget_lab_score_s - target_lab_score_s + self.kappa
            )
        elif self.mode == "PN":
            Loss_Attack = tf.maximum(
                0.0, -self.max_nontarget_lab_score + self.target_lab_score + self.kappa
            )
            Loss_Attack_s = tf.maximum(
                0.0, -max_nontarget_lab_score_s + target_lab_score_s + self.kappa
            )

        # sum up the losses
        self.Loss_L1Dist = tf.reduce_sum(self.L1_dist)
        self.Loss_L1Dist_s = tf.reduce_sum(self.L1_dist_s)

        self.Loss_L2Dist = tf.reduce_sum(self.L2_dist)
        self.Loss_L2Dist_s = tf.reduce_sum(self.L2_dist_s)

        self.Loss_Attack = tf.reduce_sum(self.const * Loss_Attack)
        self.Loss_Attack_s = tf.reduce_sum(self.const * Loss_Attack_s)

        if self.mode == "PP":
            self.Loss_AE_Dist = self.gamma * tf.square(
                tf.norm(self.AE(self.delta_img) - self.delta_img)
            )
            self.Loss_AE_Dist_s = self.gamma * tf.square(
                tf.norm(self.AE(self.delta_img) - self.delta_img_s)
            )
        elif self.mode == "PN":
            self.Loss_AE_Dist = self.gamma * tf.square(
                tf.norm(self.AE(self.adv_img) - self.adv_img)
            )
            self.Loss_AE_Dist_s = self.gamma * tf.square(
                tf.norm(self.AE(self.adv_img_s) - self.adv_img_s)
            )

        self.Loss_ToOptimize = (
            self.Loss_Attack_s + self.Loss_L2Dist_s + self.Loss_AE_Dist_s
        )
        self.Loss_Overall = (
            self.Loss_Attack
            + self.Loss_L2Dist
            + self.Loss_AE_Dist
            + tf.multiply(self.beta, self.Loss_L1Dist)
        )

        self.learning_rate = tf.train.polynomial_decay(
            self.INIT_LEARNING_RATE, self.global_step, self.MAX_ITERATIONS, 0, power=0.5
        )
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        start_vars = set(x.name for x in tf.global_variables())
        self.train = optimizer.minimize(
            self.Loss_ToOptimize,
            var_list=[self.adv_img_s],
            global_step=self.global_step,
        )
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.orig_img.assign(self.assign_orig_img))
        self.setup.append(self.target_lab.assign(self.assign_target_lab))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.adv_img.assign(self.assign_adv_img))
        self.setup.append(self.adv_img_s.assign(self.assign_adv_img_s))

        self.init = tf.variables_initializer(
            var_list=[self.global_step] + [self.adv_img_s] + [self.adv_img] + new_vars
        )

    def attack(self, X, Y):
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

        batch_size = self.batch_size

        # set the lower and upper bounds accordingly
        Const_LB = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.init_const
        Const_UB = np.ones(batch_size) * 1e10

        # the best l2, score, and image attack
        overall_best_dist = [1e10] * batch_size
        overall_best_attack = [np.zeros(X[0].shape)] * batch_size

        for binary_search_steps_idx in range(self.BINARY_SEARCH_STEPS):
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

            for iteration in range(self.MAX_ITERATIONS):
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
                Loss_Attack, Loss_L2Dist, Loss_L1Dist, Loss_AE_Dist = self.sess.run(
                    [
                        self.Loss_Attack,
                        self.Loss_L2Dist,
                        self.Loss_L1Dist,
                        self.Loss_AE_Dist,
                    ]
                )
                target_lab_score, max_nontarget_lab_score_s = self.sess.run(
                    [self.target_lab_score, self.max_nontarget_lab_score]
                )

                """
                if iteration%(self.MAX_ITERATIONS//10) == 0:
                    print("iter:{} const:{}". format(iteration, CONST))
                    print("Loss_Overall:{:.4f}, Loss_Attack:{:.4f}". format(Loss_Overall, Loss_Attack))
                    print("Loss_L2Dist:{:.4f}, Loss_L1Dist:{:.4f}, AE_loss:{}". format(Loss_L2Dist, Loss_L1Dist, Loss_AE_Dist))
                    print("target_lab_score:{:.4f}, max_nontarget_lab_score:{:.4f}". format(target_lab_score[0], max_nontarget_lab_score_s[0]))
                    print("")
                    sys.stdout.flush()
                """

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
        overall_best_attack = overall_best_attack[0]
        return overall_best_attack.reshape((1,) + overall_best_attack.shape)

    def counterfactual_search(self, instance):
        # # load the generation model: AE
        # if data_name == 'adult':
        #     dataset_filename = dataset_filename.split('.')[0]
        #     AE_model = util.load_AE(dataset_filename)
        #
        # elif data_name == 'compas':
        #     dataset_filename = dataset_filename.split('.')[0]
        #     AE_model = util.load_AE(dataset_filename)
        #
        # elif data_name == 'give-me':
        #     dataset_filename = dataset_filename.split('.')[0]
        #     AE_model = util.load_AE(dataset_filename)

        orig_prob, orig_class, orig_prob_str = model_prediction(
            self.model, np.expand_dims(instance, axis=0)
        )

        target_label = orig_class
        orig_sample, target = generate_data(instance, target_label)

        # start the search
        counterfactual = self.attack(orig_sample, target)

        adv_prob, adv_class, adv_prob_str = model_prediction(self.model, counterfactual)
        delta_prob, delta_class, delta_prob_str = model_prediction(
            self.model, orig_sample - counterfactual
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

        if np.argmax(self.model.model.predict(instance.reshape(1, -1))) != np.argmax(
            self.model.model.predict(counterfactual.reshape(1, -1))
        ):
            counterfactual = counterfactual
        else:
            counterfactual = counterfactual
            counterfactual[:] = np.nan

        return instance, counterfactual.reshape(-1)

    def get_counterfactuals(self, factuals: pd.DataFrame):
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
        target_name = self.data.target
        instances = factuals.drop(columns=[target_name])

        # normalize
        # TODO robust_binarization
        instances = self.catalog_model.perform_pipeline(instances)

        counterfactuals = []
        times_list = []

        for i in range(instances.values.shape[0]):
            start = timeit.default_timer()
            _, counterfactual = self.counterfactual_search(instances.values[i, :])
            stop = timeit.default_timer()
            time_taken = stop - start

            counterfactuals.append(counterfactual)
            times_list.append(time_taken)

        counterfactuals_df = pd.DataFrame(np.array(counterfactuals))
        counterfactuals_df.columns = instances.columns

        # Success rate & drop not successful counterfactuals & process remainder
        success_rate, counterfactuals_indices = success_rate_and_indices(
            counterfactuals_df
        )
        counterfactuals_df = counterfactuals_df.iloc[counterfactuals_indices]
        instances = instances.iloc[counterfactuals_indices]

        # Obtain labels
        instance_label = np.argmax(self.catalog_model.predict(instances.values), axis=1)
        counterfactual_label = np.argmax(
            self.catalog_model.predict(counterfactuals_df.values), axis=1
        )

        # TODO binary cols?
        binary_cols = self.data.categoricals
        # Round binary columns to integer
        counterfactuals_df[binary_cols] = (
            counterfactuals_df[binary_cols].round(0).astype(int)
        )

        # Order counterfactuals and instances in original data order
        counterfactuals_df = counterfactuals_df[self.data.columns]
        instances = instances[self.data.columns]

        if len(binary_cols) > 0:
            # Convert binary cols of counterfactuals and instances into strings: Required for >>Measurement<< in script
            counterfactuals_df[binary_cols] = counterfactuals_df[binary_cols].astype(
                "string"
            )
            instances[binary_cols] = instances[binary_cols].astype("string")

            # Convert binary cols back to original string encoding
            # TODO scipy solution should be used here?
            # counterfactuals_df = preprocessing.map_binary_backto_string(
            #     self.data, counterfactuals_df, binary_cols
            # )
            # instances = preprocessing.map_binary_backto_string(
            #     self.data, instances, binary_cols
            # )

        # Add labels
        counterfactuals_df[target_name] = counterfactual_label
        instances[target_name] = instance_label

        # Collect in list making use of pandas
        instances_list = []
        counterfactuals_list = []

        for i in range(counterfactuals_df.shape[0]):
            counterfactuals_list.append(
                pd.DataFrame(
                    counterfactuals_df.iloc[i].values.reshape((1, -1)),
                    columns=counterfactuals_df.columns,
                )
            )
            instances_list.append(
                pd.DataFrame(
                    instances.iloc[i].values.reshape((1, -1)), columns=instances.columns
                )
            )

        return instances_list, counterfactuals_list, times_list, success_rate
