import tensorflow as tf
from keras import activations
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.utils import to_categorical


def weighted_binary_cross_entropy(target, output):
    loss = 0.7 * (target * tf.log(output)) + 0.3 * ((1 - target) * tf.log(1 - output))
    return tf.negative(tf.reduce_mean(loss, axis=-1))


class AnnModel:
    def __init__(
        self,
        dim_input,
        dim_hidden_layer1,
        dim_hidden_layer2,
        dim_output_layer,
        num_of_classes,
        data_name,
        restore=None,
        use_prob=False,
    ):

        # For model loading
        """
        :param dim_input: int > 0, number of neurons for this layer
        :param dim_hidden_layer_1: int > 0, number of neurons for this layer
        :param dim_hidden_layer_2: int > 0, number of neurons for this layer
        :param dim_output_layer: int > 0, number of neurons for this layer
        :param num_of_classes: int > 0, number of classes
        :param use_prob: boolean; FALSE required for CEM; all others should use True
        """  #

        self.data_name = data_name
        self.dim_input = dim_input
        self.dim_hidden_layer1 = dim_hidden_layer1
        self.dim_hidden_layer2 = dim_hidden_layer2
        self.dim_output_layer = dim_output_layer
        self.num_of_classes = num_of_classes

        model = Sequential(
            [
                Dense(
                    self.dim_hidden_layer1, input_dim=self.dim_input, activation="relu"
                ),
                Dense(self.dim_hidden_layer2, activation="relu"),
                Dense(self.dim_output_layer, activation="relu"),
                Dense(self.num_of_classes),
            ]
        )

        # whether to output probability
        if use_prob:
            model.add(Activation(activations.softmax))
        if restore:
            model.load_weights(restore)
            model.summary()

        self.model = model

    def __call__(self, data):
        return self.predict(data)

    def predict(self, data):
        return self.model(data)

    def build_train_save_model(
        self,
        xtrain,
        ytrain,
        xtest,
        ytest,
        learning_rate,
        epochs,
        batch_size,
        model_name="ann_tf",
        model_directory="saved_models",
    ):
        model = Sequential(
            [
                Dense(
                    self.dim_hidden_layer1, input_dim=self.dim_input, activation="relu"
                ),
                Dense(self.dim_hidden_layer2, activation="relu"),
                Dense(self.dim_output_layer, activation="relu"),
                Dense(self.num_of_classes, activation="softmax"),
            ]
        )

        # sgd = optimizers.SGD(lr=self.learning_rate, momentum=0.9, decay=0, nesterov=False)

        # Compile the model
        model.compile(
            optimizer="rmsprop",  # works better than sgd
            loss=weighted_binary_cross_entropy,
            metrics=["accuracy"],
        )

        # Train the model
        model.fit(
            xtrain,
            to_categorical(ytrain),
            epochs=epochs,
            shuffle=True,
            batch_size=batch_size,
            validation_data=(xtest, to_categorical(ytest)),
        )

        # hist = model
        # test_error = 1 - hist.history.history["val_accuracy"][-1]

        # save model
        model.save(
            f"{model_directory}/{model_name}_{self.data_name}_input_{self.dim_input:.0f}.h5"
        )
