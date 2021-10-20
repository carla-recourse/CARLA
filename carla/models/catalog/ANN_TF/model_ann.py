import os

import tensorflow as tf
from keras.layers import Dense
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
    ):
        """

        Parameters
        ----------
        dim_input: int > 0
            Number of neurons for this layer.
        dim_hidden_layer1: int > 0
            Number of neurons for this layer.
        dim_hidden_layer2: int > 0
            Number of neurons for this layer.
        dim_output_layer: int > 0
            Number of neurons for this layer.
        num_of_classes: int > 0
            Number of classes.
        data_name: str
            Name of the dataset.
        """
        self.data_name = data_name
        self.dim_input = dim_input
        self.dim_hidden_layer1 = dim_hidden_layer1
        self.dim_hidden_layer2 = dim_hidden_layer2
        self.dim_output_layer = dim_output_layer
        self.num_of_classes = num_of_classes

        self.model = Sequential(
            [
                Dense(
                    self.dim_hidden_layer1, input_dim=self.dim_input, activation="relu"
                ),
                Dense(self.dim_hidden_layer2, activation="relu"),
                Dense(self.dim_output_layer, activation="relu"),
                Dense(self.num_of_classes, activation="softmax"),
            ]
        )

    def __call__(self, data):
        return self.predict(data)

    def predict(self, data):
        return self.model(data)

    def build_train_save_model(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs,
        batch_size,
        model_name="ann_tf",
        model_directory="saved_models",
    ):
        model = self.model

        # Compile the model
        model.compile(
            optimizer="rmsprop",  # works better than sgd
            loss=weighted_binary_cross_entropy,
            metrics=["accuracy"],
        )

        # Train the model
        model.fit(
            x_train,
            to_categorical(y_train),
            epochs=epochs,
            shuffle=True,
            batch_size=batch_size,
            validation_data=(x_test, to_categorical(y_test)),
        )

        hist = model
        test_error = 1 - hist.history.history["val_accuracy"][-1]
        print(f"Test {model_name} on {self.data_name}:", test_error)

        # save model
        if not os.path.exists(model_directory):
            os.mkdir(model_directory)
        model.save(
            f"{model_directory}/{model_name}_{self.data_name}_input_{self.dim_input:.0f}.h5"
        )

        self.model = model
