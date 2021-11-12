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
        dim_hidden_layers,
        num_of_classes,
        data_name,
    ):
        """

        Parameters
        ----------
        dim_input: int > 0
            Dimension of the input / number of features
        dim_hidden_layers: list
            List where each element is the number of neurons in the ith hidden layer
        num_of_classes: int > 0
            Dimension of the output / number of classes.
        data_name: str
            Name of the dataset.
        """
        self.data_name = data_name
        self.dim_input = dim_input
        self.dim_hidden_layers = dim_hidden_layers
        self.num_of_classes = num_of_classes

        self.model = Sequential()
        for i, dim_layer in enumerate(dim_hidden_layers):
            # first layer requires input dimension
            if i == 0:
                self.model.add(
                    Dense(
                        units=dim_layer,
                        input_dim=self.dim_input,
                        activation="relu",
                    )
                )
            else:
                self.model.add(Dense(units=dim_layer, activation="relu"))
        # layer that does the classification
        self.model.add(Dense(units=self.num_of_classes, activation="softmax"))

        print(self.model.summary())

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
