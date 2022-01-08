from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical


class LinearModel:
    def __init__(self, dim_input, num_of_classes, data_name):
        """

        Parameters
        ----------
        dim_input: int > 0
            number of neurons for this layer
        num_of_classes: int > 0
            number of classes
        data_name: str
            name of the dataset
        """

        self.data_name = data_name
        self.dim_input = dim_input
        self.num_of_classes = num_of_classes

        self.model = Sequential(
            [
                Dense(
                    self.num_of_classes, input_dim=self.dim_input, activation="softmax"
                ),
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
        model_name="linear_tf",
    ):
        model = self.model

        # Compile the model
        model.compile(
            optimizer="rmsprop",  # works better than sgd
            loss="categorical_crossentropy",
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

        self.model = model
