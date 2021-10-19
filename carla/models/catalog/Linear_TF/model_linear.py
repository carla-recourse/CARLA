import os

from keras import activations
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.utils import to_categorical


class LinearModel:
    def __init__(
        self, dim_input, num_of_classes, data_name, restore=None, use_prob=False
    ):

        # For model loading
        """
        :param dim_input: int > 0, number of neurons for this layer
        :param num_of_classes: int > 0, number of classes
        :param use_prob: boolean; FALSE required for CEM; all others should use True
        """  #

        self.data_name = data_name
        self.dim_input = dim_input
        self.num_of_classes = num_of_classes

        model = Sequential(
            [
                Dense(
                    self.num_of_classes, input_dim=self.dim_input, activation="linear"
                ),
                Dense(self.num_of_classes),
            ]
        )

        self.model = model

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

    def get_coefficients(self, c):
        """
        Calculates the coefficients of the model.
        :param c: int, defines the coefficients
        :return: Tuple of tensor (coefficients), tensor (intersection)
        """
        coef = self.model.weights[0]
        coef = coef[:, c]

        inter = self.model.weights[1]
        inter = inter[c]

        return coef, inter

    def build_train_save_model(
        self,
        xtrain,
        ytrain,
        xtest,
        ytest,
        learning_rate,
        epochs,
        batch_size,
        model_name="linear_tf",
        model_directory="saved_models",
    ):
        model = Sequential(
            [
                Dense(
                    self.num_of_classes, input_dim=self.dim_input, activation="linear"
                ),
                Dense(self.num_of_classes, activation="softmax"),
            ]
        )

        # sgd = optimizers.SGD(lr=self.learning_rate, momentum=0.9, decay=0, nesterov=False)

        # Compile the model
        model.compile(
            optimizer="rmsprop",  # works better than sgd
            loss="categorical_crossentropy",
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

        self.model = model

        # hist = model
        # test_error = 1 - hist.history.history["val_accuracy"][-1]

        # save model
        if not os.path.exists(model_directory):
            os.mkdir(model_directory)
        model.save(
            f"{model_directory}/{model_name}_{self.data_name}_input_{self.dim_input:.0f}.h5"
        )
