from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from carla import MLModel


# Custom black-box models need to inherit from
# the MLModel interface
class TreeModel(MLModel):
    def __init__(self, data):
        super().__init__(data)
        # The constructor can be used to load or build an
        # arbitrary black-box-model
        self._mymodel = DecisionTreeClassifier(max_depth=4)

        # add support for methods that can also use categorical data
        data_transformed = self.scaler.transform(data.raw[data.continous])
        target = data.raw[data.target]

        X_train, X_test, y_train, y_test = train_test_split(
            data_transformed, target, test_size=0.20
        )
        self._mymodel.fit(X=X_train, y=y_train)
        train_score = self._mymodel.score(X=X_train, y=y_train)
        test_score = self._mymodel.score(X=X_test, y=y_test)
        print(
            "model fitted with training score {} and test score {}".format(
                train_score, test_score
            )
        )
        self._feature_input_order = data.continous

    @property
    def feature_input_order(self):
        # List of the feature order the ml model was trained on
        return self._feature_input_order

    @property
    def backend(self):
        # The ML framework the model was trained on
        return "sklearn"

    @property
    def raw_model(self):
        # The black-box model object
        return self._mymodel

    # The predict function outputs
    # the continous prediction of the model
    def predict(self, x):
        return self._mymodel.predict(x)

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        return self._mymodel.predict_proba(x)


# Custom black-box models need to inherit from
# the MLModel interface
class ForestModel(MLModel):
    def __init__(self, data):
        super().__init__(data)
        # The constructor can be used to load or build an
        # arbitrary black-box-model
        self._mymodel = RandomForestClassifier(
            max_depth=4,
        )
        data_transformed = self.scaler.transform(data.raw[data.continous])
        target = data.raw[data.target]

        X_train, X_test, y_train, y_test = train_test_split(
            data_transformed, target, test_size=0.20
        )
        self._mymodel.fit(X=X_train, y=y_train)
        train_score = self._mymodel.score(X=X_train, y=y_train)
        test_score = self._mymodel.score(X=X_test, y=y_test)
        print(
            "model fitted with training score {} and test score {}".format(
                train_score, test_score
            )
        )

        self._feature_input_order = data.continous

    @property
    def feature_input_order(self):
        # List of the feature order the ml model was trained on
        return self._feature_input_order

    @property
    def backend(self):
        # The ML framework the model was trained on
        return "sklearn"

    @property
    def raw_model(self):
        # The black-box model object
        return self._mymodel

    # The predict function outputs
    # the continous prediction of the model
    def predict(self, x):
        return self._mymodel.predict(x)

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        return self._mymodel.predict_proba(x)
