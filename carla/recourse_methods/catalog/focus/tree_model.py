from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from carla import MLModel


# Custom black-box models need to inherit from
# the MLModel interface
class TreeModel(MLModel):
    def __init__(self, data):
        super().__init__(data)
        # The constructor can be used to load or build an
        # arbitrary black-box-model
        self._mymodel = DecisionTreeClassifier()
        data_transformed = self.scaler.transform(data.raw[data.continous])
        self._mymodel.fit(X=data_transformed, y=data.raw[data.target])
        print(
            "model fitted with training score {}".format(
                self._mymodel.score(X=data_transformed, y=data.raw[data.target])
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
        self._mymodel = RandomForestClassifier()
        data_transformed = self.scaler.transform(data.raw[data.continous])
        self._mymodel.fit(X=data_transformed, y=data.raw[data.target])
        print(
            "model fitted with training score {}".format(
                self._mymodel.score(X=data_transformed, y=data.raw[data.target])
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
