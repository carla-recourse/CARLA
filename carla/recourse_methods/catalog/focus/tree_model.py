import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from carla.models.api import MLModel


# Custom black-box models need to inherit from
# the MLModel interface
class TreeModel(MLModel):
    def __init__(self, data):
        super().__init__(data)
        # The constructor can be used to load or build an
        # arbitrary black-box-model
        self._mymodel = DecisionTreeClassifier(max_depth=4)

        # add support for methods that can also use categorical data
        data_transformed = self.scaler.transform(data.df[data.continuous])
        target = data.df[data.target]

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

        self._feature_input_order = data.continuous

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
    # the continuous prediction of the model
    def predict(self, x):
        return self._mymodel.predict(self.get_ordered_features(x))

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        return self._mymodel.predict_proba(self.get_ordered_features(x))


# Custom black-box models need to inherit from
# the MLModel interface
class ForestModel(MLModel):
    def __init__(self, data):
        super().__init__(data)
        # The constructor can be used to load or build an
        # arbitrary black-box-model
        self._mymodel = RandomForestClassifier(
            n_estimators=5,
            max_depth=2,
        )
        data_transformed = data.df[data.continuous]
        target = data.df[data.target]

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

        self._feature_input_order = data.continuous

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

    @property
    def tree_iterator(self):
        return self.raw_model

    # The predict function outputs
    # the continuous prediction of the model
    def predict(self, x):
        return self._mymodel.predict(self.get_ordered_features(x))

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        return self._mymodel.predict_proba(self.get_ordered_features(x))


class XGBoostModel(MLModel):
    """The default way of implementing XGBoost
    https://xgboost.readthedocs.io/en/latest/python/python_intro.html"""

    def __init__(self, data):
        super().__init__(data)
        self._feature_input_order = data.continuous

        data_transformed = data.df[self._feature_input_order]
        target = data.df[data.target]

        X_train, X_test, y_train, y_test = train_test_split(
            data_transformed, target, test_size=0.20
        )

        self.X = X_train

        param = {
            "max_depth": 2,  # determines how deep the tree can go
            "objective": "binary:logistic",  # determines the loss function
            "n_estimators": 5,
        }
        self._mymodel = xgboost.XGBClassifier(**param)
        self._mymodel.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric="logloss",
            verbose=True,
        )

    @property
    def feature_input_order(self):
        # List of the feature order the ml model was trained on
        return self._feature_input_order

    @property
    def backend(self):
        # The ML framework the model was trained on
        return "xgboost"

    @property
    def raw_model(self):
        # The black-box model object
        return self._mymodel

    @property
    def tree_iterator(self):
        # make a copy of the trees, else feature names are not saved
        booster_it = [booster for booster in self.raw_model.get_booster()]
        # set the feature names
        for booster in booster_it:
            booster.feature_names = self.feature_input_order
        return booster_it

    # The predict function outputs
    # the continuous prediction of the model
    def predict(self, x):
        return self._mymodel.predict(self.get_ordered_features(x))

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        return self._mymodel.predict_proba(self.get_ordered_features(x))
