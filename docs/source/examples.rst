Examples
========

To get a better insight of how to use CARLA for the different purposes, we will provide here some short example
implementations.

To benchmark an arbitrary recourse method, we provide an example implementation based on :ref:`quick`, in the
section :ref:`ex_bench`

.. _quick:

Quickstart
----------
In this case, we want to use a recourse method to generate counterfactual examples with pre-implemented
dataset and black-box-model.

.. code-block:: python
   :linenos:

    from carla import DataCatalog, MLModelCatalog
    from carla.recourse_methods import GrowingSpheres

    # 1. Load data set from the DataCatalog
    data_name = "adult"
    dataset = DataCatalog(data_name)

    # 2. Load pre-trained black-box model from the MLModelCatalog
    model = MLModelCatalog(dataset, "ann")

    # 3. Load recourse model with model specific hyperparameters
    gs = GrowingSpheres(model)

    # 4. Generate counterfactual examples
    factuals = dataset.raw.sample(10)
    counterfactuals = gs.get_counterfactuals(factuals)

Customization
-------------
The following examples will contain some pseudo-code to visualize how a :ref:`cstm_data`, :ref:`cstm_model`, or
:ref:`cstm_rec` implementation would look like. The structure to generate counterfactuals with these user specific
implementations would still resemble :ref:`quick`.

.. _cstm_data:

Data
^^^^
.. code-block:: python
   :linenos:

   from carla import Data

   # Custom data set implementations need to inherit from the Data interface
   class MyOwnDataSet(Data):
       def __init__(self):
           # The data set can e.g. be loaded in the constructor
           self._dataset = load_dataset_from_disk()

       # List of all categorical features
       @property
       def categorical(self):
           return [...]

       # List of all continuous features
       @property
       def continuous(self):
           return [...]

       # List of all immutable features which
       # should not be changed by the recourse method
       @property
       def immutables(self):
           return [...]

       # Feature name of the target column
       @property
       def target(self):
           return "label"

       # Non-encoded and  non-normalized, raw data set
       @property
       def raw(self):
           return self._dataset

.. _cstm_model:

Black-Box-Model
^^^^^^^^^^^^^^^
.. code-block:: python
   :linenos:

    from carla import MLModel

    # Custom black-box models need to inherit from
    # the MLModel interface
    class MyOwnModel(MLModel):
        def __init__(self, data):
            super().__init__(data)
            # The constructor can be used to load or build an
            # arbitrary black-box-model
            self._mymodel = load_model()

        # List of the feature order the ml model was trained on
        @property
        def feature_input_order(self):
            return [...]

        # The ML framework the model was trained on
        @property
        def backend(self):
            return "pytorch"

        # The black-box model object
        @property
        def raw_model(self):
            return self._mymodel

        # The predict function outputs
        # the continuous prediction of the model
        def predict(self, x):
            return self._mymodel.predict(x)

        # The predict_proba method outputs
        # the prediction as class probabilities
        def predict_proba(self, x):
            return self._mymodel.predict_proba(x)

See below a concrete example on how to use a sklearn model in our framework.

.. code-block:: python
   :linenos:

   from carla.models.api import MLModel

   from sklearn.model_selection import train_test_split
   from sklearn.tree import DecisionTreeClassifier

   # Custom black-box models need to inherit from
   # the MLModel interface
   class TreeModel(MLModel):
       def __init__(self, data):
           # initialize the superclass using the data object
           super().__init__(data)

           # define your model object
           self._mymodel = DecisionTreeClassifier(max_depth=4)

           # you can use the train-test split of the data object
           features = data.continuous + data.categorical

           X_train = data.df_train[features]
           y_train = data.df_train[data.target]

           X_test = data.df_test[data.continuous + data.categorical]
           y_test = data.df_test[data.target]

           # fit your model
           self._mymodel.fit(X=X_train, y_train)
           train_score = self._mymodel.score(X=X_train, y=y_train)
           test_score = self._mymodel.score(X=X_test, y=y_test)
           print(
               "model fitted with training score {} and test score {}".format(
                   train_score, test_score
               )
           )

           # save the feature order the model was trained on
           self._feature_input_order = features

       # List of the feature order the ml model was trained on
       @property
       def feature_input_order(self):
           return self._feature_input_order

       # The ML framework the model was trained on
       @property
       def backend(self):
           return "sklearn"

       # The black-box model object
       @property
       def raw_model(self):
           return self._mymodel

       # The predict function outputs
       # the continuous prediction of the model
       def predict(self, x):
           return self._mymodel.predict(self.get_ordered_features(x))

       # The predict_proba method outputs
       # the prediction as class probabilities
       def predict_proba(self, x):
           return self._mymodel.predict_proba(self.get_ordered_features(x))

.. _cstm_rec:

Recourse Method
^^^^^^^^^^^^^^^
.. code-block:: python
   :linenos:

   from carla import RecourseMethod

    # Custom recourse implementations need to
    # inherit from the RecourseMethod interface
    class MyRecourseMethod(RecourseMethod):
        def __init__(self, mlmodel):
            super().__init__(mlmodel)

        # Generate and return encoded and
        # scaled counterfactual examples
        def get_counterfactuals(self, factuals: pd.DataFrame):
    		[...]
    		return counterfactual_examples

.. _ex_bench:

Benchmarking
------------
.. code-block:: python
   :linenos:

    from carla import Benchmark

    # 1. Initilize the benchmarking class by passing
    # black-box-model, recourse method, and factuals into it
    benchmark = Benchmark(model, gs, factuals)

    # 2. Either only compute the distance measures
    distances = benchmark.compute_distances()

    # 3. Or run all implemented measurements and create a
    # DataFrame which consists of all results
    results = benchmark.run_benchmark()
