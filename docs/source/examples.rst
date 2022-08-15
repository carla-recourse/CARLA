Examples
========

To get a better insight of how to use CARLA for the different purposes, we will provide here some short example implementations.

We also have Tutorial notebooks available, which can be found under contents.

To benchmark an arbitrary recourse method, we provide an example implementation based on :ref:`quick`, in the
section :ref:`ex_bench`

.. _quick:

Quickstart
----------
In this case, we want to use a recourse method to generate counterfactual examples with pre-implemented
dataset and black-box-model.

.. code-block:: python
   :linenos:

    from carla import OnlineCatalog, MLModelCatalog
    from carla.recourse_methods import GrowingSpheres

    # 1. Load data set from the OnlineCatalog
    data_name = "adult"
    dataset = OnlineCatalog(data_name)

    # 2. Load pre-trained black-box model from the MLModelCatalog
    model = MLModelCatalog(dataset, "ann")

    # 3. Load recourse model with model specific hyperparameters
    hyperparameters = {}
    gs = GrowingSpheres(model, hyperparameters)

    # 4. Generate counterfactual examples
    factuals = dataset.df.sample(10)
    counterfactuals = gs.get_counterfactuals(factuals)

Customization
-------------
The following examples will contain some pseudo-code to visualize how a :ref:`cstm_data`, :ref:`cstm_model`, or
:ref:`cstm_rec` implementation would look like. The structure to generate counterfactuals with these user specific
implementations would still resemble :ref:`quick`.

.. _cstm_data:

Data
^^^^

The easiest way to use your own data is using the CsvDataset. For this you need to define the continuous and categorical features, which of those are immutable, and the target. Then you just give the file path to your .csv file and you are good to go!

.. code-block:: python
   :linenos:

   from carla.data.catalog import CsvCatalog

   continuous = ["age", "fnlwgt", "education-num", "capital-gain","hours-per-week", "capital-loss"]
   categorical = ["marital-status", "native-country", "occupation", "race", "relationship", "sex", "workclass"]
   immutable = ["age", "sex"]

   dataset = CsvCatalog(file_path="adult.csv",
                        continuous=continuous,
                        categorical=categorical,
                        immutables=immutable,
                        target='income')

If you want full control over your dataset, you can also implement it from scratch using our api.

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

       # The full dataset
       @property
       def df(self):
           return self._dataset

       # The training split of the dataset
       @property
       def df_train(self):
           return self._dataset_train

       # The test split of the dataset
       @property
       def df_test(self):
            return self._dataset_test

       # Data transformation, for example normalization of continuous features
       # and encoding of categorical features
       def transform(self, df):
            return transformed_df

       # Inverts transform operation
       def inverse_transform(self, df):
            return original_df

For reference you can always take a look at the `data api <https://github.com/carla-recourse/CARLA/blob/main/carla/data/api/data.py>`_. In addition we also have a concrete example of an implementation of our data api in our `DataCatalog <https://github.com/carla-recourse/CARLA/blob/main/carla/data/catalog/catalog.py>`_.

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

See below a concrete example on how to use a custom model in our framework. Note that the tree_iterator method is specific for tree methods, and is not used for other recourse methods.

.. code-block:: python
   :linenos:

   from carla import MLModel
   import xgboost

   class XGBoostModel(MLModel):
       """The default way of implementing XGBoost
       https://xgboost.readthedocs.io/en/latest/python/python_intro.html"""

       def __init__(self, data):
           super().__init__(data)

           # get preprocessed data
           df_train = self.data.df_train
           df_test = self.data.df_test

           # we only use the continuous features here
           # so you might want to also include the categorical
           # features
           x_train = df_train[self.data.continuous]
           y_train = df_train[self.data.target]
           x_test = df_test[self.data.continuous]
           y_test = df_test[self.data.target]

           # you can not only use the feature input order to
           # order the data but also to e.g. restrict the input
           # to only the continous features
           self._feature_input_order = self.data.continuous

           param = {
               "max_depth": 2,  # determines how deep the tree can go
               "objective": "binary:logistic",  # determines the loss function
               "n_estimators": 5,
           }
           self._mymodel = xgboost.XGBClassifier(**param)
           self._mymodel.fit(
                   x_train,
                   y_train,
                   eval_set=[(x_train, y_train), (x_test, y_test)],
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

.. _cstm_rec:

Recourse Method
^^^^^^^^^^^^^^^

This short code example shows you how to implement a recourse method. Any configuration options should be passed to the initializer, not the :code:`get_counterfactuals` method, trough the :code:`hyperparameters` dictionary. Also note that the recourse method has access to the :code:`mlmodel`, which in turn has access to the :code:`data` object. So for example, if you want some property of the scaler, you can access that by :code:`mlmodel.data.scaler`.

.. code-block:: python
   :linenos:

   from carla import RecourseMethod

    # Custom recourse implementations need to
    # inherit from the RecourseMethod interface
    class MyRecourseMethod(RecourseMethod):
        def __init__(self, mlmodel, hyperparameters):
            super().__init__(mlmodel)
            # the constructor can be used to load the recourse method,
            # or construct everything necessary

        # Generate and return encoded and
        # scaled counterfactual examples
        def get_counterfactuals(self, factuals: pd.DataFrame):
            # This property is responsible to generate and output
            # encoded and scaled (i.e. transformed) counterfactual examples
            # as pandas DataFrames.
            # Concretely this means that e.g. the counterfactuals should have
            # the same one-hot encoding as the factuals, and e.g. they both
            # should be min-max normalized with the same range.
            # It's expected that there is a single counterfactual per factual,
            # however in case a counterfactual cannot be found it should be NaN.
    		[...]
            return counterfactual_examples

For lots of different example of how to do this, you can take a look at all the methods in our `recourse catalog <https://github.com/carla-recourse/CARLA/tree/main/carla/recourse_methods/catalog>`_. For example the Wachter method is a clean example.

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
