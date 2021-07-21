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
       def categoricals(self):
           return [...]

       # List of all continous features
       @property
       def continous(self):
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

            # Define a fitted sklearn scaler to normalize input data
            self.scaler = MySklearnScaler().fit()

            # Define a fitted sklearn encoder for binary input data
            self.encoder = MySklearnEncoder.fit()

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
        # the continous prediction of the model
        def predict(self, x):
            return self._mymodel.predict(x)

        # The predict_proba method outputs
        # the prediction as class probabilities
        def predict_proba(self, x):
            return self._mymodel.predict_proba(x)


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
