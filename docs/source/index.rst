Welcome to CARLA's documentation!
=================================

CARLA is a python library to benchmark counterfactual explanation and recourse models.
It comes out-of-the box with commonly used datasets and various machine learning models.
Designed with extensibility in mind: Easily include your own counterfactual methods,
new machine learning models or other datasets.

Available Datasets
------------------

- Adult Data Set: `Adult <https://archive.ics.uci.edu/ml/datasets/adult>`_
- COMPAS: `Compas <https://www.kaggle.com/danofer/compass>`_
- Give Me Some Credit (GMC): `GMC <https://www.kaggle.com/c/GiveMeSomeCredit/data>`_

Implemented Counterfactual Methods
----------------------------------

- Actionable Recourse (AR): `AR <https://arxiv.org/pdf/1809.06514.pdf>`_
- Contrastive Explanations Method (CEM): `CEM <https://arxiv.org/pdf/1802.07623.pdf>`_
- Counterfactual Latent Uncertainty Explanations (CLUE): `CLUE <https://arxiv.org/pdf/2006.06848.pdf>`_
- Diverse Counterfactual Explanations (DiCE): `DiCE <https://arxiv.org/pdf/1905.07697.pdf>`_
- Feasible and Actionable Counterfactual Explanations (FACE): `FACE <https://arxiv.org/pdf/1909.09369.pdf>`_
- Growing Sphere (GS): `GS <https://arxiv.org/pdf/1910.09398.pdf>`_

Provided Machine Learning Models
--------------------------------

- **ANN**: Artificial Neural Network with 2 hidden layers and ReLU activation function
- **LR**: Linear Model with no hidden layer and no activation function


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   examples
   data
   mlmodel
   recourse
   benchmarking
   license



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
