Welcome to CARLA's documentation!
=================================

CARLA is a python library to benchmark counterfactual explanation and recourse models.
It comes out-of-the box with commonly used datasets and various machine learning models.
Designed with extensibility in mind: Easily include your own counterfactual methods,
new machine learning models or other datasets.

*Disclaimer*: We are currently in the process of open-sourcing, not all functionality is available yet.

Available Datasets
------------------

- Adult Data Set: `Source <https://archive.ics.uci.edu/ml/datasets/adult>`_
- COMPAS: `Source <https://www.kaggle.com/danofer/compass>`_
- Give Me Some Credit (GMC): `Source <https://www.kaggle.com/c/GiveMeSomeCredit/data>`_

Implemented Counterfactual Methods
----------------------------------

- Actionable Recourse (AR): `Paper <https://arxiv.org/pdf/1809.06514.pdf>`_
- Contrastive Explanations Method (CEM): `Paper <https://arxiv.org/pdf/1802.07623.pdf>`_
- Counterfactual Latent Uncertainty Explanations (CLUE): `Paper <https://arxiv.org/pdf/2006.06848.pdf>`_
- Diverse Counterfactual Explanations (DiCE): `Paper <https://arxiv.org/pdf/1905.07697.pdf>`_
- Feasible and Actionable Counterfactual Explanations (FACE): `Paper <https://arxiv.org/pdf/1909.09369.pdf>`_
- Growing Sphere (GS): `Paper <https://arxiv.org/pdf/1910.09398.pdf>`_

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
