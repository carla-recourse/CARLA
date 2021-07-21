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
- CCHVAE: `CCHVAE <https://arxiv.org/pdf/1910.09398.pdf>`_
- Contrastive Explanations Method (CEM): `CEM <https://arxiv.org/pdf/1802.07623.pdf>`_
- Counterfactual Latent Uncertainty Explanations (CLUE): `CLUE <https://arxiv.org/pdf/2006.06848.pdf>`_
- CRUDS: `CRUDS <https://finale.seas.harvard.edu/files/finale/files/cruds-_counterfactual_recourse_using_disentangled_subspaces.pdf>`_
- Diverse Counterfactual Explanations (DiCE): `DiCE <https://arxiv.org/pdf/1905.07697.pdf>`_
- Feasible and Actionable Counterfactual Explanations (FACE): `FACE <https://arxiv.org/pdf/1909.09369.pdf>`_
- Growing Sphere (GS): `GS <https://arxiv.org/pdf/1910.09398.pdf>`_
- Revise: `Revise <https://arxiv.org/pdf/1907.09615.pdf>`_
- Wachter: `Wachter <https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf>`_

Provided Machine Learning Models
--------------------------------

- **ANN**: Artificial Neural Network with 2 hidden layers and ReLU activation function
- **LR**: Linear Model with no hidden layer and no activation function

Which Recourse Methods work with which ML framework?
-----------------------------------------------
The framework a counterfactual method currently works with is dependent on its underlying implementation.
It is planned to make all recourse methods available for all ML frameworks. The latest state can be found here:

+-----------------+------------+---------+
| Recourse Method | Tensorflow | Pytorch |
+=================+============+=========+
| AR              |     X      |    X    |
+-----------------+------------+---------+
| CCHVAE          |            |    X    |
+-----------------+------------+---------+
| CEM             |     X      |         |
+-----------------+------------+---------+
| CLUE            |            |    X    |
+-----------------+------------+---------+
| CRUDS           |            |    X    |
+-----------------+------------+---------+
| DiCE            |     X      |    X    |
+-----------------+------------+---------+
| FACE            |     X      |    X    |
+-----------------+------------+---------+
| Growing Spheres |     X      |    X    |
+-----------------+------------+---------+
| Revise          |            |    X    |
+-----------------+------------+---------+
| Wachter         |            |    X    |
+-----------------+------------+---------+

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
