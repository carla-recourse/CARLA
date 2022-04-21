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
- HELOC: `HELOC <https://community.fico.com/s/explainable-machine-learning-challenge?tabset-158d9=2>`_

Implemented Counterfactual Methods
----------------------------------

- Actionable Recourse (AR): `AR <https://arxiv.org/pdf/1809.06514.pdf>`_
- Causal recourse: `Causal <https://arxiv.org/pdf/2002.06278.pdf>`_
- CCHVAE: `CCHVAE <https://arxiv.org/pdf/1910.09398.pdf>`_
- Contrastive Explanations Method (CEM): `CEM <https://arxiv.org/pdf/1802.07623.pdf>`_
- Counterfactual Latent Uncertainty Explanations (CLUE): `CLUE <https://arxiv.org/pdf/2006.06848.pdf>`_
- CRUDS: `CRUDS <https://finale.seas.harvard.edu/files/finale/files/cruds-_counterfactual_recourse_using_disentangled_subspaces.pdf>`_
- Diverse Counterfactual Explanations (DiCE): `DiCE <https://arxiv.org/pdf/1905.07697.pdf>`_
- Feasible and Actionable Counterfactual Explanations (FACE): `FACE <https://arxiv.org/pdf/1909.09369.pdf>`_
- FeatureTweak: `FeatureTweak <https://arxiv.org/pdf/1706.06691.pdf>`_
- FOCUS: `FOCUS <https://arxiv.org/pdf/1911.12199.pdf>`_
- Growing Sphere (GS): `GS <https://arxiv.org/pdf/1910.09398.pdf>`_
- Revise: `Revise <https://arxiv.org/pdf/1907.09615.pdf>`_
- Wachter: `Wachter <https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf>`_

Provided Machine Learning Models
--------------------------------

- **ANN**: Artificial Neural Network with 2 hidden layers and ReLU activation function.
- **LR**: Linear Model with no hidden layer and no activation function.
- **Forest**: Tree Ensemble Model.

Which Recourse Methods work with which ML framework?
-----------------------------------------------
The framework a counterfactual method currently works with is dependent on its underlying implementation.
It is planned to make all recourse methods available for all ML frameworks. The latest state can be found here:

+-----------------+------------+---------+---------+---------+
| Recourse Method | Tensorflow | Pytorch | Sklearn | XGBoost |
+=================+============+=========+=========+=========+
| AR              |     X      |    X    |         |         |
+-----------------+------------+---------+---------+---------+
| Causal          |     X      |    X    |         |         |
+-----------------+------------+---------+---------+---------+
| CCHVAE          |            |    X    |         |         |
+-----------------+------------+---------+---------+---------+
| CEM             |     X      |         |         |         |
+-----------------+------------+---------+---------+---------+
| CLUE            |            |    X    |         |         |
+-----------------+------------+---------+---------+---------+
| CRUDS           |            |    X    |         |         |
+-----------------+------------+---------+---------+---------+
| DiCE            |     X      |    X    |         |         |
+-----------------+------------+---------+---------+---------+
| FACE            |     X      |    X    |         |         |
+-----------------+------------+---------+---------+---------+
| FeatureTweak    |            |         |    X    |    X    |
+-----------------+------------+---------+---------+---------+
| FOCUS           |            |         |    X    |    X    |
+-----------------+------------+---------+---------+---------+
| Growing Spheres |     X      |    X    |         |         |
+-----------------+------------+---------+---------+---------+
| Revise          |            |    X    |         |         |
+-----------------+------------+---------+---------+---------+
| Wachter         |            |    X    |         |         |
+-----------------+------------+---------+---------+---------+

Citation
--------

This project was recently accepted to NeurIPS 2021 (Benchmark & Data Sets Track).

If you use this codebase, please cite: ::

    @misc{pawelczyk2021carla,
          title={CARLA: A Python Library to Benchmark Algorithmic Recourse and Counterfactual Explanation Algorithms},
          author={Martin Pawelczyk and Sascha Bielawski and Johannes van den Heuvel and Tobias Richter and Gjergji Kasneci},
          year={2021},
          eprint={2108.00783},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }

Please also cite the original authors' work.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   examples
   data
   mlmodel
   recourse
   benchmarking
   plotting
   license
   tutorial

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
