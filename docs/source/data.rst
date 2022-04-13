.. _autodoc:

.. contents::

Data
=====
This data wrapper contains the possibilities to

- use pre-processed datasets from our :ref:`online_catalog` or
- use local datasets using
  :ref:`csv_catalog`
- implement every possible dataset by either inheriting our :ref:`data_catalog` or for even more control the :ref:`data_api`

Example implementations for the first two use-cases can be found in our section :doc:`examples`.

.. _data_api:

Interface
------------------
.. automodule:: data.api.data
   :members:
   :undoc-members:

.. _data_catalog:

DataCatalog
------------------
.. automodule:: data.catalog.catalog
   :members:
   :undoc-members:

.. _online_catalog:

OnlineCatalog
------------------
.. automodule:: data.catalog.online_catalog
    :members:
    :undoc-members:

.. _csv_catalog:

CsvCatalog
------------------
.. automodule:: data.catalog.csv_catalog
    :members:
    :undoc-members:


Causal Model
------------------
.. automodule:: data.causal_model.causal_model
   :members:
   :undoc-members:

Synthetic Data
------------------
.. automodule:: data.causal_model.synthethic_data
   :members:
   :undoc-members:
