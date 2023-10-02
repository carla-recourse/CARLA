.. _autodoc:

.. contents::

Recourse Methods
================
The recourse method module contains

- a :ref:`recourse_catalog` with pre-implemented and ready-to-use recourse methods for arbitrary datasets and black-box-models or
- an :ref:`recourse_api` to implement new recourse methods for benchmarking and comparison.

Example implementations for both use-cases can be found in our section :doc:`examples`.

.. _recourse_api:

Interface
---------
.. automodule:: recourse_methods.api.recourse_method
   :members:
   :undoc-members:

.. _recourse_catalog:

Catalog
-------
The following catalog lists all currently implemented recourse methods.
Important hyperparameters that are needed for each method are specified in the notes of each section,
and have to be passed in the constructor as dictionary.

.. _ar:

Actionable Recourse
^^^^^^^^^^^^^^^^^^^
.. automodule:: recourse_methods.catalog.actionable_recourse.model
   :members:
   :undoc-members:

.. _causal:

Causal Recourse
^^^^^^^^^^^^^^^
.. automodule:: recourse_methods.catalog.causal_recourse.model
   :members:
   :undoc-members:

.. _cchvae:

CCHVAE
^^^^^^
.. automodule:: recourse_methods.catalog.cchvae.model
   :members:
   :undoc-members:

.. _cem:

CEM
^^^
.. automodule:: recourse_methods.catalog.cem.model
   :members:
   :undoc-members:

.. _clue:

CLUE
^^^^
.. automodule:: recourse_methods.catalog.clue.model
   :members:
   :undoc-members:

.. _dice:

Dice
^^^^
.. automodule:: recourse_methods.catalog.dice.model
   :members:
   :undoc-members:

.. _face:

FACE
^^^^
.. automodule:: recourse_methods.catalog.face.model
   :members:
   :undoc-members:

.. _featuretweak:

FeatureTweak
^^^^^^^
.. automodule:: recourse_methods.catalog.feature_tweak.model
   :members:
   :undoc-members:

.. _focus:

FOCUS
^^^^^^^
.. automodule:: recourse_methods.catalog.focus.model
   :members:
   :undoc-members:


.. _gs:

Growing Spheres
^^^^^^^^^^^^^^^
.. automodule:: recourse_methods.catalog.growing_spheres.model
   :members:
   :undoc-members:

.. _revise:

REVISE
^^^^^^
.. automodule:: recourse_methods.catalog.revise.model
   :members:
   :undoc-members:

.. _roar:

ROAR
^^^^^^
.. automodule:: recourse_methods.catalog.roar.model
   :members:
   :undoc-members:

.. _wachter:

Wachter
^^^^^^^
.. automodule:: recourse_methods.catalog.wachter.model
   :members:
   :undoc-members:
