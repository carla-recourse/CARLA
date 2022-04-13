Installation
============

Requirements
------------------

- ``python3.7``
- ``pip``

Install via pip
------------------

::

   pip install git+https://github.com/carla-recourse/carla.git#egg=carla


Contributing
------------

Before you can start contributing to CARLA, you need to install some development tools.
These tools makes sure all the code committed is up to standards.
The following steps guide you trough this process.

Then when your code is ready, you can submit a pull request for review. This way we can take a look if everything works and looks good.

Requirements
^^^^^^^^^^^^

- ``python3.7-venv`` (when not already shipped with python3.7)
- Recommended: `GNU Make <https://www.gnu.org/software/make/>`_

Installation
^^^^^^^^^^^^

Using make: ::

   make requirements


Using python directly or within activated virtual environment: ::

   pip install -U pip setuptools wheel
   pip install -e .


Testing
^^^^^^^

Using make: ::

   make test


Using python directly or within activated virtual environment: ::

   pip install -r requirements-dev.txt
   python -m pytest test/*


Linting and Styling
^^^^^^^^^^^^^^^^^^^

We use pre-commit hooks within our build pipelines to enforce:

- Python linting with `flake8 <https://flake8.pycqa.org/en/latest/>`_.
- Python styling with `black <https://github.com/psf/black)>`_.

Install pre-commit with: ::

   make install-dev

Using python directly or within activated virtual environment: ::

   pip install -r requirements-dev.txt
   pre-commit install
