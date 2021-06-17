# CARLA - Counterfactual And Recourse Library

CARLA is a python library to benchmark counterfactual explanation and recourse models. It comes out-of-the box with commonly used datasets and various machine learning models. Designed with extensibility in mind: Easily include your own counterfactual methods, new machine learning models or other datasets.

### Available Datasets

- Adult Data Set: [Source](https://archive.ics.uci.edu/ml/datasets/adult)
- COMPAS: [Source](https://www.kaggle.com/danofer/compass)
- Give Me Some Credit (GMC): [Source](https://www.kaggle.com/c/GiveMeSomeCredit/data)

### Implemented Counterfactual Methods

- Actionable Recourse (AR): [Paper](https://arxiv.org/pdf/1809.06514.pdf)
- Action Sequence (AS): [Paper](https://arxiv.org/pdf/1910.00057.pdf)
- Contrastive Explanations Method (CEM): [Paper](https://arxiv.org/pdf/1802.07623.pdf)
- Counterfactual Latent Uncertainty Explanations (CLUE): [Paper](https://arxiv.org/pdf/2006.06848.pdf)
- Diverse Counterfactual Explanations (DiCE): [Paper](https://arxiv.org/pdf/1905.07697.pdf)
- EB-CF: [Paper](https://arxiv.org/pdf/1912.03277.pdf)
- Feasible and Actionable Counterfactual Explanations (FACE): [Paper](https://arxiv.org/pdf/1909.09369.pdf)
- Growing Sphere (GS): [Paper](https://arxiv.org/pdf/1910.09398.pdf)

### Provided Machine Learning Models

- **ANN**: Artificial Neural Network with 2 hidden layers and ReLU activation function
- **LR**: Linear Model with no hidden layer and no activation function

## Installation

### Requirements

- `python3.7`
- `pip`

### Install via pip

```sh
pip install git+https://github.com/indyfree/carla.git#egg=carla
```

## Contributing

### Requirements

- `python3.7-venv` (when not already shipped with python3.7)
- Recommended: [GNU Make](https://www.gnu.org/software/make/)

### Installation

Using make:

```sh
make requirements
```

Using python directly or within activated virtual environment:

```sh
pip install -U pip setuptools wheel
pip install -e .
```

### Testing

Using make:

```sh
make test
```

Using python directly or within activated virtual environment:

```sh
pip install -r requirements-dev.txt
python -m pytest test/*
```

### Linting and Styling

We use pre-commit hooks within our build pipelines to enforce:

- Python linting with [flake8](https://flake8.pycqa.org/en/latest/).
- Python styling with [black](https://github.com/psf/black).

Install pre-commit with:

```sh
make install-dev
```

Using python directly or within activated virtual environment:

```sh
pip install -r requirements-dev.txt
pre-commit install
```

## Licence

carla is under the MIT Licence. See the [LICENCE](github.com/indyfree/carla/blob/master/LICENSE) for more details.
