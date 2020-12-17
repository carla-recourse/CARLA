# Counterfactual Explanation Benchmarking

A Python package **`cf-benchmark`** to benchmark counterfactual explanation methods on commonly used datasets with various machine learning models. Easily extensible with new methods, models and datasets

### Available Datasets

- Adult Data Set: [Source](https://archive.ics.uci.edu/ml/datasets/adult)

### Implemented Methods

- DiCE: [Repo](https://github.com/interpretml/DiCE), [Paper](https://arxiv.org/abs/1905.07697)

### Provided Machine Learning Models

- Artificial Neural Network with 2 hidden layers and ReLU activation function

## Installation

### Requirements

- `python3.7`
- `pip`

### Install via pip

```sh
pip install git+https://github.com/indyfree/cf-benchmark.git#egg=cf-benchmark
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

cf-benchmark is under the MIT Licence. See the [LICENCE](github.com/indyfree/cf-benchmark/blob/master/LICENSE) for more details.
