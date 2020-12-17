# Counterfactual Explanation Benchmarking

Benchmark counterfactual explanation methods on commonly used datasets with various machine learning models. Easily extensible with new methods, models or datasets

### Requirements

- `python3.7`
- `python3.7-venv` (if not shipped with python3.7)

- [GNU Make](https://www.gnu.org/software/make/)

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

### Contributing

We use pre-commit hooks to enforce:

- coherent styling with [black](https://github.com/psf/black)
- python linting with [flake8](https://flake8.pycqa.org/en/latest/)

Install pre-commit with:

```sh
make install-dev
```

Using python directly or within activated virtual environment:

```sh
pip install -r requirements-dev.txt
pre-commit install
```

### Licence

cf-benchmark is under the MIT Licence. See the [LICENCE](github.com/indyfree/cf-benchmark/blob/master/LICENSE) for more details.
