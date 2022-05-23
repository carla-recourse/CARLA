[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/carla-recourse?style=for-the-badge)](https://pypi.org/project/carla-recourse/) ![GitHub Workflow Status](https://img.shields.io/github/workflow/status/carla-recourse/CARLA/CI?style=for-the-badge) [![Read the Docs](https://img.shields.io/readthedocs/carla-counterfactual-and-recourse-library?style=for-the-badge)](https://carla-counterfactual-and-recourse-library.readthedocs.io/en/latest/?badge=latest) ![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)

# CARLA - Counterfactual And Recourse Library

CARLA is a python library to benchmark counterfactual explanation and recourse models. It comes out-of-the box with commonly used datasets and various machine learning models. Designed with extensibility in mind: Easily include your own counterfactual methods, new machine learning models or other datasets.

Find extensive documentation [here](https://carla-counterfactual-and-recourse-library.readthedocs.io/en/latest/)!
Our arXiv paper can be found [here](https://arxiv.org/pdf/2108.00783.pdf).

### Notebooks / Examples

- Plotting (notebook): [Source](https://carla-counterfactual-and-recourse-library.readthedocs.io/en/feature-plotting/notebooks/plotting_example.html)

### Available Datasets

- Adult Data Set: [Source](https://archive.ics.uci.edu/ml/datasets/adult)
- COMPAS: [Source](https://www.kaggle.com/danofer/compass)
- Give Me Some Credit (GMC): [Source](https://www.kaggle.com/c/GiveMeSomeCredit/data)

### Implemented Counterfactual Methods

- Actionable Recourse (AR): [Paper](https://arxiv.org/pdf/1809.06514.pdf)
- CCHVAE: [Paper](https://arxiv.org/pdf/1910.09398.pdf)
- Contrastive Explanations Method (CEM): [Paper](https://arxiv.org/pdf/1802.07623.pdf)
- Counterfactual Latent Uncertainty Explanations (CLUE): [Paper](https://arxiv.org/pdf/2006.06848.pdf)
- CRUDS: [Paper](https://finale.seas.harvard.edu/files/finale/files/cruds-_counterfactual_recourse_using_disentangled_subspaces.pdf)
- Diverse Counterfactual Explanations (DiCE): [Paper](https://arxiv.org/pdf/1905.07697.pdf)
- Feasible and Actionable Counterfactual Explanations (FACE): [Paper](https://arxiv.org/pdf/1909.09369.pdf)
- Growing Sphere (GS): [Paper](https://arxiv.org/pdf/1910.09398.pdf)
- Revise: [Paper](https://arxiv.org/pdf/1907.09615.pdf)
- Wachter: [Paper](https://arxiv.org/ftp/arxiv/papers/1711/1711.00399.pdf)
- FOCUS: [Paper](https://arxiv.org/pdf/1911.12199.pdf)
- FeatureTweak: [Paper](https://arxiv.org/pdf/1706.06691.pdf)

### Provided Machine Learning Models

- **ANN**: Artificial Neural Network with 2 hidden layers and ReLU activation function
- **LR**: Linear Model with no hidden layer and no activation function

### Which Recourse Methods work with which ML framework?
The framework a counterfactual method currently works with is dependent on its underlying implementation.
It is planned to make all recourse methods available for all ML frameworks . The latest state can be found here:

| Recourse Method | Tensorflow | Pytorch | SKlearn | XGBoost |
| --------------- | :--------: | :-----: | :-----: | :-----: |
| Actionable Recourse |      X | X       |         |         |
| Causal |                   X | X       |         |         |
| CCHVAE |                     | X       |         |         |
| CEM |                      X |         |         |         |
| CLUE |                       | X       |         |         |
| CRUDS |                      | X       |         |         |
| DiCE |                     X | X       |         |         |
| FACE |                     X | X       |         |         |
| FeatureTweak |               |         | X       |    X    |
| FOCUS |                      |         | X       |    X    |
| Growing Spheres |          X | X       |         |         |
| Revise |                     | X       |         |         |
| Wachter |                    | X       |         |         |

## Installation

### Requirements

- `python3.7`
- `pip`

### Install via pip

```sh
pip install carla-recourse
```

## Usage Example


```python
from carla import OnlineCatalog, MLModelCatalog
from carla.recourse_methods import GrowingSpheres

# load a catalog dataset
data_name = "adult"
dataset = OnlineCatalog(data_name)

# load artificial neural network from catalog
model = MLModelCatalog(dataset, "ann")

# get factuals from the data to generate counterfactual examples
factuals = dataset.raw.iloc[:10]

# load a recourse model and pass black box model
gs = GrowingSpheres(model)

# generate counterfactual examples
counterfactuals = gs.get_counterfactuals(factuals)
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

## Citation

This project was recently accepted to NeurIPS 2021 (Benchmark & Data Sets Track).
If you use this codebase, please cite:

```sh
@misc{pawelczyk2021carla,
      title={CARLA: A Python Library to Benchmark Algorithmic Recourse and Counterfactual Explanation Algorithms},
      author={Martin Pawelczyk and Sascha Bielawski and Johannes van den Heuvel and Tobias Richter and Gjergji Kasneci},
      year={2021},
      eprint={2108.00783},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Please also cite the original authors' work.
