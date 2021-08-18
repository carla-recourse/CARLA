# flake8: noqa
# isort:skip
import logging.config

import yaml

with open("logging.yaml", "r") as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

log = logging.getLogger(__name__)

from ._version import __version__
from .data import Data, DataCatalog
from .evaluation import Benchmark
from .models import MLModel, MLModelCatalog
from .recourse_methods import RecourseMethod


def get_logger(logger: str):
    return logging.getLogger(logger)
