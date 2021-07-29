# flake8: noqa
# isort:skip
import logging

from ._logger import INFOFORMATTER

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# defines the stream handler
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter(INFOFORMATTER))

# adds the handler to the global variable: log
log.addHandler(_ch)

from ._version import __version__
from .data import Data, DataCatalog
from .evaluation import Benchmark
from .models import MLModel, MLModelCatalog
from .recourse_methods import RecourseMethod
