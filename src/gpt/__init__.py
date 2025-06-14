from .gpt import GPT, run_gpt

from .evaluate import evaluate_gpt
from .gpt_distgen import run_gpt_with_distgen, evaluate_gpt_with_distgen

#from . import _version
#__version__ = _version.get_versions()['version']

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # for Python <3.8

__version__ = version("gpt")

__all__ = [
    "GPT",
    "evaluate_gpt",
    "evaluate_gpt_with_distgen",
    "run_gpt",
    "run_gpt_with_distgen",
]
