"""
MatKit - A robust Python toolkit for materials informatics.

MatKit provides a comprehensive suite of tools for materials data processing,
featurization, and machine learning. It offers advanced features beyond
traditional materials informatics packages with a focus on robustness,
performance, and ease of use.
"""

__version__ = "0.2.0"

from matkit.core import config, exceptions, base
from matkit.data import fetchers
from matkit.featurizers import composition
from matkit.utils import validators, chemistry

__all__ = [
    "config",
    "exceptions",
    "base",
    "fetchers",
    "composition",
    "validators",
    "chemistry"
]