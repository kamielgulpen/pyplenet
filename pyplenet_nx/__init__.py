"""
PyPleNet: Large-scale population network generation and analysis.

A Python package for generating and analyzing population networks that exceed RAM capacity
using file-based graph storage with support for preferential attachment, reciprocity,
and scalable network operations.
"""

# Core object and generation functions fully exposed
from .core.graph import *
from .core.generate import *

# namespace imports for the other core files
#from .core import grn, utils

# Import top level modules
from . import analysis
from . import export

# metadata
__version__ = "0.1.0"
__author__ = "Matijs Verloo"
__email__ = "matijs.verloo@gmail.com"