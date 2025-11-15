"""
Python for Algorithmic Trading Application (py4at_app)
A comprehensive framework for algorithmic trading with backtesting and live trading capabilities.
"""

__version__ = '1.0.0'
__author__ = 'py4at_app Contributors'
__description__ = 'Comprehensive framework for algorithmic trading'

from . import backtesting
from . import trading
from . import data
from . import utils

__all__ = [
    'backtesting',
    'trading',
    'data',
    'utils',
]
