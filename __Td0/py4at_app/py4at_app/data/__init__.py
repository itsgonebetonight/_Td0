"""
Data module for py4at_app
Handles data loading, retrieval, and preparation for backtesting and trading
"""

try:
    from .loader import DataLoader
    __all__ = ['DataLoader']
except ImportError:
    __all__ = []
