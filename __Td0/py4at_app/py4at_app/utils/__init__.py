"""
Utility functions for py4at_app
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


def calculate_returns(prices: pd.Series, method: str = 'log') -> pd.Series:
    """
    Calculate returns from price series.
    
    Parameters
    ==========
    prices: pd.Series
        Price series
    method: str
        'log' for log returns, 'simple' for simple returns
        
    Returns
    =======
    pd.Series
        Returns series
    """
    if method == 'log':
        return np.log(prices / prices.shift(1))
    elif method == 'simple':
        return prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")


def calculate_performance(initial: float, final: float) -> float:
    """
    Calculate percentage performance.
    
    Parameters
    ==========
    initial: float
        Initial value
    final: float
        Final value
        
    Returns
    =======
    float
        Percentage return
    """
    return ((final - initial) / initial) * 100


def calculate_sharpe_ratio(returns: pd.Series, 
                         risk_free_rate: float = 0.0,
                         periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Parameters
    ==========
    returns: pd.Series
        Daily returns
    risk_free_rate: float
        Annual risk-free rate (default: 0.0)
    periods_per_year: int
        Number of periods per year (default: 252 for daily data)
        
    Returns
    =======
    float
        Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def calculate_drawdown(cumulative_returns: pd.Series) -> Tuple[float, float]:
    """
    Calculate maximum drawdown.
    
    Parameters
    ==========
    cumulative_returns: pd.Series
        Cumulative returns series
        
    Returns
    =======
    tuple
        (max_drawdown, max_drawdown_duration)
    """
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    
    max_drawdown = drawdown.min()
    
    # Calculate drawdown duration
    drawdown_duration = 0
    for i, dd in enumerate(drawdown):
        if dd < 0:
            drawdown_duration += 1
        else:
            drawdown_duration = 0
    
    return max_drawdown, drawdown_duration


def calculate_win_rate(trades: List[dict]) -> float:
    """
    Calculate win rate from trades.
    
    Parameters
    ==========
    trades: list
        List of trade dictionaries with 'pnl' key
        
    Returns
    =======
    float
        Win rate (0-1)
    """
    if not trades:
        return 0.0
    
    winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
    return winning_trades / len(trades)


def format_currency(value: float, decimal_places: int = 2) -> str:
    """
    Format value as currency string.
    
    Parameters
    ==========
    value: float
        Value to format
    decimal_places: int
        Number of decimal places
        
    Returns
    =======
    str
        Formatted string
    """
    return f'${value:,.{decimal_places}f}'


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format value as percentage string.
    
    Parameters
    ==========
    value: float
        Value (0-1)
    decimal_places: int
        Number of decimal places
        
    Returns
    =======
    str
        Formatted string
    """
    return f'{value * 100:.{decimal_places}f}%'


def resample_data(data: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Resample data to different frequency.
    
    Parameters
    ==========
    data: pd.DataFrame
        Input data with DatetimeIndex
    frequency: str
        Target frequency (e.g., '1H', '1D', '1W')
        
    Returns
    =======
    pd.DataFrame
        Resampled data
    """
    if 'price' not in data.columns:
        return data
    
    resampled = data.resample(frequency).agg({
        'price': ['first', 'high', 'low', 'last']
    })
    
    resampled.columns = ['open', 'high', 'low', 'close']
    return resampled


def validate_parameters(params: dict, required_keys: List[str]) -> bool:
    """
    Validate that all required parameters are present.
    
    Parameters
    ==========
    params: dict
        Parameters dictionary
    required_keys: list
        List of required keys
        
    Returns
    =======
    bool
        True if all required keys present
    """
    return all(key in params for key in required_keys)
