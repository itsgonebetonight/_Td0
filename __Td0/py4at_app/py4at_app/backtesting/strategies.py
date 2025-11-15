"""
Vectorized backtesting strategies for multiple technical indicators
"""

import numpy as np
import pandas as pd
from scipy.optimize import brute
from typing import Optional, Tuple


class SMAVectorBacktester:
    """
    Vectorized backtesting class for Simple Moving Average (SMA) based strategies.
    
    Attributes
    ==========
    symbol: str
        RIC symbol with which to work
    SMA1: int
        Time window in days for shorter SMA
    SMA2: int
        Time window in days for longer SMA
    start: str
        Start date for data retrieval
    end: str
        End date for data retrieval
        
    Methods
    =======
    get_data:
        Retrieves and prepares the base data set
    set_parameters:
        Sets new SMA parameters
    run_strategy:
        Runs the backtest for the SMA-based strategy
    plot_results:
        Plots the performance of the strategy compared to the symbol
    optimize_parameters:
        Implements brute force optimization for SMA parameters
    """
    
    def __init__(self, symbol: str, SMA1: int, SMA2: int,
                 start: str, end: str, data: Optional[pd.DataFrame] = None):
        """
        Initialize SMA Vector Backtester.
        
        Parameters
        ==========
        symbol: str
            Financial instrument symbol
        SMA1: int
            Shorter moving average window
        SMA2: int
            Longer moving average window
        start: str
            Start date
        end: str
            End date
        data: pd.DataFrame, optional
            Pre-loaded data (if not provided, must be set later)
        """
        self.symbol = symbol
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.start = start
        self.end = end
        self.results = None
        self.data = None
        
        if data is not None:
            self.set_data(data)
        else:
            self.get_data()
    
    def get_data(self):
        """Retrieves and prepares the data."""
        try:
            raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv',
                            index_col=0, parse_dates=True).dropna()
            raw = pd.DataFrame(raw[self.symbol])
        except Exception as e:
            print(f"Could not load data from URL: {e}")
            raise
        
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: 'price'}, inplace=True)
        raw['return'] = np.log(raw['price'] / raw['price'].shift(1))
        raw['SMA1'] = raw['price'].rolling(self.SMA1).mean()
        raw['SMA2'] = raw['price'].rolling(self.SMA2).mean()
        
        self.data = raw
    
    def set_data(self, data: pd.DataFrame):
        """
        Set data directly.
        
        Parameters
        ==========
        data: pd.DataFrame
            DataFrame with price data
        """
        self.data = data.copy()
        if 'price' not in self.data.columns:
            if len(self.data.columns) > 0:
                self.data.rename(columns={self.data.columns[0]: 'price'}, 
                               inplace=True)
        
        self.data['return'] = np.log(self.data['price'] / 
                                     self.data['price'].shift(1))
        self.data['SMA1'] = self.data['price'].rolling(self.SMA1).mean()
        self.data['SMA2'] = self.data['price'].rolling(self.SMA2).mean()
    
    def set_parameters(self, SMA1: Optional[int] = None,
                      SMA2: Optional[int] = None):
        """
        Update SMA parameters and recalculate moving averages.
        
        Parameters
        ==========
        SMA1: int, optional
            New shorter SMA window
        SMA2: int, optional
            New longer SMA window
        """
        if SMA1 is not None:
            self.SMA1 = SMA1
            self.data['SMA1'] = self.data['price'].rolling(self.SMA1).mean()
        
        if SMA2 is not None:
            self.SMA2 = SMA2
            self.data['SMA2'] = self.data['price'].rolling(self.SMA2).mean()
    
    def run_strategy(self) -> Tuple[float, float]:
        """
        Backtest the trading strategy.
        
        Returns
        =======
        tuple
            (absolute_performance, over_underperformance)
        """
        data = self.data.copy().dropna()
        data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
        data['strategy'] = data['position'].shift(1) * data['return']
        data.dropna(inplace=True)
        
        data['creturns'] = data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        
        self.results = data
        
        # Absolute performance of the strategy
        aperf = data['cstrategy'].iloc[-1]
        # Out-/underperformance of strategy
        operf = aperf - data['creturns'].iloc[-1]
        
        return round(aperf, 2), round(operf, 2)
    
    def plot_results(self):
        """Plot the cumulative performance of the strategy."""
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
            return
        
        try:
            import matplotlib.pyplot as plt
            title = f'{self.symbol} | SMA1={self.SMA1}, SMA2={self.SMA2}'
            self.results[['creturns', 'cstrategy']].plot(
                title=title, figsize=(10, 6)
            )
            plt.show()
        except ImportError:
            print('Matplotlib not available for plotting.')
    
    def update_and_run(self, SMA: Tuple[int, int]) -> float:
        """
        Update SMA parameters and return negative absolute performance.
        Used for optimization algorithms.
        
        Parameters
        ==========
        SMA: tuple
            (SMA1, SMA2) parameters
            
        Returns
        =======
        float
            Negative absolute performance
        """
        self.set_parameters(int(SMA[0]), int(SMA[1]))
        return -self.run_strategy()[0]
    
    def optimize_parameters(self, SMA1_range: Tuple[int, int, int],
                           SMA2_range: Tuple[int, int, int]) -> Tuple:
        """
        Find optimal SMA parameters using brute force optimization.
        
        Parameters
        ==========
        SMA1_range: tuple
            (start, end, step) for SMA1
        SMA2_range: tuple
            (start, end, step) for SMA2
            
        Returns
        =======
        tuple
            (optimal_parameters, optimal_performance)
        """
        opt = brute(self.update_and_run, (SMA1_range, SMA2_range), 
                   finish=None)
        return opt, -self.update_and_run(opt)


class MomVectorBacktester:
    """
    Vectorized backtesting class for Momentum-based strategies.
    
    Attributes
    ==========
    symbol: str
        RIC symbol
    start: str
        Start date for data retrieval
    end: str
        End date for data retrieval
    amount: int, float
        Amount to be invested at the beginning
    tc: float
        Proportional transaction costs (e.g., 0.001 for 0.1%)
        
    Methods
    =======
    run_strategy:
        Runs the backtest for the momentum-based strategy
    plot_results:
        Plots the performance of the strategy
    """
    
    def __init__(self, symbol: str, start: str, end: str,
                 amount: float, tc: float, 
                 data: Optional[pd.DataFrame] = None):
        """
        Initialize Momentum Vector Backtester.
        
        Parameters
        ==========
        symbol: str
            Financial instrument symbol
        start: str
            Start date
        end: str
            End date
        amount: float
            Initial investment amount
        tc: float
            Transaction costs
        data: pd.DataFrame, optional
            Pre-loaded data
        """
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.results = None
        self.data = None
        self.momentum = 1
        
        if data is not None:
            self.set_data(data)
        else:
            self.get_data()
    
    def get_data(self):
        """Retrieves and prepares the data."""
        try:
            raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv',
                            index_col=0, parse_dates=True).dropna()
            raw = pd.DataFrame(raw[self.symbol])
        except Exception as e:
            print(f"Could not load data from URL: {e}")
            raise
        
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: 'price'}, inplace=True)
        raw['return'] = np.log(raw['price'] / raw['price'].shift(1))
        
        self.data = raw
    
    def set_data(self, data: pd.DataFrame):
        """Set data directly."""
        self.data = data.copy()
        if 'price' not in self.data.columns:
            if len(self.data.columns) > 0:
                self.data.rename(columns={self.data.columns[0]: 'price'}, 
                               inplace=True)
        
        self.data['return'] = np.log(self.data['price'] / 
                                     self.data['price'].shift(1))
    
    def run_strategy(self, momentum: int = 1) -> Tuple[float, float]:
        """
        Backtest the momentum strategy.
        
        Parameters
        ==========
        momentum: int
            Number of periods for momentum calculation
            
        Returns
        =======
        tuple
            (absolute_performance, over_underperformance)
        """
        self.momentum = momentum
        data = self.data.copy().dropna()
        
        data['position'] = np.sign(data['return'].rolling(momentum).mean())
        data['strategy'] = data['position'].shift(1) * data['return']
        data.dropna(inplace=True)
        
        # Determine when a trade takes place
        trades = data['position'].diff().fillna(0) != 0
        # Subtract transaction costs from return when trade takes place
        data['strategy'][trades] -= self.tc
        
        data['creturns'] = self.amount * \
            data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = self.amount * \
            data['strategy'].cumsum().apply(np.exp)
        
        self.results = data
        
        # Absolute performance of the strategy
        aperf = self.results['cstrategy'].iloc[-1]
        # Out-/underperformance of strategy
        operf = aperf - self.results['creturns'].iloc[-1]
        
        return round(aperf, 2), round(operf, 2)
    
    def plot_results(self):
        """Plot the cumulative performance of the strategy."""
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
            return
        
        try:
            import matplotlib.pyplot as plt
            title = f'{self.symbol} | TC = {self.tc:.4f}'
            self.results[['creturns', 'cstrategy']].plot(
                title=title, figsize=(10, 6)
            )
            plt.show()
        except ImportError:
            print('Matplotlib not available for plotting.')


class MRVectorBacktester(MomVectorBacktester):
    """
    Vectorized backtesting class for Mean Reversion-based strategies.
    Inherits from MomVectorBacktester.
    
    Methods
    =======
    run_strategy:
        Runs the backtest for the mean reversion-based strategy
    """
    
    def run_strategy(self, SMA: int = 50, 
                    threshold: float = 5.0) -> Tuple[float, float]:
        """
        Backtest the mean reversion strategy.
        
        Parameters
        ==========
        SMA: int
            Simple moving average window
        threshold: float
            Distance threshold from SMA for signals
            
        Returns
        =======
        tuple
            (absolute_performance, over_underperformance)
        """
        data = self.data.copy().dropna()
        data['sma'] = data['price'].rolling(SMA).mean()
        data['distance'] = data['price'] - data['sma']
        data.dropna(inplace=True)
        
        # Sell signals: price above SMA by threshold
        data['position'] = np.where(data['distance'] > threshold, -1, np.nan)
        # Buy signals: price below SMA by threshold
        data['position'] = np.where(data['distance'] < -threshold, 1,
                                   data['position'])
        # Crossing of current price and SMA
        data['position'] = np.where(data['distance'] * 
                                   data['distance'].shift(1) < 0, 0,
                                   data['position'])
        
        data['position'] = data['position'].ffill().fillna(0)
        data['strategy'] = data['position'].shift(1) * data['return']
        
        # Determine when a trade takes place
        trades = data['position'].diff().fillna(0) != 0
        # Subtract transaction costs
        data['strategy'][trades] -= self.tc
        
        data['creturns'] = self.amount * \
            data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = self.amount * \
            data['strategy'].cumsum().apply(np.exp)
        
        self.results = data
        
        # Absolute performance
        aperf = self.results['cstrategy'].iloc[-1]
        # Out-/underperformance
        operf = aperf - self.results['creturns'].iloc[-1]
        
        return round(aperf, 2), round(operf, 2)
