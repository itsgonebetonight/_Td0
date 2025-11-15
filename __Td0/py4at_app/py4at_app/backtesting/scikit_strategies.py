"""
Machine Learning-based vectorized backtesting strategies using scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from typing import Optional, Tuple, Literal


class LRVectorBacktester:
    """
    Vectorized backtesting class for Linear Regression-based strategies.
    
    Attributes
    ==========
    symbol: str
        Financial instrument symbol
    start: str
        Start date for data retrieval
    end: str
        End date for data retrieval
    amount: float
        Amount to be invested at the beginning
    tc: float
        Proportional transaction costs
        
    Methods
    =======
    run_strategy:
        Runs the backtest for the regression-based strategy
    plot_results:
        Plots the performance of the strategy
    """
    
    def __init__(self, symbol: str, start: str, end: str,
                 amount: float, tc: float = 0.0,
                 data: Optional[pd.DataFrame] = None):
        """
        Initialize Linear Regression Vector Backtester.
        
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
        self.lags = 3
        self.reg = None
        
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
        raw['returns'] = np.log(raw['price'] / raw['price'].shift(1))
        
        self.data = raw.dropna()
    
    def set_data(self, data: pd.DataFrame):
        """Set data directly."""
        self.data = data.copy()
        if 'price' not in self.data.columns:
            if len(self.data.columns) > 0:
                self.data.rename(columns={self.data.columns[0]: 'price'}, 
                               inplace=True)
        
        self.data['returns'] = np.log(self.data['price'] / 
                                      self.data['price'].shift(1))
        self.data = self.data.dropna()
    
    def select_data(self, start: str, end: str) -> pd.DataFrame:
        """Select a subset of data."""
        data = self.data[(self.data.index >= start) &
                        (self.data.index <= end)].copy()
        return data
    
    def prepare_lags(self, start: str, end: str):
        """Prepare lagged data for regression."""
        data = self.select_data(start, end)
        self.cols = []
        
        for lag in range(1, self.lags + 1):
            col = f'lag_{lag}'
            data[col] = data['returns'].shift(lag)
            self.cols.append(col)
        
        data.dropna(inplace=True)
        self.lagged_data = data
    
    def fit_model(self, start: str, end: str):
        """Implement the regression step."""
        self.prepare_lags(start, end)
        self.reg = np.linalg.lstsq(
            self.lagged_data[self.cols],
            np.sign(self.lagged_data['returns']),
            rcond=None
        )[0]
    
    def run_strategy(self, start_in: str, end_in: str,
                    start_out: str, end_out: str,
                    lags: int = 3) -> Tuple[float, float]:
        """
        Backtest the regression-based strategy.
        
        Parameters
        ==========
        start_in: str
            Start date for in-sample period
        end_in: str
            End date for in-sample period
        start_out: str
            Start date for out-sample period
        end_out: str
            End date for out-sample period
        lags: int
            Number of lags for features
            
        Returns
        =======
        tuple
            (absolute_performance, over_underperformance)
        """
        self.lags = lags
        
        # Fit the model on in-sample data
        self.fit_model(start_in, end_in)
        
        # Prepare out-sample data
        self.results = self.select_data(start_out, end_out).iloc[lags:]
        self.prepare_lags(start_out, end_out)
        
        # Make predictions
        prediction = np.sign(np.dot(self.lagged_data[self.cols], self.reg))
        self.results['prediction'] = prediction
        self.results['strategy'] = (self.results['prediction'] *
                                   self.results['returns'])
        
        # Determine when a trade takes place
        trades = self.results['prediction'].diff().fillna(0) != 0
        # Subtract transaction costs
        self.results.loc[trades, 'strategy'] -= self.tc
        
        self.results['creturns'] = (self.amount *
                                   self.results['returns'].cumsum().apply(np.exp))
        self.results['cstrategy'] = (self.amount *
                                    self.results['strategy'].cumsum().apply(np.exp))
        
        # Performance calculation
        aperf = self.results['cstrategy'].iloc[-1]
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


class ScikitVectorBacktester:
    """
    Vectorized backtesting class for Machine Learning-based strategies
    using scikit-learn models.
    
    Attributes
    ==========
    symbol: str
        Financial instrument symbol
    start: str
        Start date for data retrieval
    end: str
        End date for data retrieval
    amount: float
        Amount to be invested at the beginning
    tc: float
        Proportional transaction costs
    model: str
        Either 'regression' or 'logistic' for model type
        
    Methods
    =======
    run_strategy:
        Runs the backtest for the ML-based strategy
    plot_results:
        Plots the performance of the strategy
    """
    
    def __init__(self, symbol: str, start: str, end: str,
                 amount: float, tc: float = 0.0,
                 model: Literal['regression', 'logistic'] = 'regression',
                 data: Optional[pd.DataFrame] = None):
        """
        Initialize Scikit Vector Backtester.
        
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
        model: str
            Model type ('regression' or 'logistic')
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
        self.lags = 3
        self.feature_columns = []
        self.data_subset = None
        
        # Initialize the model
        if model == 'regression':
            self.model = linear_model.LinearRegression()
        elif model == 'logistic':
            self.model = linear_model.LogisticRegression(
                C=1e6, solver='lbfgs', multi_class='ovr', max_iter=1000
            )
        else:
            raise ValueError('Model not known or not yet implemented.')
        
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
        raw['returns'] = np.log(raw['price'] / raw['price'].shift(1))
        
        self.data = raw.dropna()
    
    def set_data(self, data: pd.DataFrame):
        """Set data directly."""
        self.data = data.copy()
        if 'price' not in self.data.columns:
            if len(self.data.columns) > 0:
                self.data.rename(columns={self.data.columns[0]: 'price'}, 
                               inplace=True)
        
        self.data['returns'] = np.log(self.data['price'] / 
                                      self.data['price'].shift(1))
        self.data = self.data.dropna()
    
    def select_data(self, start: str, end: str) -> pd.DataFrame:
        """Select a subset of data."""
        data = self.data[(self.data.index >= start) &
                        (self.data.index <= end)].copy()
        return data
    
    def prepare_features(self, start: str, end: str):
        """Prepare feature columns for model fitting."""
        self.data_subset = self.select_data(start, end)
        self.feature_columns = []
        
        for lag in range(1, self.lags + 1):
            col = f'lag_{lag}'
            self.data_subset[col] = self.data_subset['returns'].shift(lag)
            self.feature_columns.append(col)
        
        self.data_subset.dropna(inplace=True)
    
    def fit_model(self, start: str, end: str):
        """Implement the model fitting step."""
        self.prepare_features(start, end)
        self.model.fit(
            self.data_subset[self.feature_columns],
            np.sign(self.data_subset['returns'])
        )
    
    def run_strategy(self, start_in: str, end_in: str,
                    start_out: str, end_out: str,
                    lags: int = 3) -> Tuple[float, float]:
        """
        Backtest the ML-based strategy.
        
        Parameters
        ==========
        start_in: str
            Start date for in-sample period
        end_in: str
            End date for in-sample period
        start_out: str
            Start date for out-sample period
        end_out: str
            End date for out-sample period
        lags: int
            Number of lags for features
            
        Returns
        =======
        tuple
            (absolute_performance, over_underperformance)
        """
        self.lags = lags
        
        # Fit model on in-sample data
        self.fit_model(start_in, end_in)
        
        # Prepare out-sample data
        self.prepare_features(start_out, end_out)
        
        # Make predictions
        prediction = self.model.predict(
            self.data_subset[self.feature_columns]
        )
        self.data_subset['prediction'] = prediction
        self.data_subset['strategy'] = (self.data_subset['prediction'] *
                                       self.data_subset['returns'])
        
        # Determine when a trade takes place
        trades = self.data_subset['prediction'].diff().fillna(0) != 0
        # Subtract transaction costs
        self.data_subset.loc[trades, 'strategy'] -= self.tc
        
        self.data_subset['creturns'] = (
            self.amount *
            self.data_subset['returns'].cumsum().apply(np.exp)
        )
        self.data_subset['cstrategy'] = (
            self.amount *
            self.data_subset['strategy'].cumsum().apply(np.exp)
        )
        
        self.results = self.data_subset
        
        # Performance calculation
        aperf = self.results['cstrategy'].iloc[-1]
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
