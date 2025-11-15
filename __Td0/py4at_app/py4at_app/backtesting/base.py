"""
Base class for event-based backtesting of trading strategies
"""

import numpy as np
import pandas as pd
from typing import Optional


class BacktestBase:
    """
    Base class for event-based backtesting of trading strategies.
    
    Attributes
    ==========
    symbol: str
        Financial instrument to be used (e.g., 'EUR/USD', 'AAPL')
    start: str
        Start date for data selection (format: 'YYYY-MM-DD')
    end: str
        End date for data selection (format: 'YYYY-MM-DD')
    amount: float
        Initial capital amount to be invested
    ftc: float
        Fixed transaction costs per trade (in currency units)
    ptc: float
        Proportional transaction costs per trade (e.g., 0.001 for 0.1%)
    verbose: bool
        If True, prints trading information
        
    Methods
    =======
    get_data:
        Retrieves and prepares the base data set
    plot_data:
        Plots the closing price for the symbol
    get_date_price:
        Returns the date and price for the given bar
    print_balance:
        Prints out the current cash balance
    print_net_wealth:
        Prints out the current net wealth
    place_buy_order:
        Places a buy order
    place_sell_order:
        Places a sell order
    close_out:
        Closes out a long or short position
    """
    
    def __init__(self, symbol: str, start: str, end: str, amount: float,
                 ftc: float = 0.0, ptc: float = 0.0, verbose: bool = True):
        """
        Initialize the BacktestBase instance.
        
        Parameters
        ==========
        symbol: str
            Financial instrument symbol
        start: str
            Start date for backtesting
        end: str
            End date for backtesting
        amount: float
            Initial capital amount
        ftc: float
            Fixed transaction costs per trade
        ptc: float
            Proportional transaction costs per trade
        verbose: bool
            Verbosity flag
        """
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_amount = amount
        self.amount = amount
        self.ftc = ftc
        self.ptc = ptc
        self.units = 0
        self.position = 0
        self.trades = 0
        self.verbose = verbose
        self.data = None
        self.get_data()
    
    def get_data(self):
        """
        Retrieves and prepares the data.
        Expects data to be loaded from a CSV file or external source.
        """
        # This should be overridden or data should be loaded externally
        pass
    
    def set_data(self, data: pd.DataFrame):
        """
        Set data directly (for testing or custom data sources).
        
        Parameters
        ==========
        data: pd.DataFrame
            DataFrame with 'price' column and DatetimeIndex
        """
        self.data = data.copy()
        if 'return' not in self.data.columns:
            self.data['return'] = np.log(self.data['price'] / 
                                         self.data['price'].shift(1))
        self.data = self.data.dropna()
    
    def plot_data(self, cols=None):
        """
        Plots the closing prices for symbol.
        
        Parameters
        ==========
        cols: list, optional
            Columns to plot (default: ['price'])
        """
        if cols is None:
            cols = ['price']
        
        if self.data is not None:
            self.data[cols].plot(figsize=(10, 6), title=self.symbol)
    
    def get_date_price(self, bar: int) -> tuple:
        """
        Return date and price for a given bar.
        
        Parameters
        ==========
        bar: int
            Bar index
            
        Returns
        =======
        tuple
            (date_string, price)
        """
        date = str(self.data.index[bar])[:10]
        price = self.data['price'].iloc[bar]
        return date, price
    
    def print_balance(self, bar: int):
        """
        Print out current cash balance information.
        
        Parameters
        ==========
        bar: int
            Bar index
        """
        date, price = self.get_date_price(bar)
        print(f'{date} | current balance {self.amount:.2f}')
    
    def print_net_wealth(self, bar: int):
        """
        Print out current net wealth (cash + position value).
        
        Parameters
        ==========
        bar: int
            Bar index
        """
        date, price = self.get_date_price(bar)
        net_wealth = self.units * price + self.amount
        print(f'{date} | current net wealth {net_wealth:.2f}')
    
    def place_buy_order(self, bar: int, units: Optional[int] = None,
                       amount: Optional[float] = None):
        """
        Place a buy order.
        
        Parameters
        ==========
        bar: int
            Bar index at which to place the order
        units: int, optional
            Number of units to buy
        amount: float, optional
            Amount of capital to invest
        """
        date, price = self.get_date_price(bar)
        
        if units is None:
            if amount is None:
                raise ValueError("Either units or amount must be specified")
            units = int(amount / price)
        
        self.amount -= (units * price) * (1 + self.ptc) + self.ftc
        self.units += units
        self.trades += 1
        
        if self.verbose:
            print(f'{date} | buying {units} units at {price:.2f}')
            self.print_balance(bar)
            self.print_net_wealth(bar)
    
    def place_sell_order(self, bar: int, units: Optional[int] = None,
                        amount: Optional[float] = None):
        """
        Place a sell order.
        
        Parameters
        ==========
        bar: int
            Bar index at which to place the order
        units: int, optional
            Number of units to sell
        amount: float, optional
            Amount to raise from selling
        """
        date, price = self.get_date_price(bar)
        
        if units is None:
            if amount is None:
                raise ValueError("Either units or amount must be specified")
            units = int(amount / price)
        
        self.amount += (units * price) * (1 - self.ptc) - self.ftc
        self.units -= units
        self.trades += 1
        
        if self.verbose:
            print(f'{date} | selling {units} units at {price:.2f}')
            self.print_balance(bar)
            self.print_net_wealth(bar)
    
    def close_out(self, bar: int):
        """
        Close out any remaining position.
        
        Parameters
        ==========
        bar: int
            Bar index at which to close the position
        """
        date, price = self.get_date_price(bar)
        
        if self.units != 0:
            self.amount += self.units * price
            self.units = 0
            self.trades += 1
            
            if self.verbose:
                print(f'{date} | closing position')
                print('=' * 55)
        
        if self.verbose:
            print('Final balance   [$] {:.2f}'.format(self.amount))
            perf = ((self.amount - self.initial_amount) /
                    self.initial_amount * 100)
            print('Net Performance [%] {:.2f}'.format(perf))
            print('Trades Executed [#] {}'.format(self.trades))
            print('=' * 55)
    
    def get_performance(self) -> float:
        """
        Get the performance of the backtest.
        
        Returns
        =======
        float
            Percentage return
        """
        return ((self.amount - self.initial_amount) / 
                self.initial_amount * 100)
