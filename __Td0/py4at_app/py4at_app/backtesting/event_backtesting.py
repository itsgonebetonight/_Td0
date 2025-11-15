"""
Event-based backtesting classes for long-only and long-short strategies
"""

from .base import BacktestBase
import numpy as np


class BacktestLongOnly(BacktestBase):
    """
    Event-based backtesting class for long-only trading strategies.
    Inherits from BacktestBase.
    
    Methods
    =======
    run_sma_strategy:
        Backtests a SMA-based strategy
    run_momentum_strategy:
        Backtests a momentum-based strategy
    run_mean_reversion_strategy:
        Backtests a mean reversion strategy
    """
    
    def run_sma_strategy(self, SMA1: int, SMA2: int):
        """
        Backtest a Simple Moving Average (SMA) based strategy.
        Only goes long when SMA1 > SMA2, exits when SMA1 < SMA2.
        
        Parameters
        ==========
        SMA1: int
            Shorter SMA window
        SMA2: int
            Longer SMA window
        """
        msg = f'\n\nRunning SMA strategy | SMA1={SMA1} & SMA2={SMA2}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        
        if self.verbose:
            print(msg)
            print('=' * 55)
        
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        
        self.data['SMA1'] = self.data['price'].rolling(SMA1).mean()
        self.data['SMA2'] = self.data['price'].rolling(SMA2).mean()
        
        for bar in range(SMA2, len(self.data)):
            if self.position == 0:
                if self.data['SMA1'].iloc[bar] > self.data['SMA2'].iloc[bar]:
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1  # long position
            
            elif self.position == 1:
                if self.data['SMA1'].iloc[bar] < self.data['SMA2'].iloc[bar]:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0  # market neutral
        
        self.close_out(bar)
    
    def run_momentum_strategy(self, momentum: int):
        """
        Backtest a momentum-based strategy.
        Goes long when momentum is positive, exits when momentum is negative.
        
        Parameters
        ==========
        momentum: int
            Number of days for mean return calculation
        """
        msg = f'\n\nRunning momentum strategy | {momentum} days'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        
        if self.verbose:
            print(msg)
            print('=' * 55)
        
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        
        self.data['momentum'] = self.data['return'].rolling(momentum).mean()
        
        for bar in range(momentum, len(self.data)):
            if self.position == 0:
                if self.data['momentum'].iloc[bar] > 0:
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1  # long position
            
            elif self.position == 1:
                if self.data['momentum'].iloc[bar] < 0:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0  # market neutral
        
        self.close_out(bar)
    
    def run_mean_reversion_strategy(self, SMA: int, threshold: float):
        """
        Backtest a mean reversion strategy.
        Goes long when price is below SMA - threshold.
        Exits when price crosses above SMA.
        
        Parameters
        ==========
        SMA: int
            Simple moving average window
        threshold: float
            Absolute value for deviation-based signal
        """
        msg = f'\n\nRunning mean reversion strategy | '
        msg += f'SMA={SMA} & thr={threshold}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        
        if self.verbose:
            print(msg)
            print('=' * 55)
        
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        
        self.data['SMA'] = self.data['price'].rolling(SMA).mean()
        
        for bar in range(SMA, len(self.data)):
            if self.position == 0:
                if (self.data['price'].iloc[bar] <
                    self.data['SMA'].iloc[bar] - threshold):
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1  # long position
            
            elif self.position == 1:
                if self.data['price'].iloc[bar] >= self.data['SMA'].iloc[bar]:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0  # market neutral
        
        self.close_out(bar)


class BacktestLongShort(BacktestBase):
    """
    Event-based backtesting class for long-short trading strategies.
    Inherits from BacktestBase.
    Allows both long and short positions.
    
    Methods
    =======
    go_long:
        Implements going long
    go_short:
        Implements going short
    run_sma_strategy:
        Backtests a SMA-based long-short strategy
    run_momentum_strategy:
        Backtests a momentum-based long-short strategy
    run_mean_reversion_strategy:
        Backtests a mean reversion long-short strategy
    """
    
    def go_long(self, bar: int, units: int = None, amount: float = None):
        """
        Implement going long.
        If in short position, closes it first.
        
        Parameters
        ==========
        bar: int
            Bar index
        units: int, optional
            Number of units to buy
        amount: float, optional
            Amount to invest (or 'all' for all available capital)
        """
        if self.position == -1:
            self.place_buy_order(bar, units=-self.units)
        
        if units:
            self.place_buy_order(bar, units=units)
        elif amount:
            if amount == 'all':
                amount = self.amount
            self.place_buy_order(bar, amount=amount)
    
    def go_short(self, bar: int, units: int = None, amount: float = None):
        """
        Implement going short.
        If in long position, closes it first.
        
        Parameters
        ==========
        bar: int
            Bar index
        units: int, optional
            Number of units to sell
        amount: float, optional
            Amount to raise from selling (or 'all')
        """
        if self.position == 1:
            self.place_sell_order(bar, units=self.units)
        
        if units:
            self.place_sell_order(bar, units=units)
        elif amount:
            if amount == 'all':
                amount = self.amount
            self.place_sell_order(bar, amount=amount)
    
    def run_sma_strategy(self, SMA1: int, SMA2: int):
        """
        Backtest a long-short SMA strategy.
        Goes long when SMA1 > SMA2, short when SMA1 < SMA2.
        
        Parameters
        ==========
        SMA1: int
            Shorter SMA window
        SMA2: int
            Longer SMA window
        """
        msg = f'\n\nRunning long-short SMA strategy | SMA1={SMA1} & SMA2={SMA2}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        
        if self.verbose:
            print(msg)
            print('=' * 55)
        
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        
        self.data['SMA1'] = self.data['price'].rolling(SMA1).mean()
        self.data['SMA2'] = self.data['price'].rolling(SMA2).mean()
        
        for bar in range(SMA2, len(self.data)):
            if self.position in [0, -1]:
                if self.data['SMA1'].iloc[bar] > self.data['SMA2'].iloc[bar]:
                    self.go_long(bar, amount='all')
                    self.position = 1  # long position
            
            if self.position in [0, 1]:
                if self.data['SMA1'].iloc[bar] < self.data['SMA2'].iloc[bar]:
                    self.go_short(bar, amount='all')
                    self.position = -1  # short position
        
        self.close_out(bar)
    
    def run_momentum_strategy(self, momentum: int):
        """
        Backtest a long-short momentum strategy.
        Goes long when momentum > 0, short when momentum <= 0.
        
        Parameters
        ==========
        momentum: int
            Number of days for mean return calculation
        """
        msg = f'\n\nRunning long-short momentum strategy | {momentum} days'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        
        if self.verbose:
            print(msg)
            print('=' * 55)
        
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        
        self.data['momentum'] = self.data['return'].rolling(momentum).mean()
        
        for bar in range(momentum, len(self.data)):
            if self.position in [0, -1]:
                if self.data['momentum'].iloc[bar] > 0:
                    self.go_long(bar, amount='all')
                    self.position = 1  # long position
            
            if self.position in [0, 1]:
                if self.data['momentum'].iloc[bar] <= 0:
                    self.go_short(bar, amount='all')
                    self.position = -1  # short position
        
        self.close_out(bar)
    
    def run_mean_reversion_strategy(self, SMA: int, threshold: float):
        """
        Backtest a long-short mean reversion strategy.
        Goes long when price < SMA - threshold.
        Goes short when price > SMA + threshold.
        
        Parameters
        ==========
        SMA: int
            Simple moving average window
        threshold: float
            Absolute value for deviation-based signal
        """
        msg = f'\n\nRunning long-short mean reversion strategy | '
        msg += f'SMA={SMA} & thr={threshold}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        
        if self.verbose:
            print(msg)
            print('=' * 55)
        
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        
        self.data['SMA'] = self.data['price'].rolling(SMA).mean()
        
        for bar in range(SMA, len(self.data)):
            if self.position == 0:
                if (self.data['price'].iloc[bar] <
                    self.data['SMA'].iloc[bar] - threshold):
                    self.go_long(bar, amount=self.initial_amount)
                    self.position = 1  # long position
                
                elif (self.data['price'].iloc[bar] >
                      self.data['SMA'].iloc[bar] + threshold):
                    self.go_short(bar, amount=self.initial_amount)
                    self.position = -1  # short position
            
            elif self.position == 1:
                if self.data['price'].iloc[bar] >= self.data['SMA'].iloc[bar]:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0  # market neutral
            
            elif self.position == -1:
                if self.data['price'].iloc[bar] <= self.data['SMA'].iloc[bar]:
                    self.place_buy_order(bar, units=-self.units)
                    self.position = 0  # market neutral
        
        self.close_out(bar)
