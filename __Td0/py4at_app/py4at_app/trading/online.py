"""
Online real-time trading algorithms and stream processing
"""

import numpy as np
import pandas as pd
from typing import Optional, Callable, Dict, Any
import threading
import time


class OnlineAlgorithm:
    """
    Base class for online (real-time) trading algorithms.
    Processes streaming market data and generates trading signals.
    
    Methods
    =======
    process_tick:
        Process a single tick/data point
    generate_signal:
        Generate trading signal based on latest data
    calculate_momentum:
        Calculate momentum-based signal
    """
    
    def __init__(self, instrument: str, window: int = 5, 
                 momentum_period: int = 3):
        """
        Initialize Online Algorithm.
        
        Parameters
        ==========
        instrument: str
            Financial instrument to trade
        window: int
            Window size for aggregation
        momentum_period: int
            Period for momentum calculation
        """
        self.instrument = instrument
        self.window = window
        self.momentum_period = momentum_period
        self.data = pd.DataFrame()
        self.returns = None
        self.momentum = None
        self.position = 0
        self.trades = []
    
    def process_tick(self, timestamp: Any, price: float) -> Optional[int]:
        """
        Process a single tick of market data.
        
        Parameters
        ==========
        timestamp: Any
            Timestamp of the tick
        price: float
            Price of the tick
            
        Returns
        =======
        int or None
            Trading signal (-1, 0, 1) or None if insufficient data
        """
        # Add new data point
        df = pd.DataFrame({'price': [price]}, index=[pd.Timestamp(timestamp)])
        self.data = pd.concat([self.data, df])
        
        # Aggregate to the specified window
        aggregated = self.data.resample(f'{self.window}s', 
                                       label='right').last().ffill()
        
        if len(aggregated) <= self.momentum_period:
            return None
        
        # Calculate returns and momentum
        aggregated['return'] = np.log(aggregated['price'] / 
                                      aggregated['price'].shift(1))
        aggregated['momentum'] = np.sign(
            aggregated['return'].rolling(self.momentum_period).mean()
        )
        
        signal = aggregated['momentum'].iloc[-1]
        return int(signal) if signal != 0 else 0
    
    def generate_signal(self) -> int:
        """
        Generate trading signal based on current data.
        
        Returns
        =======
        int
            Trading signal (-1, 0, 1)
        """
        if len(self.data) < self.momentum_period + 1:
            return 0
        
        # Calculate momentum
        returns = np.log(self.data['price'] / 
                        self.data['price'].shift(1))
        momentum = returns.rolling(self.momentum_period).mean()
        
        signal = np.sign(momentum.iloc[-1])
        return int(signal) if signal != 0 else 0
    
    def calculate_momentum(self, period: Optional[int] = None) -> float:
        """
        Calculate momentum indicator.
        
        Parameters
        ==========
        period: int, optional
            Period for momentum (default: self.momentum_period)
            
        Returns
        =======
        float
            Momentum value
        """
        if period is None:
            period = self.momentum_period
        
        if len(self.data) < period + 1:
            return 0.0
        
        returns = np.log(self.data['price'] / 
                        self.data['price'].shift(1))
        momentum = returns.rolling(period).mean()
        
        return float(momentum.iloc[-1])
    
    def record_trade(self, timestamp: Any, signal: int, 
                    price: float, units: int):
        """
        Record a trade execution.
        
        Parameters
        ==========
        timestamp: Any
            Timestamp of the trade
        signal: int
            Trading signal
        price: float
            Execution price
        units: int
            Number of units traded
        """
        trade = {
            'timestamp': timestamp,
            'signal': signal,
            'price': price,
            'units': units,
            'value': price * units
        }
        self.trades.append(trade)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get trading statistics.
        
        Returns
        =======
        dict
            Dictionary with trading statistics
        """
        if not self.trades:
            return {'trades': 0, 'total_value': 0}
        
        trades_df = pd.DataFrame(self.trades)
        return {
            'trades': len(self.trades),
            'total_value': trades_df['value'].sum(),
            'avg_price': trades_df['price'].mean(),
            'first_trade': trades_df['timestamp'].iloc[0],
            'last_trade': trades_df['timestamp'].iloc[-1]
        }


class TickDataProcessor:
    """
    Processes tick-level market data in real-time.
    
    Methods
    =======
    process_stream:
        Process a stream of tick data
    aggregate_bars:
        Aggregate ticks into bar data
    """
    
    def __init__(self, instruments: list):
        """
        Initialize Tick Data Processor.
        
        Parameters
        ==========
        instruments: list
            List of instrument symbols to process
        """
        self.instruments = instruments
        self.data = {instr: pd.DataFrame() for instr in instruments}
        self.callbacks = {instr: [] for instr in instruments}
    
    def add_callback(self, instrument: str, callback: Callable):
        """
        Register a callback function for an instrument.
        
        Parameters
        ==========
        instrument: str
            Instrument symbol
        callback: Callable
            Function to call when new data arrives
        """
        if instrument in self.callbacks:
            self.callbacks[instrument].append(callback)
    
    def process_tick(self, instrument: str, timestamp: Any, 
                    bid: float, ask: float):
        """
        Process a single tick.
        
        Parameters
        ==========
        instrument: str
            Instrument symbol
        timestamp: Any
            Timestamp of the tick
        bid: float
            Bid price
        ask: float
            Ask price
        """
        if instrument not in self.data:
            return
        
        # Create tick data
        tick = pd.DataFrame(
            {'bid': [bid], 'ask': [ask]},
            index=[pd.Timestamp(timestamp)]
        )
        
        # Append to existing data
        self.data[instrument] = pd.concat([self.data[instrument], tick])
        
        # Calculate mid price
        tick['mid'] = (bid + ask) / 2
        
        # Call registered callbacks
        for callback in self.callbacks[instrument]:
            callback(self.data[instrument])
    
    def aggregate_bars(self, instrument: str, timeframe: str = '1min',
                      label: str = 'right') -> pd.DataFrame:
        """
        Aggregate tick data into bars.
        
        Parameters
        ==========
        instrument: str
            Instrument symbol
        timeframe: str
            Timeframe for aggregation (e.g., '1min', '5min')
        label: str
            Label for aggregation ('right' or 'left')
            
        Returns
        =======
        pd.DataFrame
            Bar data
        """
        if instrument not in self.data or len(self.data[instrument]) == 0:
            return pd.DataFrame()
        
        data = self.data[instrument].copy()
        
        # Resample to specified timeframe
        bars = data.resample(timeframe, label=label).agg({
            'bid': 'first',
            'ask': 'first',
            'mid': ['first', 'high', 'low', 'last', 'mean', 'std']
        })
        
        return bars
