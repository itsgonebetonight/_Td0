"""
Momentum-based trading strategies for live markets
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime


class MomentumTrader:
    """
    Live momentum trading strategy implementation.
    Trades based on momentum signals calculated from real-time data.
    
    Attributes
    ==========
    instrument: str
        Trading instrument
    bar_length: str or int
        Timeframe for bar aggregation
    momentum: int
        Period for momentum calculation
    units: int
        Number of units to trade per signal
        
    Methods
    =======
    on_tick:
        Process incoming tick data
    calculate_momentum:
        Calculate momentum indicator
    place_order:
        Place a trading order
    close_position:
        Close current position
    """
    
    def __init__(self, instrument: str, bar_length: int = 60,
                 momentum: int = 6, units: int = 100000):
        """
        Initialize Momentum Trader.
        
        Parameters
        ==========
        instrument: str
            Trading instrument (e.g., 'EUR_USD')
        bar_length: int
            Bar length in seconds (default: 60 = 1 minute)
        momentum: int
            Momentum period in bars
        units: int
            Number of units to trade
        """
        self.instrument = instrument
        self.bar_length = bar_length
        self.momentum = momentum
        self.units = units
        
        # Position tracking
        self.position = 0  # 1 for long, -1 for short, 0 for flat
        self.entry_price = None
        self.entry_time = None
        
        # Data storage
        self.raw_data = pd.DataFrame()
        self.bar_data = pd.DataFrame()
        self.trades = []
        
        # Configuration
        self.min_length = self.momentum + 1
        self.verbose = True
    
    def on_tick(self, timestamp: Any, bid: float, ask: float) -> Optional[int]:
        """
        Process incoming tick data.
        
        Parameters
        ==========
        timestamp: Any
            Timestamp of the tick
        bid: float
            Bid price
        ask: float
            Ask price
            
        Returns
        =======
        int or None
            Signal (-1, 0, 1) or None if no signal
        """
        # Store raw tick data
        tick = pd.DataFrame(
            {'bid': [bid], 'ask': [ask]},
            index=[pd.Timestamp(timestamp)]
        )
        self.raw_data = pd.concat([self.raw_data, tick])
        
        # Aggregate to bars
        self.bar_data = self.raw_data.resample(
            f'{self.bar_length}s', label='right'
        ).last().ffill().iloc[:-1]
        
        # Calculate indicators
        self.bar_data['mid'] = self.bar_data.mean(axis=1)
        self.bar_data['returns'] = np.log(
            self.bar_data['mid'] / self.bar_data['mid'].shift(1)
        )
        
        # Calculate momentum signal
        if len(self.bar_data) > self.min_length:
            self.bar_data['position'] = np.sign(
                self.bar_data['returns'].rolling(self.momentum).mean()
            )
            
            signal = self.bar_data['position'].iloc[-1]
            
            # Execute orders based on signal
            return self._execute_signal(int(signal), timestamp, 
                                       self.bar_data['mid'].iloc[-1])
        
        return None
    
    def _execute_signal(self, signal: int, timestamp: Any, 
                       price: float) -> int:
        """
        Execute trading orders based on signal.
        
        Parameters
        ==========
        signal: int
            Trading signal (-1, 0, 1)
        timestamp: Any
            Current timestamp
        price: float
            Current price
            
        Returns
        =======
        int
            Order units placed (0 if no order)
        """
        order_units = 0
        
        if signal == 1:  # Go long
            if self.position == 0:
                order_units = self.units
                self.position = 1
                self._record_trade('BUY', order_units, price, timestamp)
            elif self.position == -1:
                # Close short and open long
                order_units = self.units * 2
                self.position = 1
                self._record_trade('CLOSE SHORT', self.units, price, timestamp)
                self._record_trade('BUY', self.units, price, timestamp)
        
        elif signal == -1:  # Go short
            if self.position == 0:
                order_units = -self.units
                self.position = -1
                self._record_trade('SELL', self.units, price, timestamp)
            elif self.position == 1:
                # Close long and open short
                order_units = -self.units * 2
                self.position = -1
                self._record_trade('CLOSE LONG', self.units, price, timestamp)
                self._record_trade('SELL', self.units, price, timestamp)
        
        return order_units
    
    def _record_trade(self, action: str, units: int, price: float,
                     timestamp: Any):
        """Record a trade in the trades log."""
        trade = {
            'timestamp': timestamp,
            'action': action,
            'units': units,
            'price': price,
            'value': units * price
        }
        self.trades.append(trade)
        
        if self.verbose:
            print(f'{timestamp} | {action}: {units} units @ {price:.5f}')
    
    def calculate_momentum(self, period: Optional[int] = None) -> float:
        """
        Calculate current momentum value.
        
        Parameters
        ==========
        period: int, optional
            Period for calculation (default: self.momentum)
            
        Returns
        =======
        float
            Current momentum value
        """
        if period is None:
            period = self.momentum
        
        if len(self.bar_data) < period + 1:
            return 0.0
        
        momentum = np.sign(
            self.bar_data['returns'].rolling(period).mean()
        )
        
        return float(momentum.iloc[-1])
    
    def place_order(self, side: str, units: int) -> Dict[str, Any]:
        """
        Place a manual order.
        
        Parameters
        ==========
        side: str
            'BUY' or 'SELL'
        units: int
            Number of units
            
        Returns
        =======
        dict
            Order information
        """
        if len(self.bar_data) == 0:
            return {}
        
        price = self.bar_data['mid'].iloc[-1]
        timestamp = self.bar_data.index[-1]
        
        if side == 'BUY':
            order_units = units
            self.position = 1
        elif side == 'SELL':
            order_units = -units
            self.position = -1
        else:
            return {}
        
        self._record_trade(side, units, price, timestamp)
        
        return {
            'side': side,
            'units': units,
            'price': price,
            'timestamp': timestamp,
            'position': self.position
        }
    
    def close_position(self) -> Optional[Dict[str, Any]]:
        """
        Close current position.
        
        Returns
        =======
        dict or None
            Order information if position was closed
        """
        if self.position == 0:
            return None
        
        if len(self.bar_data) == 0:
            return None
        
        price = self.bar_data['mid'].iloc[-1]
        timestamp = self.bar_data.index[-1]
        units = abs(self.position * self.units)
        
        if self.position == 1:
            self._record_trade('CLOSE LONG', units, price, timestamp)
            side = 'SELL'
        else:
            self._record_trade('CLOSE SHORT', units, price, timestamp)
            side = 'BUY'
        
        self.position = 0
        
        return {
            'side': side,
            'units': units,
            'price': price,
            'timestamp': timestamp,
            'position': self.position
        }
    
    def get_performance(self) -> Dict[str, Any]:
        """
        Calculate trading performance metrics.
        
        Returns
        =======
        dict
            Performance metrics
        """
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate PnL
        entry_trades = trades_df[trades_df['action'].isin(['BUY', 'SELL'])]
        if len(entry_trades) == 0:
            return {}
        
        # Group by position
        entry_price = entry_trades['price'].iloc[0]
        last_price = self.bar_data['mid'].iloc[-1] if len(
            self.bar_data) > 0 else entry_price
        
        pnl = (last_price - entry_price) * self.units
        
        return {
            'total_trades': len(self.trades),
            'buy_trades': len(trades_df[trades_df['action'] == 'BUY']),
            'sell_trades': len(trades_df[trades_df['action'] == 'SELL']),
            'current_position': self.position,
            'current_pnl': pnl,
            'entry_price': entry_price,
            'current_price': last_price,
            'trades': trades_df.to_dict('records')
        }
    
    def reset(self):
        """Reset the trader state."""
        self.position = 0
        self.entry_price = None
        self.entry_time = None
        self.raw_data = pd.DataFrame()
        self.bar_data = pd.DataFrame()
        self.trades = []
