"""
Strategy monitoring and logging for live trading systems
"""

import logging
import json
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
from io import StringIO


class StrategyMonitor:
    """
    Monitor and log trading strategy execution in real-time.
    
    Methods
    =======
    log_trade:
        Log a trade execution
    log_signal:
        Log a trading signal
    log_error:
        Log an error
    get_statistics:
        Get current trading statistics
    export_log:
        Export logs to file
    """
    
    def __init__(self, strategy_name: str, log_file: Optional[str] = None):
        """
        Initialize Strategy Monitor.
        
        Parameters
        ==========
        strategy_name: str
            Name of the strategy being monitored
        log_file: str, optional
            Path to log file (if None, uses string buffer)
        """
        self.strategy_name = strategy_name
        self.log_file = log_file
        self.start_time = datetime.now()
        
        # Initialize logger
        self.logger = logging.getLogger(strategy_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler if log_file provided
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        
        # Setup console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # Data storage
        self.trades = []
        self.signals = []
        self.errors = []
        self.events = []
        self.lock = threading.Lock()
    
    def log_trade(self, timestamp: Any, side: str, units: int, 
                 price: float, pnl: Optional[float] = None,
                 comment: str = ''):
        """
        Log a trade execution.
        
        Parameters
        ==========
        timestamp: Any
            Trade timestamp
        side: str
            'BUY' or 'SELL'
        units: int
            Number of units
        price: float
            Execution price
        pnl: float, optional
            Profit/loss if available
        comment: str
            Additional comment
        """
        with self.lock:
            trade = {
                'timestamp': timestamp,
                'side': side,
                'units': units,
                'price': price,
                'pnl': pnl,
                'comment': comment
            }
            self.trades.append(trade)
            
            msg = (f'TRADE | {side} {units} @ {price:.5f} | '
                  f'PnL: {pnl if pnl else "N/A"}')
            if comment:
                msg += f' | {comment}'
            
            self.logger.info(msg)
    
    def log_signal(self, timestamp: Any, signal: str, 
                  indicator: str, value: float,
                  comment: str = ''):
        """
        Log a trading signal.
        
        Parameters
        ==========
        timestamp: Any
            Signal timestamp
        signal: str
            Signal type ('BUY', 'SELL', 'NEUTRAL')
        indicator: str
            Name of indicator (e.g., 'MOMENTUM', 'SMA')
        value: float
            Indicator value
        comment: str
            Additional comment
        """
        with self.lock:
            sig = {
                'timestamp': timestamp,
                'signal': signal,
                'indicator': indicator,
                'value': value,
                'comment': comment
            }
            self.signals.append(sig)
            
            msg = (f'SIGNAL | {signal} from {indicator} '
                  f'(value: {value:.4f})')
            if comment:
                msg += f' | {comment}'
            
            self.logger.info(msg)
    
    def log_error(self, timestamp: Any, error_type: str,
                 message: str, severity: str = 'WARNING'):
        """
        Log an error or warning.
        
        Parameters
        ==========
        timestamp: Any
            Error timestamp
        error_type: str
            Type of error
        message: str
            Error message
        severity: str
            'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        """
        with self.lock:
            error = {
                'timestamp': timestamp,
                'error_type': error_type,
                'message': message,
                'severity': severity
            }
            self.errors.append(error)
            
            log_func = getattr(self.logger, severity.lower())
            log_func(f'{error_type} | {message}')
    
    def log_event(self, timestamp: Any, event_type: str,
                 details: Dict[str, Any]):
        """
        Log a generic event.
        
        Parameters
        ==========
        timestamp: Any
            Event timestamp
        event_type: str
            Type of event
        details: dict
            Event details
        """
        with self.lock:
            event = {
                'timestamp': timestamp,
                'event_type': event_type,
                'details': details
            }
            self.events.append(event)
            
            self.logger.debug(f'{event_type} | {json.dumps(details)}')
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current trading statistics.
        
        Returns
        =======
        dict
            Trading statistics
        """
        with self.lock:
            if not self.trades:
                return {
                    'strategy': self.strategy_name,
                    'total_trades': 0,
                    'total_buys': 0,
                    'total_sells': 0,
                    'runtime': str(datetime.now() - self.start_time)
                }
            
            trades_df = pd.DataFrame(self.trades)
            
            buys = trades_df[trades_df['side'] == 'BUY']
            sells = trades_df[trades_df['side'] == 'SELL']
            
            stats = {
                'strategy': self.strategy_name,
                'total_trades': len(self.trades),
                'total_buys': len(buys),
                'total_sells': len(sells),
                'avg_buy_price': buys['price'].mean() if len(buys) > 0 else 0,
                'avg_sell_price': sells['price'].mean() if len(
                    sells) > 0 else 0,
                'total_volume': trades_df['units'].sum(),
                'total_errors': len(self.errors),
                'runtime': str(datetime.now() - self.start_time)
            }
            
            # Calculate PnL if available
            if trades_df['pnl'].notna().any():
                stats['total_pnl'] = trades_df['pnl'].sum()
            
            return stats
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """
        Get signal statistics.
        
        Returns
        =======
        dict
            Signal statistics
        """
        with self.lock:
            if not self.signals:
                return {}
            
            signals_df = pd.DataFrame(self.signals)
            
            return {
                'total_signals': len(self.signals),
                'buy_signals': len(signals_df[signals_df['signal'] == 'BUY']),
                'sell_signals': len(signals_df[signals_df['signal'] == 'SELL']),
                'neutral_signals': len(
                    signals_df[signals_df['signal'] == 'NEUTRAL']
                ),
                'signal_types': signals_df['indicator'].value_counts().to_dict()
            }
    
    def export_log(self, filepath: str, format: str = 'csv'):
        """
        Export logs to file.
        
        Parameters
        ==========
        filepath: str
            Output file path
        format: str
            Format: 'csv', 'json', or 'excel'
        """
        with self.lock:
            if format == 'csv':
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv(filepath, index=False)
                self.logger.info(f'Logs exported to {filepath}')
            
            elif format == 'json':
                data = {
                    'strategy': self.strategy_name,
                    'trades': self.trades,
                    'signals': self.signals,
                    'errors': self.errors,
                    'events': self.events
                }
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                self.logger.info(f'Logs exported to {filepath}')
            
            elif format == 'excel':
                try:
                    with pd.ExcelWriter(filepath) as writer:
                        if self.trades:
                            trades_df = pd.DataFrame(self.trades)
                            trades_df.to_excel(writer, sheet_name='Trades', 
                                             index=False)
                        if self.signals:
                            signals_df = pd.DataFrame(self.signals)
                            signals_df.to_excel(writer, sheet_name='Signals', 
                                              index=False)
                        if self.errors:
                            errors_df = pd.DataFrame(self.errors)
                            errors_df.to_excel(writer, sheet_name='Errors', 
                                             index=False)
                    self.logger.info(f'Logs exported to {filepath}')
                except ImportError:
                    self.logger.warning(
                        'openpyxl not installed. Export to CSV instead.'
                    )
    
    def print_summary(self):
        """Print a summary of current statistics."""
        stats = self.get_statistics()
        signal_stats = self.get_signal_statistics()
        
        print('\n' + '='*60)
        print(f'Strategy: {stats.get("strategy", "Unknown")}')
        print(f'Runtime: {stats.get("runtime", "N/A")}')
        print('-'*60)
        print(f'Total Trades: {stats.get("total_trades", 0)}')
        print(f'  Buys: {stats.get("total_buys", 0)}')
        print(f'  Sells: {stats.get("total_sells", 0)}')
        print(f'Total Volume: {stats.get("total_volume", 0)} units')
        if 'total_pnl' in stats:
            print(f'Total PnL: {stats["total_pnl"]:.2f}')
        print(f'Errors: {stats.get("total_errors", 0)}')
        print('-'*60)
        if signal_stats:
            print(f'Total Signals: {signal_stats.get("total_signals", 0)}')
            print(f'  Buys: {signal_stats.get("buy_signals", 0)}')
            print(f'  Sells: {signal_stats.get("sell_signals", 0)}')
            print(f'  Neutral: {signal_stats.get("neutral_signals", 0)}')
        print('='*60 + '\n')
