"""
ML Strategy Integration for py4at_app
======================================

This module integrates the enhanced ML predictor with your existing
backtesting framework, providing improved trading strategies.

Usage:
    python ml_strategy_integration.py
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory and __Td0/py4at_app to path
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '__Td0', 'py4at_app'))

# Import from your existing framework
from py4at_app.backtesting import BacktestBase
from py4at_app.trading import StrategyMonitor
from py4at_app.data import DataLoader
from py4at_app import utils


class MLEnhancedBacktester(BacktestBase):
    """
    Enhanced backtester using ML predictions.
    Integrates with your existing BacktestBase class.
    """
    
    def __init__(self, symbol: str, start: str, end: str, 
                 amount: float, ml_model=None, confidence_threshold: float = 0.6,
                 ftc: float = 0.0, ptc: float = 0.0, verbose: bool = True,
                 risk_fraction: float = 0.01, stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.04, slippage: float = 0.0005,
                 retrain_interval: int = 0, confirmation_bars: int = 1, dynamic_atr_k: float = 0.0):
        """
        Initialize ML Enhanced Backtester.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        start : str
            Start date
        end : str
            End date
        amount : float
            Initial capital
        ml_model : EnhancedTradingPredictor
            Trained ML model
        confidence_threshold : float
            Minimum confidence for taking positions (0-1)
        ftc : float
            Fixed transaction costs
        ptc : float
            Proportional transaction costs
        verbose : bool
            Print trade information
        """
        super().__init__(symbol, start, end, amount, ftc, ptc, verbose)
        self.ml_model = ml_model
        self.confidence_threshold = confidence_threshold
        self.predictions = None
        self.probabilities = None
        # Risk management parameters
        self.risk_fraction = risk_fraction
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.slippage = slippage
        # Walk-forward retrain interval (bars). 0 => no retrain during backtest
        self.retrain_interval = retrain_interval
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None
        # Advanced execution filters
        self.confirmation_bars = confirmation_bars
        # If > 0, use ATR-based stop: stop = entry_price - dynamic_atr_k * atr_14 (long)
        self.dynamic_atr_k = dynamic_atr_k
        # Shorting and trade log
        self.short_units = 0
        self.short_entry_price = None
        self.trade_log = []  # list of dicts with per-trade P&L and metadata
        self.allow_short = True

    def _record_trade_entry(self, side: str, bar: int, price: float, units: int):
        """Record a trade entry in the trade log."""
        date, _ = self.get_date_price(bar)
        self.trade_log.append({
            'side': side,
            'entry_bar': bar,
            'entry_date': date,
            'entry_price': price,
            'units': units,
            'exit_bar': None,
            'exit_date': None,
            'exit_price': None,
            'pnl': None,
            'return_pct': None,
            'hold_bars': None,
            'status': 'open'
        })

    def _record_trade_exit(self, side: str, bar: int, price: float):
        """Record a trade exit and compute P&L. Finds last open trade of this side."""
        for trade in reversed(self.trade_log):
            if trade['side'] == side and trade['status'] == 'open':
                entry_price = trade['entry_price']
                units = trade['units']
                entry_bar = trade['entry_bar']
                
                trade['exit_bar'] = bar
                trade['exit_date'] = str(self.data.index[bar])[:10]
                trade['exit_price'] = price
                trade['hold_bars'] = bar - entry_bar
                trade['status'] = 'closed'

                if side == 'long':
                    pnl = (price - entry_price) * units
                    trade['pnl'] = pnl
                    trade['return_pct'] = (price - entry_price) / entry_price if entry_price != 0 else 0
                else:  # short
                    pnl = (entry_price - price) * units
                    trade['pnl'] = pnl
                    trade['return_pct'] = (entry_price - price) / entry_price if entry_price != 0 else 0
                
                return True  # Successfully recorded
        
        return False  # No open trade found

    def enter_long(self, bar: int, units: int):
        """Enter a long position and record it."""
        date, price = self.get_date_price(bar)
        self._record_trade_entry('long', bar, price, units)
        self.place_buy_order(bar, units=units)
        self.amount -= units * price * self.slippage
        self.trades += 1

    def exit_long(self, bar: int):
        """Exit a long position and record the exit."""
        date, price = self.get_date_price(bar)
        units = self.units
        if units <= 0:
            return
        self._record_trade_exit('long', bar, price)
        self.place_sell_order(bar, units=units)
        self.amount -= units * price * self.slippage
        self.trades += 1

    def enter_short(self, bar: int, units: int):
        """Enter a short position and record it."""
        date, price = self.get_date_price(bar)
        self._record_trade_entry('short', bar, price, units)
        proceeds = units * price * (1 - self.ptc) - self.ftc
        self.amount += proceeds
        self.short_units += units
        self.short_entry_price = price
        self.amount -= units * price * self.slippage
        self.trades += 1

    def cover_short(self, bar: int):
        """Cover a short position and record the exit."""
        date, price = self.get_date_price(bar)
        units = self.short_units
        if units <= 0:
            return
        self._record_trade_exit('short', bar, price)
        cost = units * price * (1 + self.ptc) + self.ftc
        self.amount -= cost
        self.amount -= units * price * self.slippage
        self.trades += 1
        # reset short state
        self.short_units = 0
        self.short_entry_price = None
        self.position = 0

    def summarize_trade_log(self, export_equity_csv: str = None) -> dict:
        """Summarize trade_log with P&L, drawdown, and per-side statistics.
        
        Parameters:
        -----------
        export_equity_csv : str, optional
            If provided, export equity curve to CSV file.
        
        Returns:
        --------
        dict : Comprehensive trade summary including per-side stats
        """
        if not self.trade_log:
            return {}

        # Separate by side
        long_trades = [t for t in self.trade_log if t['side'] == 'long' and t['status'] == 'closed']
        short_trades = [t for t in self.trade_log if t['side'] == 'short' and t['status'] == 'closed']
        
        # Overall stats
        total_pnl = 0.0
        wins = 0
        losses = 0
        pnls = []
        hold_times = []
        
        for t in self.trade_log:
            if t.get('status') == 'closed' and t.get('pnl') is not None:
                pnl = t['pnl']
                pnls.append(pnl)
                total_pnl += pnl
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1
                else:
                    losses += 1  # zero pnl counts as loss
                
                if t.get('hold_bars') is not None:
                    hold_times.append(t['hold_bars'])

        total_trades = len(pnls)
        win_rate = (wins / total_trades) if total_trades > 0 else 0
        avg_pnl = (total_pnl / total_trades) if total_trades > 0 else 0
        avg_hold_bars = (sum(hold_times) / len(hold_times)) if hold_times else 0

        # Per-side stats
        def compute_side_stats(trades_list):
            if not trades_list:
                return {}
            side_pnl = sum(t['pnl'] for t in trades_list if t.get('pnl') is not None)
            side_wins = sum(1 for t in trades_list if t.get('pnl', 0) > 0)
            side_trades = len([t for t in trades_list if t.get('pnl') is not None])
            side_win_rate = (side_wins / side_trades) if side_trades > 0 else 0
            side_avg_pnl = (side_pnl / side_trades) if side_trades > 0 else 0
            return {
                'trades': side_trades,
                'total_pnl': side_pnl,
                'win_rate': side_win_rate,
                'avg_pnl': side_avg_pnl,
                'wins': side_wins
            }
        
        long_stats = compute_side_stats(long_trades)
        short_stats = compute_side_stats(short_trades)

        # Compute running equity curve from trade-level pnl starting at initial_amount
        equity = [self.initial_amount]
        equity_ts = [0]  # time step
        ts = 0
        for t in self.trade_log:
            if t.get('status') == 'closed' and t.get('pnl') is not None:
                equity.append(equity[-1] + t['pnl'])
                ts += 1
                equity_ts.append(ts)

        # Compute max drawdown
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e)
            if dd > max_dd:
                max_dd = dd

        # Export equity curve if requested
        if export_equity_csv and equity:
            try:
                df_equity = pd.DataFrame({'trade_step': equity_ts, 'equity': equity})
                df_equity.to_csv(export_equity_csv, index=False)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not export equity curve to {export_equity_csv}: {e}")

        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'max_drawdown': max_dd,
            'avg_hold_bars': avg_hold_bars,
            'long_stats': long_stats,
            'short_stats': short_stats,
            'equity_curve': equity
        }
    
    def run_ml_strategy(self):
        """
        Run backtest using ML predictions.
        """
        if self.ml_model is None:
            raise ValueError("ML model not provided")
        
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data available")
        
        msg = f'\n\nRunning ML Enhanced Strategy'
        msg += f'\nConfidence threshold: {self.confidence_threshold}'
        msg += f'\nfixed costs {self.ftc} | proportional costs {self.ptc}'
        
        if self.verbose:
            print(msg)
            print('=' * 55)
        
        self.position = 0
        self.trades = 0
        self.amount = self.initial_amount
        
        # Get predictions from ML model
        self.predictions = self.ml_model.predict(self.data)
        self.probabilities = self.ml_model.predict_proba(self.data)
        
        # Add predictions to data
        self.data['ml_prediction'] = self.predictions
        self.data['ml_probability'] = self.probabilities
        
        # Determine minimum bars needed (based on feature engineering).
        # Use 200 as safe default, but adapt to shorter datasets to allow testing.
        min_bars = min(200, max(1, len(self.data) - 1))

        # Retrieve arrays for easy indexing
        probs = self.data['ml_probability'].values
        preds = self.data['ml_prediction'].values

        for bar in range(min_bars, len(self.data)):
            current_prob = float(probs[bar])
            current_pred = int(preds[bar])
            date, price = self.get_date_price(bar)

            # Confirmation: require same prediction for last `confirmation_bars` bars
            if self.confirmation_bars and self.confirmation_bars > 1:
                start_idx = max(min_bars, bar - (self.confirmation_bars - 1))
                window_preds = preds[start_idx:bar + 1]
                window_probs = probs[start_idx:bar + 1]
                # require all preds equal to current_pred and probs >= threshold
                if not all(int(p) == current_pred for p in window_preds):
                    continue
                if not all(float(pv) >= self.confidence_threshold for pv in window_probs):
                    continue

            # Walk-forward retrain if requested (simple full retrain on expanding window)
            if self.retrain_interval and (bar - min_bars) % self.retrain_interval == 0 and bar > min_bars:
                # retrain on all data up to current bar
                try:
                    train_window = self.data.iloc[:bar].copy()
                    # Ensure features present for retraining
                    self.ml_model.partial_retrain(train_window)
                    # refresh predictions/probas for remaining data if possible
                    self.predictions = self.ml_model.predict(self.data)
                    self.probabilities = self.ml_model.predict_proba(self.data)
                    self.data['ml_prediction'] = self.predictions
                    self.data['ml_probability'] = self.probabilities
                    if self.verbose:
                        print(f"{date} | Walk-forward retrain performed at bar {bar}")
                except Exception as e:
                    if self.verbose:
                        print(f"Retrain error at bar {bar}: {e}")

            # Check for stop-loss / take-profit when in position
            if self.position == 1:
                # check stop-loss
                if self.stop_price is not None and price <= self.stop_price:
                    # execute sell due to stop-loss
                    # account for slippage cost
                    slippage_cost = self.units * price * self.slippage
                    self.place_sell_order(bar, units=self.units)
                    self.amount -= slippage_cost
                    if self.verbose:
                        print(f"{date} | Stop-loss triggered at {price:.4f}")
                    self.position = 0
                    # clear entry/stop/tp
                    self.entry_price = self.stop_price = self.tp_price = None
                    continue

                # check take-profit
                if self.tp_price is not None and price >= self.tp_price:
                    slippage_cost = self.units * price * self.slippage
                    self.place_sell_order(bar, units=self.units)
                    self.amount -= slippage_cost
                    if self.verbose:
                        print(f"{date} | Take-profit hit at {price:.4f}")
                    self.position = 0
                    self.entry_price = self.stop_price = self.tp_price = None
                    continue

            # Only trade if confidence exceeds threshold
            if current_prob < self.confidence_threshold:
                continue

            # Entry logic for flat book
            if self.position == 0:
                # Long entry
                if current_pred == 1:
                    invest_amount = max(self.amount * self.risk_fraction, 0)
                    units = int(invest_amount / price)
                    if units <= 0:
                        units = int((self.amount * 0.1) / price)
                        if units <= 0:
                            continue

                    # Set dynamic ATR-based stop if configured
                    atr = None
                    if self.dynamic_atr_k and 'atr_14' in self.data.columns:
                        atr = float(self.data['atr_14'].iloc[bar])

                    self.enter_long(bar, units)
                    self.position = 1
                    self.entry_price = price
                    if atr is not None and atr > 0 and self.dynamic_atr_k > 0:
                        self.stop_price = price - (self.dynamic_atr_k * atr)
                    else:
                        self.stop_price = price * (1 - self.stop_loss_pct)
                    self.tp_price = price * (1 + self.take_profit_pct)

                # Short entry (if allowed)
                elif current_pred == 0 and self.allow_short:
                    invest_amount = max(self.amount * self.risk_fraction, 0)
                    units = int(invest_amount / price)
                    if units <= 0:
                        units = int((self.amount * 0.1) / price)
                        if units <= 0:
                            continue

                    atr = None
                    if self.dynamic_atr_k and 'atr_14' in self.data.columns:
                        atr = float(self.data['atr_14'].iloc[bar])

                    self.enter_short(bar, units)
                    self.position = -1
                    self.entry_price = price
                    if atr is not None and atr > 0 and self.dynamic_atr_k > 0:
                        self.stop_price = price + (self.dynamic_atr_k * atr)
                    else:
                        self.stop_price = price * (1 + self.stop_loss_pct)
                    self.tp_price = price * (1 - self.take_profit_pct)

            # Manage existing long position
            elif self.position == 1:
                # Stop-loss
                if self.stop_price is not None and price <= self.stop_price:
                    self.exit_long(bar)
                    if self.verbose:
                        print(f"{date} | Stop-loss triggered at {price:.4f}")
                    self.position = 0
                    self.entry_price = self.stop_price = self.tp_price = None
                    continue

                # Take-profit
                if self.tp_price is not None and price >= self.tp_price:
                    self.exit_long(bar)
                    if self.verbose:
                        print(f"{date} | Take-profit hit at {price:.4f}")
                    self.position = 0
                    self.entry_price = self.stop_price = self.tp_price = None
                    continue

                # Model flip to exit
                if current_pred == 0:
                    self.exit_long(bar)
                    if self.verbose:
                        print(f"{date} | Model signaled SELL at prob {current_prob:.3f}")
                    self.position = 0
                    self.entry_price = self.stop_price = self.tp_price = None

            # Manage existing short position
            elif self.position == -1:
                # Stop-loss for short (price moves up)
                if self.stop_price is not None and price >= self.stop_price:
                    self.cover_short(bar)
                    if self.verbose:
                        print(f"{date} | Short stop-loss triggered at {price:.4f}")
                    self.position = 0
                    self.entry_price = self.stop_price = self.tp_price = None
                    continue

                # Take-profit for short (price moves down)
                if self.tp_price is not None and price <= self.tp_price:
                    self.cover_short(bar)
                    if self.verbose:
                        print(f"{date} | Short take-profit hit at {price:.4f}")
                    self.position = 0
                    self.entry_price = self.stop_price = self.tp_price = None
                    continue

                # Model flip to cover
                if current_pred == 1:
                    self.cover_short(bar)
                    if self.verbose:
                        print(f"{date} | Model signaled COVER at prob {current_prob:.3f}")
                    self.position = 0
                    self.entry_price = self.stop_price = self.tp_price = None

        # Final close out
        self.close_out(len(self.data) - 1)


    def close_out(self, bar: int):
        """Override to record any open trades before closing out.
        Ensures all open positions are exited and logged before base close.
        """
        date, price = self.get_date_price(bar)
        
        # If we have a long position open, record exit
        if self.units > 0:
            self._record_trade_exit('long', bar, price)

        # If we have an open short, record exit
        if self.short_units > 0:
            self._record_trade_exit('short', bar, price)

        # Call base class close_out to handle cash/unit reset and prints
        super().close_out(bar)
    
    def get_signal_statistics(self) -> dict:
        """
        Get statistics about ML signals.
        
        Returns:
        --------
        dict : Signal statistics
        """
        if self.predictions is None:
            return {}
        
        total_signals = len(self.predictions)
        buy_signals = np.sum(self.predictions == 1)
        sell_signals = np.sum(self.predictions == 0)
        
        high_conf_signals = np.sum(self.probabilities > self.confidence_threshold)
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'high_confidence_signals': high_conf_signals,
            'high_conf_percentage': (high_conf_signals / total_signals * 100) if total_signals > 0 else 0,
            'avg_probability': np.mean(self.probabilities)
        }


class MLTradingStrategy:
    """
    Complete ML trading strategy with training and backtesting.
    """
    
    def __init__(self, symbol: str = 'EUR/USD'):
        """
        Initialize ML Trading Strategy.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        """
        self.symbol = symbol
        self.predictor = None
        self.data_with_features = None
        self.backtest_results = None
    
    def prepare_data(self, data: pd.DataFrame, future_h: int = 5, ret_threshold: float = 0.0003) -> pd.DataFrame:
        """
        Prepare data with technical indicators.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw price data
            
        Returns:
        --------
        pd.DataFrame : Data with features
        """
        from enhanced_predictor import EnhancedFeatureEngineering

        fe = EnhancedFeatureEngineering()
        data_with_features = fe.add_technical_indicators(data, future_h=future_h, ret_threshold=ret_threshold)
        
        return data_with_features
    
    def train(self, train_data: pd.DataFrame, test_size: float = 0.2) -> dict:
        """
        Train the ML model.
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        test_size : float
            Proportion for testing
            
        Returns:
        --------
        dict : Training results
        """
        from enhanced_predictor import EnhancedTradingPredictor
        
        print("Training ML model...")
        
        self.predictor = EnhancedTradingPredictor(use_ensemble=True)
        results = self.predictor.train(train_data, test_size=test_size)
        
        return results
    
    def backtest(self, data: pd.DataFrame, confidence_threshold: float = 0.6,
                initial_amount: float = 10000, tc: float = 0.001,
                risk_fraction: float = 0.01, stop_loss_pct: float = 0.02,
                take_profit_pct: float = 0.04, slippage: float = 0.0005,
                retrain_interval: int = 0, allow_short: bool = True,
                confirmation_bars: int = 1, dynamic_atr_k: float = 0.0) -> dict:
        """
        Backtest the ML strategy.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data for backtesting
        confidence_threshold : float
            Minimum confidence for trades
        initial_amount : float
            Initial capital
        tc : float
            Transaction costs
            
        Returns:
        --------
        dict : Backtest results
        """
        if self.predictor is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print()
        print("=" * 70)
        print("BACKTESTING ML STRATEGY")
        print("=" * 70)
        
        # Create backtester
        backtester = MLEnhancedBacktester(
            symbol=self.symbol,
            start=str(data.index[0]),
            end=str(data.index[-1]),
            amount=initial_amount,
            ml_model=self.predictor,
            confidence_threshold=confidence_threshold,
            ftc=0.0,
            ptc=tc,
            verbose=True,
            # risk and execution params (defaults can be tuned)
            risk_fraction=risk_fraction,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            slippage=slippage,
            retrain_interval=retrain_interval
            ,confirmation_bars=confirmation_bars, dynamic_atr_k=dynamic_atr_k
        )
        backtester.allow_short = allow_short
        
        # Set data
        backtester.set_data(data[['price']])
        
        # Add features to backtest data
        data_with_features = self.prepare_data(data[['price']])
        backtester.data = data_with_features
        
        # Run strategy
        backtester.run_ml_strategy()
        
        # Calculate performance and trade summary
        performance = backtester.get_performance()
        signal_stats = backtester.get_signal_statistics()
        trade_summary = backtester.summarize_trade_log()
        
        self.backtest_results = {
            'performance': performance,
            'signal_stats': signal_stats,
            'trades': backtester.trades,
            'final_amount': backtester.amount,
            'trade_log': backtester.trade_log,
            'trade_summary': trade_summary
        }
        
        return self.backtest_results
    
    def print_results(self):
        """Print backtest results."""
        if self.backtest_results is None:
            print("No backtest results available.")
            return
        
        print()
        print("=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        print()
        
        perf = self.backtest_results['performance']
        stats = self.backtest_results['signal_stats']
        
        print(f"Performance:               {perf:.2f}%")
        print(f"Total Trades:              {self.backtest_results['trades']}")
        print(f"Final Amount:              ${self.backtest_results['final_amount']:.2f}")
        print()
        
        print("Signal Statistics:")
        print(f"  Total Signals:           {stats['total_signals']}")
        print(f"  Buy Signals:             {stats['buy_signals']}")
        print(f"  Sell Signals:            {stats['sell_signals']}")
        print(f"  High Confidence:         {stats['high_confidence_signals']} "
              f"({stats['high_conf_percentage']:.1f}%)")
        print(f"  Average Probability:     {stats['avg_probability']:.4f}")
        
        # Trade log summary
        if 'trade_log' in self.backtest_results:
            print()
            print("Trade Log Summary:")
            trade_log = self.backtest_results.get('trade_log', [])
            if trade_log:
                try:
                    # Count closed trades with P&L
                    closed_trades = [t for t in trade_log if t.get('status') == 'closed' and t.get('pnl') is not None]
                    
                    if closed_trades:
                        total_pnl = sum(t['pnl'] for t in closed_trades)
                        wins = sum(1 for t in closed_trades if t['pnl'] > 0)
                        losses = sum(1 for t in closed_trades if t['pnl'] <= 0)
                        trades = len(closed_trades)
                        win_rate = (wins / trades) if trades > 0 else 0
                        avg_pnl = (total_pnl / trades) if trades > 0 else 0
                        avg_hold = sum(t.get('hold_bars', 0) for t in closed_trades) / trades if trades > 0 else 0
                        
                        # Long stats
                        long_trades = [t for t in closed_trades if t['side'] == 'long']
                        long_pnl = sum(t['pnl'] for t in long_trades)
                        long_wins = sum(1 for t in long_trades if t['pnl'] > 0)
                        long_count = len(long_trades)
                        
                        # Short stats
                        short_trades = [t for t in closed_trades if t['side'] == 'short']
                        short_pnl = sum(t['pnl'] for t in short_trades)
                        short_wins = sum(1 for t in short_trades if t['pnl'] > 0)
                        short_count = len(short_trades)
                        
                        print(f"  Closed Trades:           {trades}")
                        print(f"  Total P&L:               ${total_pnl:.2f}")
                        print(f"  Win Rate:                {win_rate*100:.1f}% ({wins} wins, {losses} losses)")
                        print(f"  Avg P&L per trade:       ${avg_pnl:.2f}")
                        print(f"  Avg Hold (bars):         {avg_hold:.1f}")
                        
                        if long_count > 0:
                            print(f"\n  LONG trades:             {long_count} | P&L ${long_pnl:.2f} | {long_wins} wins ({long_wins*100//long_count}%)")
                        if short_count > 0:
                            print(f"  SHORT trades:            {short_count} | P&L ${short_pnl:.2f} | {short_wins} wins ({short_wins*100//short_count if short_count > 0 else 0}%)")
                        
                        # Equity curve
                        equity = [self.backtest_results['final_amount'] - total_pnl]
                        for t in closed_trades:
                            equity.append(equity[-1] + t['pnl'])
                        peak = max(equity)
                        max_dd = max(peak - e for e in equity)
                        print(f"  Max Drawdown (currency): ${max_dd:.2f}")
                    else:
                        print("  No closed trades with P&L recorded.")
                except Exception as e:
                    print(f"  (unable to summarize trade log: {e})")
        print()


def compare_strategies():
    """
    Compare ML strategy with traditional SMA strategy.
    """
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  STRATEGY COMPARISON: ML vs SMA  ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╗")
    print()
    
    # Load data
    try:
        data = pd.read_csv('sample_data.csv', index_col=0, parse_dates=True)
        data.rename(columns={'Close': 'price'}, inplace=True)
        print(f"✓ Loaded {len(data)} rows of data")
        print()
    except FileNotFoundError:
        print("Error: sample_data.csv not found")
        return
    
    # Split into train and test
    train_size = int(len(data) * 0.7)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    print(f"Training period: {train_data.index[0].date()} to {train_data.index[-1].date()}")
    print(f"Testing period:  {test_data.index[0].date()} to {test_data.index[-1].date()}")
    print()
    
    # 1. Train and test ML strategy
    print("=" * 70)
    print("TESTING ML STRATEGY")
    print("=" * 70)
    print()
    
    ml_strategy = MLTradingStrategy()
    
    # Prepare and train
    train_with_features = ml_strategy.prepare_data(train_data)
    train_results = ml_strategy.train(train_with_features, test_size=0.2)
    
    # Backtest on test data
    ml_results = ml_strategy.backtest(test_data, confidence_threshold=0.55)
    ml_strategy.print_results()
    
    # 2. Test traditional SMA strategy
    print()
    print("=" * 70)
    print("TESTING TRADITIONAL SMA STRATEGY")
    print("=" * 70)
    print()
    
    from py4at_app.backtesting import SMAVectorBacktester
    
    sma_bt = SMAVectorBacktester(
        symbol='EUR/USD',
        SMA1=10,
        SMA2=20,
        start=str(test_data.index[0]),
        end=str(test_data.index[-1]),
        data=test_data
    )
    
    sma_perf, sma_outperf = sma_bt.run_strategy()
    
    print()
    print("SMA Strategy Results:")
    print(f"  Performance:     {sma_perf * 100:.2f}%")
    print(f"  Outperformance:  {sma_outperf * 100:.2f}%")
    print()
    
    # 3. Comparison
    print("=" * 70)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 70)
    print()
    
    ml_perf = ml_results['performance']
    
    print(f"ML Strategy:       {ml_perf:>8.2f}%")
    print(f"SMA Strategy:      {sma_perf * 100:>8.2f}%")
    print()
    
    if ml_perf > sma_perf * 100:
        diff = ml_perf - (sma_perf * 100)
        print(f"✓ ML strategy outperforms by {diff:.2f} percentage points")
    else:
        diff = (sma_perf * 100) - ml_perf
        print(f"⚠ SMA strategy outperforms by {diff:.2f} percentage points")
    
    print()
    print("=" * 70)
    print()


if __name__ == '__main__':
    try:
        # Import the enhanced predictor
        from enhanced_predictor import EnhancedTradingPredictor, EnhancedFeatureEngineering
        
        # Run comparison
        compare_strategies()
        
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure enhanced_predictor.py is in the same directory")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
