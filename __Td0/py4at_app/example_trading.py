"""
EXAMPLE 2: ADVANCED TRADING EXAMPLE
====================================

This example shows how to use the trading and monitoring modules
for real-time trading simulation and performance tracking.

USAGE:
  python example_trading.py
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from py4at_app.trading import MomentumTrader, StrategyMonitor
from py4at_app.data import DataLoader


def example_momentum_trader():
    """
    Example: Live Momentum Trading Simulation
    
    Simulates real-time trading with:
    - Tick data processing
    - Bar aggregation
    - Momentum signal generation
    - Trade execution
    - Performance monitoring
    """
    print("=" * 70)
    print("EXAMPLE: MOMENTUM TRADER WITH MONITORING")
    print("=" * 70)
    print()
    
    # Load data
    print("📊 Loading historical data...")
    data_path = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
    
    try:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        data_prepared = DataLoader.prepare_data(data)
        print(f"✓ Loaded {len(data_prepared)} bars of data")
    except FileNotFoundError:
        print(f"✗ Error: sample_data.csv not found")
        return
    
    print()
    
    # Initialize trader and monitor
    print("🔧 Initializing Momentum Trader...")
    trader = MomentumTrader(
        instrument='EUR/USD',
        bar_length=1,              # 1-bar lookback
        momentum_length=10,        # 10-bar momentum
        units=1,                   # 1 unit per trade
        verbose=False
    )
    print("✓ Trader initialized")
    print()
    
    print("📋 Initializing Strategy Monitor...")
    monitor = StrategyMonitor(strategy_name="Momentum Trading Example")
    print("✓ Monitor initialized")
    print()
    
    # Simulate processing tick data (bars in this case)
    print("⚙️  Simulating tick processing (using bar data)...")
    print()
    
    trades_executed = 0
    signals_generated = 0
    
    for idx in range(10, len(data_prepared)):
        bar_data = data_prepared.iloc[:idx+1]
        
        # Process the latest bar as a "tick"
        latest_price = bar_data['Close'].iloc[-1]
        
        # Generate signals and execute trades
        signal = trader.on_tick(
            time=bar_data.index[-1],
            bid=latest_price * 0.99,  # Simulate bid/ask spread
            ask=latest_price * 1.01
        )
        
        if signal is not None:
            signals_generated += 1
            # Log the signal
            monitor.log_signal(
                timestamp=bar_data.index[-1],
                signal=signal,
                price=latest_price,
                instrument='EUR/USD'
            )
            
            # If signal is a trade, log it
            if signal in ['BUY', 'SELL']:
                trades_executed += 1
                monitor.log_trade(
                    timestamp=bar_data.index[-1],
                    instrument='EUR/USD',
                    action=signal,
                    quantity=1,
                    price=latest_price,
                    commission=0.002 * latest_price
                )
    
    print(f"✓ Processing complete")
    print(f"  - Bars processed: {len(data_prepared) - 10}")
    print(f"  - Signals generated: {signals_generated}")
    print(f"  - Trades executed: {trades_executed}")
    print()
    
    # Display performance summary
    print("📊 TRADING PERFORMANCE SUMMARY")
    print("-" * 70)
    monitor.print_summary()
    print()
    
    # Get performance metrics
    performance = monitor.get_performance_metrics()
    if performance:
        print("📈 DETAILED METRICS")
        print("-" * 70)
        for key, value in performance.items():
            print(f"  {key:.<50} {value}")
    
    print()
    print("=" * 70)
    
    return monitor


def example_monitoring_system():
    """
    Example: Advanced Monitoring System
    
    Shows how to track and export trading data
    """
    print("\n\n")
    print("=" * 70)
    print("EXAMPLE: MONITORING SYSTEM FEATURES")
    print("=" * 70)
    print()
    
    print("📋 Creating monitoring instance...")
    monitor = StrategyMonitor(strategy_name="Example Strategy")
    print("✓ Monitor created")
    print()
    
    print("📝 Logging sample trades...")
    
    # Sample trades
    trades = [
        {'time': datetime(2024, 1, 2, 10, 0), 'action': 'BUY', 'price': 100.50, 'qty': 1},
        {'time': datetime(2024, 1, 2, 11, 30), 'action': 'SELL', 'price': 101.20, 'qty': 1},
        {'time': datetime(2024, 1, 3, 9, 0), 'action': 'BUY', 'price': 100.80, 'qty': 2},
        {'time': datetime(2024, 1, 3, 14, 0), 'action': 'SELL', 'price': 102.10, 'qty': 2},
        {'time': datetime(2024, 1, 4, 10, 0), 'action': 'BUY', 'price': 101.95, 'qty': 1},
        {'time': datetime(2024, 1, 4, 15, 0), 'action': 'SELL', 'price': 103.20, 'qty': 1},
    ]
    
    for i, trade in enumerate(trades, 1):
        monitor.log_trade(
            timestamp=trade['time'],
            instrument='EUR/USD',
            action=trade['action'],
            quantity=trade['qty'],
            price=trade['price'],
            commission=0.002 * trade['price'] * trade['qty']
        )
        print(f"  ✓ Trade {i}: {trade['action']:>4} {trade['qty']} units @ ${trade['price']:.2f}")
    
    print()
    print("📊 Trade Summary:")
    print("-" * 70)
    monitor.print_summary()
    print()
    
    # Get logs
    print("📋 Trade Log:")
    print("-" * 70)
    trade_log = monitor.get_trade_log()
    for idx, trade in enumerate(trade_log.tail(5).itertuples(), 1):
        print(f"  {idx}. {trade.Index.strftime('%Y-%m-%d %H:%M')} | {trade.action:>4} | "
              f"Qty: {trade.quantity:>2} | Price: ${trade.price:>7.2f} | "
              f"Commission: ${trade.commission:>7.4f}")
    
    print()
    print("=" * 70)


if __name__ == '__main__':
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  ADVANCED TRADING EXAMPLES  ".center(68) + "║")
    print("║" + "  Real-time trading simulation  ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    try:
        # Run examples
        example_momentum_trader()
        example_monitoring_system()
        
        print("\n")
        print("╔" + "=" * 68 + "╗")
        print("║" + "  ALL EXAMPLES COMPLETED ✓  ".center(68) + "║")
        print("╚" + "=" * 68 + "╝")
        print()
        
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
