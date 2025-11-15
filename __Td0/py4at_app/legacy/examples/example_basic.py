"""
EXAMPLE 1: BASIC WORKING EXAMPLE - SMA Backtest
================================================

This is a complete working example that you can run immediately.
It demonstrates the core functionality of the py4at_app framework.

REQUIREMENTS:
- numpy, pandas installed (should be in requirements.txt)
- sample_data.csv in the same directory

USAGE:
  python example_basic.py

EXPECTED OUTPUT:
- Backtesting complete message
- Performance metrics (Returns, Sharpe Ratio, Win Rate, etc.)
- Graph will be displayed (if matplotlib is available)
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add the app to path so we can import it
sys.path.insert(0, os.path.dirname(__file__))

from py4at_app.backtesting import SMAVectorBacktester
from py4at_app.data import DataLoader
from py4at_app import utils


def example_sma_backtest():
    """
    Example 1: Simple Moving Average (SMA) Backtest
    
    This is the easiest strategy to understand:
    - Buy when fast SMA crosses above slow SMA
    - Sell when fast SMA crosses below slow SMA
    """
    print("=" * 70)
    print("EXAMPLE 1: SMA VECTOR BACKTEST")
    print("=" * 70)
    print()
    
    # Step 1: Load data
    print("📊 Step 1: Loading sample data...")
    data_path = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
    
    try:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(data)} rows of data")
        print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"  Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    except FileNotFoundError:
        print(f"✗ Error: sample_data.csv not found at {data_path}")
        print("  Please ensure sample_data.csv is in the same directory as this script")
        return
    
    print()
    
    # Step 2: Create backtester
    print("🔧 Step 2: Setting up SMA Backtest strategy...")
    sma_bt = SMAVectorBacktester(
        symbol='EUR/USD',           # Symbol
        SMA1=10,                    # Short SMA window
        SMA2=20,                    # Long SMA window
        start=str(data.index[0]),   # Start date
        end=str(data.index[-1]),    # End date
        data=data                   # Pre-loaded data
    )
    print("✓ Backtester created with:")
    print(f"  - Symbol: EUR/USD")
    print(f"  - SMA1: 10 bars")
    print(f"  - SMA2: 20 bars")
    print()
    
    # Step 3: Run backtest
    print("⚙️  Step 3: Running backtest...")
    print()
    
    results = sma_bt.run_strategy()
    
    print("✓ Backtest completed!")
    print()
    
    # Step 4: Display results
    print("📈 BACKTEST RESULTS")
    print("-" * 70)
    
    # Calculate performance metrics
    returns = results['return'].dropna()
    cumulative_returns = (1 + returns).cumprod()
    
    # Strategy return (log returns)
    strategy_return = results['return'].sum() * 100
    buy_hold_return = (np.log(data['Close'][-1]) - np.log(data['Close'][0])) * 100
    
    # Sharpe ratio
    sharpe = utils.calculate_sharpe_ratio(returns)
    
    # Drawdown
    max_dd = utils.calculate_max_drawdown(cumulative_returns)
    
    # Win rate
    positive_returns = (returns > 0).sum()
    win_rate = (positive_returns / len(returns) * 100) if len(returns) > 0 else 0
    
    print(f"Strategy Return:     {strategy_return:>10.2f} %")
    print(f"Buy & Hold Return:   {buy_hold_return:>10.2f} %")
    print(f"Sharpe Ratio:        {sharpe:>10.2f}")
    print(f"Max Drawdown:        {max_dd * 100:>10.2f} %")
    print(f"Win Rate:            {win_rate:>10.2f} %")
    print()
    
    # Step 5: Show portfolio value over time
    print("💰 PRICE AND SIGNALS OVER TIME")
    print("-" * 70)
    
    key_points = [0, len(results)//4, len(results)//2, 
                   3*len(results)//4, -1]
    
    for idx in key_points:
        date = results.index[idx].date()
        price = results['price'].iloc[idx]
        sma1 = results['SMA1'].iloc[idx]
        sma2 = results['SMA2'].iloc[idx]
        
        if pd.notna(sma1) and pd.notna(sma2):
            signal = "BUY" if sma1 > sma2 else "SELL"
        else:
            signal = "N/A"
        
        print(f"  {date}  |  Price: ${price:>8.2f}  |  SMA1: ${sma1:>8.2f}  |  SMA2: ${sma2:>8.2f}  |  Signal: {signal}")
    
    print()
    
    # Step 6: Show sample trades
    print("🤝 SAMPLE DATA POINTS")
    print("-" * 70)
    
    data_points = results[results['SMA1'].notna()].head(10)
    for idx, (date, row) in enumerate(data_points.iterrows(), 1):
        price = row['price']
        sma1 = row['SMA1']
        sma2 = row['SMA2']
        print(f"  {date.date()} | Price: ${price:.2f} | SMA1: ${sma1:.2f} | SMA2: ${sma2:.2f}")
    
    print()
    print("=" * 70)
    print("✓ BACKTEST COMPLETE!")
    print("=" * 70)
    
    return results


def example_momentum_backtest():
    """
    Example 2: Momentum Strategy Backtest
    
    This strategy uses the momentum indicator:
    - Calculate momentum (price change over N periods)
    - Buy when momentum is positive
    - Sell when momentum turns negative
    """
    print("\n\n")
    print("=" * 70)
    print("EXAMPLE 2: MOMENTUM VECTOR BACKTEST")
    print("=" * 70)
    print()
    
    # Load data
    print("📊 Loading data...")
    data_path = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
    
    try:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(data)} rows")
    except FileNotFoundError:
        print(f"✗ Error: sample_data.csv not found")
        return
    
    print()
    
    from py4at_app.backtesting import MomVectorBacktester
    
    print("🔧 Setting up Momentum strategy...")
    mom_bt = MomVectorBacktester(
        symbol='EUR/USD',
        momentum=10,
        start=str(data.index[0]),
        end=str(data.index[-1]),
        data=data
    )
    print("✓ Backtester created")
    print()
