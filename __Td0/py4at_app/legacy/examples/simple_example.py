"""
SIMPLE WORKING EXAMPLE - START HERE!
====================================

This is a basic example that demonstrates the core functionality.
Run this to test if everything is working correctly.

USAGE:
  python simple_example.py
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the app to path
sys.path.insert(0, os.path.dirname(__file__))

from py4at_app.backtesting import SMAVectorBacktester
from py4at_app.data import DataLoader
from py4at_app import utils


def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  SIMPLE WORKING EXAMPLE - SMA BACKTEST  ".center(68) + "║")
    print("║" + "  Test the framework  ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    try:
        # Step 1: Load data
        print("📊 Step 1: Loading sample data...")
        data_path = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
        
        if not os.path.exists(data_path):
            print(f"❌ Error: sample_data.csv not found at {data_path}")
            print("\nPlease ensure sample_data.csv exists in the same directory")
            return False
        
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(data)} rows of data")
        print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"  Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        print()
        
        # Step 2: Prepare data (rename Close to price for compatibility)
        print("🔧 Step 2: Preparing data...")
        data_prep = data[['Close']].copy()
        data_prep.rename(columns={'Close': 'price'}, inplace=True)
        print(f"✓ Data prepared")
        print()
        
        # Step 3: Create and run backtest
        print("⚙️  Step 3: Creating SMA Backtest strategy...")
        start_date = str(data_prep.index[0])
        end_date = str(data_prep.index[-1])
        
        sma_bt = SMAVectorBacktester(
            symbol='EUR/USD',
            SMA1=10,              # Short moving average
            SMA2=20,              # Long moving average
            start=start_date,
            end=end_date,
            data=data_prep
        )
        print(f"✓ Backtester created")
        print(f"  - Symbol: EUR/USD")
        print(f"  - SMA1: 10 bars")
        print(f"  - SMA2: 20 bars")
        print(f"  - Date range: {start_date.split()[0]} to {end_date.split()[0]}")
        print()
        
        print("🚀 Running strategy...")
        performance, outperformance = sma_bt.run_strategy()
        print(f"✓ Backtest complete!")
        print()
        
        # Step 4: Display results
        print("=" * 70)
        print("📊 BACKTEST RESULTS")
        print("=" * 70)
        print()
        
        # Get results
        results = sma_bt.results
        
        # Calculate metrics
        returns = results['return'].dropna()
        cumulative_returns = (1 + returns).cumprod()
        
        # Sharpe ratio
        sharpe = utils.calculate_sharpe_ratio(returns)
        
        # Max drawdown
        max_dd, dd_duration = utils.calculate_drawdown(cumulative_returns)
        
        # Win rate
        positive_returns = (returns > 0).sum()
        total_returns = len(returns)
        win_rate = (positive_returns / total_returns * 100) if total_returns > 0 else 0
        
        # Print results
        print(f"Cumulative Return (Strategy):    {performance * 100:>10.2f} %")
        print(f"Outperformance vs Buy & Hold:    {outperformance * 100:>10.2f} %")
        print(f"Sharpe Ratio:                    {sharpe:>10.2f}")
        print(f"Maximum Drawdown:                {abs(max_dd) * 100:>10.2f} %")
        print(f"Win Rate:                        {win_rate:>10.2f} %")
        print()
        
        # Step 5: Show sample data points
        print("=" * 70)
        print("📈 SAMPLE DATA POINTS (with SMA signals)")
        print("=" * 70)
        print()
        
        # Get data points where both SMAs are available
        valid_data = results[results['SMA1'].notna() & results['SMA2'].notna()].head(10)
        
        print("Date       | Price    | SMA1     | SMA2     | Signal")
        print("-" * 70)
        
        for idx, (date, row) in enumerate(valid_data.iterrows()):
            price = row['price']
            sma1 = row['SMA1']
            sma2 = row['SMA2']
            signal = "BUY ↑" if sma1 > sma2 else "SELL ↓"
            print(f"{date.date()} | ${price:>7.2f} | ${sma1:>7.2f} | ${sma2:>7.2f} | {signal}")
        
        print()
        print("=" * 70)
        print()
        
        # Step 6: Show tail of results
        print("📊 LATEST DATA POINTS")
        print("=" * 70)
        print()
        
        tail_data = results[results['SMA1'].notna() & results['SMA2'].notna()].tail(5)
        
        print("Date       | Price    | SMA1     | SMA2     | Return   | Strategy Return")
        print("-" * 70)
        
        for idx, (date, row) in enumerate(tail_data.iterrows()):
            price = row['price']
            sma1 = row['SMA1']
            sma2 = row['SMA2']
            ret = row['return'] * 100
            strat = row['strategy'] * 100
            signal = "BUY ↑" if sma1 > sma2 else "SELL ↓"
            print(f"{date.date()} | ${price:>7.2f} | ${sma1:>7.2f} | ${sma2:>7.2f} | {ret:>6.2f}% | {strat:>6.2f}%")
        
        print()
        print("=" * 70)
        print()
        
        # Step 7: Summary
        print("✅ EXAMPLE COMPLETED SUCCESSFULLY!")
        print()
        print("Summary:")
        print(f"  ✓ Loaded {len(data)} bars of historical data")
        print(f"  ✓ Created SMA(10)/SMA(20) strategy")
        print(f"  ✓ Strategy return: {performance * 100:.2f}%")
        print(f"  ✓ Sharpe ratio: {sharpe:.2f}")
        print()
        print("Next steps:")
        print("  1. Try modifying SMA1 and SMA2 parameters")
        print("  2. Run example_basic.py for more examples")
        print("  3. Run example_trading.py for trading examples")
        print("  4. Check README.md for full documentation")
        print()
        
        return True
        
    except Exception as e:
        print()
        print("❌ ERROR OCCURRED:")
        print(f"   {str(e)}")
        print()
        print("Troubleshooting:")
        print("  1. Make sure all dependencies are installed:")
        print("     pip install -r requirements.txt")
        print("  2. Verify sample_data.csv exists in the same directory")
        print("  3. Check that py4at_app folder is in the same directory")
        print()
        
        # Print full traceback for debugging
        import traceback
        print("Full traceback:")
        print("-" * 70)
        traceback.print_exc()
        print("-" * 70)
        
        return False
