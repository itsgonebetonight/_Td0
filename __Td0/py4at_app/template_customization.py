"""
CUSTOMIZATION GUIDE - How to Modify & Create Your Own Strategies
==================================================================

This file shows you how to modify the framework for your own use cases.
It provides several templates you can copy and adapt.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from py4at_app.backtesting import SMAVectorBacktester, MomVectorBacktester
from py4at_app.data import DataLoader
from py4at_app import utils


# ============================================================================
# TEMPLATE 1: Modify SMA Parameters
# ============================================================================

def template_1_sma_parameter_optimization():
    """
    Test multiple SMA combinations to find the best parameters.
    
    This shows how to loop through different parameters and find the
    combination that gives the best results.
    """
    print("=" * 70)
    print("TEMPLATE 1: SMA PARAMETER OPTIMIZATION")
    print("=" * 70)
    print()
    
    # Load data
    data = pd.read_csv('sample_data.csv', index_col=0, parse_dates=True)
    data.rename(columns={'Close': 'price'}, inplace=True)
    
    print("Testing different SMA combinations...")
    print()
    
    results_list = []
    
    # Test different parameter combinations
    for sma1 in [5, 10, 15, 20]:
        for sma2 in [20, 30, 50]:
            if sma1 >= sma2:  # Fast SMA must be shorter than slow SMA
                continue
            
            try:
                # Create backtester
                bt = SMAVectorBacktester(
                    symbol='EUR/USD',
                    SMA1=sma1,
                    SMA2=sma2,
                    start=str(data.index[0]),
                    end=str(data.index[-1]),
                    data=data
                )
                
                # Run strategy
                perf, outperf = bt.run_strategy()
                
                # Store results
                results_list.append({
                    'SMA1': sma1,
                    'SMA2': sma2,
                    'Return': perf * 100,
                    'Outperformance': outperf * 100
                })
                
                print(f"SMA({sma1:2d}/{sma2:2d}): Return={perf*100:>7.2f}% | "
                      f"Outperf={outperf*100:>7.2f}%")
                
            except Exception as e:
                print(f"SMA({sma1:2d}/{sma2:2d}): Error - {str(e)[:40]}")
    
    # Find best parameters
    print()
    best = max(results_list, key=lambda x: x['Return'])
    print(f"🏆 Best parameters: SMA({best['SMA1']}/{best['SMA2']})")
    print(f"   Return: {best['Return']:.2f}%")
    print()
    
    return results_list


# ============================================================================
# TEMPLATE 2: Backtest with Different Data
# ============================================================================

def template_2_backtest_different_data():
    """
    Load your own CSV data and run backtest.
    
    Your CSV file should have a 'Close' or 'price' column.
    """
    print("=" * 70)
    print("TEMPLATE 2: BACKTEST WITH DIFFERENT DATA")
    print("=" * 70)
    print()
    
    # Modify this to load your own file
    csv_file = 'sample_data.csv'
    
    print(f"Loading data from: {csv_file}")
    
    try:
        # Load your data
        data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(data)} rows")
        print(f"  Columns: {list(data.columns)}")
        print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print()
        
        # Rename column if needed (expects 'price' column)
        if 'Close' in data.columns:
            data.rename(columns={'Close': 'price'}, inplace=True)
        elif 'close' in data.columns:
            data.rename(columns={'close': 'price'}, inplace=True)
        elif 'Price' in data.columns:
            data.rename(columns={'Price': 'price'}, inplace=True)
        
        print("✓ Data prepared")
        print()
        
        # Run backtest
        bt = SMAVectorBacktester(
            symbol='EUR/USD',
            SMA1=10,
            SMA2=20,
            start=str(data.index[0]),
            end=str(data.index[-1]),
            data=data
        )
        
        perf, outperf = bt.run_strategy()
        
        print(f"Strategy Return: {perf * 100:.2f}%")
        print(f"Outperformance: {outperf * 100:.2f}%")
        print()
        
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
        print("   Please check the filename and try again")


# ============================================================================
# TEMPLATE 3: Calculate Additional Performance Metrics
# ============================================================================

def template_3_advanced_metrics():
    """
    Show how to calculate additional performance metrics from results.
    """
    print("=" * 70)
    print("TEMPLATE 3: ADVANCED PERFORMANCE METRICS")
    print("=" * 70)
    print()
    
    # Load and run backtest
    data = pd.read_csv('sample_data.csv', index_col=0, parse_dates=True)
    data.rename(columns={'Close': 'price'}, inplace=True)
    
    bt = SMAVectorBacktester(
        symbol='EUR/USD',
        SMA1=10,
        SMA2=20,
        start=str(data.index[0]),
        end=str(data.index[-1]),
        data=data
    )
    
    perf, outperf = bt.run_strategy()
    results = bt.results
    
    # Extract returns
    returns = results['return'].dropna()
    strategy_returns = results['strategy'].dropna()
    cumulative_returns = (1 + returns).cumprod()
    
    print("📊 BASIC METRICS:")
    print(f"  Strategy Return:  {perf * 100:.2f}%")
    print(f"  Buy & Hold Return: {(cumulative_returns.iloc[-1] - 1) * 100:.2f}%")
    print()
    
    print("📈 RETURN METRICS:")
    print(f"  Average Daily Return:     {returns.mean() * 100:.4f}%")
    print(f"  Daily Return Std Dev:     {returns.std() * 100:.4f}%")
    print(f"  Min Daily Return:         {returns.min() * 100:.4f}%")
    print(f"  Max Daily Return:         {returns.max() * 100:.4f}%")
    print()
    
    print("📊 RISK METRICS:")
    sharpe = utils.calculate_sharpe_ratio(returns)
    max_dd, dd_dur = utils.calculate_drawdown(cumulative_returns)
    print(f"  Sharpe Ratio:             {sharpe:.2f}")
    print(f"  Max Drawdown:             {abs(max_dd) * 100:.2f}%")
    print(f"  Drawdown Duration:        {int(dd_dur)} bars")
    print()
    
    print("🎯 TRADE METRICS:")
    win_count = (strategy_returns > 0).sum()
    loss_count = (strategy_returns < 0).sum()
    win_rate = (win_count / (win_count + loss_count) * 100) if (win_count + loss_count) > 0 else 0
    avg_win = strategy_returns[strategy_returns > 0].mean() * 100 if len(strategy_returns[strategy_returns > 0]) > 0 else 0
    avg_loss = strategy_returns[strategy_returns < 0].mean() * 100 if len(strategy_returns[strategy_returns < 0]) > 0 else 0
    
    print(f"  Winning Trades:           {int(win_count)}")
    print(f"  Losing Trades:            {int(loss_count)}")
    print(f"  Win Rate:                 {win_rate:.2f}%")
    print(f"  Average Win:              {avg_win:.4f}%")
    print(f"  Average Loss:             {avg_loss:.4f}%")
    print()


# ============================================================================
# TEMPLATE 4: Export Results for Analysis
# ============================================================================

def template_4_export_results():
    """
    Save backtest results to CSV for further analysis in Excel or other tools.
    """
    print("=" * 70)
    print("TEMPLATE 4: EXPORT RESULTS TO CSV")
    print("=" * 70)
    print()
    
    # Run backtest
    data = pd.read_csv('sample_data.csv', index_col=0, parse_dates=True)
    data.rename(columns={'Close': 'price'}, inplace=True)
    
    bt = SMAVectorBacktester(
        symbol='EUR/USD',
        SMA1=10,
        SMA2=20,
        start=str(data.index[0]),
        end=str(data.index[-1]),
        data=data
    )
    
    perf, outperf = bt.run_strategy()
    results = bt.results
    
    # Save results
    output_file = 'backtest_results.csv'
    results.to_csv(output_file)
    
    print(f"✓ Results saved to: {output_file}")
    print(f"  Rows: {len(results)}")
    print(f"  Columns: {list(results.columns)}")
    print()
    print("You can now open this file in Excel to analyze further!")
    print()


# ============================================================================
# TEMPLATE 5: Compare Multiple Strategies
# ============================================================================

def template_5_compare_strategies():
    """
    Compare different strategies on the same data.
    """
    print("=" * 70)
    print("TEMPLATE 5: COMPARE MULTIPLE STRATEGIES")
    print("=" * 70)
    print()
    
    # Load data
    data = pd.read_csv('sample_data.csv', index_col=0, parse_dates=True)
    data.rename(columns={'Close': 'price'}, inplace=True)
    
    strategies = [
        {'name': 'Conservative', 'SMA1': 20, 'SMA2': 50},
        {'name': 'Balanced', 'SMA1': 10, 'SMA2': 20},
        {'name': 'Aggressive', 'SMA1': 5, 'SMA2': 15},
    ]
    
    print("Comparing strategies...")
    print()
    
    results_df = []
    
    for strat in strategies:
        bt = SMAVectorBacktester(
            symbol='EUR/USD',
            SMA1=strat['SMA1'],
            SMA2=strat['SMA2'],
            start=str(data.index[0]),
            end=str(data.index[-1]),
            data=data
        )
        
        perf, outperf = bt.run_strategy()
        backtest_results = bt.results
        returns = backtest_results['return'].dropna()
        sharpe = utils.calculate_sharpe_ratio(returns)
        cumulative_returns = (1 + returns).cumprod()
        max_dd, _ = utils.calculate_drawdown(cumulative_returns)
        
        results_df.append({
            'Strategy': strat['name'],
            'SMA1': strat['SMA1'],
            'SMA2': strat['SMA2'],
            'Return %': perf * 100,
            'Sharpe': sharpe,
            'Max DD %': abs(max_dd) * 100,
        })
    
    # Display as table
    results_table = pd.DataFrame(results_df)
    print(results_table.to_string(index=False))
    print()
    
    # Find best
    best_return = max(results_df, key=lambda x: x['Return %'])
    best_sharpe = max(results_df, key=lambda x: x['Sharpe'])
    
    print(f"🏆 Best Return: {best_return['Strategy']} "
          f"({best_return['Return %']:.2f}%)")
    print(f"🏆 Best Risk-Adjusted: {best_sharpe['Strategy']} "
          f"(Sharpe: {best_sharpe['Sharpe']:.2f})")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  CUSTOMIZATION TEMPLATES  ".center(68) + "║")
    print("║" + "  Learn how to adapt the framework  ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    try:
        # Run all templates
        template_1_sma_parameter_optimization()
        print()
        
        template_2_backtest_different_data()
        print()
        
        template_3_advanced_metrics()
        print()
        
        template_4_export_results()
        print()
        
        template_5_compare_strategies()
        
        print()
        print("╔" + "=" * 68 + "╗")
        print("║" + "  ALL TEMPLATES COMPLETED  ".center(68) + "║")
        print("╚" + "=" * 68 + "╝")
        print()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
