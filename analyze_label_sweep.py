"""
Label Sweep Results Analysis
============================
Analyzes results from run_label_sweep.py to identify optimal label configuration.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_label_sweep():
    """Analyze label sweep results and generate recommendations."""
    
    # Load results
    results_path = Path(__file__).parent / "label_sweep_results" / "label_sweep_results.csv"
    if not results_path.exists():
        print(f"ERROR: Results file not found at {results_path}")
        return None
    
    results = pd.read_csv(results_path)
    
    # Clean numeric columns
    numeric_cols = ['total_trades', 'closed_trades']
    for col in numeric_cols:
        results[col] = pd.to_numeric(results[col], errors='coerce')
    
    # Parse win_rate, avg_hold_bars, net_pnl, max_drawdown, signal_density
    results['win_rate_pct'] = results['win_rate'].str.rstrip('%').astype(float, errors='ignore') / 100.0 if 'win_rate' in results.columns else None
    results['avg_hold_bars_num'] = pd.to_numeric(results['avg_hold_bars'], errors='coerce')
    results['net_pnl_num'] = pd.to_numeric(results['net_pnl'], errors='coerce')
    results['max_drawdown_pct'] = results['max_drawdown'].str.rstrip('%').astype(float, errors='ignore') / 100.0 if 'max_drawdown' in results.columns else None
    results['signal_density_pct'] = results['signal_density'].str.rstrip('%').astype(float, errors='ignore') if 'signal_density' in results.columns else None
    
    print("=" * 150)
    print("LABEL SWEEP ANALYSIS - RESULTS")
    print("=" * 150)
    print()
    
    # Display all results
    print("All Label Configurations (sorted by Net P&L, descending):")
    print("-" * 150)
    results_sorted = results.sort_values('net_pnl_num', ascending=False, na_position='last')
    display_cols = ['future_h', 'ret_threshold', 'total_trades', 'closed_trades', 'win_rate', 'avg_hold_bars', 'net_pnl', 'signal_density']
    print(results_sorted[display_cols].to_string(index=False))
    print()
    
    # Analysis and recommendations
    print("=" * 150)
    print("KEY METRICS & ANALYSIS")
    print("=" * 150)
    print()
    
    # Find best configurations by different criteria
    print("1. BEST CONFIGURATIONS BY NET P&L:")
    print("-" * 150)
    best_pnl = results_sorted.iloc[0]
    print(f"   Rank 1: future_h={int(best_pnl['future_h'])}, ret_threshold={best_pnl['ret_threshold']:.5f}")
    print(f"      -> Net P&L: {best_pnl['net_pnl']}")
    print(f"      -> Total Trades: {int(best_pnl['total_trades'])}, Closed: {int(best_pnl['closed_trades'])}")
    print(f"      -> Win Rate: {best_pnl['win_rate']}, Avg Hold: {best_pnl['avg_hold_bars']} bars")
    print(f"      -> Signal Density: {best_pnl['signal_density']}")
    print()
    
    if len(results_sorted) > 1:
        second_best = results_sorted.iloc[1]
        print(f"   Rank 2: future_h={int(second_best['future_h'])}, ret_threshold={second_best['ret_threshold']:.5f}")
        print(f"      -> Net P&L: {second_best['net_pnl']}")
        print(f"      -> Total Trades: {int(second_best['total_trades'])}, Closed: {int(second_best['closed_trades'])}")
        print(f"      -> Signal Density: {second_best['signal_density']}")
    print()
    
    # Best trade frequency (maximize trades while keeping P&L non-negative)
    print("2. TRADE FREQUENCY ANALYSIS (maximize signal density + trades):")
    print("-" * 150)
    positive_or_breakeven = results[results['net_pnl_num'] >= -10]  # Allow small losses
    if len(positive_or_breakeven) > 0:
        most_trades = positive_or_breakeven.nlargest(1, 'total_trades').iloc[0]
        print(f"   Config with most trades (P&L >= -10): future_h={int(most_trades['future_h'])}, ret_threshold={most_trades['ret_threshold']:.5f}")
        print(f"      -> Total Trades: {int(most_trades['total_trades'])}, Signal Density: {most_trades['signal_density']}")
        print(f"      -> Net P&L: {most_trades['net_pnl']}")
    else:
        print("   No configs with non-negative P&L found")
    print()
    
    # Analysis by future_h
    print("3. IMPACT OF FUTURE_H (future price prediction horizon):")
    print("-" * 150)
    for fh in sorted(results['future_h'].unique()):
        subset = results[results['future_h'] == fh]
        avg_pnl = subset['net_pnl_num'].mean()
        avg_trades = subset['total_trades'].mean()
        avg_signal = subset['signal_density_pct'].mean()
        print(f"   future_h={int(fh)}: Avg P&L=${avg_pnl:.2f}, Avg Trades={avg_trades:.1f}, Avg Signal Density={avg_signal:.2f}%")
    print()
    
    # Analysis by ret_threshold
    print("4. IMPACT OF RET_THRESHOLD (minimum return for label):")
    print("-" * 150)
    for rt in sorted(results['ret_threshold'].unique()):
        subset = results[results['ret_threshold'] == rt]
        avg_pnl = subset['net_pnl_num'].mean()
        avg_trades = subset['total_trades'].mean()
        avg_signal = subset['signal_density_pct'].mean()
        print(f"   ret_threshold={rt:.5f}: Avg P&L=${avg_pnl:.2f}, Avg Trades={avg_trades:.1f}, Avg Signal Density={avg_signal:.2f}%")
    print()
    
    # Final recommendation
    print("=" * 150)
    print("RECOMMENDATION FOR PRODUCTION")
    print("=" * 150)
    print()
    
    optimal = results_sorted.iloc[0]
    print(f"Optimal Label Configuration:")
    print(f"  • future_h: {int(optimal['future_h'])}")
    print(f"  • ret_threshold: {optimal['ret_threshold']:.5f} ({optimal['ret_threshold']*100:.3f}%)")
    print()
    
    print(f"Expected Performance (from backtests):")
    print(f"  • Net P&L: {optimal['net_pnl']}")
    print(f"  • Total Trades: {int(optimal['total_trades'])}")
    print(f"  • Closed Trades: {int(optimal['closed_trades'])}")
    print(f"  • Win Rate: {optimal['win_rate']}")
    print(f"  • Avg Hold Time: {optimal['avg_hold_bars']} bars (minutes in 1min data)")
    print(f"  • Signal Density: {optimal['signal_density']} ({int(optimal['total_trades'])} trades out of ~{len(results_sorted)} trading days)")
    print(f"  • Max Drawdown: {optimal['max_drawdown']}")
    print()
    
    print(f"Strategy Configuration (fixed):")
    print(f"  • confidence_threshold: 0.60")
    print(f"  • confirmation_bars: 2")
    print(f"  • dynamic_atr_k: 2.0")
    print(f"  • risk_fraction: 0.05")
    print(f"  • stop_loss_pct: 2%")
    print(f"  • take_profit_pct: 4%")
    print()
    
    print(f"Rationale:")
    if optimal['net_pnl_num'] > 0:
        print(f"  ✓ Configuration shows positive P&L while maintaining reasonable trade frequency")
    elif optimal['net_pnl_num'] >= -10:
        print(f"  ✓ Configuration shows near-breakeven P&L with good signal generation")
    else:
        print(f"  ! Configuration shows losses but best among tested options")
    
    if optimal['total_trades'] > 5:
        print(f"  ✓ High signal density ({optimal['signal_density']}) provides enough trading opportunities")
    elif optimal['total_trades'] > 2:
        print(f"  - Moderate signal density; consider stricter labels if more trades needed")
    else:
        print(f"  ! Low signal density; consider using lower ret_threshold for more trades")
    print()
    
    return results_sorted

if __name__ == "__main__":
    results = analyze_label_sweep()
