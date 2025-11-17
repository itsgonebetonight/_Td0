"""
Grid search over future_h and ret_threshold parameters.
Tests AAPL 1-minute data with various label horizons and thresholds.
Generates a summary table of backtest results for each combination.
"""

import os
import sys
import time
import traceback
import pandas as pd
import numpy as np

# Ensure package path imports the integrated package
pkg_path = r"c:\Users\HP\Downloads\_\_Td0\__Td0\py4at_app"
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

try:
    import yfinance as yf
except Exception:
    yf = None

from py4at_app.ml_strategy_integration import MLTradingStrategy

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'grid_search_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grid parameters
FUTURE_H_VALUES = [3, 5, 8, 10]
RET_THRESHOLD_VALUES = [0.0002, 0.0003, 0.0005, 0.001]

def fetch_aapl_1m_data():
    """Fetch AAPL 1m data for the last 7 days."""
    if yf is None:
        raise RuntimeError('yfinance not installed')
    try:
        t = yf.Ticker('AAPL')
        df = t.history(period='7d', interval='1m', auto_adjust=False)
        if df.empty:
            return None
        if 'Close' in df.columns:
            df = df.rename(columns={'Close': 'price'})
        elif 'close' in df.columns:
            df = df.rename(columns={'close': 'price'})
        else:
            df['price'] = df.iloc[:, 0]
        df = df[['price']].dropna()
        return df
    except Exception:
        traceback.print_exc()
        return None


def run_grid_search_combo(symbol, future_h, ret_threshold, train_data, test_data):
    """Run a single grid search combination (future_h, ret_threshold)."""
    results = {
        'future_h': future_h,
        'ret_threshold': ret_threshold,
        'total_trades': 0,
        'closed_trades': 0,
        'net_pnl': 0.0,
        'win_rate': 0.0,
        'wins': 0,
        'losses': 0,
        'avg_pnl': 0.0,
        'avg_hold_bars': 0.0,
        'high_confidence_signals': 0,
        'final_amount': 10000.0,
        'error': None
    }
    
    try:
        ml = MLTradingStrategy(symbol=symbol)
        
        # Prepare training data with specific parameters
        train_with_features = ml.prepare_data(train_data, future_h=future_h, ret_threshold=ret_threshold)
        
        # Train model
        train_results = ml.train(train_with_features, test_size=0.2)
        
        # Backtest on test data
        backtest_results = ml.backtest(test_data, confidence_threshold=0.65, retrain_interval=60, allow_short=True)
        
        # Extract metrics
        results['total_trades'] = backtest_results.get('trades', 0)
        results['final_amount'] = backtest_results.get('final_amount', 10000.0)
        
        trade_log = backtest_results.get('trade_log', [])
        if trade_log:
            closed_trades = [t for t in trade_log if t.get('status') == 'closed' and t.get('pnl') is not None]
            results['closed_trades'] = len(closed_trades)
            
            if closed_trades:
                pnls = [t['pnl'] for t in closed_trades]
                results['net_pnl'] = sum(pnls)
                results['avg_pnl'] = results['net_pnl'] / len(closed_trades)
                results['wins'] = sum(1 for p in pnls if p > 0)
                results['losses'] = sum(1 for p in pnls if p <= 0)
                results['win_rate'] = results['wins'] / len(closed_trades) if closed_trades else 0.0
                
                hold_bars = [t.get('hold_bars', 0) for t in closed_trades if t.get('hold_bars') is not None]
                results['avg_hold_bars'] = sum(hold_bars) / len(hold_bars) if hold_bars else 0.0
        
        signal_stats = backtest_results.get('signal_stats', {})
        results['high_confidence_signals'] = signal_stats.get('high_confidence_signals', 0)
        
    except Exception as e:
        results['error'] = str(e)
        print(f"  Error: {e}")
    
    return results


def main():
    print("\n" + "=" * 80)
    print("GRID SEARCH: FUTURE_H × RET_THRESHOLD")
    print("=" * 80)
    print()
    
    if yf is None:
        print('yfinance not available. Aborting.')
        return
    
    # Fetch data
    print("Fetching AAPL 1m data...")
    df = fetch_aapl_1m_data()
    if df is None or len(df) < 100:
        print(f"Not enough data (rows={len(df) if df is not None else 0}). Aborting.")
        return
    
    print(f"✓ Fetched {len(df)} rows")
    
    # Split train/test
    split = int(len(df) * 0.7)
    train_data = df.iloc[:split]
    test_data = df.iloc[split:]
    print(f"  Train: {len(train_data)} | Test: {len(test_data)}")
    print()
    
    # Run grid search
    all_results = []
    total_combos = len(FUTURE_H_VALUES) * len(RET_THRESHOLD_VALUES)
    combo_idx = 0
    
    for future_h in FUTURE_H_VALUES:
        for ret_threshold in RET_THRESHOLD_VALUES:
            combo_idx += 1
            print(f"[{combo_idx}/{total_combos}] Running: future_h={future_h}, ret_threshold={ret_threshold:.5f}")
            
            result = run_grid_search_combo('AAPL', future_h, ret_threshold, train_data, test_data)
            all_results.append(result)
            
            # Print summary for this combo
            if result['error']:
                print(f"  → ERROR: {result['error']}")
            else:
                print(f"  → Total trades: {result['total_trades']} | Closed: {result['closed_trades']} | "
                      f"P&L: ${result['net_pnl']:.2f} | Win rate: {result['win_rate']*100:.1f}% | "
                      f"Avg hold: {result['avg_hold_bars']:.1f} bars")
            
            time.sleep(0.5)  # brief pause between runs
    
    print()
    print("=" * 80)
    print("GRID SEARCH RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Create summary table
    summary_df = pd.DataFrame(all_results)
    
    # Display table
    display_cols = ['future_h', 'ret_threshold', 'total_trades', 'closed_trades', 'net_pnl', 
                    'win_rate', 'avg_pnl', 'avg_hold_bars', 'high_confidence_signals']
    summary_table = summary_df[display_cols].copy()
    
    # Format for display
    summary_table['ret_threshold'] = summary_table['ret_threshold'].apply(lambda x: f"{x:.5f}")
    summary_table['net_pnl'] = summary_table['net_pnl'].apply(lambda x: f"${x:>7.2f}")
    summary_table['win_rate'] = summary_table['win_rate'].apply(lambda x: f"{x*100:>5.1f}%")
    summary_table['avg_pnl'] = summary_table['avg_pnl'].apply(lambda x: f"${x:>6.2f}")
    summary_table['avg_hold_bars'] = summary_table['avg_hold_bars'].apply(lambda x: f"{x:>6.1f}")
    
    print(summary_table.to_string(index=False))
    print()
    
    # Save detailed results to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'grid_search_results.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"✓ Detailed results saved to: {csv_path}")
    print()
    
    # Find best combinations
    print("=" * 80)
    print("TOP CONFIGURATIONS (by net P&L)")
    print("=" * 80)
    print()
    
    valid_results = [r for r in all_results if r['error'] is None and r['closed_trades'] > 0]
    if valid_results:
        sorted_by_pnl = sorted(valid_results, key=lambda x: x['net_pnl'], reverse=True)
        for i, r in enumerate(sorted_by_pnl[:5], 1):
            print(f"{i}. future_h={r['future_h']}, ret_threshold={r['ret_threshold']:.5f}")
            print(f"   P&L: ${r['net_pnl']:.2f} | Closed trades: {r['closed_trades']} | "
                  f"Win rate: {r['win_rate']*100:.1f}% | Avg hold: {r['avg_hold_bars']:.1f} bars")
            print()
    else:
        print("No valid results with closed trades.")
    
    print("=" * 80)
    print("TOP CONFIGURATIONS (by win rate, min 2 closed trades)")
    print("=" * 80)
    print()
    
    valid_results_min_trades = [r for r in all_results if r['error'] is None and r['closed_trades'] >= 2]
    if valid_results_min_trades:
        sorted_by_wr = sorted(valid_results_min_trades, key=lambda x: x['win_rate'], reverse=True)
        for i, r in enumerate(sorted_by_wr[:5], 1):
            print(f"{i}. future_h={r['future_h']}, ret_threshold={r['ret_threshold']:.5f}")
            print(f"   Win rate: {r['win_rate']*100:.1f}% | Closed trades: {r['closed_trades']} | "
                  f"P&L: ${r['net_pnl']:.2f} | Avg hold: {r['avg_hold_bars']:.1f} bars")
            print()
    else:
        print("No valid results with ≥2 closed trades.")
    
    print()


if __name__ == '__main__':
    main()
