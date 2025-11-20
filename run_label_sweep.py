"""
Label Configuration Sweep Experiment
=====================================
Tests different label configurations (future_h, ret_threshold) with fixed best strategy config.

Best strategy config (from confidence sweep):
  - confidence_threshold: 0.60
  - confirmation_bars: 2
  - dynamic_atr_k: 2.0
  - risk_fraction: 0.05

Label configurations to test:
  - future_h: [3, 5]
  - ret_threshold: [0.00005, 0.0001, 0.00015, 0.0002]
  Total combinations: 2 * 4 = 8
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add py4at_app to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_predictor import EnhancedFeatureEngineering, EnhancedTradingPredictor
from ml_strategy_integration import MLTradingStrategy

def run_label_sweep():
    """Run label sweep with fixed best strategy config."""
    
    # Fixed best strategy config
    BEST_CONFIG = {
        'confidence_threshold': 0.60,
        'confirmation_bars': 2,
        'dynamic_atr_k': 2.0,
        'risk_fraction': 0.05,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'slippage': 0.0005,
    }
    
    # Label configurations to test
    FUTURE_H_VALUES = [3, 5]
    RET_THRESHOLD_VALUES = [round(x, 5) for x in [0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008]]
    
    # Output directory
    output_dir = Path(__file__).parent / "label_sweep_results"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    # Try multiple possible locations (prefer minute data for more bars)
    possible_paths = [
        Path(__file__).parent / "py4at" / "data" / "AAPL_7d_1m.csv"
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        print(f"ERROR: Data file not found at any of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        return None
    
    # Skip first three header rows to get valid numeric data
    raw_data = pd.read_csv(data_path, index_col=0, parse_dates=True, skiprows=3)
    # Rename first column to 'price'
    raw_data.columns = ['price'] + list(raw_data.columns[1:])
    
    # Keep only price column
    raw_data = raw_data[['price']].copy()
    print(f"[+] Loaded {len(raw_data)} bars of data from {data_path}")
    print()
    
    # Initialize results list
    results = []
    
    print("=" * 140)
    print("LABEL CONFIGURATION SWEEP (fixed best strategy config)")
    print("=" * 140)
    print()
    print(f"Best Strategy Config:")
    for key, val in BEST_CONFIG.items():
        print(f"  • {key}: {val}")
    print()
    print(f"Label configurations: {len(FUTURE_H_VALUES)} x {len(RET_THRESHOLD_VALUES)} = {len(FUTURE_H_VALUES) * len(RET_THRESHOLD_VALUES)} combinations")
    print()
    
    combination_num = 0
    total_combinations = len(FUTURE_H_VALUES) * len(RET_THRESHOLD_VALUES)
    
    for future_h in FUTURE_H_VALUES:
        for ret_threshold in RET_THRESHOLD_VALUES:
            combination_num += 1
            
            print("-" * 140)
            print(f"[{combination_num}/{total_combinations}] future_h={future_h}, ret_threshold={ret_threshold}")
            print("-" * 140)
            
            try:
                # Engineer features with label config
                fe = EnhancedFeatureEngineering()
                data_with_features = fe.add_technical_indicators(
                    raw_data.copy(),
                    future_h=future_h,
                    ret_threshold=ret_threshold
                )
                
                # Initialize ML trading strategy
                strategy = MLTradingStrategy(symbol='AAPL')
                
                # Train model on data
                train_results = strategy.train(data_with_features, test_size=0.2)
                
                # Run backtest with best config
                backtest_results = strategy.backtest(
                    data=data_with_features,
                    confidence_threshold=BEST_CONFIG['confidence_threshold'],
                    initial_amount=10000,
                    tc=0.001,
                    risk_fraction=BEST_CONFIG['risk_fraction'],
                    stop_loss_pct=BEST_CONFIG['stop_loss_pct'],
                    take_profit_pct=BEST_CONFIG['take_profit_pct'],
                    slippage=BEST_CONFIG['slippage'],
                    retrain_interval=60,
                    allow_short=False,
                    confirmation_bars=BEST_CONFIG['confirmation_bars'],
                    dynamic_atr_k=BEST_CONFIG['dynamic_atr_k']
                )
                
                # Extract results
                final_amount = backtest_results['final_amount']
                net_pnl = final_amount - 10000
                
                # Get trade summary
                trade_summary = backtest_results.get('trade_summary', {})
                total_trades = len(backtest_results.get('trade_log', []))
                closed_trades = trade_summary.get('total_trades', 0)
                win_rate = trade_summary.get('win_rate', 0.0)
                avg_hold = trade_summary.get('avg_hold_bars', 0.0)
                max_drawdown = trade_summary.get('max_drawdown', 0.0)
                sharpe = trade_summary.get('sharpe_ratio', None)
                
                # Calculate signal density (trades per bar)
                total_bars = len(data_with_features)
                signal_density = (total_trades / total_bars * 100) if total_bars > 0 else 0.0
                
                result = {
                    'future_h': future_h,
                    'ret_threshold': ret_threshold,
                    'total_trades': total_trades,
                    'closed_trades': closed_trades,
                    'win_rate': f"{win_rate*100:.1f}%" if win_rate is not None else "N/A",
                    'avg_hold_bars': f"{avg_hold:.1f}" if avg_hold is not None else "N/A",
                    'net_pnl': f"{net_pnl:.2f}",
                    'max_drawdown': f"{max_drawdown:.2f}%" if max_drawdown is not None else "N/A",
                    'sharpe': f"{sharpe:.3f}" if sharpe is not None else "N/A",
                    'signal_density': f"{signal_density:.2f}%",
                    'final_amount': f"{final_amount:.2f}",
                }
                
                results.append(result)
                
                print(f"  Total Trades: {total_trades}")
                print(f"  Closed Trades: {closed_trades}")
                print(f"  Win Rate: {win_rate*100:.1f}%" if win_rate is not None else "  Win Rate: N/A")
                print(f"  Avg Hold: {avg_hold:.1f} bars" if avg_hold is not None else "  Avg Hold: N/A")
                print(f"  Net P&L: ${net_pnl:.2f}")
                print(f"  Max Drawdown: {max_drawdown:.2f}%" if max_drawdown is not None else "  Max Drawdown: N/A")
                print(f"  Sharpe: {sharpe:.3f}" if sharpe is not None else "  Sharpe: N/A")
                print(f"  Signal Density: {signal_density:.2f}%")
                print()
                
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                result = {
                    'future_h': future_h,
                    'ret_threshold': ret_threshold,
                    'total_trades': 'ERROR',
                    'closed_trades': 'ERROR',
                    'win_rate': 'ERROR',
                    'avg_hold_bars': 'ERROR',
                    'net_pnl': 'ERROR',
                    'max_drawdown': 'ERROR',
                    'sharpe': 'ERROR',
                    'signal_density': 'ERROR',
                    'final_amount': 'ERROR',
                }
                results.append(result)
                print()
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_csv = output_dir / "label_sweep_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"[+] Results saved to: {output_csv}")
    print()
    
    return results_df

if __name__ == "__main__":
    results_df = run_label_sweep()
    if results_df is not None:
        print("\n" + "=" * 140)
        print("RESULTS SUMMARY")
        print("=" * 140)
        print(results_df.to_string(index=False))
