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
