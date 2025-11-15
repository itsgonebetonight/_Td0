#!/usr/bin/env python3
"""
Main entry point for py4at_app
Command-line interface for algorithmic trading application
"""

import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Optional

# Import modules
from py4at_app.backtesting import (
    SMAVectorBacktester, MomVectorBacktester, MRVectorBacktester,
    LRVectorBacktester, ScikitVectorBacktester,
    BacktestLongOnly, BacktestLongShort
)
from py4at_app.trading import MomentumTrader, StrategyMonitor, OnlineAlgorithm
from py4at_app.data import DataLoader
from py4at_app import utils


def backtest_sma(args) -> dict:
    """Run SMA vectorized backtest."""
    print(f"\nRunning SMA Vectorized Backtest")
    print(f"Symbol: {args.symbol}")
    print(f"SMA1: {args.sma1}, SMA2: {args.sma2}")
    print(f"Period: {args.start} to {args.end}")
    print("=" * 60)
    
    try:
        # Load data if provided
        data = None
        if args.data_file:
            data = DataLoader.load_from_csv(args.data_file, args.symbol,
                                           args.start, args.end)
        
        backtester = SMAVectorBacktester(
            args.symbol, args.sma1, args.sma2,
            args.start, args.end, data
        )
        
        aperf, operf = backtester.run_strategy()
        
        print(f"\nAbsolute Performance: {aperf:.2f}")
        print(f"Out-/Underperformance: {operf:.2f}")
        
        if args.optimize:
            print("\nOptimizing parameters...")
            opt, perf = backtester.optimize_parameters(
                (args.sma1_min, args.sma1_max, args.sma1_step),
                (args.sma2_min, args.sma2_max, args.sma2_step)
            )
            print(f"Optimal SMA1: {opt[0]:.0f}, SMA2: {opt[1]:.0f}")
            print(f"Optimized Performance: {perf:.2f}")
        
        if args.plot:
            backtester.plot_results()
        
        return {
            'strategy': 'SMA',
            'absolute_performance': aperf,
            'over_underperformance': operf
        }
    
    except Exception as e:
        print(f"Error: {e}")
        return {}


def backtest_momentum(args) -> dict:
    """Run Momentum vectorized backtest."""
    print(f"\nRunning Momentum Vectorized Backtest")
    print(f"Symbol: {args.symbol}")
    print(f"Momentum Period: {args.momentum}")
    print(f"Period: {args.start} to {args.end}")
    print("=" * 60)
    
    try:
        data = None
        if args.data_file:
            data = DataLoader.load_from_csv(args.data_file, args.symbol,
                                           args.start, args.end)
        
        backtester = MomVectorBacktester(
            args.symbol, args.start, args.end,
            args.amount, args.tc, data
        )
        
        aperf, operf = backtester.run_strategy(args.momentum)
        
        print(f"\nAbsolute Performance: {aperf:.2f}")
        print(f"Out-/Underperformance: {operf:.2f}")
        
        if args.plot:
            backtester.plot_results()
        
        return {
            'strategy': 'Momentum',
            'momentum_period': args.momentum,
            'absolute_performance': aperf,
            'over_underperformance': operf
        }
    
    except Exception as e:
        print(f"Error: {e}")
        return {}


def backtest_mean_reversion(args) -> dict:
    """Run Mean Reversion vectorized backtest."""
    print(f"\nRunning Mean Reversion Vectorized Backtest")
    print(f"Symbol: {args.symbol}")
    print(f"SMA: {args.sma}, Threshold: {args.threshold}")
    print(f"Period: {args.start} to {args.end}")
    print("=" * 60)
    
    try:
        data = None
        if args.data_file:
            data = DataLoader.load_from_csv(args.data_file, args.symbol,
                                           args.start, args.end)
        
        backtester = MRVectorBacktester(
            args.symbol, args.start, args.end,
            args.amount, args.tc, data
        )
        
        aperf, operf = backtester.run_strategy(args.sma, args.threshold)
        
        print(f"\nAbsolute Performance: {aperf:.2f}")
        print(f"Out-/Underperformance: {operf:.2f}")
        
        if args.plot:
            backtester.plot_results()
        
        return {
            'strategy': 'Mean Reversion',
            'sma': args.sma,
            'threshold': args.threshold,
            'absolute_performance': aperf,
            'over_underperformance': operf
        }
    
    except Exception as e:
        print(f"Error: {e}")
        return {}


def backtest_ml(args) -> dict:
    """Run Machine Learning vectorized backtest."""
    print(f"\nRunning ML Vectorized Backtest")
    print(f"Symbol: {args.symbol}")
    print(f"Model: {args.ml_model}")
    print(f"Period: {args.start} to {args.end}")
    print("=" * 60)
    
    try:
        data = None
        if args.data_file:
            data = DataLoader.load_from_csv(args.data_file, args.symbol,
                                           args.start, args.end)
        
        backtester = ScikitVectorBacktester(
            args.symbol, args.start, args.end,
            args.amount, args.tc, args.ml_model, data
        )
        
        aperf, operf = backtester.run_strategy(
            args.train_start, args.train_end,
            args.test_start, args.test_end,
            args.lags
        )
        
        print(f"\nAbsolute Performance: {aperf:.2f}")
        print(f"Out-/Underperformance: {operf:.2f}")
        
        if args.plot:
            backtester.plot_results()
        
        return {
            'strategy': 'ML',
            'model': args.ml_model,
            'lags': args.lags,
            'absolute_performance': aperf,
            'over_underperformance': operf
        }
    
    except Exception as e:
        print(f"Error: {e}")
        return {}


def backtest_event(args) -> dict:
    """Run event-based backtest."""
    print(f"\nRunning Event-Based Backtest")
    print(f"Symbol: {args.symbol}")
    print(f"Strategy: {args.event_strategy}")
    print(f"Initial Amount: ${args.amount:.2f}")
    print("=" * 60)
    
    try:
        data = None
        if args.data_file:
            data = DataLoader.load_from_csv(args.data_file, args.symbol,
                                           args.start, args.end)
        
        if args.long_short:
            backtester = BacktestLongShort(
                args.symbol, args.start, args.end,
                args.amount, args.ftc, args.ptc,
                verbose=args.verbose
            )
        else:
            backtester = BacktestLongOnly(
                args.symbol, args.start, args.end,
                args.amount, args.ftc, args.ptc,
                verbose=args.verbose
            )
        
        if data is not None:
            backtester.set_data(data)
        
        if args.event_strategy == 'sma':
            backtester.run_sma_strategy(args.sma1, args.sma2)
        elif args.event_strategy == 'momentum':
            backtester.run_momentum_strategy(args.momentum)
        elif args.event_strategy == 'mean_reversion':
            backtester.run_mean_reversion_strategy(args.sma, args.threshold)
        
        perf = backtester.get_performance()
        
        return {
            'strategy': 'Event-Based',
            'sub_strategy': args.event_strategy,
            'performance': perf,
            'trades': backtester.trades
        }
    
    except Exception as e:
        print(f"Error: {e}")
        return {}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Python for Algorithmic Trading Application (py4at_app)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--symbol', type=str, default='EUR=',
                       help='Financial instrument symbol')
    common.add_argument('--start', type=str, default='2010-01-01',
                       help='Start date (YYYY-MM-DD)')
    common.add_argument('--end', type=str, default='2020-12-31',
                       help='End date (YYYY-MM-DD)')
    common.add_argument('--amount', type=float, default=10000,
                       help='Initial investment amount')
    common.add_argument('--tc', type=float, default=0.001,
                       help='Transaction costs (proportional)')
    common.add_argument('--data-file', type=str, 
                       help='Path to CSV data file')
    common.add_argument('--plot', action='store_true',
                       help='Plot results')
    common.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    # SMA backtest
    sma_parser = subparsers.add_parser('backtest-sma', parents=[common],
                                       help='SMA vectorized backtest')
    sma_parser.add_argument('--sma1', type=int, default=42,
                           help='Short SMA period')
    sma_parser.add_argument('--sma2', type=int, default=252,
                           help='Long SMA period')
    sma_parser.add_argument('--optimize', action='store_true',
                           help='Optimize parameters')
    sma_parser.add_argument('--sma1-min', type=int, default=30,
                           help='SMA1 min for optimization')
    sma_parser.add_argument('--sma1-max', type=int, default=56,
                           help='SMA1 max for optimization')
    sma_parser.add_argument('--sma1-step', type=int, default=4,
                           help='SMA1 step for optimization')
    sma_parser.add_argument('--sma2-min', type=int, default=200,
                           help='SMA2 min for optimization')
    sma_parser.add_argument('--sma2-max', type=int, default=300,
                           help='SMA2 max for optimization')
    sma_parser.add_argument('--sma2-step', type=int, default=4,
                           help='SMA2 step for optimization')
    sma_parser.set_defaults(func=backtest_sma)
    
    # Momentum backtest
    mom_parser = subparsers.add_parser('backtest-momentum', parents=[common],
                                       help='Momentum vectorized backtest')
    mom_parser.add_argument('--momentum', type=int, default=1,
                           help='Momentum period')
    mom_parser.set_defaults(func=backtest_momentum)
    
    # Mean Reversion backtest
    mr_parser = subparsers.add_parser('backtest-mean-reversion', 
                                      parents=[common],
                                      help='Mean reversion vectorized backtest')
    mr_parser.add_argument('--sma', type=int, default=50,
                          help='SMA period')
    mr_parser.add_argument('--threshold', type=float, default=5.0,
                          help='Deviation threshold')
    mr_parser.set_defaults(func=backtest_mean_reversion)
    
    # ML backtest
    ml_parser = subparsers.add_parser('backtest-ml', parents=[common],
                                      help='Machine Learning vectorized backtest')
    ml_parser.add_argument('--ml-model', type=str, default='logistic',
                          choices=['regression', 'logistic'],
                          help='ML model type')
    ml_parser.add_argument('--train-start', type=str, default='2010-01-01',
                          help='Training period start')
    ml_parser.add_argument('--train-end', type=str, default='2015-12-31',
                          help='Training period end')
    ml_parser.add_argument('--test-start', type=str, default='2016-01-01',
                          help='Testing period start')
    ml_parser.add_argument('--test-end', type=str, default='2020-12-31',
                          help='Testing period end')
    ml_parser.add_argument('--lags', type=int, default=5,
                          help='Number of lags for features')
    ml_parser.set_defaults(func=backtest_ml)
    
    # Event-based backtest
    event_parser = subparsers.add_parser('backtest-event', parents=[common],
                                         help='Event-based backtest')
    event_parser.add_argument('--event-strategy', type=str, 
                             default='sma',
                             choices=['sma', 'momentum', 'mean_reversion'],
                             help='Event-based strategy')
    event_parser.add_argument('--sma1', type=int, default=42,
                             help='Short SMA period')
    event_parser.add_argument('--sma2', type=int, default=252,
                             help='Long SMA period')
    event_parser.add_argument('--sma', type=int, default=50,
                             help='SMA period for mean reversion')
    event_parser.add_argument('--threshold', type=float, default=5.0,
                             help='Threshold for mean reversion')
    event_parser.add_argument('--momentum', type=int, default=60,
                             help='Momentum period')
    event_parser.add_argument('--ftc', type=float, default=0.0,
                             help='Fixed transaction costs')
    event_parser.add_argument('--long-short', action='store_true',
                             help='Use long-short strategy')
    event_parser.set_defaults(func=backtest_event)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    result = args.func(args)
    
    if args.verbose or args.command.startswith('backtest'):
        print("\nCommand executed successfully!")
        if result:
            print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
