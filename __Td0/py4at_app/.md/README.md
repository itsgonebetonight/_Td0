# Python for Algorithmic Trading Application (py4at_app)

A comprehensive Python framework for algorithmic trading with vectorized backtesting, event-based backtesting, machine learning strategies, and live trading capabilities.

## Features

### 1. **Vectorized Backtesting**
- **SMA (Simple Moving Average) Strategy**: Dual-parameter moving average crossover strategy with parameter optimization
- **Momentum Strategy**: Trend-following momentum indicator-based trading
- **Mean Reversion Strategy**: Statistical reversal trading based on deviation from moving averages

### 2. **Machine Learning Backtesting**
- **Linear Regression Strategy**: Lag-based linear regression for price direction prediction
- **Scikit-learn Strategies**: Support for logistic regression and linear regression models with feature engineering

### 3. **Event-Based Backtesting**
- **Long-Only Strategies**: Traditional long-only position management
- **Long-Short Strategies**: Market-neutral and short-selling capabilities
- Implemented strategies:
  - SMA crossover with long/short positions
  - Momentum-based long/short trading
  - Mean reversion with bidirectional signals

### 4. **Live Trading (Online Algorithms)**
- Real-time tick data processing
- Momentum-based online trading
- Strategy monitoring and logging
- Trade execution tracking
- Performance metrics calculation

### 5. **Data Management**
- CSV data loading
- URL-based data retrieval
- Log returns calculation
- Technical indicator computation (SMA, momentum, etc.)

### 6. **Strategy Monitoring**
- Real-time trade logging
- Signal generation tracking
- Error logging and alerting
- Performance statistics
- Multi-format export (CSV, JSON, Excel)

## Installation

### Requirements
- Python 3.7+
- pip or conda

### Setup

```bash
# Clone or download the repository
cd py4at_app

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Dependencies
- **numpy**: Numerical computations
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning models
- **scipy**: Scientific computing (optimization)
- **matplotlib**: Data visualization

## Usage

### Quick Start

```python
from py4at_app.backtesting import SMAVectorBacktester
from py4at_app.data import DataLoader

# Load data
data = DataLoader.load_from_csv('data.csv', 'EUR', '2010-01-01', '2020-12-31')

# Create and run backtest
backtester = SMAVectorBacktester('EUR=', 42, 252, '2010-01-01', '2020-12-31', data)
abs_perf, over_perf = backtester.run_strategy()

print(f"Absolute Performance: {abs_perf}")
print(f"Out/Underperformance: {over_perf}")

# Optimize parameters
opt_params, opt_perf = backtester.optimize_parameters(
    (30, 56, 4),   # SMA1 range
    (200, 300, 4)  # SMA2 range
)
```

### Command Line Interface

#### SMA Vectorized Backtest
```bash
python main.py backtest-sma \
    --symbol EUR= \
    --sma1 42 \
    --sma2 252 \
    --start 2010-01-01 \
    --end 2020-12-31 \
    --plot
```

#### With Parameter Optimization
```bash
python main.py backtest-sma \
    --symbol EUR= \
    --sma1 42 \
    --sma2 252 \
    --optimize \
    --sma1-min 30 --sma1-max 56 --sma1-step 4 \
    --sma2-min 200 --sma2-max 300 --sma2-step 4
```

#### Momentum Strategy
```bash
python main.py backtest-momentum \
    --symbol XAU= \
    --momentum 2 \
    --amount 10000 \
    --tc 0.001
```

#### Mean Reversion Strategy
```bash
python main.py backtest-mean-reversion \
    --symbol GDX \
    --sma 25 \
    --threshold 5.0 \
    --amount 10000
```

#### Machine Learning Strategy
```bash
python main.py backtest-ml \
    --symbol .SPX \
    --ml-model logistic \
    --train-start 2010-01-01 --train-end 2015-12-31 \
    --test-start 2016-01-01 --test-end 2020-12-31 \
    --lags 5
```

#### Event-Based Backtest (Long-Short)
```bash
python main.py backtest-event \
    --symbol AAPL.O \
    --event-strategy sma \
    --sma1 42 \
    --sma2 252 \
    --long-short \
    --amount 10000 \
    --ftc 10.0 --tc 0.01
```

## Project Structure

```
py4at_app/
├── py4at_app/                          # Main package
│   ├── __init__.py
│   ├── backtesting/                    # Backtesting module
│   │   ├── __init__.py
│   │   ├── base.py                     # BacktestBase class
│   │   ├── strategies.py               # Vectorized strategies (SMA, Momentum, MR)
│   │   ├── scikit_strategies.py        # ML-based strategies
│   │   └── event_backtesting.py        # Event-based strategies (Long-Only, Long-Short)
│   ├── trading/                        # Live trading module
│   │   ├── __init__.py
│   │   ├── online.py                   # Online algorithms and real-time processing
│   │   ├── momentum.py                 # Momentum trading strategy
│   │   └── monitoring.py               # Strategy monitoring and logging
│   ├── data/                           # Data module
│   │   ├── __init__.py
│   │   └── loader.py                   # Data loading utilities
│   ├── utils/                          # Utility functions
│   │   └── __init__.py
│   └── config/                         # Configuration (expandable)
├── main.py                             # CLI entry point
├── requirements.txt                    # Package dependencies
└── README.md                           # This file
```

## Core Classes

### Backtesting

#### SMAVectorBacktester
Simple Moving Average crossover strategy with optimization.
```python
backtester = SMAVectorBacktester(symbol, SMA1, SMA2, start, end, data)
aperf, operf = backtester.run_strategy()
opt_params, opt_perf = backtester.optimize_parameters(SMA1_range, SMA2_range)
```

#### MomVectorBacktester
Momentum-based vectorized strategy.
```python
backtester = MomVectorBacktester(symbol, start, end, amount, tc, data)
aperf, operf = backtester.run_strategy(momentum=1)
```

#### MRVectorBacktester
Mean reversion strategy (inherits from MomVectorBacktester).
```python
backtester = MRVectorBacktester(symbol, start, end, amount, tc, data)
aperf, operf = backtester.run_strategy(SMA=50, threshold=5.0)
```

#### ScikitVectorBacktester
Machine learning strategies using scikit-learn models.
```python
backtester = ScikitVectorBacktester(symbol, start, end, amount, tc, 'logistic', data)
aperf, operf = backtester.run_strategy(train_start, train_end, test_start, test_end, lags=5)
```

#### BacktestLongOnly / BacktestLongShort
Event-based backtesting for long-only or long-short strategies.
```python
bt = BacktestLongOnly(symbol, start, end, amount, ftc, ptc, verbose)
bt.set_data(data)
bt.run_sma_strategy(SMA1, SMA2)
bt.run_momentum_strategy(momentum)
bt.run_mean_reversion_strategy(SMA, threshold)
```

### Trading

#### MomentumTrader
Live momentum trading with real-time data processing.
```python
trader = MomentumTrader(instrument, bar_length=60, momentum=6, units=100000)
signal = trader.on_tick(timestamp, bid, ask)
perf = trader.get_performance()
```

#### OnlineAlgorithm
Real-time online trading algorithm framework.
```python
algo = OnlineAlgorithm(instrument, window=5, momentum_period=3)
signal = algo.process_tick(timestamp, price)
stats = algo.get_statistics()
```

#### StrategyMonitor
Comprehensive strategy monitoring and logging.
```python
monitor = StrategyMonitor('MyStrategy', 'trading.log')
monitor.log_trade(timestamp, 'BUY', 100, 95.50, pnl=100.0)
monitor.log_signal(timestamp, 'BUY', 'MOMENTUM', 0.025)
stats = monitor.get_statistics()
monitor.export_log('results.csv', format='csv')
```

### Data

#### DataLoader
Utility class for loading and preparing financial data.
```python
# From CSV file
data = DataLoader.load_from_csv('data.csv', 'EUR', '2010-01-01', '2020-12-31')

# From URL
data = DataLoader.load_from_url('http://example.com/data.csv', 'EUR', '2010-01-01', '2020-12-31')

# Prepare returns
data = DataLoader.prepare_data(data)

# Add technical indicators
data = DataLoader.add_sma(data, 50)
data = DataLoader.add_momentum(data, window=1)
```

## Examples

### Example 1: Basic SMA Backtest

```python
from py4at_app.backtesting import SMAVectorBacktester
from py4at_app.data import DataLoader

# Load data
data = DataLoader.load_from_csv('EURUSD.csv', 'EUR', '2015-01-01', '2020-12-31')

# Create backtester
bt = SMAVectorBacktester('EUR=', SMA1=42, SMA2=252, 
                         start='2015-01-01', end='2020-12-31', data=data)

# Run strategy
absolute_perf, over_perf = bt.run_strategy()

# Optimize parameters
opt_params, opt_perf = bt.optimize_parameters(
    SMA1_range=(30, 56, 4),
    SMA2_range=(200, 300, 4)
)

print(f"Optimal Parameters: SMA1={opt_params[0]}, SMA2={opt_params[1]}")
print(f"Optimized Performance: {opt_perf:.2f}")

# Plot results
bt.plot_results()
```

### Example 2: Event-Based Long-Short Backtest

```python
from py4at_app.backtesting import BacktestLongShort
from py4at_app.data import DataLoader

# Load data
data = DataLoader.load_from_csv('AAPL.csv', 'AAPL', '2015-01-01', '2020-12-31')

# Create backtester
bt = BacktestLongShort('AAPL.O', '2015-01-01', '2020-12-31', 
                       amount=10000, ftc=10.0, ptc=0.01, verbose=True)

# Set data
bt.set_data(data)

# Run strategies
print("\n=== SMA Strategy ===")
bt.run_sma_strategy(SMA1=50, SMA2=200)

print("\n=== Momentum Strategy ===")
bt.run_momentum_strategy(momentum=30)

print("\n=== Mean Reversion Strategy ===")
bt.run_mean_reversion_strategy(SMA=50, threshold=5.0)
```

### Example 3: Machine Learning Strategy

```python
from py4at_app.backtesting import ScikitVectorBacktester
from py4at_app.data import DataLoader

# Load data
data = DataLoader.load_from_csv('SPX.csv', '.SPX', '2010-01-01', '2020-12-31')

# Create backtester
bt = ScikitVectorBacktester('.SPX', '2010-01-01', '2020-12-31',
                            amount=10000, tc=0.001, 
                            model='logistic', data=data)

# Run strategy
aperf, operf = bt.run_strategy(
    start_in='2010-01-01', end_in='2015-12-31',
    start_out='2016-01-01', end_out='2020-12-31',
    lags=5
)

print(f"Absolute Performance: {aperf:.2f}")
print(f"Out/Underperformance: {operf:.2f}")

# Plot results
bt.plot_results()
```

### Example 4: Live Momentum Trading

```python
from py4at_app.trading import MomentumTrader, StrategyMonitor

# Create trader and monitor
trader = MomentumTrader('EUR_USD', bar_length=60, momentum=6, units=100000)
monitor = StrategyMonitor('MomentumTrader', 'momentum.log')

# Simulate tick data
import pandas as pd
timestamps = pd.date_range('2023-01-01', periods=1000, freq='1s')
prices_bid = [1.0550 + i*0.0001 for i in range(1000)]
prices_ask = [1.0551 + i*0.0001 for i in range(1000)]

for ts, bid, ask in zip(timestamps, prices_bid, prices_ask):
    signal = trader.on_tick(ts, bid, ask)
    
    if signal is not None and signal != 0:
        monitor.log_signal(ts, 'BUY' if signal > 0 else 'SELL', 
                          'MOMENTUM', trader.calculate_momentum())

# Get performance
performance = trader.get_performance()
monitor.print_summary()
monitor.export_log('trades.json', format='json')
```

## Performance Considerations

1. **Vectorized Backtesting**: Much faster than event-based, suitable for parameter optimization
2. **Event-Based Backtesting**: More realistic, includes transaction costs and slippage simulation
3. **Machine Learning**: Requires training/testing split, uses lag-based features
4. **Live Trading**: Designed for real-time processing with minimal latency

## Output & Metrics

### Backtest Results
- **Absolute Performance**: Strategy cumulative return
- **Over/Underperformance**: Difference vs. buy-and-hold return
- **Number of Trades**: Total trades executed
- **Transaction Costs**: Impact of fees and slippage

### Live Trading Metrics
- **Win Rate**: Percentage of profitable trades
- **Average Trade PnL**: Mean profit/loss per trade
- **Largest Win/Loss**: Maximum positive/negative trade
- **Sharpe Ratio**: Risk-adjusted return (with utils module)
- **Maximum Drawdown**: Largest peak-to-trough decline

## Configuration

Create a `config.py` file for custom settings:

```python
# py4at_app/config/config.py

# Data settings
DATA_SOURCE = 'http://hilpisch.com/pyalgo_eikon_eod_data.csv'

# Trading parameters
DEFAULT_AMOUNT = 10000
DEFAULT_TC = 0.001
DEFAULT_FTC = 0.0

# Strategy parameters
SMA_DEFAULT_WINDOW1 = 42
SMA_DEFAULT_WINDOW2 = 252
MOMENTUM_DEFAULT_PERIOD = 6
MR_DEFAULT_SMA = 50
MR_DEFAULT_THRESHOLD = 5.0

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'trading.log'
```

## Contributing

Contributions are welcome! Please:
1. Follow PEP 8 style guidelines
2. Add docstrings to all functions and classes
3. Write tests for new functionality
4. Update documentation as needed

## License

This project is inspired by "Python for Algorithmic Trading" by Dr. Yves Hilpisch.

## References

- **Original Course**: Python for Algorithmic Trading (py4at)
- **Author**: Dr. Yves J. Hilpisch
- **Publisher**: The Python Quants GmbH
- **Chapters Implemented**: 3-10 (Data, Strategies, Backtesting, Online Trading, Monitoring)

## Support

For issues or questions:
1. Check the examples directory
2. Review the docstrings in source files
3. Consult the original py4at course materials

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Production Ready
