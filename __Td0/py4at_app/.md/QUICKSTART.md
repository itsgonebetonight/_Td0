# Quick Start Guide - py4at_app

## Installation

```bash
# Navigate to project directory
cd py4at_app

# Install dependencies
pip install -r requirements.txt
```

## 5-Minute Quick Start

### 1. Prepare Sample Data

Create a CSV file `sample_data.csv` with columns: Date, Price
```csv
Date,Price
2020-01-01,1.1000
2020-01-02,1.1010
2020-01-03,1.0990
...
```

### 2. Run Your First Backtest

```bash
# SMA Strategy
python main.py backtest-sma --symbol EUR --data-file sample_data.csv --plot

# Momentum Strategy
python main.py backtest-momentum --symbol EUR --data-file sample_data.csv --amount 10000

# Mean Reversion Strategy
python main.py backtest-mean-reversion --symbol EUR --data-file sample_data.csv
```

### 3. Python Script Usage

```python
from py4at_app.backtesting import SMAVectorBacktester
from py4at_app.data import DataLoader

# Load data
data = DataLoader.load_from_csv('sample_data.csv', 'Price', '2020-01-01', '2020-12-31')

# Create strategy
strategy = SMAVectorBacktester('EUR', SMA1=42, SMA2=252, 
                               start='2020-01-01', end='2020-12-31', 
                               data=data)

# Run backtest
absolute_performance, over_performance = strategy.run_strategy()

# Print results
print(f"Performance: {absolute_performance:.2f}")
print(f"Over/Under: {over_performance:.2f}")
```

## Available Strategies

| Strategy | Command | Python Class | Use Case |
|----------|---------|--------------|----------|
| SMA | `backtest-sma` | `SMAVectorBacktester` | Trend following |
| Momentum | `backtest-momentum` | `MomVectorBacktester` | Price momentum |
| Mean Reversion | `backtest-mean-reversion` | `MRVectorBacktester` | Reversion to mean |
| ML (Scikit) | `backtest-ml` | `ScikitVectorBacktester` | Prediction-based |
| Event-Based | `backtest-event` | `BacktestLongOnly/LongShort` | Realistic simulation |
| Live Trading | (API) | `MomentumTrader` | Real-time execution |

## Common Tasks

### Backtest with Custom Parameters

```bash
python main.py backtest-sma \
    --symbol EUR \
    --sma1 50 \
    --sma2 200 \
    --start 2015-01-01 \
    --end 2020-12-31 \
    --amount 50000 \
    --tc 0.001 \
    --plot
```

### Optimize Strategy Parameters

```bash
python main.py backtest-sma \
    --symbol EUR \
    --optimize \
    --sma1-min 20 --sma1-max 60 --sma1-step 5 \
    --sma2-min 100 --sma2-max 300 --sma2-step 10
```

### Run Long-Short Strategy

```bash
python main.py backtest-event \
    --symbol AAPL \
    --event-strategy sma \
    --sma1 50 --sma2 200 \
    --long-short \
    --ftc 10.0 --tc 0.01 \
    --verbose
```

### Machine Learning Strategy

```bash
python main.py backtest-ml \
    --symbol .SPX \
    --ml-model logistic \
    --train-start 2010-01-01 --train-end 2015-12-31 \
    --test-start 2016-01-01 --test-end 2020-12-31 \
    --lags 5
```

## Python API Examples

### Example 1: Compare Multiple Strategies

```python
from py4at_app.backtesting import (
    SMAVectorBacktester, MomVectorBacktester, MRVectorBacktester
)
from py4at_app.data import DataLoader

# Load data
data = DataLoader.load_from_csv('eurusd.csv', 'EUR', '2015-01-01', '2020-12-31')

# Test multiple strategies
strategies = [
    ('SMA', SMAVectorBacktester('EUR', 42, 252, '2015-01-01', '2020-12-31', data)),
    ('Momentum', MomVectorBacktester('EUR', '2015-01-01', '2020-12-31', 10000, 0.001, data)),
    ('MeanRev', MRVectorBacktester('EUR', '2015-01-01', '2020-12-31', 10000, 0.001, data))
]

results = {}
for name, strategy in strategies:
    aperf, operf = strategy.run_strategy()
    results[name] = {'abs': aperf, 'over': operf}
    print(f"{name}: {aperf:.2f} ({operf:+.2f})")
```

### Example 2: Event-Based with Transaction Costs

```python
from py4at_app.backtesting import BacktestLongOnly
from py4at_app.data import DataLoader

# Load data
data = DataLoader.load_from_csv('aapl.csv', 'AAPL', '2018-01-01', '2020-12-31')

# Create backtester with costs
bt = BacktestLongOnly(
    symbol='AAPL',
    start='2018-01-01',
    end='2020-12-31',
    amount=10000,
    ftc=10.0,        # $10 per trade
    ptc=0.01,        # 1% proportional cost
    verbose=True
)

# Set data
bt.set_data(data)

# Run strategies
bt.run_sma_strategy(SMA1=50, SMA2=200)
```

### Example 3: Live Trading Monitoring

```python
from py4at_app.trading import MomentumTrader, StrategyMonitor
import pandas as pd

# Setup
trader = MomentumTrader('EUR_USD', bar_length=60, momentum=6, units=100000)
monitor = StrategyMonitor('MyMomentumTrader', 'trades.log')

# Simulate data
import numpy as np
n_ticks = 500
timestamps = pd.date_range('2023-01-01', periods=n_ticks, freq='1s')
prices = 1.0550 + np.cumsum(np.random.randn(n_ticks) * 0.0001)

# Process ticks
for ts, price in zip(timestamps, prices):
    bid = price - 0.0001
    ask = price + 0.0001
    
    order_units = trader.on_tick(ts, bid, ask)
    
    if order_units != 0:
        side = 'BUY' if order_units > 0 else 'SELL'
        monitor.log_trade(ts, side, abs(order_units), price)

# Print summary
monitor.print_summary()
monitor.export_log('results.json', format='json')
```

## File Formats

### CSV Data Format
```csv
Date,Symbol
2020-01-01,1.1000
2020-01-02,1.1010
2020-01-03,1.0990
```

### CLI Output Format
```
=== SMA Vectorized Backtest ===
Symbol: EUR
SMA1: 42, SMA2: 252
Period: 2010-01-01 to 2020-12-31
============================================================

Absolute Performance: 1.45
Out-/Underperformance: 0.25
```

## Troubleshooting

### Import Error
```
ImportError: No module named 'py4at_app'
```
**Solution**: Make sure you're running from the correct directory and have installed the package:
```bash
pip install -e .
```

### Data Not Found
```
FileNotFoundError: 'sample_data.csv' not found
```
**Solution**: Provide correct path or use URL:
```bash
python main.py backtest-sma --data-file ./data/sample.csv
```

### No Module matplotlib
```
ModuleNotFoundError: No module named 'matplotlib'
```
**Solution**: Install it with:
```bash
pip install matplotlib
```

## Performance Tips

1. **Vectorized vs Event-Based**: Use vectorized for quick backtests, event-based for realistic simulations
2. **Large Datasets**: Use date range filters (--start, --end) to reduce data
3. **Parameter Optimization**: Start with coarse step sizes, then refine
4. **ML Models**: Use appropriate number of lags (5-15 typically good)

## Next Steps

1. ✅ Install and run a simple SMA backtest
2. ✅ Compare multiple strategies
3. ✅ Optimize parameters for best performance
4. ✅ Run event-based backtest with realistic costs
5. ✅ Try machine learning strategies
6. ✅ Explore live trading with monitoring

## Help & Support

```bash
# Show all commands
python main.py --help

# Show command-specific help
python main.py backtest-sma --help
python main.py backtest-momentum --help
python main.py backtest-event --help
```

## Resources

- **Documentation**: See `README.md` for comprehensive guide
- **Examples**: Check docstrings in source code
- **Original Course**: py4at by Dr. Yves Hilpisch
- **Source Code**: See `py4at_app/` directory structure

---

Enjoy algorithmic trading with py4at_app! 🚀
