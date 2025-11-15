# 🚀 QUICK START - RUN YOUR FIRST BACKTEST

## Installation (First Time Only)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning
- **scipy** - Scientific computing
- **matplotlib** - Plotting (optional)

### 2. Verify Installation
```bash
python -c "import numpy, pandas, sklearn; print('✓ All dependencies installed!')"
```

---

## Running the Examples

### Example 1: Basic SMA Backtest (EASIEST - START HERE)
```bash
python example_basic.py
```

**What it does:**
- ✅ Loads sample historical data
- ✅ Runs SMA crossover strategy
- ✅ Runs momentum strategy
- ✅ Shows performance metrics
- ✅ Demonstrates data preparation

**Expected Output:**
```
======================================================================
EXAMPLE 1: SMA VECTOR BACKTEST
======================================================================

📊 Step 1: Loading sample data...
✓ Loaded 100 rows of data
  Date range: 2024-01-02 to 2024-05-10
  Price range: $100.50 - $151.45

🔧 Step 2: Setting up SMA Backtest strategy...
✓ Backtester created with:
  - Initial Capital: $100,000
  - Transaction Cost: 0.2%
  - Symbol: EUR/USD

⚙️  Step 3: Running backtest with SMA parameters...

✓ Backtest completed!

📈 BACKTEST RESULTS
----------------------------------------------------------------------
Strategy Return:           45.50 %
Buy & Hold Return:         50.60 %
Number of Trades:              8
Sharpe Ratio:               1.23
Max Drawdown:              -5.30 %
Win Rate:                  62.50 %
```

---

### Example 2: Advanced Trading Simulation
```bash
python example_trading.py
```

**What it does:**
- ✅ Simulates real-time tick data processing
- ✅ Demonstrates momentum trader
- ✅ Shows strategy monitoring system
- ✅ Logs trades and signals

**Expected Output:**
```
======================================================================
EXAMPLE: MOMENTUM TRADER WITH MONITORING
======================================================================

📊 Loading historical data...
✓ Loaded 100 bars of data

🔧 Initializing Momentum Trader...
✓ Trader initialized

⚙️  Simulating tick processing (using bar data)...

✓ Processing complete
  - Bars processed: 90
  - Signals generated: 15
  - Trades executed: 8
```

---

## Using the CLI Interface

### View Available Commands
```bash
python main.py --help
```

### Run SMA Strategy
```bash
python main.py backtest-sma --symbol EUR --output sma_results.csv
```

### Run Momentum Strategy
```bash
python main.py backtest-momentum --symbol GBP --output momentum_results.csv
```

### Run Machine Learning Backtest
```bash
python main.py backtest-ml --symbol USD --output ml_results.csv
```

---

## Python API Usage

### Quick Example: Run a Backtest Directly

Create a new file `my_test.py`:

```python
from py4at_app.backtesting import SMAVectorBacktester
from py4at_app.data import DataLoader
import pandas as pd

# Load data
data = pd.read_csv('sample_data.csv', index_col=0, parse_dates=True)

# Create backtester
bt = SMAVectorBacktester('EUR/USD', data, 100000, 0.002)

# Run backtest
results = bt.backtest(SMA1=10, SMA2=20)

# Get metrics
returns = results['returns'].dropna()
total_return = (results['Close'][-1] / results['Close'][0] - 1) * 100

print(f"Total Return: {total_return:.2f}%")
print(f"Final Price: ${results['Close'][-1]:.2f}")
```

Run it:
```bash
python my_test.py
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'py4at_app'"

**Solution:** Make sure you're running scripts from the correct directory:
```bash
# Navigate to py4at_app directory
cd c:\Users\HP\Downloads\__Td0\__Td0\py4at_app

# Then run
python example_basic.py
```

### Issue: "FileNotFoundError: sample_data.csv"

**Solution:** Make sure `sample_data.csv` is in the same directory as the scripts:
```bash
# Check if file exists
dir sample_data.csv

# If not found, make sure you're in the right directory:
cd c:\Users\HP\Downloads\__Td0\__Td0\py4at_app
```

### Issue: "No module named 'pandas'" (or numpy, sklearn)

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
c:\Users\HP\Downloads\__Td0\__Td0\py4at_app\
│
├── sample_data.csv              ← Sample historical data
├── example_basic.py             ← START HERE! Basic examples
├── example_trading.py           ← Advanced trading examples
├── main.py                      ← CLI interface
├── requirements.txt             ← Dependencies
│
└── py4at_app/                   ← Main package
    ├── backtesting/             ← Backtesting strategies
    ├── trading/                 ← Live trading simulation
    ├── data/                    ← Data loading utilities
    └── utils/                   ← Helper functions
```

---

## What Each Module Does

### 🔍 Backtesting Module
Vectorized backtesting for various strategies:
- **SMAVectorBacktester** - Simple Moving Average crossover
- **MomVectorBacktester** - Momentum-based strategy
- **MRVectorBacktester** - Mean reversion
- **LRVectorBacktester** - Linear regression
- **ScikitVectorBacktester** - Machine learning models
- **BacktestLongOnly** - Event-based long only
- **BacktestLongShort** - Event-based long/short

### 📊 Trading Module
Real-time trading simulation:
- **OnlineAlgorithm** - Streaming algorithm base
- **TickDataProcessor** - Tick data aggregation
- **MomentumTrader** - Live momentum strategy
- **StrategyMonitor** - Trade logging and monitoring

### 📈 Data Module
Data loading and preparation:
- **DataLoader** - Load CSV, add indicators, prepare data

### 🛠️ Utils Module
Helper functions:
- Performance metrics (Sharpe ratio, drawdown, returns)
- Formatting utilities
- Data validation

---

## Common Tasks

### Task 1: Backtest a Strategy
```python
from py4at_app.backtesting import SMAVectorBacktester
import pandas as pd

data = pd.read_csv('sample_data.csv', index_col=0, parse_dates=True)
bt = SMAVectorBacktester('EUR/USD', data, 100000, 0.002)
results = bt.backtest(SMA1=10, SMA2=20)
print(results.tail())
```

### Task 2: Load Data with Indicators
```python
from py4at_app.data import DataLoader
import pandas as pd

data = DataLoader.load_from_csv('sample_data.csv')
data = DataLoader.prepare_data(data)
data = DataLoader.add_sma(data, [10, 20])
data = DataLoader.add_momentum(data, [10])
print(data.head())
```

### Task 3: Monitor Trading Activity
```python
from py4at_app.trading import StrategyMonitor
from datetime import datetime

monitor = StrategyMonitor('My Strategy')
monitor.log_trade(
    timestamp=datetime.now(),
    instrument='EUR/USD',
    action='BUY',
    quantity=1,
    price=100.50,
    commission=0.01
)
monitor.print_summary()
```

---

## Next Steps

1. **✅ Run Example 1** → `python example_basic.py`
2. **✅ Run Example 2** → `python example_trading.py`
3. **✅ Modify Examples** → Edit files to test different parameters
4. **✅ Create Your Own** → Use the examples as templates
5. **✅ Read Full Docs** → Check README.md for comprehensive guide

---

## File Locations

**Examples:**
- Basic: `c:\Users\HP\Downloads\__Td0\__Td0\py4at_app\example_basic.py`
- Trading: `c:\Users\HP\Downloads\__Td0\__Td0\py4at_app\example_trading.py`

**Sample Data:**
- CSV: `c:\Users\HP\Downloads\__Td0\__Td0\py4at_app\sample_data.csv`

**Documentation:**
- Index: `c:\Users\HP\Downloads\__Td0\__Td0\py4at_app\INDEX.md`
- Full Guide: `c:\Users\HP\Downloads\__Td0\__Td0\py4at_app\README.md`
- Architecture: `c:\Users\HP\Downloads\__Td0\__Td0\py4at_app\PACKAGE_STRUCTURE.md`

---

**Now run: `python example_basic.py`** 🚀
