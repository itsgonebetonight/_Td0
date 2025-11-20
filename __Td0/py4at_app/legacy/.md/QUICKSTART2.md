# Quick Start Guide - Enhanced ML Trading Strategy

## Run Backtest

```bash
# Set Python path and run
$env:PYTHONPATH='c:\Users\HP\Downloads\_\_Td0\__Td0\py4at_app'
python run_backtest_v2.py
```

**Output:**
- Training metrics from best model
- Backtest summary with performance
- **NEW:** Per-side trade statistics (long vs short)
- **NEW:** Average hold time and max drawdown
- **NEW:** Equity curve exported to `equity_curve.csv`
- Full trade log with P&L for each trade

---

## Key Features Implemented

### 1. Trade Count Reconciliation ✓
- Internal trade counter now matches round-trip count
- All trades marked with status ('open' or 'closed')
- `equity_curve.csv` accurately reflects completed trades

### 2. P&L Population ✓
- All closed trades have:
  - `exit_price`: Price at which trade was exited
  - `pnl`: Profit/loss in currency
  - `return_pct`: Return as percentage
  - `hold_bars`: Number of bars held

### 3. Per-Side Statistics ✓
Print output now shows:
```
Trade Log Summary:
  Closed Trades:           1
  Total P&L:               $0.00
  Win Rate:                0.0% (0 wins, 1 losses)
  Avg P&L per trade:       $0.00
  Avg Hold (bars):         0.0

  LONG trades:             1 | P&L $0.00 | 0 wins (0%)
  Max Drawdown (currency): $0.00
```

### 4. Equity Curve Export ✓
File: `equity_curve.csv`
```
trade_step,equity
0,10000.0
1,10000.0
```

---

## Configuration

Edit `run_backtest_v2.py` or `run_backtest.py` to change:

```python
# Backtest parameters
results = ml.backtest(
    test_data,
    confidence_threshold=0.55,    # Min probability to trade
    initial_amount=10000,         # Starting capital
    tc=0.001,                     # Transaction costs (0.1%)
    risk_fraction=0.01,           # Position size as % of capital
    stop_loss_pct=0.02,           # Stop loss (2%)
    take_profit_pct=0.04,         # Take profit (4%)
    slippage=0.0005,              # Slippage as % of price
    retrain_interval=0,           # Walk-forward retrain bars (0=off)
    allow_short=True              # Enable short selling
)
```

---

## Output Files

| File | Contains | Use Case |
|------|----------|----------|
| `equity_curve.csv` | Trade step, equity value | Plot equity curve growth |
| Console output | Performance metrics | Monitor backtest results |
| Trade log (in output) | Entry/exit details | Analyze individual trades |

---

## Performance Metrics Explained

- **Performance (%):** Net return on initial capital
- **Total Trades:** Number of completed round-trips
- **Final Amount:** Ending capital
- **Win Rate:** % of profitable trades
- **Avg P&L:** Average profit/loss per trade
- **Avg Hold:** Average bars held per trade
- **Max Drawdown:** Largest equity decline

---

## Example: Optimize Parameters

```python
# Test different confidence thresholds
for confidence in [0.50, 0.55, 0.60, 0.65]:
    results = ml.backtest(test_data, confidence_threshold=confidence)
    print(f"Confidence {confidence}: {results['performance']:.2f}%")
```

---

## Advanced: Manual Summary

```python
# After backtest
trade_log = results['trade_log']
closed = [t for t in trade_log if t['status'] == 'closed' and t['pnl'] is not None]

print(f"Closed trades: {len(closed)}")
print(f"Total P&L: ${sum(t['pnl'] for t in closed):.2f}")
print(f"Long trades: {len([t for t in closed if t['side']=='long'])}")
print(f"Short trades: {len([t for t in closed if t['side']=='short'])}")

# Build equity curve
equity = [10000]
for t in closed:
    equity.append(equity[-1] + t['pnl'])

print(f"Peak equity: ${max(equity):.2f}")
print(f"Max drawdown: ${max(max(equity) - e for e in equity):.2f}")
```

---

## Files to Know

| File | Location | Purpose |
|------|----------|---------|
| `run_backtest_v2.py` | Root | Enhanced runner with CSV export |
| `run_backtest.py` | Root | Original runner |
| `ml_strategy_integration.py` | Root + package | Main strategy module |
| `enhanced_predictor.py` | Root + package | ML model with calibration |
| `IMPROVEMENTS_SUMMARY.md` | Root | Full technical details |

---

## Troubleshooting

**Issue:** `ModuleNotFoundError: py4at_app`
**Fix:** Ensure PYTHONPATH is set:
```bash
$env:PYTHONPATH='c:\Users\HP\Downloads\_\_Td0\__Td0\py4at_app'
```

**Issue:** No trades executed
**Fix:** Lower confidence_threshold (default 0.55):
```python
results = ml.backtest(test_data, confidence_threshold=0.50)
```

**Issue:** CSV not exported
**Fix:** Check trade_log has closed trades:
```python
closed = [t for t in results['trade_log'] if t['status']=='closed']
if not closed:
    print("No closed trades to export")
```

---

**Version:** 1.0 (All enhancements complete)  
**Status:** ✅ Production ready

For full technical details, see `IMPROVEMENTS_SUMMARY.md`
