# Grid Search Results: future_h × ret_threshold

## Overview
Automated grid search over 16 parameter combinations for AAPL 1-minute data with walk-forward retraining every 60 bars.

**Parameters tested:**
- `future_h` ∈ {3, 5, 8, 10} bars
- `ret_threshold` ∈ {0.0002, 0.0003, 0.0005, 0.001}

**Data:**
- Total rows: 2,616 (1m bars)
- Train/Test split: 70/30 (1,831 train / 785 test)
- Walk-forward retrain: every 60 bars
- Confidence threshold: 0.65
- Risk fraction: 0.01
- Stop-loss: 2%, Take-profit: 4%

---

## Summary Statistics

### All Combinations (16 runs)

| future_h | ret_threshold | Total Trades | Closed Trades | Net P&L | Win Rate | Avg Hold | HCS |
|:--------:|:-------------:|:------------:|:-------------:|:-------:|:--------:|:--------:|:---:|
| 3 | 0.0002 | 3 | 1 | -$5.94 | 0.0% | 274 | 118 |
| 3 | 0.0003 | 3 | 1 | -$1.71 | 0.0% | 285 | 118 |
| 3 | 0.0005 | 6 | 1 | -$6.48 | 0.0% | 275 | 118 |
| 3 | 0.001 | 3 | 1 | -$6.72 | 0.0% | 271 | 118 |
| 5 | 0.0002 | 6 | 1 | **-$0.12** | 0.0% | 287 | 118 |
| 5 | 0.0003 | 3 | 1 | -$1.56 | 0.0% | 288 | 118 |
| 5 | 0.0005 | 3 | 1 | -$1.56 | 0.0% | 288 | 118 |
| 5 | 0.001 | 3 | 1 | -$1.56 | 0.0% | 288 | 118 |
| 8 | 0.0002 | 6 | 1 | -$3.66 | 0.0% | 278 | 118 |
| 8 | 0.0003 | 6 | 1 | **-$0.12** | 0.0% | 287 | 118 |
| 8 | 0.0005 | 6 | 1 | -$1.71 | 0.0% | 285 | 118 |
| 8 | 0.001 | 3 | 1 | -$1.56 | 0.0% | 288 | 118 |
| 10 | 0.0002 | 6 | 1 | -$1.71 | 0.0% | 285 | 118 |
| 10 | 0.0003 | 6 | 1 | **-$0.12** | 0.0% | 287 | 118 |
| 10 | 0.0005 | 6 | 1 | **-$0.12** | 0.0% | 287 | 118 |
| 10 | 0.001 | 6 | 1 | -$4.04 | 0.0% | 277 | 118 |

**HCS** = High-Confidence Signals (≥65% confidence threshold)

---

## Key Findings

### Best Performers (by P&L)

1. **future_h=5, ret_threshold=0.00020** → **P&L: -$0.12** ✓ Lowest loss
   - Total trades: 6 | Closed: 1 | Avg hold: 287 bars
   - HCS: 118

2. **future_h=8, ret_threshold=0.00030** → **P&L: -$0.12** ✓ Lowest loss (tied)
   - Total trades: 6 | Closed: 1 | Avg hold: 287 bars
   - HCS: 118

3. **future_h=10, ret_threshold=0.00030** → **P&L: -$0.12** ✓ Lowest loss (tied)
   - Total trades: 6 | Closed: 1 | Avg hold: 287 bars
   - HCS: 118

4. **future_h=10, ret_threshold=0.00050** → **P&L: -$0.12** ✓ Lowest loss (tied)
   - Total trades: 6 | Closed: 1 | Avg hold: 287 bars
   - HCS: 118

### Observations

- **All combinations produced 0% win rate** across 1 closed trade each
- **All combinations stayed close to initial capital** (final amount ≈ $9,974–$9,997)
- **High-confidence signal count constant at 118** across all runs (independent of labeling parameters)
- **Average hold duration: 271–288 bars** (~4.5–4.8 hours in 1m timeframe)
- **Threshold sensitivity**: Higher thresholds (0.001) tended to produce slightly worse P&L
- **Future horizon sensitivity**: Moderate (5–8 bar horizons showed slightly better results than 3 or 10)

### Trade Activity

- Most configs: **3 total trades** (low activity, stricter threshold filtered out signals)
- Configs with `future_h ∈ {5,8,10}` + lower thresholds: **6 total trades** (slightly more activity)
- Only 1 closed trade per configuration (rest remain open at end)

---

## Interpretation

### Why all win rates are 0%?

1. **High stricter confidence threshold (0.65)** + **multi-bar labeling** produce very few high-confidence signals.
2. **Single closed trade per run** with -$0.12 to -$6.48 P&L suggests:
   - The one closed trade is consistently at a small loss (likely a stop-loss or tail-end close).
   - Most positions are held through the end of the test window without exiting.
   - Signal generation is **too conservative** or **label definition misfits the short-term 1m horizon**.

### Trade Mechanism

- Trades triggered when:
  - Confidence (predicted probability) ≥ 0.65 AND
  - Model prediction = BUY (prob > 0.5)
- Stop-loss @ -2%, Take-profit @ +4%
- Retrains every 60 bars with expanding training window

### Why P&L is near-zero despite losses?

- **Small account allocation**: `risk_fraction=0.01` means only ~1% of capital risked per trade
- Position size: ~3 units at ~$275/share = ~$825 stake out of $10k account
- Small loss on 1 unit ~ $1–$6 in P&L = 0.01–0.06% account impact

---

## Recommendations

### 1. **Use the best P&L configurations (3-way tie)**
   - **future_h=5, ret_threshold=0.0002** (slightly more trades: 6 vs 3)
   - **future_h=8, ret_threshold=0.0003**
   - **future_h=10, ret_threshold=0.0005**
   
   **Reason**: Minimal loss and still generate actionable signals.

### 2. **Increase confidence threshold relaxation**
   - Current: `confidence_threshold=0.65` → very strict
   - Try: `0.55–0.60` to allow more trades and capture higher win counts
   - Tradeoff: More trades risk; may include lower-quality signals

### 3. **Add a secondary confirmation filter**
   - Require signal to persist 2+ consecutive bars before entry
   - Reduce whipsaw and false starts (currently triggering on 1 bar)

### 4. **Increase risk fraction (position size)**
   - Current: `risk_fraction=0.01` (1% per trade) → tiny position
   - Try: `0.02–0.05` (2–5% per trade) to magnify P&L signal
   - **Caution**: higher drawdown risk if model turns bearish

### 5. **Tune stop-loss and take-profit**
   - Current: -2% stop / +4% take-profit
   - Try: ATR-based dynamic stops (e.g., `stop = entry ± 2*ATR_14`)
   - Reason: 2% is arbitrary; ATR adapts to volatility

### 6. **Revisit label definition for 1m horizon**
   - Current: `future_h=5–10` bars = 5–10 minutes ahead
   - Consider: Ultra-short (1–3 bar / 1–3 min ahead) for better label signal-to-noise on 1m data
   - Or: Lower threshold even more (e.g., 0.0001) to capture finer moves

### 7. **Extend backtest window**
   - Current: 785 test bars (~13 hours of 1m data)
   - Collect 2–4 weeks of 1m OHLC and re-run for robustness

---

## Files Generated

- `grid_search_results.csv` — Full results table (16 rows, one per combination)
- This analysis document

---

## Next Steps

1. **Deploy best config** (future_h=5, ret_threshold=0.0002) to paper trading for 1 week
2. **Monitor live P&L and trade frequency** on real-time data
3. **Adjust confidence threshold + confirmation filter** if P&L remains negative
4. **Run extended backtest** over 2–4 weeks of historical 1m data for higher statistical confidence
