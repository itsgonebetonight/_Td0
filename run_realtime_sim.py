import os
import sys
import time
import traceback

# Ensure package path imports the integrated package
pkg_path = r"c:\Users\HP\Downloads\_\_Td0\__Td0\py4at_app"
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

import pandas as pd

# Try to import yfinance; user allowed installation if missing
try:
    import yfinance as yf
except Exception:
    yf = None

from py4at_app.ml_strategy_integration import MLTradingStrategy

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'realtime_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYMBOLS = ['AAPL', 'MSFT', 'EURUSD=X']  # random examples
INTERVALS = ['1m', '5m', '1d']

# Map interval to default period to download
PERIOD_FOR_INTERVAL = {
    '1m': '7d',   # yfinance provides 1m for last 7 days
    '5m': '60d',
    '1d': '1y'
}


def fetch_data(sym: str, interval: str):
    if yf is None:
        raise RuntimeError('yfinance not installed')
    period = PERIOD_FOR_INTERVAL.get(interval, '60d')
    try:
        t = yf.Ticker(sym)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        if df.empty:
            return None
        # Keep close price and drop NA
        if 'Close' in df.columns:
            df = df.rename(columns={'Close': 'price'})
        elif 'close' in df.columns:
            df = df.rename(columns={'close': 'price'})
        else:
            # try first numeric column
            df['price'] = df.iloc[:, 0]
        df = df[['price']].dropna()
        return df
    except Exception:
        traceback.print_exc()
        return None


def run_for_symbol_interval(symbol, interval, quick_demo=True):
    print(f"\n=== SYMBOL: {symbol} | INTERVAL: {interval} ===")
    df = fetch_data(symbol, interval)
    if df is None or len(df) < 30:
        print(f"Not enough data for {symbol} {interval} (rows={len(df) if df is not None else 0}), skipping")
        return None

    # Train/test split: 70/30
    split = int(len(df) * 0.7)
    train = df.iloc[:split]
    test = df.iloc[split:]
    print(f"Rows total: {len(df)} | Train: {len(train)} | Test: {len(test)}")

    ml = MLTradingStrategy(symbol=symbol)
    # engineer features and use a 5-bar future horizon with small threshold to reduce noise
    train_with_features = ml.prepare_data(train, future_h=5, ret_threshold=0.0003)
    print('Training features shape:', train_with_features.shape)
    try:
        train_results = ml.train(train_with_features, test_size=0.2)
        print('Train results best model:', train_results.get('best_model'))
    except Exception as e:
        print('Training failed:', e)
        traceback.print_exc()
        return None

    # Backtest on test (simulate recent live run)
    try:
        # Use stricter confidence threshold and periodic retrain (simulate real-time retraining every 60 bars)
        results = ml.backtest(test, confidence_threshold=0.65, retrain_interval=60, allow_short=True)
        ml.print_results()

        # save results and trade_log
        out_prefix = f"{symbol.replace('/','_').replace('=','')}_{interval}"
        out_json = os.path.join(OUTPUT_DIR, f"{out_prefix}_results.json")
        out_csv = os.path.join(OUTPUT_DIR, f"{out_prefix}_trade_log.csv")

        # write trade log to CSV if present
        trade_log = results.get('trade_log', [])
        if trade_log:
            try:
                pd.DataFrame(trade_log).to_csv(out_csv, index=False)
                print(f"Saved trade log to {out_csv}")
            except Exception as e:
                print('Failed saving trade log:', e)
        # Save summary
        try:
            import json
            with open(out_json, 'w') as f:
                json.dump({'results': results}, f, default=str, indent=2)
            print(f"Saved summary to {out_json}")
        except Exception as e:
            print('Failed saving summary:', e)

        # Optionally export equity curve via backtester.summarize_trade_log
        try:
            backtester = ml  # ml has backtest_results but not backtester instance; re-run backtest to construct backtester? skip
            # we already saved trade_log csv
        except Exception:
            pass

        # Quick demo: if quick_demo, run only a small subset (we already did)
        return results
    except Exception as e:
        print('Backtest failed:', e)
        traceback.print_exc()
        return None


def main():
    if yf is None:
        print('yfinance not available. Abort. Please install yfinance first.')
        return

    # Run limited demo across symbols and intervals
    runs = []
    # For a longer live-sim, run a single symbol & 1m interval step-through
    demo_symbol = SYMBOLS[0]
    demo_interval = '1m'
    res = run_for_symbol_interval(demo_symbol, demo_interval, quick_demo=False)
    runs.append((demo_symbol, demo_interval, res))
    # sleep to avoid throttling
    time.sleep(2)
    print('\nDemo complete. Outputs in', OUTPUT_DIR)


if __name__ == '__main__':
    main()
