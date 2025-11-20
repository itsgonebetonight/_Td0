"""
Run extended backtest on a provided CSV file.
Usage:
  python run_extended_backtest.py --csv path/to/AAPL_multi_month.csv

The script will:
 - Load CSV (expects a date index and Close/CLOSE column)
 - Compute features using EnhancedFeatureEngineering
 - Train ML model (MLTradingStrategy.train)
 - Backtest with the recommended production config and label (future_h=5, ret_threshold=0.0002)
 - Save results to `extended_backtest_results.csv`

Note: For minute-level data you'll need a file covering the period you want (3+ months).
"""
import argparse
from pathlib import Path
import pandas as pd
import sys

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_predictor import EnhancedFeatureEngineering
from ml_strategy_integration import MLTradingStrategy


def load_price_csv(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    # Try to read normally first
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    # If the first data rows contain non-numeric 'price' values (common with yfinance CSVs
    # that include extra header rows), try re-reading while skipping the extraneous rows.
    def _is_price_numeric(series):
        try:
            return pd.to_numeric(series.dropna()).notna().all()
        except Exception:
            return False

    # Find candidate close columns
    cols = [c for c in df.columns]
    # If the first column contains non-numeric values, attempt to skip the second row (ticker row)
    if not _is_price_numeric(df.iloc[:, 0]):
        # Try re-reading while skipping the 2nd row (index 1) which often contains ticker info
        try:
            df2 = pd.read_csv(csv_path, index_col=0, parse_dates=True, skiprows=[1])
            if _is_price_numeric(df2.iloc[:, 0]):
                df = df2
        except Exception:
            # fallback: try skipping first two rows
            try:
                df3 = pd.read_csv(csv_path, index_col=0, parse_dates=True, skiprows=[1,2])
                if _is_price_numeric(df3.iloc[:, 0]):
                    df = df3
            except Exception:
                pass
    # Normalize Close column (support several possible column names)
    if 'Close' in df.columns:
        df.rename(columns={'Close': 'price'}, inplace=True)
    elif 'CLOSE' in df.columns:
        df.rename(columns={'CLOSE': 'price'}, inplace=True)
    elif 'close' in df.columns:
        df.rename(columns={'close': 'price'}, inplace=True)
    elif 'Price' in df.columns:
        # yfinance sometimes outputs 'Price' as the first column
        df.rename(columns={'Price': 'price'}, inplace=True)
    else:
        raise ValueError(f"CSV must contain 'Close'/'CLOSE'/'Price' column. Found: {df.columns.tolist()}")
    return df[['price']].copy()


def run(csv_file: str, future_h: int = 5, ret_threshold: float = 0.0002):
    csv_path = Path(csv_file)
    print(f"Loading data from: {csv_path}")
    data = load_price_csv(csv_path)
    print(f"Loaded {len(data)} bars")

    # Prepare features
    fe = EnhancedFeatureEngineering()
    data_with_features = fe.add_technical_indicators(data.copy(), future_h=future_h, ret_threshold=ret_threshold)
    print(f"Prepared features: {len(data_with_features.columns)} cols, {len(data_with_features)} rows")

    # Initialize strategy
    strategy = MLTradingStrategy(symbol=csv_path.stem)

    # Train
    print("Training model...")
    train_res = strategy.train(data_with_features, test_size=0.2)

    # Backtest using recommended production config
    backtest_res = strategy.backtest(
        data_with_features,
        confidence_threshold=0.60,
        initial_amount=10000,
        tc=0.001,
        risk_fraction=0.05,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        slippage=0.0005,
        retrain_interval=60,
        allow_short=False,
        confirmation_bars=2,
        dynamic_atr_k=2.0
    )

    # Save results
    out = Path(__file__).parent / 'extended_backtest_results.csv'
    df = pd.DataFrame([{
        'csv': str(csv_path),
        'rows': len(data),
        'future_h': future_h,
        'ret_threshold': ret_threshold,
        'final_amount': backtest_res.get('final_amount'),
        'trades': backtest_res.get('trades'),
        'trade_summary': backtest_res.get('trade_summary')
    }])
    df.to_csv(out, index=False)
    print(f"Saved summary to {out}")
    print("Backtest complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to price CSV file')
    parser.add_argument('--future_h', type=int, default=5)
    parser.add_argument('--ret_threshold', type=float, default=0.0002)
    args = parser.parse_args()
    run(args.csv, args.future_h, args.ret_threshold)
