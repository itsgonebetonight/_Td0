"""
Download historical price data using yfinance and save to CSV.
Downloads daily data for the past 6 months for a symbol (default AAPL).

Usage:
  python download_data.py --symbol AAPL --period 6mo --interval 1d --out data/AAPL_6mo_daily.csv
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--symbol', default='AAPL')
parser.add_argument('--period', default='6mo')
parser.add_argument('--interval', default='1d')
parser.add_argument('--out', default='data/AAPL_6mo_daily.csv')
args = parser.parse_args()

import yfinance as yf
import pandas as pd

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Downloading {args.symbol} period={args.period} interval={args.interval} ...")

df = yf.download(args.symbol, period=args.period, interval=args.interval, progress=False)
if df.empty:
    raise SystemExit('No data downloaded')

# Ensure CSV has Date index and CLOSE column
if 'Close' in df.columns:
    df.rename(columns={'Close':'CLOSE'}, inplace=True)

print(f"Saving to {out_path} ({len(df)} rows)")
df.to_csv(out_path)
print('Done')
