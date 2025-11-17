"""
Run a confidence-threshold sweep using the best configuration
(future_h=5, ret_threshold=0.0002). Test confidence_threshold = [0.50, 0.55, 0.60, 0.65].
Collect metrics and save CSV.

Also run improved-strategy experiments with confirmation (2 bars), ATR-based stop (k=2),
and risk_fraction in [0.02, 0.05] for thresholds 0.55 and 0.60.
"""

import os
import sys
import time
import traceback
import pandas as pd
import numpy as np

# ensure package path
pkg_path = r"c:\Users\HP\Downloads\_\_Td0\__Td0\py4at_app"
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

try:
    import yfinance as yf
except Exception:
    yf = None

from py4at_app.ml_strategy_integration import MLTradingStrategy

OUT_DIR = os.path.join(os.path.dirname(__file__), 'confidence_sweep_results')
os.makedirs(OUT_DIR, exist_ok=True)

CONF_THRESHOLDS = [0.50, 0.55, 0.60, 0.65]

def fetch_aapl_1m():
    if yf is None:
        raise RuntimeError('yfinance not installed')
    t = yf.Ticker('AAPL')
    df = t.history(period='7d', interval='1m', auto_adjust=False)
    if df.empty:
        return None
    if 'Close' in df.columns:
        df = df.rename(columns={'Close':'price'})
    else:
        df['price'] = df.iloc[:,0]
    return df[['price']].dropna()

def compute_sharpe(pnls, initial_amount=10000.0):
    if not pnls:
        return None
    # use pnl percentage per trade
    pct = np.array(pnls) / float(initial_amount)
    if len(pct) < 2:
        return None
    mean = pct.mean()
    std = pct.std(ddof=1)
    if std == 0:
        return None
    # annualize by sqrt(252) for illustration
    return (mean / std) * np.sqrt(252)

def run_backtest_for_threshold(df, threshold, future_h=5, ret_threshold=0.0002,
                               retrain_interval=60, confirmation_bars=1,
                               dynamic_atr_k=0.0, risk_fraction=0.01):
    split = int(len(df) * 0.7)
    train = df.iloc[:split]
    test = df.iloc[split:]

    ml = MLTradingStrategy(symbol='AAPL')
    train_with_features = ml.prepare_data(train, future_h=future_h, ret_threshold=ret_threshold)
    ml.train(train_with_features, test_size=0.2)

    res = ml.backtest(test, confidence_threshold=threshold, retrain_interval=retrain_interval,
                      allow_short=True, risk_fraction=risk_fraction,
                      confirmation_bars=confirmation_bars, dynamic_atr_k=dynamic_atr_k)
    # collect metrics
    trade_log = res.get('trade_log', [])
    trade_summary = res.get('trade_summary', {})
    closed = [t for t in trade_log if t.get('status')=='closed' and t.get('pnl') is not None]
    pnls = [t['pnl'] for t in closed]
    wins = sum(1 for p in pnls if p>0)
    losses = sum(1 for p in pnls if p<=0)
    win_rate = (wins / len(closed)) if closed else 0.0

    sharpe = compute_sharpe(pnls)

    return {
        'confidence_threshold': threshold,
        'total_trades': res.get('trades', 0),
        'closed_trades': len(closed),
        'win_rate': win_rate,
        'net_pnl': trade_summary.get('total_pnl', 0.0),
        'avg_hold_bars': trade_summary.get('avg_hold_bars', 0.0),
        'max_drawdown': trade_summary.get('max_drawdown', 0.0),
        'sharpe': sharpe,
        'final_amount': res.get('final_amount', 10000.0)
    }

def main():
    if yf is None:
        print('yfinance not installed. Install to run sweep.')
        return

    df = fetch_aapl_1m()
    if df is None or len(df) < 200:
        print('Not enough data')
        return
    print(f'Rows: {len(df)}')

    # baseline sweep using best config
    baseline_results = []
    for th in CONF_THRESHOLDS:
        print(f'Running threshold {th} ...')
        r = run_backtest_for_threshold(df, th, future_h=5, ret_threshold=0.0002,
                                       retrain_interval=60, confirmation_bars=1,
                                       dynamic_atr_k=0.0, risk_fraction=0.01)
        baseline_results.append(r)
        print('  ->', r)
        time.sleep(0.5)

    pd.DataFrame(baseline_results).to_csv(os.path.join(OUT_DIR, 'baseline_confidence_sweep.csv'), index=False)

    # Now apply improvements: confirmation=2, ATR stop k=2, test risk_fraction 0.02 and 0.05
    improved_results = []
    for risk in [0.02, 0.05]:
        for th in [0.55, 0.60]:
            print(f'Running improved config threshold={th}, risk={risk} ...')
            r = run_backtest_for_threshold(df, th, future_h=5, ret_threshold=0.0002,
                                           retrain_interval=60, confirmation_bars=2,
                                           dynamic_atr_k=2.0, risk_fraction=risk)
            r.update({'risk_fraction': risk, 'confirmation_bars': 2, 'dynamic_atr_k': 2.0})
            improved_results.append(r)
            print('  ->', r)
            time.sleep(0.5)

    pd.DataFrame(improved_results).to_csv(os.path.join(OUT_DIR, 'improved_strategy_results.csv'), index=False)

    print('\nBaseline results saved to', os.path.join(OUT_DIR, 'baseline_confidence_sweep.csv'))
    print('Improved strategy results saved to', os.path.join(OUT_DIR, 'improved_strategy_results.csv'))

if __name__ == '__main__':
    main()
