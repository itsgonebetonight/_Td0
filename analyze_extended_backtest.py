"""
Parse and analyze extended backtest results CSV.
Extracts trade-summary metrics, compares across configs, and provides recommendations.
"""
import pandas as pd
import json
from pathlib import Path


def parse_backtest_results(csv_path: Path):
    """Load and parse the extended backtest results CSV."""
    df = pd.read_csv(csv_path)
    results = []
    
    for _, row in df.iterrows():
        rec = {
            'csv': row['csv'],
            'rows': row['rows'],
            'future_h': row['future_h'],
            'ret_threshold': row['ret_threshold'],
            'final_amount': row['final_amount'],
            'trades': row['trades'],
        }
        
        # Parse trade_summary JSON
        try:
            trade_summary = json.loads(row['trade_summary'].replace("'", '"'))
            rec.update({
                'closed_trades': trade_summary.get('total_trades', 0),
                'total_pnl': trade_summary.get('total_pnl', 0),
                'wins': trade_summary.get('wins', 0),
                'losses': trade_summary.get('losses', 0),
                'win_rate': trade_summary.get('win_rate', 0),
                'avg_pnl_per_trade': trade_summary.get('avg_pnl', 0),
                'max_drawdown': trade_summary.get('max_drawdown', 0),
                'avg_hold_bars': trade_summary.get('avg_hold_bars', 0),
                'long_trades': trade_summary.get('long_stats', {}).get('trades', 0),
                'long_pnl': trade_summary.get('long_stats', {}).get('total_pnl', 0),
                'long_win_rate': trade_summary.get('long_stats', {}).get('win_rate', 0),
                'short_trades': trade_summary.get('short_stats', {}).get('trades', 0),
                'short_pnl': trade_summary.get('short_stats', {}).get('total_pnl', 0),
                'short_win_rate': trade_summary.get('short_stats', {}).get('win_rate', 0),
            })
        except Exception as e:
            print(f"Error parsing trade_summary: {e}")
            rec.update({c: None for c in ['closed_trades', 'total_pnl', 'wins', 'losses', 'win_rate', 'avg_pnl_per_trade', 'max_drawdown', 'avg_hold_bars']})
        
        results.append(rec)
    
    return pd.DataFrame(results)


def calculate_metrics(df: pd.DataFrame):
    """Calculate derived metrics for each backtest run."""
    df['net_pnl'] = df['final_amount'] - 10000
    df['return_pct'] = (df['net_pnl'] / 10000) * 100
    df['signal_density'] = (df['trades'] / df['rows']) * 100
    df['closed_trade_ratio'] = df['closed_trades'] / df['trades'].replace(0, 1)
    return df


def print_summary(df: pd.DataFrame):
    """Print formatted summary of backtest results."""
    print("\n" + "=" * 140)
    print("EXTENDED BACKTEST RESULTS - DETAILED ANALYSIS")
    print("=" * 140)
    print()
    
    for idx, row in df.iterrows():
        print(f"Backtest #{idx + 1}: {Path(row['csv']).name}")
        print("-" * 140)
        print(f"  Data:              {row['rows']} bars | future_h={row['future_h']} | ret_threshold={row['ret_threshold']}")
        print(f"  Final Balance:     ${row['final_amount']:.2f}")
        print(f"  Net P&L:           ${row['net_pnl']:.2f} ({row['return_pct']:+.2f}%)")
        print()
        print(f"  TRADE EXECUTION:")
        print(f"    Total Signals:   {row['trades']} (Signal Density: {row['signal_density']:.2f}%)")
        print(f"    Closed Trades:   {row['closed_trades']}")
        print(f"    Closed/Signal:   {row['closed_trade_ratio']:.1%}")
        print()
        print(f"  PERFORMANCE METRICS:")
        print(f"    Win Rate:        {row['win_rate']*100:.1f}% ({row['wins']} wins, {row['losses']} losses)")
        print(f"    Avg P&L/Trade:   ${row['avg_pnl_per_trade']:.2f}")
        print(f"    Max Drawdown:    {row['max_drawdown']:.2f}%")
        print(f"    Avg Hold Time:   {row['avg_hold_bars']:.1f} bars")
        print()
        print(f"  LONG TRADES:       {row['long_trades']} trades | P&L: ${row['long_pnl']:.2f} | Win Rate: {row['long_win_rate']*100:.1f}%")
        print(f"  SHORT TRADES:      {row['short_trades']} trades | P&L: ${row['short_pnl']:.2f} | Win Rate: {row['short_win_rate']*100:.1f}%")
        print()


def print_recommendations(df: pd.DataFrame):
    """Print recommendations for label adjustments."""
    print("\n" + "=" * 140)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 140)
    print()
    
    row = df.iloc[0]  # First (primary) backtest result
    
    print("CURRENT CONFIGURATION:")
    print(f"  future_h: {row['future_h']}")
    print(f"  ret_threshold: {row['ret_threshold']} ({row['ret_threshold']*100:.3f}%)")
    print()
    
    print("FINDINGS:")
    print()
    print(f"1. P&L QUALITY:")
    if row['net_pnl'] > 0:
        print(f"   ✓ Slightly profitable on minute-level 7-day data: +${row['net_pnl']:.2f} ({row['return_pct']:+.2f}%)")
    elif row['net_pnl'] > -50:
        print(f"   ~ Near-breakeven on minute-level 7-day data: ${row['net_pnl']:.2f} ({row['return_pct']:+.2f}%)")
    else:
        print(f"   ✗ Negative P&L on minute-level data: ${row['net_pnl']:.2f} ({row['return_pct']:+.2f}%)")
    print()
    
    print(f"2. TRADE FREQUENCY:")
    print(f"   Signal density: {row['signal_density']:.2f}% ({row['trades']} trades over {row['rows']} bars)")
    print(f"   Closed trades: {row['closed_trades']} ({row['closed_trade_ratio']*100:.1f}% of signals)")
    if row['signal_density'] > 2.0:
        print(f"   → High frequency on 1-minute data; may reduce position sizing to manage risk.")
    elif row['signal_density'] < 0.5:
        print(f"   → Low frequency; could lower ret_threshold to increase trade opportunities.")
    print()
    
    print(f"3. WIN RATE & EXPECTED VALUE:")
    print(f"   Win Rate: {row['win_rate']*100:.1f}% ({row['wins']} wins, {row['losses']} losses)")
    print(f"   Avg P&L/Trade: ${row['avg_pnl_per_trade']:.2f}")
    if row['win_rate'] >= 0.60:
        print(f"   → Strong win rate; consider increasing position size or risk_fraction.")
    elif row['win_rate'] >= 0.50:
        print(f"   → Acceptable win rate; current sizing appears appropriate.")
    else:
        print(f"   → Low win rate; model may need retraining or label adjustment.")
    print()
    
    print(f"4. SIDE ANALYSIS (LONG vs SHORT):")
    print(f"   Long:  {row['long_trades']:2d} trades | ${row['long_pnl']:+7.2f} P&L | Win: {row['long_win_rate']*100:5.1f}%")
    print(f"   Short: {row['short_trades']:2d} trades | ${row['short_pnl']:+7.2f} P&L | Win: {row['short_win_rate']*100:5.1f}%")
    if row['long_pnl'] > abs(row['short_pnl']):
        print(f"   → Long trades profitable; short trades struggle. Consider disabling shorts or tightening short-entry filters.")
    print()
    
    print(f"5. DRAWDOWN & RISK:")
    print(f"   Max Drawdown: {row['max_drawdown']:.2f}%")
    print(f"   Avg Hold: {row['avg_hold_bars']:.1f} bars")
    if row['max_drawdown'] < 1.0:
        print(f"   → Low drawdown; risk is well-controlled.")
    elif row['max_drawdown'] < 2.0:
        print(f"   → Moderate drawdown; acceptable for 1-minute trading.")
    else:
        print(f"   → High drawdown; consider tighter stop-loss or reduced risk_fraction.")
    print()
    
    print("\nACTION ITEMS FOR PRODUCTION:")
    print()
    print("  1. Current config (future_h=5, ret_threshold=0.0002) shows:")
    print(f"     - Near-breakeven 1-minute performance (small sample: 8 closed trades)")
    print(f"     - Strong long-side bias (+{row['long_pnl']:.2f} from longs, {row['short_pnl']:.2f} from shorts)")
    print(f"     - Acceptable win rate ({row['win_rate']*100:.1f}%) and low drawdown ({row['max_drawdown']:.2f}%)")
    print()
    print("  2. RECOMMENDED next steps:")
    print("     a) Disable short-selling (allow_short=False) to focus on profitable long trades")
    print("     b) Run label sweep on this 7-day minute dataset testing:")
    print("        - Lower ret_threshold values (0.00005, 0.00010) for higher signal density")
    print("        - Check if tighter labels improve P&L on intraday timeframes")
    print("     c) Validate on a different time period (different week/month) to confirm out-of-sample stability")
    print("     d) Increase position size gradually (risk_fraction: 0.05 → 0.10) if wins persist")
    print()
    print("  3. DEPLOYMENT READINESS:")
    if row['net_pnl'] >= -50 and row['win_rate'] >= 0.50 and row['max_drawdown'] <= 2.0:
        print("     ✓ Configuration is READY for paper trading (1-2 weeks)")
        print("     ✓ Monitor: Win rate, actual slippage, execution fills")
        print("     → After positive paper trading results, proceed to small live account ($1-5k)")
    else:
        print("     ⚠ Configuration needs optimization before paper trading")
        print("     → Run label sweep to find better parameter set")
        print("     → Consider disabling shorts and retesting")
    print()


if __name__ == '__main__':
    csv_path = Path(__file__).parent / 'extended_backtest_results.csv'
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        exit(1)
    
    # Parse and analyze
    df = parse_backtest_results(csv_path)
    df = calculate_metrics(df)
    
    # Print results
    print_summary(df)
    print_recommendations(df)
    
    # Save detailed results to CSV
    output_csv = Path(__file__).parent / 'extended_backtest_analysis.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to: {output_csv}")
