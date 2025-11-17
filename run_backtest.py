import os
import sys
import pandas as pd

# Add package path so `py4at_app` imports resolve to the integrated package
pkg_path = r"c:\Users\HP\Downloads\_\_Td0\__Td0\py4at_app"
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

from py4at_app import ml_strategy_integration as m
from py4at_app import enhanced_predictor as e

sample_path = r'c:\Users\HP\Downloads\_\_Td0\__Td0\py4at_app\legacy\examples\sample_data.csv'
print('Loading', sample_path)

data = pd.read_csv(sample_path, index_col=0, parse_dates=True)
data.rename(columns={'Close':'price'}, inplace=True)
print(data.head())

# split
train_size = int(len(data) * 0.7)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]
print('Train rows', len(train_data), 'Test rows', len(test_data))

ml = m.MLTradingStrategy()
train_with_features = ml.prepare_data(train_data)
print('Training features shape:', train_with_features.shape)
try:
    train_results = ml.train(train_with_features, test_size=0.2)
    print('Train results best model:', train_results.get('best_model'))
except Exception as ex:
    print('Training failed:', ex)

# Backtest
try:
    results = ml.backtest(test_data, confidence_threshold=0.55, retrain_interval=0, allow_short=True)
    print('\nBacktest results dict:')
    print(results)
    ml.print_results()
    # show trade log
    print('\nTrade log entries:')
    for t in results.get('trade_log', []):
        print(t)
except Exception as ex:
    import traceback
    traceback.print_exc()
    print('Backtest failed:', ex)
