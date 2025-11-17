"""
Enhanced Trading Predictor (package copy)
Placed inside the package for direct imports like `from py4at_app import enhanced_predictor`.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')


class EnhancedFeatureEngineering:
    @staticmethod
    def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = df['price'].rolling(window).mean()
            df[f'ema_{window}'] = df['price'].ewm(span=window, adjust=False).mean()
        df['price_to_sma_20'] = df['price'] / df['sma_20']
        df['price_to_sma_50'] = df['price'] / df['sma_50']
        df['price_to_sma_200'] = df['price'] / df['sma_200']
        exp1 = df['price'].ewm(span=12, adjust=False).mean()
        exp2 = df['price'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        for window in [20, 50]:
            rolling_mean = df['price'].rolling(window).mean()
            rolling_std = df['price'].rolling(window).std()
            df[f'bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
            df[f'bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
            df[f'bb_width_{window}'] = df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']
            df[f'bb_position_{window}'] = (df['price'] - df[f'bb_lower_{window}']) / df[f'bb_width_{window}']
        for period in [5, 10, 20, 50]:
            df[f'momentum_{period}'] = df['price'] - df['price'].shift(period)
            df[f'roc_{period}'] = ((df['price'] - df['price'].shift(period)) / df['price'].shift(period)) * 100
        for window in [10, 20, 50]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            df[f'volatility_ratio_{window}'] = (df[f'volatility_{window}'] / df[f'volatility_{window}'].rolling(50).mean())
        df['tr'] = df['price'].diff().abs()
        for window in [14, 20]:
            df[f'atr_{window}'] = df['tr'].rolling(window).mean()
        for window in [10, 20, 50]:
            df[f'high_{window}'] = df['price'].rolling(window).max()
            df[f'low_{window}'] = df['price'].rolling(window).min()
            df[f'channel_position_{window}'] = ((df['price'] - df[f'low_{window}']) / (df[f'high_{window}'] - df[f'low_{window}']))
        df['adx'] = EnhancedFeatureEngineering._calculate_adx(df['price'], window=14)
        df['volume_proxy'] = df['returns'].rolling(20).std() * 100
        for lag in range(1, 11):
            df[f'lag_{lag}'] = df['returns'].shift(lag)
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
        df['target'] = np.where(df['returns'].shift(-1) > 0, 1, 0)
        return df

    @staticmethod
    def _calculate_adx(prices: pd.Series, window: int = 14) -> pd.Series:
        high = prices.rolling(window).max()
        low = prices.rolling(window).min()
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        tr = high - low
        atr = tr.rolling(window).mean()
        plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window).mean()
        return adx


class EnhancedTradingPredictor:
    def __init__(self, use_ensemble: bool = True):
        self.use_ensemble = use_ensemble
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = None
        self.best_model = None
        self.model_scores = {}
        self.calibrated = {}
        self.ensemble_weights = {}

    def _initialize_models(self):
        xgb_model = None
        try:
            from xgboost import XGBClassifier
            xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=6, random_state=42)
        except Exception:
            xgb_model = None
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', max_iter=500, random_state=42, early_stopping=True)
        }
        if xgb_model is not None:
            self.models['xgboost'] = xgb_model

    def prepare_features(self, data: pd.DataFrame) -> tuple:
        exclude_cols = ['price', 'target', 'returns', 'log_returns', 'tr']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        valid_cols = []
        for col in feature_cols:
            if data[col].notna().sum() / len(data) > 0.7:
                valid_cols.append(col)
        X = data[valid_cols].copy()
        y = data['target'].copy()
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        return X, y, valid_cols

    def _calibrate_model(self, model, X_train, y_train):
        try:
            calibrated = CalibratedClassifierCV(base_estimator=model, cv=3, method='sigmoid')
            calibrated.fit(X_train, y_train)
            return calibrated
        except Exception:
            try:
                model.fit(X_train, y_train)
            except Exception:
                pass
            return model

    def train_test_split_timeseries(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple:
        split_idx = int(len(X) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        y_pred = model.predict(X_test)
        metrics = {'accuracy': accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred, zero_division=0), 'recall': recall_score(y_test, y_pred, zero_division=0), 'f1': f1_score(y_test, y_pred, zero_division=0)}
        return metrics

    def train(self, data: pd.DataFrame, test_size: float = 0.2) -> dict:
        print("Preparing features...")
        X, y, feature_names = self.prepare_features(data)
        # store feature names for prediction time
        self.feature_names = feature_names
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        print(f"Total samples: {len(X)}")
        print(f"Features: {len(feature_names)}")
        print(f"Positive class: {y.sum()} ({y.mean()*100:.1f}%)")
        print()
        X_train, X_test, y_train, y_test = self.train_test_split_timeseries(X, y, test_size)
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self._initialize_models()
        results = {}
        print("Training models...")
        print("-" * 70)
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            try:
                model.fit(X_train_scaled, y_train)
                calibrated_model = None
                try:
                    calibrated_model = self._calibrate_model(model, X_train_scaled, y_train)
                except Exception:
                    calibrated_model = model
                metrics = self.evaluate_model(calibrated_model, X_test_scaled, y_test)
                results[name] = metrics
                self.model_scores[name] = metrics['f1']
                self.calibrated[name] = calibrated_model if calibrated_model is not None else model
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1 Score:  {metrics['f1']:.4f}")
            except Exception as e:
                print(f"  Error: {str(e)}")
                results[name] = None
        if self.model_scores:
            self.best_model = max(self.model_scores, key=self.model_scores.get)
            print()
            print("=" * 70)
            print(f"Best model: {self.best_model} (F1: {self.model_scores[self.best_model]:.4f})")
            print("=" * 70)
        if 'random_forest' in self.models:
            rf = self.models['random_forest']
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
            self.feature_importance = importance_df
            print()
            print("Top 10 Most Important Features:")
            print("-" * 70)
            for idx, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']:<30} {row['importance']:.4f}")
        return {'results': results, 'best_model': self.best_model, 'feature_importance': self.feature_importance, 'test_accuracy': results.get(self.best_model, {}).get('accuracy', 0)}

    def fit_ensemble_weights(self):
        if not self.model_scores:
            return
        total = sum(max(v, 0) for v in self.model_scores.values())
        if total <= 0:
            n = len(self.model_scores)
            for k in self.model_scores:
                self.ensemble_weights[k] = 1.0 / n
        else:
            for k, v in self.model_scores.items():
                self.ensemble_weights[k] = max(v, 0) / total

    def partial_retrain(self, data: pd.DataFrame, test_size: float = 0.2):
        results = self.train(data, test_size=test_size)
        self.fit_ensemble_weights()
        return results

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, 'feature_names') or not self.feature_names:
            X, _, _ = self.prepare_features(data)
        else:
            X = data.copy()
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names].copy()
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        X_scaled = self.scaler.transform(X)
        if self.use_ensemble and len(self.models) > 1:
            preds = []
            weights = []
            for name, model in self.models.items():
                try:
                    mdl = self.calibrated.get(name, model)
                    p = mdl.predict(X_scaled)
                    preds.append(p)
                    weights.append(self.ensemble_weights.get(name, 1.0))
                except Exception:
                    pass
            if preds:
                preds = np.array(preds)
                weights = np.array(weights).reshape((-1, 1))
                vote_scores = (preds * weights).sum(axis=0)
                threshold = weights.sum()
                ensemble_pred = (vote_scores >= (threshold / 2)).astype(int)
                return ensemble_pred
        if self.best_model and self.best_model in self.models:
            mdl = self.calibrated.get(self.best_model, self.models[self.best_model])
            return mdl.predict(X_scaled)
        return np.zeros(len(X))

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, 'feature_names') or not self.feature_names:
            X, _, _ = self.prepare_features(data)
        else:
            X = data.copy()
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names].copy()
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        X_scaled = self.scaler.transform(X)
        if self.use_ensemble and len(self.models) > 1:
            probas = []
            weights = []
            for name, model in self.models.items():
                try:
                    mdl = self.calibrated.get(name, model)
                    proba = mdl.predict_proba(X_scaled)[:, 1]
                    probas.append(proba)
                    weights.append(self.ensemble_weights.get(name, 1.0))
                except Exception:
                    pass
            if probas:
                probas = np.array(probas)
                weights = np.array(weights).reshape((-1, 1))
                weighted = (probas * weights).sum(axis=0) / weights.sum()
                return weighted
        if self.best_model and self.best_model in self.models:
            mdl = self.calibrated.get(self.best_model, self.models[self.best_model])
            try:
                return mdl.predict_proba(X_scaled)[:, 1]
            except Exception:
                return self.models[self.best_model].predict_proba(X_scaled)[:, 1]
        return np.zeros(len(X))