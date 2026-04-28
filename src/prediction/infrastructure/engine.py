import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging

logger = logging.getLogger(__name__)

class TrafficModelEngine:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.classifiers = {} # dict of time_horizon -> model
        self.regressors = {} # dict of time_horizon -> model (for vehicle count)
        self.label_map = {0: "Normal", 1: "High", 2: "Heavy"}
        
    def train(self, df: pd.DataFrame, feature_cols: list, target_col_class: str, target_col_reg: str, horizon_name: str):
        """Trains both classifier (congestion) and regressor (vehicle count) for a horizon."""
        # 1. Train Classifier
        train_df = df[feature_cols + [target_col_class]].dropna()
        if len(train_df) > 10:
            X = train_df[feature_cols]
            y = train_df[target_col_class]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            logger.info(f"Classifier {horizon_name} Accuracy: {acc:.2f}")
            
            self.classifiers[horizon_name] = clf
            self.save_model(clf, f"class_{horizon_name}")
        else:
            logger.warning(f"Not enough data for classifier {horizon_name} ({len(train_df)} rows)")

        # 2. Train Regressor
        train_df_reg = df[feature_cols + [target_col_reg]].dropna()
        if len(train_df_reg) > 10:
            X_reg = train_df_reg[feature_cols]
            y_reg = train_df_reg[target_col_reg]
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
            
            reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            reg.fit(X_train_r, y_train_r)
            
            # Simple eval
            score = reg.score(X_test_r, y_test_r)
            logger.info(f"Regressor {horizon_name} R2 Score: {score:.2f}")
            
            self.regressors[horizon_name] = reg
            self.save_model(reg, f"reg_{horizon_name}")
        else:
            logger.warning(f"Not enough data for regressor {horizon_name} ({len(train_df_reg)} rows)")

    def save_model(self, model, name: str):
        path = os.path.join(self.model_dir, f"traffic_rf_{name}.joblib")
        joblib.dump(model, path)
        logger.info(f"Saved model to {path}")

    def load_models(self):
        """Loads all available models from the directory."""
        horizons = ['current', '15min', '30min', '45min']
        for h in horizons:
            # Load Classifiers
            path_c = os.path.join(self.model_dir, f"traffic_rf_class_{h}.joblib")
            if os.path.exists(path_c):
                try:
                    self.classifiers[h] = joblib.load(path_c)
                except Exception as e:
                    logger.error(f"Failed to load classifier {h}: {e}")
            
            # Load Regressors
            path_r = os.path.join(self.model_dir, f"traffic_rf_reg_{h}.joblib")
            if os.path.exists(path_r):
                try:
                    self.regressors[h] = joblib.load(path_r)
                except Exception as e:
                    logger.error(f"Failed to load regressor {h}: {e}")

    def predict(self, features: dict) -> dict:
        """
        Predicts for all loaded horizons (classification and regression).
        """
        feature_order = [
            'total_vehicles', 'occupancy_rate', 'flow_rate_per_min', 
            'avg_speed', 'avg_density', 'hour', 'day_of_week'
        ]
        
        row = pd.DataFrame([features])
        for col in feature_order:
            if col not in row.columns:
                row[col] = 0
        row = row[feature_order].fillna(0)
        
        predictions = {}
        
        # Classification
        for name, model in self.classifiers.items():
            try:
                pred_class = model.predict(row)[0]
                pred_label = self.label_map.get(pred_class, "Unknown")
                predictions[f"predicted_congestion_{name}"] = pred_label
            except Exception as e:
                predictions[f"predicted_congestion_{name}"] = "Error"
                
        # Regression
        for name, model in self.regressors.items():
            try:
                pred_val = model.predict(row)[0]
                predictions[f"predicted_vehicles_{name}"] = int(pred_val)
            except Exception as e:
                predictions[f"predicted_vehicles_{name}"] = -1
        
        return predictions
