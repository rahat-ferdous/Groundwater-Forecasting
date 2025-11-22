import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from typing import Dict, Any, Tuple

class GroundwaterForecaster:
    def __init__(self):
        self.models: Dict = {}
        self.feature_importance: Dict = {}
        self.performance: Dict = {}
        
    def prepare_features(self, df: pd.DataFrame, target_lag: int = 1) -> Tuple[pd.DataFrame, list]:
        """Prepare features for training"""
        # Use current features + lagged features to predict future groundwater
        feature_columns = [col for col in df.columns if col not in 
                          ['date', 'region', 'groundwater_level', 'target']]
        
        # Create target (groundwater level n months ahead)
        df = df.sort_values(['region', 'date'])
        df['target'] = df.groupby('region')['groundwater_level'].shift(-target_lag)
        
        # Remove rows with NaN targets
        df = df.dropna(subset=['target'])
        
        return df, feature_columns
    
    def train_models(self, df: pd.DataFrame, target_lag: int = 1) -> Dict:
        """Train multiple models"""
        df, feature_columns = self.prepare_features(df, target_lag)
        
        models_performance = {}
        
        for region in df['region'].unique():
            region_data = df[df['region'] == region]
            
            X = region_data[feature_columns]
            y = region_data['target']
            
            # Split data chronologically
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=10
            )
            rf_model.fit(X_train, y_train)
            
            # Train XGBoost
            xgb_model = xgb.XGBRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            )
            xgb_model.fit(X_train, y_train)
            
            # Evaluate models
            rf_pred = rf_model.predict(X_test)
            xgb_pred = xgb_model.predict(X_test)
            
            rf_mae = mean_absolute_error(y_test, rf_pred)
            xgb_mae = mean_absolute_error(y_test, xgb_pred)
            rf_r2 = r2_score(y_test, rf_pred)
            xgb_r2 = r2_score(y_test, xgb_pred)
            
            self.models[region] = {
                'random_forest': rf_model,
                'xgboost': xgb_model,
                'feature_columns': feature_columns
            }
            
            models_performance[region] = {
                'Random Forest MAE': round(rf_mae, 3),
                'XGBoost MAE': round(xgb_mae, 3),
                'Random Forest R²': round(rf_r2, 3),
                'XGBoost R²': round(xgb_r2, 3),
                'Test Samples': len(X_test)
            }
            
            # Store feature importance
            self.feature_importance[region] = {
                'random_forest': dict(zip(feature_columns, rf_model.feature_importances_)),
                'xgboost': dict(zip(feature_columns, xgb_model.feature_importances_))
            }
        
        self.performance = models_performance
        return models_performance
    
    def predict(self, region: str, features: list) -> Dict[str, float]:
        """Make predictions for a region"""
        if region not in self.models:
            raise ValueError(f"No model trained for region: {region}")
        
        model_info = self.models[region]
        predictions = {}
        
        for model_name, model in model_info.items():
            if model_name != 'feature_columns':
                predictions[model_name] = float(model.predict([features])[0])
        
        return predictions
