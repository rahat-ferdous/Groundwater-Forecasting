import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class GroundwaterDataGenerator:
    """Generate synthetic groundwater data for demonstration"""
    
    def __init__(self, start_date='2010-01-01', end_date='2023-12-31'):
        self.start_date = start_date
        self.end_date = end_date
        
    def generate_sample_data(self, n_regions: int = 3) -> pd.DataFrame:
        """Generate realistic synthetic groundwater data"""
        dates = pd.date_range(self.start_date, self.end_date, freq='M')
        regions = [f'Aquifer_{i+1}' for i in range(n_regions)]
        
        data = []
        for region in regions:
            # Base groundwater level with seasonal pattern and trend
            time_index = np.arange(len(dates))
            base_level = 100 + np.sin(2 * np.pi * time_index / 12) * 5
            
            # Long-term decline trend
            trend = -0.02 * time_index
            
            # Add some noise
            noise = np.random.normal(0, 0.5, len(dates))
            
            groundwater_level = base_level + trend + noise
            
            # Generate features
            for i, date in enumerate(dates):
                # Seasonal precipitation (more in winter)
                precip = max(0, np.random.normal(50, 20) + 20 * np.sin(2 * np.pi * i/12 - np.pi/2))
                
                # Temperature (seasonal)
                temp = 15 + 10 * np.sin(2 * np.pi * i/12)
                
                # NDVI (vegetation health - proxy for irrigation)
                ndvi = 0.3 + 0.4 * np.sin(2 * np.pi * i/12 + np.pi/4)
                ndvi = np.clip(ndvi + np.random.normal(0, 0.1), 0, 1)
                
                # Drought index (SPEI)
                spei = np.random.normal(0, 1)
                
                data.append({
                    'date': date,
                    'region': region,
                    'groundwater_level': float(groundwater_level[i]),
                    'precipitation': float(precip),
                    'temperature': float(temp),
                    'ndvi': float(ndvi),
                    'spei': float(spei),
                    'month': date.month,
                    'year': date.year
                })
        
        return pd.DataFrame(data)
    
    def create_lagged_features(self, df: pd.DataFrame, lags: list[int] = [1, 2, 3, 12]) -> pd.DataFrame:
        """Create lagged features for time series forecasting"""
        df = df.sort_values(['region', 'date'])
        
        for lag in lags:
            df[f'precipitation_lag_{lag}'] = df.groupby('region')['precipitation'].shift(lag)
            df[f'ndvi_lag_{lag}'] = df.groupby('region')['ndvi'].shift(lag)
            df[f'groundwater_lag_{lag}'] = df.groupby('region')['groundwater_level'].shift(lag)
            
        return df.dropna()

    def save_sample_data(self, filename: str = "sample_groundwater_data.csv"):
        """Generate and save sample data"""
        data = self.generate_sample_data()
        data_with_lags = self.create_lagged_features(data)
        data_with_lags.to_csv(f"data/processed/{filename}", index=False)
        return data_with_lags
