import pandas as pd
import numpy as np
from data_processor import GroundwaterDataGenerator

def generate_exploration_report():
    """Generate comprehensive data exploration"""
    print("Generating Groundwater Data Exploration Report...")
    
    # Generate sample data
    generator = GroundwaterDataGenerator()
    data = generator.generate_sample_data(n_regions=3)
    data = generator.create_lagged_features(data)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Regions: {list(data['region'].unique())}")
    
    # Basic statistics
    print("\n📊 Basic Statistics:")
    print(data[['groundwater_level', 'precipitation', 'temperature', 'ndvi']].describe())
    
    # Regional analysis
    print("\n🏞️ Regional Analysis:")
    regional_stats = data.groupby('region').agg({
        'groundwater_level': ['mean', 'std', 'min', 'max'],
        'precipitation': 'mean',
        'temperature': 'mean',
        'ndvi': 'mean'
    }).round(2)
    print(regional_stats)
    
    # Correlation analysis
    print("\n📈 Correlation with Groundwater Levels:")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    correlations = data[numeric_cols].corr()['groundwater_level'].sort_values(ascending=False)
    
    for feature, corr in correlations.items():
        if feature != 'groundwater_level':
            print(f"  {feature:25}: {corr:.3f}")
    
    print("\n✅ Exploration complete! Data is ready for modeling.")

if __name__ == "__main__":
    generate_exploration_report()
