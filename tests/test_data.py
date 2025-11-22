import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import GroundwaterDataGenerator
import pandas as pd

def test_data_generation():
    """Test that data generation works correctly"""
    generator = GroundwaterDataGenerator()
    data = generator.generate_sample_data(n_regions=2)
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert 'groundwater_level' in data.columns
    assert 'region' in data.columns
    assert 'date' in data.columns
    print("✅ Data generation test passed!")

if __name__ == "__main__":
    test_data_generation()
    print("All tests passed! 🎉")
