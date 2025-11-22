import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import GroundwaterDataGenerator
from model import GroundwaterForecaster

# Page configuration
st.set_page_config(
    page_title="Groundwater Forecasting",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">💧 Groundwater Forecasting Prototype</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="feature-card">
This prototype demonstrates a machine learning approach to groundwater forecasting using climate and satellite data.
The model integrates precipitation, temperature, vegetation health (NDVI), and historical groundwater levels to predict future groundwater availability.
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'performance' not in st.session_state:
    st.session_state.performance = None

# Sidebar
st.sidebar.header("🔧 Configuration")

# Data generation
st.sidebar.subheader("1. Data Generation")
n_regions = st.sidebar.slider("Number of Regions", 1, 5, 3)

if st.sidebar.button("🚀 Generate Sample Data"):
    with st.spinner("Generating synthetic groundwater data..."):
        generator = GroundwaterDataGenerator()
        data = generator.generate_sample_data(n_regions=n_regions)
        data = generator.create_lagged_features(data)
        st.session_state.data = data
    st.sidebar.success(f"✅ Data generated for {n_regions} regions!")

# Model training
st.sidebar.subheader("2. Model Training")
forecast_months = st.sidebar.slider("Forecast Horizon (months)", 1, 6, 1)

if st.sidebar.button("🤖 Train Forecasting Models"):
    if st.session_state.data is not None:
        with st.spinner("Training machine learning models..."):
            forecaster = GroundwaterForecaster()
            performance = forecaster.train_models(st.session_state.data, target_lag=forecast_months)
            st.session_state.models = forecaster
            st.session_state.performance = performance
        st.sidebar.success("✅ Models trained successfully!")
    else:
        st.sidebar.error("❌ Please generate data first!")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Model Results", "🔮 Predictions", "📚 Methodology"])

with tab1:
    if st.session_state.data is not None:
        st.header("Data Overview")
        
        # Data preview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sample Data")
            st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Regions Summary")
            region_summary = st.session_state.data.groupby('region').agg({
                'groundwater_level': ['mean', 'std', 'min', 'max'],
                'precipitation': 'mean',
                'temperature': 'mean',
                'ndvi': 'mean'
            }).round(2)
            st.dataframe(region_summary, use_container_width=True)

        # Time series visualization
        st.subheader("Groundwater Level Time Series")
        
        fig = px.line(st.session_state.data, x='date', y='groundwater_level', 
                      color='region', title='Groundwater Levels Over Time',
                      labels={'groundwater_level': 'Groundwater Level', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlations
        st.subheader("Feature Correlations with Groundwater Levels")
        
        corr_data = st.session_state.data.select_dtypes(include=[np.number]).corr()
        groundwater_corrs = corr_data['groundwater_level'].sort_values(ascending=False)
        
        fig_corr = px.bar(groundwater_corrs[1:6],  # Top 5 correlations
                         title="Top Feature Correlations with Groundwater Levels",
                         labels={'value': 'Correlation', 'index': 'Feature'})
        fig_corr.update_layout(showlegend=False)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    else:
        st.info("👆 Please generate sample data using the sidebar controls to get started.")

with tab2:
    if st.session_state.models is not None and st.session_state.performance is not None:
        st.header("Model Performance Analysis")
        
        # Performance metrics
        st.subheader("📈 Model Performance Metrics")
        perf_df = pd.DataFrame(st.session_state.performance).T
        st.dataframe(perf_df.style.highlight_min(subset=['Random Forest MAE', 'XGBoost MAE'], color='#ffcccc'), 
                    use_container_width=True)
        
        # Feature importance
        st.subheader("🔍 Feature Importance Analysis")
        
        region_select = st.selectbox("Select Region", list(st.session_state.models.feature_importance.keys()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Random Forest Feature Importance**")
            rf_importance = pd.DataFrame({
                'feature': list(st.session_state.models.feature_importance[region_select]['random_forest'].keys()),
                'importance': list(st.session_state.models.feature_importance[region_select]['random_forest'].values())
            }).sort_values('importance', ascending=True).tail(10)  # Top 10 features
            
            fig_rf = px.bar(rf_importance, x='importance', y='feature', 
                           title=f'Random Forest - {region_select}',
                           orientation='h')
            st.plotly_chart(fig_rf, use_container_width=True)
        
        with col2:
            st.write("**XGBoost Feature Importance**")
            xgb_importance = pd.DataFrame({
                'feature': list(st.session_state.models.feature_importance[region_select]['xgboost'].keys()),
                'importance': list(st.session_state.models.feature_importance[region_select]['xgboost'].values())
            }).sort_values('importance', ascending=True).tail(10)
            
            fig_xgb = px.bar(xgb_importance, x='importance', y='feature', 
                            title=f'XGBoost - {region_select}',
                            orientation='h')
            st.plotly_chart(fig_xgb, use_container_width=True)

    else:
        st.info("👆 Please train models using the sidebar to see performance results.")

with tab3:
    st.header("Groundwater Level Predictions")
    
    if st.session_state.models is not None:
        st.subheader("Input Features for Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            precipitation = st.number_input("Precipitation (mm)", value=50.0, min_value=0.0, max_value=500.0)
            temperature = st.number_input("Temperature (°C)", value=15.0, min_value=-10.0, max_value=40.0)
        
        with col2:
            ndvi = st.slider("NDVI (Vegetation Health)", 0.0, 1.0, 0.5)
            spei = st.slider("SPEI (Drought Index)", -3.0, 3.0, 0.0, step=0.1)
        
        with col3:
            gw_lag1 = st.number_input("Groundwater Lag 1 month", value=95.0)
            precip_lag1 = st.number_input("Precipitation Lag 1 month", value=45.0)
        
        # Create feature vector (simplified for demo)
        feature_vector = [precipitation, temperature, ndvi, spei, precip_lag1, ndvi, gw_lag1]
        
        if st.button("🎯 Generate Predictions", type="primary"):
            predictions = {}
            for region in st.session_state.models.feature_importance.keys():
                try:
                    # Use the first few features that match our model expectations
                    model_features = feature_vector[:len(st.session_state.models.models[region]['feature_columns'])]
                    pred = st.session_state.models.predict(region, model_features)
                    predictions[region] = pred
                except Exception as e:
                    st.error(f"Error predicting for {region}: {str(e)}")
            
            # Display predictions
            st.subheader("📊 Prediction Results")
            pred_df = pd.DataFrame(predictions).T
            st.dataframe(pred_df.style.highlight_max(axis=1, color='#ccffcc'), 
                        use_container_width=True)
            
            # Visualization
            fig_pred = px.bar(pred_df, barmode='group', 
                             title="Groundwater Level Predictions by Model and Region",
                             labels={'value': 'Predicted Groundwater Level', 'index': 'Region'})
            st.plotly_chart(fig_pred, use_container_width=True)
            
    else:
        st.info("👆 Please generate data and train models first to make predictions.")

with tab4:
    st.header("Research Methodology")
    
    st.markdown("""
    ## 🧠 How This Prototype Works

    ### 1. Data Integration
    - **Climate Data**: Precipitation, temperature, drought indices (SPEI)
    - **Satellite Observations**: NDVI (Normalized Difference Vegetation Index) as proxy for irrigation
    - **Groundwater Measurements**: Historical groundwater level data
    - **Temporal Features**: Lagged variables to capture memory effects

    ### 2. Machine Learning Models
    - **Random Forest**: Robust ensemble method, handles non-linear relationships
    - **XGBoost**: Gradient boosting optimized for performance
    - **Temporal Validation**: Chronological train-test split to prevent data leakage

    ### 3. Model Interpretation
    - Feature importance analysis
    - Correlation analysis with groundwater levels

    ## 🔮 Real-World Application

    This approach can be extended with real data from:
    - **GRACE/GRACE-FO** satellites for total water storage
    - **InSAR** data for land subsidence measurements  
    - **MODIS/Landsat** for vegetation indices
    - **Local monitoring wells** for groundwater validation

    ## 📚 Scientific Foundation

    Inspired by remote sensing research in hydrology, particularly:
    - Multi-sensor data fusion approaches
    - Machine learning for hydrological forecasting
    - Climate-groundwater interactions
    - Anthropogenic impacts on water resources
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Prototype inspired by remote sensing research for water resources • "
    "Built with Streamlit and Python 3.13"
    "</div>", 
    unsafe_allow_html=True
)
