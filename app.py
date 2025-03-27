import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objs as go

# Custom CSS for enhanced styling
def set_custom_style():
    st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .title {
        color: #2C3E50;
        font-size: 48px;
        text-align: center;
        margin-bottom: 30px;
        font-weight: bold;
    }
    .info-box {
        background-color: #ECF0F1;
        border-left: 5px solid #3498DB;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    .prediction-card {
        background-color: #2ECC71;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Comprehensive data generation function
def generate_detailed_crop_yield_data():
    np.random.seed(42)
    n_samples = 1500
    
    # More comprehensive feature generation
    data = {
        'Rainfall': np.random.normal(1200, 300, n_samples),
        'Temperature': np.random.normal(28, 3, n_samples),
        'Soil_Nitrogen': np.random.normal(50, 15, n_samples),
        'Soil_Phosphorus': np.random.normal(40, 10, n_samples),
        'Soil_Potassium': np.random.normal(45, 12, n_samples),
        'Farming_Practice': np.random.choice(['Traditional', 'Modern'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # More sophisticated yield calculation
    df['Crop_Yield'] = (
        0.35 * df['Rainfall'] / 1000 + 
        0.25 * df['Temperature'] / 30 + 
        0.15 * df['Soil_Nitrogen'] / 100 + 
        0.10 * df['Soil_Phosphorus'] / 50 + 
        0.05 * df['Soil_Potassium'] / 50 + 
        0.1 * (df['Farming_Practice'] == 'Modern') + 
        np.random.normal(2, 0.5, n_samples)
    ).clip(1, 10)
    
    return df

# Main Streamlit application
def main():
    # Set custom styling
    set_custom_style()
    
    # App Title and Introduction
    st.markdown('<div class="title">🌾 Nigerian Crop Yield Predictor</div>', unsafe_allow_html=True)
    
    # Informative Introduction
    st.markdown("""
    <div class="info-box">
    <h3>🚜 Agricultural Intelligence Platform</h3>
    
    This advanced prediction tool helps farmers and agricultural professionals:
    - Estimate potential crop yields
    - Understand key factors influencing agricultural productivity
    - Make data-driven decisions for improved farming strategies
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for Detailed Inputs
    st.sidebar.header('🌱 Agricultural Parameters')
    
    # Detailed Input Sliders with Explanations
    st.sidebar.markdown("### Environmental Conditions")
    rainfall = st.sidebar.slider(
        'Rainfall (mm)', 
        min_value=500, 
        max_value=2000, 
        value=1200, 
        help="Total rainfall affects water availability for crops. Higher rainfall typically supports better crop growth."
    )
    
    temperature = st.sidebar.slider(
        'Average Temperature (°C)', 
        min_value=20, 
        max_value=35, 
        value=28, 
        help="Temperature impacts plant metabolism, photosynthesis, and overall crop development."
    )
    
    st.sidebar.markdown("### Soil Nutrients")
    nitrogen = st.sidebar.slider(
        'Soil Nitrogen Level', 
        min_value=20, 
        max_value=80, 
        value=50, 
        help="Nitrogen is crucial for leaf growth and chlorophyll production. Optimal levels support better crop yield."
    )
    
    phosphorus = st.sidebar.slider(
        'Soil Phosphorus Level', 
        min_value=20, 
        max_value=80, 
        value=40, 
        help="Phosphorus supports root development, energy transfer, and flower/fruit production."
    )
    
    potassium = st.sidebar.slider(
        'Soil Potassium Level', 
        min_value=20, 
        max_value=80, 
        value=45, 
        help="Potassium enhances overall plant health, disease resistance, and water regulation."
    )
    
    farming_practice = st.sidebar.selectbox(
        'Farming Practice', 
        ['Traditional', 'Modern'], 
        help="Modern farming practices often include improved techniques, irrigation, and technology."
    )
    
    # Generate and prepare data
    df = generate_detailed_crop_yield_data()
    
    # Prepare features
    features = ['Rainfall', 'Temperature', 'Soil_Nitrogen', 'Soil_Phosphorus', 'Soil_Potassium']
    X = df[features]
    y = df['Crop_Yield']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    
    # Predict
    input_data = np.array([[rainfall, temperature, nitrogen, phosphorus, potassium]])
    prediction = model.predict(input_data)
    
    # Results Section
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction Display
        st.markdown('<div class="prediction-card">Predicted Crop Yield:<br>' + 
                    f'{prediction[0]:.2f} tons/hectare</div>', 
                    unsafe_allow_html=True)
    
    with col2:
        # Feature Importance Visualization
        feature_importance = model.feature_importances_
        feature_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Plotly Bar Chart for Feature Importance
        fig = px.bar(
            feature_df, 
            x='Feature', 
            y='Importance',
            title='Impact of Different Factors on Crop Yield',
            labels={'Importance': 'Relative Importance', 'Feature': 'Agricultural Factors'},
            color='Importance',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Insights Section
    st.markdown("### 💡 Insights and Recommendations")
    
    insights = [
        f"**Rainfall Impact**: The current rainfall of {rainfall} mm suggests {['moderate' if 800 <= rainfall <= 1500 else 'extreme'] [0]} water availability.",
        f"**Temperature Analysis**: At {temperature}°C, the temperature is {'optimal' if 25 <= temperature <= 30 else 'outside ideal range'} for crop growth.",
        f"**Soil Nutrient Status**: Nitrogen level of {nitrogen} indicates {['good' if 40 <= nitrogen <= 60 else 'sub-optimal'] [0]} nutrient conditions."
    ]
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    # Additional Resources Link
    st.markdown("""
    ### 🌐 Learn More
    [Agricultural Extension Resources](https://www.fao.org/agricultural-extension-services)
    """)

# Run the application
if __name__ == '__main__':
    main()