import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objs as go

# Enhanced Custom CSS for better UX
def set_custom_style():
    st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp {
        max-width: 1100px;
        margin: 0 auto;
        padding: 10px;
    }
    .app-header {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 25px;
    }
    .app-title {
        color: #2C3E50;
        font-size: 36px;
        font-weight: bold;
        margin: 0;
        line-height: 1.2;
    }
    .app-logo {
        font-size: 36px;
    }
    .info-box {
        background-color: #F1F8FF;
        border-left: 5px solid #0078D7;
        padding: 15px;
        margin-bottom: 25px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .prediction-card {
        background-color: #27AE60;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .prediction-value {
        font-size: 38px;
        margin: 10px 0;
    }
    .insight-item {
        background-color: #F8F9FA;
        padding: 12px 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 4px solid #3498DB;
    }
    .section-header {
        color: #34495E;
        font-size: 24px;
        font-weight: 600;
        margin: 30px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #E9ECEF;
    }
    .sidebar-section {
        background-color: #F5F7FA;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .sidebar-header {
        font-weight: 600;
        color: #2C3E50;
        margin-bottom: 10px;
    }
    /* Mobile responsive adjustments */
    @media (max-width: 768px) {
        .app-title {
            font-size: 28px;
        }
        .prediction-card {
            font-size: 22px;
        }
        .prediction-value {
            font-size: 32px;
        }
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

# Generate insight recommendations based on inputs
def generate_insights(rainfall, temperature, nitrogen, phosphorus, potassium, practice, importance_df):
    insights = []
    
    # Most important factor insight
    top_factor = importance_df.iloc[0]['Feature']
    insights.append({
        'title': f"Focus on {top_factor}",
        'content': f"{top_factor} is the most influential factor for your crop yield. Consider prioritizing management of this factor.",
        'color': '#3498DB'
    })
    
    # Rainfall insight
    if rainfall < 800:
        insights.append({
            'title': 'Low Rainfall Alert',
            'content': f"Current rainfall ({rainfall} mm) is below optimal levels. Consider supplementary irrigation or drought-resistant varieties.",
            'color': '#E74C3C'
        })
    elif rainfall > 1600:
        insights.append({
            'title': 'High Rainfall Management',
            'content': f"Rainfall ({rainfall} mm) is above optimal range. Consider improved drainage systems to prevent waterlogging.",
            'color': '#F39C12'
        })
    else:
        insights.append({
            'title': 'Optimal Rainfall',
            'content': f"Current rainfall of {rainfall} mm is within optimal range for crop growth.",
            'color': '#27AE60'
        })
    
    # Temperature insight
    if temperature < 25 or temperature > 30:
        insights.append({
            'title': 'Temperature Adjustment Needed',
            'content': f"Current temperature ({temperature}°C) is outside ideal range (25-30°C). Consider timing adjustments or heat-tolerant varieties.",
            'color': '#F39C12'
        })
    
    # Soil nutrients insight
    nutrients = {'Nitrogen': nitrogen, 'Phosphorus': phosphorus, 'Potassium': potassium}
    deficient = [n for n, v in nutrients.items() if v < 35]
    if deficient:
        insights.append({
            'title': 'Nutrient Deficiency',
            'content': f"Low levels of {', '.join(deficient)}. Consider appropriate fertilization to improve soil quality.",
            'color': '#E74C3C'
        })
    
    # Farming practice insight
    if practice == 'Traditional':
        insights.append({
            'title': 'Consider Modern Techniques',
            'content': "Switching to modern farming practices could potentially increase yield by approximately 10%.",
            'color': '#9B59B6'
        })
    
    return insights

# Main Streamlit application
def main():
    # Set custom styling
    set_custom_style()
    
    # App Title and Introduction with improved layout
    st.markdown("""
    <div class="app-header">
        <div class="app-logo">🌾</div>
        <h1 class="app-title">Nigerian Crop Yield Predictor</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Informative Introduction with better styling
    st.markdown("""
    <div class="info-box">
    <h3>🚜 Agricultural Intelligence Platform</h3>
    <p>This advanced prediction tool helps farmers and agricultural professionals make data-driven decisions for improved farming strategies.</p>
    
    <ul>
    <li>Estimate potential crop yields based on environmental conditions</li>
    <li>Understand key factors influencing agricultural productivity</li>
    <li>Receive tailored recommendations for your specific farming context</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["Prediction", "Learn More"])
    
    with tab1:
        # Sidebar for Detailed Inputs with improved organization
        with st.sidebar:
            st.markdown('<div class="sidebar-header">🌱 Input Parameters</div>', unsafe_allow_html=True)
            
            # Environmental Conditions section
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### Environmental Conditions")
            rainfall = st.slider(
                'Rainfall (mm)', 
                min_value=500, 
                max_value=2000, 
                value=1200, 
                help="Total rainfall affects water availability for crops."
            )
            
            temperature = st.slider(
                'Average Temperature (°C)', 
                min_value=20, 
                max_value=35, 
                value=28, 
                help="Temperature impacts plant metabolism and development."
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Soil Nutrients section
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### Soil Nutrients")
            nitrogen = st.slider(
                'Soil Nitrogen Level', 
                min_value=20, 
                max_value=80, 
                value=50, 
                help="Nitrogen is crucial for leaf growth."
            )
            
            phosphorus = st.slider(
                'Soil Phosphorus Level', 
                min_value=20, 
                max_value=80, 
                value=40, 
                help="Phosphorus supports root development."
            )
            
            potassium = st.slider(
                'Soil Potassium Level', 
                min_value=20, 
                max_value=80, 
                value=45, 
                help="Potassium enhances overall plant health."
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Farming Practice section
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### Farming Method")
            farming_practice = st.selectbox(
                'Farming Practice', 
                ['Traditional', 'Modern'], 
                help="Modern practices often include improved techniques."
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate and prepare data
        df = generate_detailed_crop_yield_data()
        
        # Prepare features
        features = ['Rainfall', 'Temperature', 'Soil_Nitrogen', 'Soil_Phosphorus', 'Soil_Potassium']
        X = df[features]
        y = df['Crop_Yield']
        
        # Train model
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        
        # Feature Importance calculation for insights
        feature_importance = model.feature_importances_
        feature_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Predict
        input_data = np.array([[rainfall, temperature, nitrogen, phosphorus, potassium]])
        prediction = model.predict(input_data)
        
        # Prediction Display with improved styling
        st.markdown("""
        <div class="prediction-card">
            <div>Predicted Crop Yield:</div>
            <div class="prediction-value">{:.2f} tons/hectare</div>
        </div>
        """.format(prediction[0]), unsafe_allow_html=True)
        
        # Feature Importance Visualization - improved for better mobile viewing
        st.markdown('<div class="section-header">Impact of Different Factors</div>', unsafe_allow_html=True)
        
        # Plotly Bar Chart with better mobile responsiveness
        fig = px.bar(
            feature_df, 
            x='Feature', 
            y='Importance',
            title='Factors Influencing Crop Yield',
            labels={'Importance': 'Relative Importance', 'Feature': 'Agricultural Factors'},
            color='Importance',
            color_continuous_scale='viridis',
            height=400  # Fixed height for better mobile viewing
        )
        
        # Improve the figure layout for better readability
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="",
            yaxis_title="Impact Level",
            title_font_size=18,
            font=dict(family="Segoe UI", size=14),
            legend_title_font_size=14
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate insights based on inputs and visualize
        insights = generate_insights(rainfall, temperature, nitrogen, phosphorus, potassium, 
                                    farming_practice, feature_df)
        
        # Display insights with improved styling
        st.markdown('<div class="section-header">💡 Analysis & Recommendations</div>', unsafe_allow_html=True)
        
        for insight in insights:
            st.markdown(f"""
            <div class="insight-item" style="border-left-color: {insight['color']}">
                <strong>{insight['title']}</strong><br>
                {insight['content']}
            </div>
            """, unsafe_allow_html=True)
    
    # Additional Resources Tab with more organized information
    with tab2:
        st.markdown('<div class="section-header">Resources & Information</div>', unsafe_allow_html=True)
        
        # More detailed information about crop factors
        st.markdown("### Understanding Crop Factors")
        st.markdown("""
        - **Rainfall**: Water is essential for nutrient transport and photosynthesis.
          Optimal range: 900-1500mm annually depending on crop type.
          
        - **Temperature**: Affects germination, growth rate, and metabolism.
          Most crops prefer 22-30°C for optimal growth.
          
        - **Soil Nutrients**: The three primary nutrients are:
          * **Nitrogen (N)**: Essential for leaf growth and protein formation
          * **Phosphorus (P)**: Critical for root development and flowering
          * **Potassium (K)**: Improves overall plant health and stress resistance
        """)
        
        # Add a "Further Reading" section
        st.markdown("### Further Reading")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            - [FAO Agricultural Extension Services](https://www.fao.org/agricultural-extension-services)
            - [Nigerian Agricultural Research Council](https://arcn.gov.ng/)
            - [Crop Modeling Best Practices](https://www.nature.com/articles/s41597-020-0453-3)
            """)
        with cols[1]:
            st.markdown("""
            - [Soil Health Management Guide](https://www.soilhealthinstitute.org/)
            - [Weather Patterns and Crop Growth](https://www.wmo.int/pages/prog/wcp/agm/)
            - [Modern Farming Techniques](https://www.fao.org/agriculture/crops/thematic-sitemap/theme/spi/en/)
            """)

# Run the application
if __name__ == '__main__':
    main()