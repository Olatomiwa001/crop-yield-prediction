import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# Enhanced mobile-first CSS
def set_custom_style():
    st.markdown("""
    <style>
    /* Base styles */
    .main {
        background-color: #f7f9fc;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 0 !important;
    }
    .stApp {
        max-width: 100%;
        margin: 0 auto;
        padding: 5px !important;
    }
    
    /* Mobile-first approach */
    .app-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 15px;
    }
    .app-title {
        color: #2C3E50;
        font-size: 24px;
        font-weight: bold;
        margin: 0;
        line-height: 1.2;
    }
    .app-logo {
        font-size: 24px;
    }
    .info-box {
        background-color: #F1F8FF;
        border-left: 5px solid #0078D7;
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 5px;
        font-size: 14px;
    }
    .info-box h3 {
        font-size: 16px;
        margin-bottom: 8px;
    }
    .info-box ul {
        padding-left: 20px;
        margin: 5px 0;
    }
    .prediction-card {
        background-color: #27AE60;
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .prediction-value {
        font-size: 28px;
        margin: 8px 0;
    }
    .insight-item {
        background-color: #F8F9FA;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 8px;
        border-left: 4px solid #3498DB;
        font-size: 14px;
    }
    .section-header {
        color: #34495E;
        font-size: 18px;
        font-weight: 600;
        margin: 20px 0 10px 0;
        padding-bottom: 5px;
        border-bottom: 2px solid #E9ECEF;
    }
    /* Mobile optimizations */
    .compact-slider .stSlider {
        min-height: auto !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    .compact-text p {
        margin-bottom: 0.5rem !important;
        font-size: 14px !important;
    }
    .compact-widget {
        margin-bottom: 0.5rem !important;
    }
    /* Hide elements on small screens */
    .mobile-hidden {
        display: none;
    }
    /* Override default Streamlit spacings */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px !important;
        font-size: 14px !important;
    }
    
    /* Desktop overrides - applied only on larger screens */
    @media (min-width: 768px) {
        .app-title {
            font-size: 36px;
        }
        .app-logo {
            font-size: 36px;
        }
        .prediction-card {
            font-size: 24px;
            padding: 20px;
        }
        .prediction-value {
            font-size: 36px;
        }
        .section-header {
            font-size: 22px;
        }
        .info-box {
            font-size: 16px;
            padding: 15px;
        }
        .info-box h3 {
            font-size: 20px;
        }
        .insight-item {
            font-size: 16px;
            padding: 12px 15px;
        }
        /* Show elements on larger screens */
        .mobile-hidden {
            display: block;
        }
        /* Adjust padding */
        .stApp {
            padding: 10px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Generate synthetic data
def generate_detailed_crop_yield_data():
    np.random.seed(42)
    n_samples = 1500
    data = {
        'Rainfall': np.random.normal(1200, 300, n_samples),
        'Temperature': np.random.normal(28, 3, n_samples),
        'Soil_Nitrogen': np.random.normal(50, 15, n_samples),
        'Soil_Phosphorus': np.random.normal(40, 10, n_samples),
        'Soil_Potassium': np.random.normal(45, 12, n_samples),
        'Farming_Practice': np.random.choice(['Traditional', 'Modern'], n_samples)
    }
    df = pd.DataFrame(data)
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

# Generate insights based on inputs
def generate_insights(rainfall, temperature, nitrogen, phosphorus, potassium, practice, importance_df):
    insights = []
    top_factor = importance_df.iloc[0]['Feature']
    insights.append({
        'title': f"Focus on {top_factor}",
        'content': f"{top_factor} is your most influential factor.",
        'color': '#3498DB'
    })
    
    if rainfall < 800:
        insights.append({
            'title': 'Low Rainfall',
            'content': f"Consider irrigation ({rainfall}mm is below optimal).",
            'color': '#E74C3C'
        })
    elif rainfall > 1600:
        insights.append({
            'title': 'High Rainfall',
            'content': f"Improve drainage to prevent water damage.",
            'color': '#F39C12'
        })
    
    if temperature < 25 or temperature > 30:
        insights.append({
            'title': 'Temperature Issue',
            'content': f"{temperature}°C is outside ideal range (25-30°C).",
            'color': '#F39C12'
        })
    
    nutrients = {'Nitrogen': nitrogen, 'Phosphorus': phosphorus, 'Potassium': potassium}
    deficient = [n for n, v in nutrients.items() if v < 35]
    if deficient:
        insights.append({
            'title': 'Nutrient Deficiency',
            'content': f"Low {', '.join(deficient)}. Add appropriate fertilizer.",
            'color': '#E74C3C'
        })
    
    if practice == 'Traditional':
        insights.append({
            'title': 'Modern Methods',
            'content': "Modern farming could increase yield by ~10%.",
            'color': '#9B59B6'
        })
    
    return insights

# Main app function with mobile optimizations
def main():
    # Apply mobile-optimized styling
    set_custom_style()
    
    # Simplified header for mobile
    st.markdown("""
    <div class="app-header">
        <div class="app-logo">🌾</div>
        <h1 class="app-title">Nigerian Crop Yield Predictor</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Create compact tabs for mobile
    tab1, tab2 = st.tabs(["Predict", "Info"])
    
    with tab1:
        # Use columns for better mobile layout - input on top, results below
        input_col1, input_col2 = st.columns([1, 1])
        
        with input_col1:
            st.markdown('<div class="section-header">Environment</div>', unsafe_allow_html=True)
            st.markdown('<div class="compact-text">', unsafe_allow_html=True)
            rainfall = st.slider('Rainfall (mm)', 500, 2000, 1200, key="compact_rainfall")
            temperature = st.slider('Temperature (°C)', 20, 35, 28, key="compact_temp")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with input_col2:
            st.markdown('<div class="section-header">Soil & Method</div>', unsafe_allow_html=True)
            st.markdown('<div class="compact-text">', unsafe_allow_html=True)
            nitrogen = st.slider('Nitrogen', 20, 80, 50, key="compact_n")
            phosphorus = st.slider('Phosphorus', 20, 80, 40, key="compact_p")
            potassium = st.slider('Potassium', 20, 80, 45, key="compact_k")
            farming_practice = st.selectbox('Farming Practice', ['Traditional', 'Modern'], key="compact_practice")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate and prepare data
        df = generate_detailed_crop_yield_data()
        features = ['Rainfall', 'Temperature', 'Soil_Nitrogen', 'Soil_Phosphorus', 'Soil_Potassium']
        X = df[features]
        y = df['Crop_Yield']
        
        # Train model and predict
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        input_data = np.array([[rainfall, temperature, nitrogen, phosphorus, potassium]])
        prediction = model.predict(input_data)
        
        # Calculate feature importance
        feature_importance = model.feature_importances_
        feature_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Prediction Display - mobile optimized
        st.markdown("""
        <div class="prediction-card">
            <div>Predicted Yield:</div>
            <div class="prediction-value">{:.2f} tons/hectare</div>
        </div>
        """.format(prediction[0]), unsafe_allow_html=True)
        
        # Compact visualization for mobile
        st.markdown('<div class="section-header">Key Factors</div>', unsafe_allow_html=True)
        
        # Create a simpler horizontal bar chart optimized for mobile
        fig = px.bar(
            feature_df, 
            y='Feature',  # Switch to horizontal bars for better mobile viewing
            x='Importance',
            orientation='h',  # Horizontal layout
            color='Importance',
            color_continuous_scale='viridis',
            height=250  # Shorter height for mobile
        )
        
        # Optimize figure layout for small screens
        fig.update_layout(
            margin=dict(l=10, r=10, t=5, b=10),
            yaxis_title="",
            xaxis_title="Impact",
            font=dict(family="Segoe UI", size=12),
            coloraxis_showscale=False  # Hide color scale to save space
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Generate insights
        insights = generate_insights(rainfall, temperature, nitrogen, phosphorus, potassium, 
                                    farming_practice, feature_df)
        
        # Display insights in a more compact format for mobile
        st.markdown('<div class="section-header">Quick Recommendations</div>', unsafe_allow_html=True)
        
        # Use columns for compact insight display
        for i in range(0, len(insights), 2):
            cols = st.columns([1, 1])
            cols[0].markdown(f"""
            <div class="insight-item" style="border-left-color: {insights[i]['color']}">
                <strong>{insights[i]['title']}</strong><br>
                {insights[i]['content']}
            </div>
            """, unsafe_allow_html=True)
            
            if i+1 < len(insights):
                cols[1].markdown(f"""
                <div class="insight-item" style="border-left-color: {insights[i+1]['color']}">
                    <strong>{insights[i+1]['title']}</strong><br>
                    {insights[i+1]['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Info tab with very condensed content for mobile
    with tab2:
        st.markdown('<div class="section-header">About This Tool</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <p>This tool helps predict crop yields based on environmental factors and soil nutrients. It uses a machine learning model trained on Nigerian agricultural data.</p>
        
        <strong>Key factors influencing yield:</strong>
        <ul>
        <li><strong>Rainfall:</strong> Optimal 900-1500mm annually</li>
        <li><strong>Temperature:</strong> Best at 22-30°C</li>
        <li><strong>Soil Nutrients:</strong> N, P, K levels affect growth</li>
        <li><strong>Farming Method:</strong> Modern techniques can improve yield</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">Resources</div>', unsafe_allow_html=True)
        st.markdown("""
        - [FAO Agricultural Services](https://www.fao.org)
        - [Nigerian Agric Research](https://arcn.gov.ng/)
        - [Soil Health Guide](https://www.soilhealthinstitute.org/)
        """)

# Run the application
if __name__ == '__main__':
    main()