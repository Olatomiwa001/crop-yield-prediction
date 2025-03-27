import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Enhanced Data Generation
def generate_advanced_crop_yield_data():
    np.random.seed(42)
    n_samples = 1000
    
    # More complex data generation
    data = {
        'Rainfall': np.random.normal(1200, 300, n_samples),
        'Temperature': np.random.normal(28, 3, n_samples),
        'Soil_Nitrogen': np.random.normal(50, 15, n_samples),
        'Soil_Phosphorus': np.random.normal(30, 10, n_samples),
        'Soil_Potassium': np.random.normal(40, 12, n_samples),
        'Farming_Practice': np.random.choice(['Traditional', 'Modern'], n_samples),
        'Region': np.random.choice(['North', 'South', 'Central'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Encode categorical variables
    df['Farming_Practice_Encoded'] = df['Farming_Practice'].map({'Traditional': 0, 'Modern': 1})
    df['Region_Encoded'] = df['Region'].map({'North': 0, 'Central': 1, 'South': 2})
    
    # More sophisticated yield calculation
    df['Crop_Yield'] = (
        0.4 * df['Rainfall'] / 1000 + 
        0.2 * df['Temperature'] / 30 + 
        0.15 * df['Soil_Nitrogen'] / 100 + 
        0.1 * df['Soil_Phosphorus'] / 50 + 
        0.05 * df['Soil_Potassium'] / 50 + 
        0.1 * df['Farming_Practice_Encoded'] + 
        0.05 * df['Region_Encoded'] + 
        np.random.normal(2, 0.5, n_samples)
    ).clip(1, 10)
    
    return df

# Visualization Functions
def create_visualizations(df):
    plt.figure(figsize=(20, 15))
    
    # 1. Correlation Heatmap
    plt.subplot(2, 2, 1)
    numeric_cols = ['Rainfall', 'Temperature', 'Soil_Nitrogen', 
                    'Soil_Phosphorus', 'Soil_Potassium', 'Crop_Yield']
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    
    # 2. Scatter Plot Matrix for Numeric Features
    plt.subplot(2, 2, 2)
    scatter_cols = ['Rainfall', 'Temperature', 'Soil_Nitrogen', 'Crop_Yield']
    scatter_df = df[scatter_cols]
    sns.scatterplot(data=scatter_df, x='Rainfall', y='Crop_Yield', hue='Temperature', palette='viridis')
    plt.title('Rainfall vs Crop Yield (Colored by Temperature)')
    
    # 3. Box Plot of Crop Yield by Region and Farming Practice
    plt.subplot(2, 2, 3)
    sns.boxplot(x='Region', y='Crop_Yield', hue='Farming_Practice', data=df)
    plt.title('Crop Yield by Region and Farming Practice')
    
    # 4. Distribution of Crop Yield
    plt.subplot(2, 2, 4)
    sns.histplot(df['Crop_Yield'], kde=True)
    plt.title('Distribution of Crop Yield')
    
    plt.tight_layout()
    plt.savefig('crop_yield_analysis.png')
    plt.close()

# Machine Learning Pipeline
def advanced_crop_yield_prediction(df):
    # Prepare features and target
    features = ['Rainfall', 'Temperature', 'Soil_Nitrogen', 
                'Soil_Phosphorus', 'Soil_Potassium', 
                'Farming_Practice_Encoded', 'Region_Encoded']
    X = df[features]
    y = df['Crop_Yield']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        results[name] = {
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        
        # Feature Importance Visualization
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title(f"Feature Importances - {name}")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(f'{name.replace(" ", "_")}_feature_importance.png')
        plt.close()
    
    return results

# Main Execution
def main():
    # Generate advanced dataset
    df = generate_advanced_crop_yield_data()
    
    # Create visualizations
    create_visualizations(df)
    
    # Run machine learning prediction
    results = advanced_crop_yield_prediction(df)
    
    # Print results
    print("Model Performance Metrics:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()