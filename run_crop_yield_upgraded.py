"""
Crop Yield Prediction - UPGRADED VERSION
Enhanced Neural Network with better accuracy
Author: Krishna
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("      CROP YIELD PREDICTION - UPGRADED VERSION")
print("      Author: Krishna")
print("="*70)

# =====================================================
# 1. CREATE ENHANCED DATASET
# =====================================================
print("\n[1] Creating Enhanced Dataset...")

def create_enhanced_dataset():
    np.random.seed(42)
    n_samples = 15000  # More samples for better learning
    
    crops = ['Wheat', 'Rice', 'Corn', 'Soybean', 'Barley', 'Cotton', 'Potato', 'Sugarcane']
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    data = {
        'Area': np.random.uniform(1000, 100000, n_samples),
        'Year': np.random.randint(1990, 2024, n_samples),
        'Rainfall_mm': np.random.uniform(50, 3000, n_samples),
        'Pesticides_tonnes': np.random.uniform(0.1, 500, n_samples),
        'Avg_temp': np.random.uniform(5, 40, n_samples),
        'Soil_quality': np.random.uniform(1, 10, n_samples),  # NEW: Soil quality 1-10
        'Fertilizer': np.random.uniform(0, 200, n_samples),  # NEW: Fertilizer usage
    }
    
    df = pd.DataFrame(data)
    df['Crop'] = np.random.choice(crops, n_samples)
    df['Region'] = np.random.choice(regions, n_samples)
    
    # Enhanced crop yield base with more realistic values
    crop_yield_base = {
        'Wheat': 3.5, 'Rice': 4.5, 'Corn': 9.0, 'Soybean': 2.8,
        'Barley': 3.8, 'Cotton': 1.8, 'Potato': 22.0, 'Sugarcane': 55.0
    }
    
    # Region multipliers
    region_multiplier = {'North': 1.1, 'South': 1.05, 'East': 1.0, 'West': 0.95, 'Central': 1.02}
    
    yield_values = []
    for idx, row in df.iterrows():
        base = crop_yield_base[row['Crop']]
        region_mult = region_multiplier[row['Region']]
        
        # Enhanced feature engineering
        rainfall_factor = (row['Rainfall_mm'] / 1000) * 0.25
        temp_factor = (1 - abs(row['Avg_temp'] - 25) / 25) * 0.25
        pesticide_factor = min(row['Pesticides_tonnes'] / 100, 1) * 0.08
        area_factor = (row['Area'] / 100000) * 0.15
        soil_factor = (row['Soil_quality'] / 10) * 0.15
        fertilizer_factor = (row['Fertilizer'] / 200) * 0.12
        
        # Add some realistic noise
        noise = np.random.normal(0, 0.3)
        
        yield_val = base * region_mult * (1 + rainfall_factor + temp_factor + 
                                           pesticide_factor + area_factor + 
                                           soil_factor + fertilizer_factor) + noise
        yield_values.append(max(0.1, yield_val))
    
    df['Yield'] = yield_values
    return df

df = create_enhanced_dataset()
print(f"    Enhanced Dataset: {df.shape[0]} samples, {df.shape[1]} columns")
print(f"    Crops: {list(df['Crop'].unique())}")
print(f"    Regions: {list(df['Region'].unique())}")
print(f"    New Features: Soil Quality, Fertilizer Usage")

# =====================================================
# 2. DATA EXPLORATION
# =====================================================
print("\n[2] Data Exploration...")
print(f"\n    Statistical Summary:")
print(df.describe().round(2).to_string().replace('\n', '\n    '))

print(f"\n    Crop Distribution:")
for crop, count in df['Crop'].value_counts().items():
    print(f"      {crop}: {count}")

# =====================================================
# 3. ADVANCED DATA PREPROCESSING
# =====================================================
print("\n[3] Advanced Data Preprocessing...")

# Encode categorical variables
le_crop = LabelEncoder()
le_region = LabelEncoder()

df['Crop_encoded'] = le_crop.fit_transform(df['Crop'])
df['Region_encoded'] = le_region.fit_transform(df['Region'])

print("    Crop Encoding:")
for i, crop in enumerate(le_crop.classes_):
    print(f"      {crop}: {i}")

print("    Region Encoding:")
for i, region in enumerate(le_region.classes_):
    print(f"      {region}: {i}")

# Feature engineering - create interaction features
feature_columns = ['Area', 'Year', 'Rainfall_mm', 'Pesticides_tonnes', 'Avg_temp', 
                   'Soil_quality', 'Fertilizer', 'Crop_encoded', 'Region_encoded']

# Add derived features
df['Rainfall_Temp_Ratio'] = df['Rainfall_mm'] / (df['Avg_temp'] + 1)
df['Area_Rainfall'] = df['Area'] * df['Rainfall_mm'] / 1000000
df['Soil_Fertilizer'] = df['Soil_quality'] * df['Fertilizer'] / 10

feature_columns_extended = feature_columns + ['Rainfall_Temp_Ratio', 'Area_Rainfall', 'Soil_Fertilizer']

X = df[feature_columns_extended]
y = df['Yield']

# Split with stratification-like approach
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"    Training samples: {len(X_train)}")
print(f"    Test samples: {len(X_test)}")

# Advanced scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"    Features: {len(feature_columns_extended)} (including engineered features)")

# =====================================================
# 4. BUILD UPGRADED ENSEMBLE MODEL
# =====================================================
print("\n[4] Building UPGRADED Neural Network Model...")

# Model 1: Enhanced MLP
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 16),  # Deeper network
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=64,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=25,
    random_state=42,
    verbose=False
)

# Model 2: Random Forest for ensemble
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Model 3: Gradient Boosting
gb = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    min_samples_split=5,
    random_state=42
)

# Create Voting Ensemble
ensemble = VotingRegressor([
    ('mlp', mlp),
    ('rf', rf),
    ('gb', gb)
])

print("    Ensemble Model Architecture:")
print("    ├── Neural Network (MLP): 256→128→64→32→16 neurons")
print("    ├── Random Forest: 100 trees, max_depth=15")
print("    └── Gradient Boosting: 100 estimators")

# =====================================================
# 5. TRAIN UPGRADED MODEL
# =====================================================
print("\n[5] Training UPGRADED Model...")

# Train individual models first
print("    Training Neural Network...")
mlp.fit(X_train_scaled, y_train)
print(f"    MLP iterations: {mlp.n_iter_}")

print("    Training Random Forest...")
rf.fit(X_train_scaled, y_train)

print("    Training Gradient Boosting...")
gb.fit(X_train_scaled, y_train)

# Train ensemble
print("    Training Ensemble...")
ensemble.fit(X_train_scaled, y_train)

print("    Training completed!")

# =====================================================
# 6. MODEL EVALUATION
# =====================================================
print("\n[6] Model Evaluation - UPGRADED RESULTS...")

# Evaluate all models
models = {
    'Neural Network': mlp,
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'ENSEMBLE': ensemble
}

results = []

print("\n" + "="*80)
print("                         PERFORMANCE COMPARISON")
print("="*80)
print(f"\n  {'Model':<20} {'Train R²':<12} {'Test R²':<12} {'Test RMSE':<12} {'Test MAE':<12}")
print("  " + "-"*80)

best_r2 = 0
best_model_name = ""

for name, model in models.items():
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    results.append({
        'model': name,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'rmse': test_rmse,
        'mae': test_mae
    })
    
    print(f"  {name:<20} {train_r2:<12.6f} {test_r2:<12.6f} {test_rmse:<12.6f} {test_mae:<12.6f}")
    
    if test_r2 > best_r2:
        best_r2 = test_r2
        best_model_name = name
        best_model = model

print("  " + "-"*80)
print(f"\n  🏆 BEST MODEL: {best_model_name} with R² = {best_r2:.6f}")
print("="*80)

# Cross-validation for best model
print("\n[7] Cross-Validation (5-Fold)...")
cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"    CV R² Scores: {cv_scores.round(6)}")
print(f"    Mean CV R²: {cv_scores.mean():.6f} (+/- {cv_scores.std()*2:.6f})")

# =====================================================
# 8. EXAMPLE PREDICTIONS
# =====================================================
print("\n[8] Example Predictions with UPGRADED Model:")

def predict_yield(area, year, rainfall, pesticides, temperature, crop_type, region, soil_quality=7, fertilizer=100):
    crop_encoded = le_crop.transform([crop_type])[0]
    region_encoded = le_region.transform([region])[0]
    
    rainfall_temp_ratio = rainfall / (temperature + 1)
    area_rainfall = area * rainfall / 1000000
    soil_fertilizer = soil_quality * fertilizer / 10
    
    features = np.array([[area, year, rainfall, pesticides, temperature, 
                         soil_quality, fertilizer, crop_encoded, region_encoded,
                         rainfall_temp_ratio, area_rainfall, soil_fertilizer]])
    
    features_scaled = scaler.transform(features)
    prediction = ensemble.predict(features_scaled)
    return prediction[0]

print("\n  " + "-"*85)
print(f"  {'Crop':<12} {'Region':<10} {'Area(ha)':<10} {'Rain(mm)':<10} {'Temp':<8} {'Yield':<10}")
print("  " + "-"*85)

test_cases = [
    {'area': 50000, 'year': 2024, 'rainfall': 1500, 'pesticides': 100, 'temp': 25, 'crop': 'Wheat', 'region': 'North', 'soil': 8, 'fert': 120},
    {'area': 30000, 'year': 2024, 'rainfall': 2000, 'pesticides': 50, 'temp': 30, 'crop': 'Rice', 'region': 'South', 'soil': 9, 'fert': 150},
    {'area': 40000, 'year': 2024, 'rainfall': 1000, 'pesticides': 150, 'temp': 28, 'crop': 'Corn', 'region': 'West', 'soil': 7, 'fert': 100},
    {'area': 25000, 'year': 2024, 'rainfall': 800, 'pesticides': 30, 'temp': 22, 'crop': 'Soybean', 'region': 'Central', 'soil': 6, 'fert': 80},
    {'area': 35000, 'year': 2024, 'rainfall': 1200, 'pesticides': 80, 'temp': 20, 'crop': 'Barley', 'region': 'East', 'soil': 7, 'fert': 90},
    {'area': 45000, 'year': 2024, 'rainfall': 1800, 'pesticides': 120, 'temp': 27, 'crop': 'Cotton', 'region': 'West', 'soil': 5, 'fert': 60},
    {'area': 20000, 'year': 2024, 'rainfall': 900, 'pesticides': 40, 'temp': 18, 'crop': 'Potato', 'region': 'North', 'soil': 9, 'fert': 180},
    {'area': 55000, 'year': 2024, 'rainfall': 2200, 'pesticides': 200, 'temp': 32, 'crop': 'Sugarcane', 'region': 'South', 'soil': 8, 'fert': 150},
]

for case in test_cases:
    prediction = predict_yield(
        case['area'], case['year'], case['rainfall'], 
        case['pesticides'], case['temp'], case['crop'],
        case['region'], case['soil'], case['fert']
    )
    print(f"  {case['crop']:<12} {case['region']:<10} {case['area']:<10} {case['rainfall']:<10} {case['temp']:<8} {prediction:<10.2f}")

print("  " + "-"*85)

# =====================================================
# 9. SAVE UPGRADED MODEL
# =====================================================
print("\n[9] Saving UPGRADED Model...")
import joblib

joblib.dump(ensemble, 'crop_yield_model_upgraded.joblib')
joblib.dump(mlp, 'crop_yield_mlp.joblib')
joblib.dump(rf, 'crop_yield_rf.joblib')
joblib.dump(gb, 'crop_yield_gb.joblib')
joblib.dump(scaler, 'scaler_upgraded.joblib')
joblib.dump(le_crop, 'label_encoder_crop.joblib')
joblib.dump(le_region, 'label_encoder_region.joblib')

feature_info = {
    'feature_columns': feature_columns_extended, 
    'crop_classes': list(le_crop.classes_),
    'region_classes': list(le_region.classes_)
}
joblib.dump(feature_info, 'feature_info_upgraded.joblib')

print("    ✅ crop_yield_model_upgraded.joblib (Ensemble)")
print("    ✅ crop_yield_mlp.joblib")
print("    ✅ crop_yield_rf.joblib")
print("    ✅ crop_yield_gb.joblib")
print("    ✅ scaler_upgraded.joblib")
print("    ✅ label_encoder_crop.joblib")
print("    ✅ label_encoder_region.joblib")
print("    ✅ feature_info_upgraded.joblib")

# =====================================================
# COMPLETION
# =====================================================
print("\n" + "="*70)
print("   🎉 CROP YIELD PREDICTION - UPGRADED VERSION COMPLETED!")
print("   Author: Krishna")
print("="*70)
print(f"\n   Best Model: {best_model_name}")
print(f"   Test R² Score: {best_r2:.6f} ({best_r2*100:.2f}%)")
print(f"   Cross-Validation R²: {cv_scores.mean():.6f}")
print("\n   Run with: python run_crop_yield_upgraded.py")

