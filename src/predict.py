import pandas as pd
import torch
import pickle
import numpy as np
from src.model import CarPricePredictor

# Load your trained model
model = CarPricePredictor(input_features=52, hidden_size=32)
model.load_state_dict(torch.load('models/car_price_model.pth'))
model.eval()

# Load the scalers and feature columns from training
try:
    with open('models/scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('models/scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    with open('models/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    print(f"Loaded {len(feature_columns)} feature columns from training")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Make sure you've run training first to generate scaler files")
    exit(1)

# Create a new car as a dictionary
new_car = {
    'Car_ID': 500,
    'Brand': 'Toyota',
    'Model_Year': 2020,
    'Kilometers_Driven': 45000,
    'Fuel_Type': 'Hybrid',
    'Transmission': 'Automatic',
    'Owner_Type': 'First',
    'Engine_CC': 2400,
    'Max_Power_bhp': 180,
    'Mileage_kmpl': 18.5,
    'Seats': 5,
    'Price_USD': 50000  # Use a realistic value for feature engineering
}

# Convert to DataFrame
new_car_df = pd.DataFrame([new_car])

# Apply all feature engineering
from src.munger import (
    create_vehicle_age_feature, 
    create_price_per_year_feature,
    create_mileage_category,
    create_price_category,
    create_engine_power_efficiency,
    create_fuel_efficiency_power_ratio,
    create_usage_intensity,
    create_depreciation_index,
    create_engine_displacement_category,
    create_power_rating,
    create_fuel_type_category,
    create_transmission_type_numeric,
    create_owner_history_numeric,
    create_seat_efficiency
)

new_car_df = create_vehicle_age_feature(new_car_df)
new_car_df = create_price_per_year_feature(new_car_df)
new_car_df = create_mileage_category(new_car_df)
new_car_df = create_price_category(new_car_df)
new_car_df = create_engine_power_efficiency(new_car_df)
new_car_df = create_fuel_efficiency_power_ratio(new_car_df)
new_car_df = create_usage_intensity(new_car_df)
new_car_df = create_depreciation_index(new_car_df)
new_car_df = create_engine_displacement_category(new_car_df)
new_car_df = create_power_rating(new_car_df)
new_car_df = create_fuel_type_category(new_car_df)
new_car_df = create_transmission_type_numeric(new_car_df)
new_car_df = create_owner_history_numeric(new_car_df)
new_car_df = create_seat_efficiency(new_car_df)

# Drop the target column
X_new = new_car_df.drop(columns=['Price_USD'])

# Apply one-hot encoding with drop_first=False to match training
X_new = pd.get_dummies(X_new, drop_first=False)

# Reindex to match training columns exactly
for col in feature_columns:
    if col not in X_new.columns:
        X_new[col] = 0

# Keep only columns from training in the same order
X_new = X_new[feature_columns]

print(f"Features shape: {X_new.shape}")
print(f"Expected features: {len(feature_columns)}")

# Scale features using the training scaler
X_new_scaled = scaler_X.transform(X_new.values)

# Convert to tensor and predict
X_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

with torch.no_grad():
    predicted_price_scaled = model(X_tensor).item()

# Inverse transform to get actual price
predicted_price = scaler_y.inverse_transform([[predicted_price_scaled]])[0, 0]

print(f"\nPredicted Price for {new_car['Brand']}: ${predicted_price:,.2f}")
