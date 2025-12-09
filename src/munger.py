import pandas as pd
import numpy as np
from pathlib import Path

def load_data(filepath):
    return pd.read_csv(filepath)

def create_vehicle_age_feature(df):
    current_year = 2025
    df['Vehicle_Age'] = current_year - df['Model_Year']
    return df

def create_price_per_year_feature(df):
    df['Price_Per_Year'] = df['Price_USD'] / (df['Vehicle_Age'] + 1)
    return df

def create_mileage_category(df):
    mileage_percentiles = df['Kilometers_Driven'].quantile([0.25, 0.5, 0.75])
    def categorize_mileage(km):
        if km <= mileage_percentiles[0.25]:
            return 'Low'
        elif km <= mileage_percentiles[0.5]:
            return 'Medium'
        elif km <= mileage_percentiles[0.75]:
            return 'High'
        else:
            return 'Very High'
    df['Mileage_Category'] = df['Kilometers_Driven'].apply(categorize_mileage)
    return df

def create_price_category(df):
    price_percentiles = df['Price_USD'].quantile([0.33, 0.66, 0.9])
    def categorize_price(price):
        if price <= price_percentiles[0.33]:
            return 'Budget'
        elif price <= price_percentiles[0.66]:
            return 'Mid-range'
        elif price <= price_percentiles[0.9]:
            return 'Premium'
        else:
            return 'Luxury'
    df['Price_Category'] = df['Price_USD'].apply(categorize_price)
    return df

def create_engine_power_efficiency(df):
    df['Power_Per_CC'] = df['Max_Power_bhp'] / (df['Engine_CC'] / 1000)
    return df

def create_fuel_efficiency_power_ratio(df):
    df['Efficiency_to_Power'] = df['Mileage_kmpl'] / (df['Max_Power_bhp'] + 0.1)
    return df

def create_usage_intensity(df):
    df['Usage_Intensity'] = df['Kilometers_Driven'] / (df['Vehicle_Age'] + 1)
    return df

def create_depreciation_index(df):
    df['Depreciation_Rate'] = (df['Vehicle_Age'] * df['Max_Power_bhp']) / (df['Price_USD'] + 1)
    return df

def create_engine_displacement_category(df):
    def categorize_engine(cc):
        if cc < 2000:
            return 'Small'
        elif cc < 3500:
            return 'Medium'
        else:
            return 'Large'
    df['Engine_Size_Category'] = df['Engine_CC'].apply(categorize_engine)
    return df

def create_power_rating(df):
    power_percentiles = df['Max_Power_bhp'].quantile([0.25, 0.5, 0.75])
    def categorize_power(bhp):
        if bhp <= power_percentiles[0.25]:
            return 'Low'
        elif bhp <= power_percentiles[0.5]:
            return 'Medium'
        elif bhp <= power_percentiles[0.75]:
            return 'High'
        else:
            return 'Very High'
    df['Power_Rating'] = df['Max_Power_bhp'].apply(categorize_power)
    return df

def create_fuel_type_category(df):
    fuel_mapping = {
        'Diesel': 'Traditional',
        'Petrol': 'Traditional',
        'Hybrid': 'Eco-Friendly',
        'Electric': 'Eco-Friendly'
    }
    df['Fuel_Type_Category'] = df['Fuel_Type'].map(fuel_mapping)
    return df

def create_transmission_type_numeric(df):
    df['Is_Automatic'] = (df['Transmission'] == 'Automatic').astype(int)
    return df

def create_owner_history_numeric(df):
    owner_mapping = {'First': 0, 'Second': 1, 'Third': 2}
    df['Owner_History_Numeric'] = df['Owner_Type'].map(owner_mapping)
    return df

def create_seat_efficiency(df):
    df['Price_Per_Seat'] = df['Price_USD'] / (df['Seats'] + 1)
    return df

def munge_data(input_path, output_path):

    df = load_data(input_path)
    
    df = create_vehicle_age_feature(df)
    df = create_price_per_year_feature(df)
    df = create_mileage_category(df)
    df = create_price_category(df)
    df = create_engine_power_efficiency(df)
    df = create_fuel_efficiency_power_ratio(df)
    df = create_usage_intensity(df)
    df = create_depreciation_index(df)
    df = create_engine_displacement_category(df)
    df = create_power_rating(df)
    df = create_fuel_type_category(df)
    df = create_transmission_type_numeric(df)
    df = create_owner_history_numeric(df)
    df = create_seat_efficiency(df)
    
    df.to_csv(output_path, index=False)

def munge_data_with_encoding(input_path, output_path):
    """Munge data and apply one-hot encoding for categorical features"""
    
    df = load_data(input_path)
    
    # Apply all feature engineering
    df = create_vehicle_age_feature(df)
    df = create_price_per_year_feature(df)
    df = create_mileage_category(df)
    df = create_price_category(df)
    df = create_engine_power_efficiency(df)
    df = create_fuel_efficiency_power_ratio(df)
    df = create_usage_intensity(df)
    df = create_depreciation_index(df)
    df = create_engine_displacement_category(df)
    df = create_power_rating(df)
    df = create_fuel_type_category(df)
    df = create_transmission_type_numeric(df)
    df = create_owner_history_numeric(df)
    df = create_seat_efficiency(df)
    
    # Drop only the original raw categorical columns (not the engineered ones)
    # Keep Brand, Fuel_Type, Transmission, Owner_Type for one-hot encoding
    # Also keep the engineered categorical features for one-hot encoding
    columns_to_drop = []  # Don't drop anything, just encode everything
    
    # One-hot encode all categorical columns, drop_first=False to keep all categories
    df = pd.get_dummies(df, drop_first=False)
    
    df.to_csv(output_path, index=False)
    print(f"Munged data with encoding saved to {output_path}")
    print(f"Total features: {df.shape[1] - 1} (excluding Price_USD)")  # -1 for Price_USD

if __name__ == "__main__":
    input_file = Path(__file__).parent.parent / "data" / "car_price_dataset_medium.csv"
    output_file = Path(__file__).parent.parent / "data" / "car_price_dataset_munged.csv"
    
    # Create the new encoded version
    munge_data_with_encoding(input_file, Path(__file__).parent.parent / "data" / "car_price_dataset_munged_encoded.csv")