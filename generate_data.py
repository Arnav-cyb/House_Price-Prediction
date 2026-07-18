import os
import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples=3000, seed=42):
    np.random.seed(seed)
    
    # 1. Generate features
    area = np.random.normal(2200, 700, num_samples).astype(int)
    # Clip area to be reasonable
    area = np.clip(area, 600, 6000)
    
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=num_samples, p=[0.1, 0.25, 0.4, 0.2, 0.05])
    
    # Full bathrooms usually correlates with bedrooms
    full_bath = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        bd = bedrooms[i]
        if bd == 1:
            full_bath[i] = 1
        elif bd in [2, 3]:
            full_bath[i] = np.random.choice([1, 2], p=[0.3, 0.7])
        else:
            full_bath[i] = np.random.choice([2, 3, 4], p=[0.2, 0.6, 0.2])
            
    half_bath = np.random.choice([0, 1, 2], size=num_samples, p=[0.5, 0.4, 0.1])
    
    locations = ['Rural', 'Suburbs', 'Standard', 'Downtown', 'Premium']
    location = np.random.choice(locations, size=num_samples, p=[0.15, 0.3, 0.35, 0.1, 0.1])
    
    year_built = np.random.randint(1960, 2026, size=num_samples)
    
    # Garage area (some houses don't have garages)
    has_garage = np.random.choice([0, 1], size=num_samples, p=[0.15, 0.85])
    garage_area = np.zeros(num_samples)
    for i in range(num_samples):
        if has_garage[i]:
            # Garage area roughly scaled with house size
            garage_area[i] = np.random.normal(400, 150)
            garage_area[i] = np.clip(garage_area[i], 150, 900)
            
    # Basement area (some houses don't have basements)
    has_basement = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
    basement_area = np.zeros(num_samples)
    for i in range(num_samples):
        if has_basement[i]:
            basement_area[i] = np.random.normal(area[i] * 0.4, 200)
            basement_area[i] = np.clip(basement_area[i], 200, 2000)
            
    luxury_score = np.random.randint(1, 101, size=num_samples) # 1 to 100
    
    # 2. Generate target variable (Price) with some non-linearities and noise
    loc_multipliers = {
        'Rural': 0.7,
        'Suburbs': 1.0,
        'Standard': 1.2,
        'Downtown': 1.6,
        'Premium': 2.1
    }
    
    base_price = 45000
    noise = np.random.normal(0, 35000, num_samples)
    
    price = np.zeros(num_samples)
    for i in range(num_samples):
        # Calculate price based on features with non-linear interaction terms
        f_area = area[i] * 95
        f_beds = bedrooms[i] * 22000
        f_baths = (full_bath[i] * 28000) + (half_bath[i] * 12000)
        f_garage = garage_area[i] * 65
        f_basement = basement_area[i] * 45
        
        # Non-linear interaction between area and luxury
        f_lux_area = (luxury_score[i] ** 1.25) * (area[i] ** 0.6) * 12
        
        # Quadratic effect for age (very new has premium, very old historical has premium, mid age is discounted)
        age = 2026 - year_built[i]
        f_age = 80000 - 3000 * age + 40 * (age ** 2)
        
        # Base combination
        val = base_price + f_area + f_beds + f_baths + f_garage + f_basement + f_lux_area + f_age
        
        # Non-linear multiplicative location factor
        val *= loc_multipliers[location[i]]
        
        # Add non-linear noise scaled with house size
        price[i] = val + np.random.normal(0, val * 0.08)
        
    # Ensure price is at least 30,000
    price = np.clip(price, 30000, None)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Area_SqFt': area,
        'Bedrooms': bedrooms,
        'Full_Bathrooms': full_bath,
        'Half_Bathrooms': half_bath,
        'Location': location,
        'Year_Built': year_built,
        'Garage_Area': garage_area,
        'Basement_Area': basement_area,
        'Luxury_Score': luxury_score,
        'Price': price
    })
    
    # 3. Inject Missing Values (NaN)
    # We will inject some NaNs to show data preprocessing cleaning
    nan_mask_garage = np.random.choice([True, False], size=num_samples, p=[0.04, 0.96])
    df.loc[nan_mask_garage, 'Garage_Area'] = np.nan
    
    nan_mask_year = np.random.choice([True, False], size=num_samples, p=[0.02, 0.98])
    df.loc[nan_mask_year, 'Year_Built'] = np.nan
    
    nan_mask_luxury = np.random.choice([True, False], size=num_samples, p=[0.03, 0.97])
    df.loc[nan_mask_luxury, 'Luxury_Score'] = np.nan
    
    # 4. Inject Outliers
    # Inject some ridiculously high and low priced rows
    outlier_indices = np.random.choice(num_samples, size=35, replace=False)
    for idx in outlier_indices[:20]:
        # Extremely high price (Upper Outlier)
        df.loc[idx, 'Price'] = df.loc[idx, 'Price'] * 3.5
    for idx in outlier_indices[20:30]:
        # Extremely low price (Lower Outlier)
        df.loc[idx, 'Price'] = df.loc[idx, 'Price'] * 0.15
    for idx in outlier_indices[30:]:
        # Area outlier
        df.loc[idx, 'Area_SqFt'] = df.loc[idx, 'Area_SqFt'] * 3
        
    # 5. Inject Duplicates
    dup_rows = df.sample(n=25, random_state=42)
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # Re-shuffle
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    return df

if __name__ == '__main__':
    print("Generating synthetic housing data...")
    df = generate_synthetic_data(num_samples=3000)
    
    # Create directory if not exists
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/housing.csv', index=False)
    print(f"Data generated and saved to data/housing.csv. Shape: {df.shape}")
    print(df.head())
