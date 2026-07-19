import os
import argparse
import joblib
import pandas as pd
import numpy as np
import utils

def predict_price(area, bedrooms, full_bath, half_bath, location, year_built, garage_area, basement_area, luxury_score):
    """
    Loads saved unified production pipeline, and predicts price for given features.
    No manual scaling, imputation, encoding, or feature engineering is needed,
    as it's all handled inside the pipeline steps!
    """
    # Check if unified pipeline exists, train if it doesn't
    if not os.path.exists('models/housing_pipeline.pkl'):
        print("Production pipeline file not found. Automatically running training pipeline first...")
        import train
        train.main()
        
    # Load unified pipeline
    pipeline = joblib.load('models/housing_pipeline.pkl')
    
    # Create DataFrame representing raw user input
    raw_data = {
        'Area_SqFt': [area],
        'Bedrooms': [bedrooms],
        'Full_Bathrooms': [full_bath],
        'Half_Bathrooms': [half_bath],
        'Location': [location],
        'Year_Built': [year_built],
        'Garage_Area': [garage_area],
        'Basement_Area': [basement_area],
        'Luxury_Score': [luxury_score]
    }
    
    df_raw = pd.DataFrame(raw_data)
    
    # Predict directly using the pipeline
    predicted_price = pipeline.predict(df_raw)[0]
    
    # Calculate confidence interval (6% spread)
    confidence_spread = predicted_price * 0.06
    lower_bound = predicted_price - confidence_spread
    upper_bound = predicted_price + confidence_spread
    
    return predicted_price, lower_bound, upper_bound

def main():
    parser = argparse.ArgumentParser(description="Predict house price using trained production model pipeline.")
    parser.add_argument('--area', type=float, default=2000.0, help="Living area in SqFt (default: 2000)")
    parser.add_argument('--bedrooms', type=int, default=3, help="Number of bedrooms (default: 3)")
    parser.add_argument('--full-bath', type=int, default=2, help="Number of full bathrooms (default: 2)")
    parser.add_argument('--half-bath', type=int, default=1, help="Number of half bathrooms (default: 1)")
    parser.add_argument('--location', type=str, default='Suburbs', choices=['Rural', 'Suburbs', 'Standard', 'Downtown', 'Premium'], help="Location (default: Suburbs)")
    parser.add_argument('--year-built', type=float, default=2010.0, help="Year built (default: 2010)")
    parser.add_argument('--garage', type=float, default=400.0, help="Garage Area in SqFt (default: 400)")
    parser.add_argument('--basement', type=float, default=600.0, help="Basement Area in SqFt (default: 600)")
    parser.add_argument('--luxury', type=float, default=50.0, help="Luxury Score from 1 to 100 (default: 50)")
    
    args = parser.parse_args()
    
    print("\nPredicting house price for the following parameters:")
    print(f"  - Area: {args.area} sqft")
    print(f"  - Bedrooms: {args.bedrooms}")
    print(f"  - Bathrooms: {args.full_bath} Full, {args.half_bath} Half")
    print(f"  - Location: {args.location}")
    print(f"  - Year Built: {int(args.year_built)}")
    print(f"  - Garage Area: {args.garage} sqft")
    print(f"  - Basement Area: {args.basement} sqft")
    print(f"  - Luxury Score: {args.luxury}")
    
    pred, low, high = predict_price(
        args.area, args.bedrooms, args.full_bath, args.half_bath, 
        args.location, args.year_built, args.garage, args.basement, args.luxury
    )
    
    print("\n==========================================")
    print(f" PREDICTED PRICE:  INR {pred:,.2f}")
    print(f" Confidence Range: INR {low:,.2f} to INR {high:,.2f}")
    print("==========================================\n")

if __name__ == '__main__':
    main()
