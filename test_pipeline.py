import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import utils

def test_dataframe_imputer():
    """Test that DataFrameImputer correctly handles missing data."""
    print("Running test_dataframe_imputer...")
    df = pd.DataFrame({
        'Area_SqFt': [2000.0, np.nan, 2200.0],
        'Location': ['Suburbs', 'Rural', np.nan]
    })
    
    imputer = utils.DataFrameImputer()
    imputer.fit(df)
    
    # Assert fitted values are computed
    assert imputer.medians_['Area_SqFt'] == 2100.0
    assert imputer.modes_['Location'] in ['Suburbs', 'Rural']
    
    # Assert transformation resolves NaNs
    transformed = imputer.transform(df)
    assert not transformed.isnull().any().any()
    assert transformed.loc[1, 'Area_SqFt'] == 2100.0
    print("test_dataframe_imputer passed.")

def test_outlier_capper():
    """Test that OutlierCapper caps extreme values based on IQR bounds."""
    print("Running test_outlier_capper...")
    # Data with a huge outlier (10000.0)
    df = pd.DataFrame({
        'Area_SqFt': [1000.0, 1200.0, 1100.0, 1300.0, 10000.0]
    })
    
    capper = utils.OutlierCapper(cols_to_cap=['Area_SqFt'])
    capper.fit(df)
    
    # Transform
    transformed = capper.transform(df)
    
    # The extreme outlier 10000.0 must be capped to the computed upper bound
    upper_bound = capper.bounds_['Area_SqFt'][1]
    assert transformed.loc[4, 'Area_SqFt'] == upper_bound
    assert transformed.loc[0, 'Area_SqFt'] == 1000.0 # Normal value untouched
    print("test_outlier_capper passed.")

def test_feature_engineer():
    """Test feature engineering calculations and column drops."""
    print("Running test_feature_engineer...")
    df = pd.DataFrame({
        'Area_SqFt': [2000.0],
        'Bedrooms': [3],
        'Full_Bathrooms': [2],
        'Half_Bathrooms': [1],
        'Year_Built': [2010.0],
        'Garage_Area': [400.0],
        'Basement_Area': [600.0],
        'Luxury_Score': [85.0]
    })
    
    engineer = utils.FeatureEngineer(current_year=2026)
    transformed = engineer.fit_transform(df)
    
    # Assert engineered features exist
    assert 'House_Age' in transformed.columns
    assert 'Total_Bathrooms' in transformed.columns
    assert 'Total_Area' in transformed.columns
    assert 'Luxury_Band' in transformed.columns
    
    # Assert column drop
    assert 'Year_Built' not in transformed.columns
    
    # Assert correct math
    assert transformed.loc[0, 'House_Age'] == 16.0
    assert transformed.loc[0, 'Total_Bathrooms'] == 2.5
    assert transformed.loc[0, 'Total_Area'] == 3000.0
    assert transformed.loc[0, 'Luxury_Band'] == 1
    print("test_feature_engineer passed.")

def test_anomaly_detector():
    """Test that InputAnomalyDetector correctly flags out-of-bounds input parameters."""
    print("Running test_anomaly_detector...")
    # Training data
    df_train = pd.DataFrame({
        'Area_SqFt': [1000.0, 1500.0, 2000.0, 2500.0],
        'Location': ['Rural', 'Suburbs', 'Standard', 'Premium']
    })
    
    detector = utils.InputAnomalyDetector()
    detector.fit(df_train)
    
    # Normal input
    normal_input = {'Area_SqFt': [1800.0], 'Location': ['Standard']}
    warnings_normal = detector.check_anomaly(normal_input)
    assert len(warnings_normal) == 0
    
    # Anomaly numerical input
    anomalous_num = {'Area_SqFt': [5000.0], 'Location': ['Standard']}
    warnings_num = detector.check_anomaly(anomalous_num)
    assert len(warnings_num) > 0
    
    # Anomaly categorical input
    anomalous_cat = {'Area_SqFt': [1800.0], 'Location': ['InvalidLoc']}
    warnings_cat = detector.check_anomaly(anomalous_cat)
    assert len(warnings_cat) > 0
    print("test_anomaly_detector passed.")

def test_preprocessing_pipeline_dimensions():
    """Test that the complete preprocessing pipeline returns expected array shapes."""
    print("Running test_preprocessing_pipeline_dimensions...")
    df = pd.DataFrame({
        'Area_SqFt': [2000.0, 2500.0, 1800.0],
        'Bedrooms': [3, 4, 3],
        'Full_Bathrooms': [2, 2, 1],
        'Half_Bathrooms': [1, 1, 0],
        'Location': ['Suburbs', 'Premium', 'Rural'],
        'Year_Built': [2010.0, 2018.0, 1995.0],
        'Garage_Area': [400.0, 500.0, 300.0],
        'Basement_Area': [600.0, 800.0, 500.0],
        'Luxury_Score': [50.0, 90.0, 30.0]
    })
    
    pipeline = utils.get_preprocessing_pipeline()
    # Fit pipeline
    processed_arr = pipeline.fit_transform(df)
    
    # 11 numerical features scaled + One-hot locations categories (3 unique categories in training = 3 elements)
    # Shape should be (3, 11 + 3) = (3, 14)
    assert processed_arr.shape == (3, 14)
    print("test_preprocessing_pipeline_dimensions passed.")

def test_sqlite_database_utilities():
    """Test SQLite initialization, CRUD operations, prediction logging, and feedback loops."""
    print("Running test_sqlite_database_utilities...")
    import tempfile
    
    # Use a temporary file for test database
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(temp_db_fd)
    
    try:
        # 1. Test database initialization
        utils.init_db(db_path=temp_db_path)
        
        # Verify tables are created by fetching properties (should be empty except for seeded items)
        # Note: Since the database seeds from housing.csv if empty, let's verify it seeds correctly
        props = utils.fetch_properties_from_db(db_path=temp_db_path)
        assert len(props) > 0, "Seeding failed during DB initialization!"
        initial_count = len(props)
        
        # 2. Test CRUD - Create
        new_prop = {
            'Area_SqFt': 1850.0, 'Bedrooms': 3, 'Full_Bathrooms': 2, 'Half_Bathrooms': 1,
            'Location': 'Suburbs', 'Year_Built': 2012.0, 'Garage_Area': 350.0, 'Basement_Area': 400.0, 'Luxury_Score': 60.0,
            'Price': 450000.0
        }
        utils.add_property_to_db(new_prop, db_path=temp_db_path)
        
        props_after_add = utils.fetch_properties_from_db(db_path=temp_db_path)
        assert len(props_after_add) == initial_count + 1, "Failed to add property record!"
        
        # Verify details of added property (it should be the first item since we query DESC by id)
        latest_prop = props_after_add.iloc[0]
        assert latest_prop['Area_SqFt'] == 1850.0
        assert latest_prop['Location'] == 'Suburbs'
        assert latest_prop['Price'] == 450000.0
        
        # 3. Test CRUD - Delete
        latest_id = latest_prop['id']
        utils.delete_property_from_db(latest_id, db_path=temp_db_path)
        props_after_del = utils.fetch_properties_from_db(db_path=temp_db_path)
        assert len(props_after_del) == initial_count, "Failed to delete property record!"
        
        # 4. Test Prediction Logging & Feedback
        test_inputs = {
            'Area_SqFt': 2200.0, 'Bedrooms': 4, 'Full_Bathrooms': 3, 'Half_Bathrooms': 1,
            'Location': 'Premium', 'Year_Built': 2018.0, 'Garage_Area': 400.0, 'Basement_Area': 600.0, 'Luxury_Score': 80.0
        }
        pred_id = utils.log_prediction_to_db(test_inputs, 850000.0, 800000.0, 900000.0, 'Stacking Ensemble', db_path=temp_db_path)
        assert pred_id is not None and pred_id > 0, "Prediction logging failed!"
        
        # Submit feedback
        utils.submit_prediction_feedback(pred_id, 5, "Valuation matches regional listing comparisons.", db_path=temp_db_path)
        
        # Verify prediction log entry
        pred_logs = utils.fetch_predictions_log(db_path=temp_db_path)
        assert len(pred_logs) == 1, "Failed to fetch prediction log records!"
        latest_log = pred_logs.iloc[0]
        assert latest_log['id'] == pred_id
        assert latest_log['Predicted_Price'] == 850000.0
        assert latest_log['Feedback_Rating'] == 5
        assert latest_log['Feedback_Comment'] == "Valuation matches regional listing comparisons."
        
        print("test_sqlite_database_utilities passed.")
        
    finally:
        # Clean up temporary database file
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

def test_multi_model_and_stats():
    """Test get_model_mae stats lookup and all_models serving pipelines."""
    print("Running test_multi_model_and_stats...")
    import joblib
    
    # 1. Test MAE Lookup utility
    mae_val = utils.get_model_mae('Stacking Ensemble')
    assert mae_val > 0, "MAE lookup returned invalid score!"
    
    # 2. Test Multi-Model Loading & Prediction serving dictionary
    models_pkl = 'models/all_models.pkl'
    if os.path.exists(models_pkl):
        all_models = joblib.load(models_pkl)
        assert 'Stacking Ensemble' in all_models, "Stacking Ensemble not found in all_models.pkl"
        assert 'XGBoost' in all_models, "XGBoost not found in all_models.pkl"
        
        # Take sample input
        raw_input = pd.DataFrame({
            'Area_SqFt': [2200.0], 'Bedrooms': [3], 'Full_Bathrooms': [2], 'Half_Bathrooms': [1],
            'Location': ['Standard'], 'Year_Built': [2010.0], 'Garage_Area': [400.0], 'Basement_Area': [500.0], 'Luxury_Score': [65.0]
        })
        
        # Test predictions with Stacking Ensemble
        stack_pred = all_models['Stacking Ensemble'].predict(raw_input)[0]
        assert stack_pred > 30000.0
        
        # Test predictions with XGBoost
        xgb_pred = all_models['XGBoost'].predict(raw_input)[0]
        assert xgb_pred > 30000.0
        
        print("Predictions served correctly from serialized multi-model dictionary.")
        
    print("test_multi_model_and_stats passed.")

if __name__ == '__main__':
    print("==========================================")
    print(" RUNNING PIPELINE TESTS")
    print("==========================================")
    test_dataframe_imputer()
    test_outlier_capper()
    test_feature_engineer()
    test_anomaly_detector()
    test_preprocessing_pipeline_dimensions()
    test_sqlite_database_utilities()
    test_multi_model_and_stats()
    print("==========================================")
    print(" ALL TESTS PASSED SUCCESSFULLY!")
    print("==========================================")
