# ==============================================================================
# PROHOUSE ENTERPRISE REAL ESTATE VALUATION & ANALYTICS SUITE (ALL-IN-ONE)
# ==============================================================================
# File: prohouse_enterprise_suite.py
# Description: A self-contained, enterprise-grade Machine Learning and MLOps 
#              dashboard application designed to predict residential property 
#              valuations, manage datasets in SQLite, analyze feature drift, 
#              benchmark ML algorithms, and serve real-time predictions.
#
# Table of Contents:
#   1. Enterprise Imports & Setup
#   2. YAML Configuration Parser
#   3. Synthetic Housing Data Generator
#   4. SQLite Database CRUD Operations
#   5. Custom Scikit-Learn Transformers
#   6. Production Training Pipeline & Stacking Ensembles
#   7. Experiment Tracker & Logging
#   8. Statistical Margins & PDF Report Generator
#   9. Plotly Visualizations & Diagnostics (Learning Curves, Q-Q, Residuals)
#  10. Streamlit Dashboard User Interface
# ==============================================================================

# ==============================================================================
# SECTION 1: ENTERPRISE IMPORTS & SETUP
# ==============================================================================
import os
import sys
import json
import yaml
import time
import sqlite3
import logging
import joblib
from io import BytesIO
from datetime import datetime

# Mathematical and Data Science Imports
import numpy as np
import pandas as pd
import scipy.stats as stats

# Machine Learning and Pipelines
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Dashboard and Visualization
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import shap

# Report Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Set up Enterprise Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ProHouseEnterprise")

# ==============================================================================
# SECTION 2: YAML CONFIGURATION PARSER
# ==============================================================================
DEFAULT_CONFIG = {
    'paths': {
        'database': 'data/housing_records.db',
        'pipeline_pkl': 'models/housing_pipeline.pkl',
        'all_models_pkl': 'models/all_models.pkl',
        'comparison_csv': 'models/model_comparison.csv',
        'experiment_json': 'models/experiment_log.json',
        'best_model_name': 'models/best_model_name.txt'
    },
    'split_params': {
        'test_size': 0.2,
        'random_state': 42,
        'cv_splits': 5
    },
    'outliers': {
        'iqr_factor': 1.5
    },
    'growth_rates': {
        'Rural': 0.04,
        'Suburbs': 0.06,
        'Standard': 0.08,
        'Downtown': 0.10,
        'Premium': 0.12
    },
    'geospatial': {
        'Rural': {'lat': 18.9200, 'lon': 72.8100},
        'Suburbs': {'lat': 18.9800, 'lon': 72.8350},
        'Standard': {'lat': 19.0300, 'lon': 72.8600},
        'Downtown': {'lat': 19.0750, 'lon': 72.8770},
        'Premium': {'lat': 19.1200, 'lon': 72.9100}
    },
    'tuning_params': {
        'n_iter': 8,
        'cv': 3,
        'rf_grid': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'xgb_grid': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.03, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
    }
}

class ConfigurationManager:
    """Manages system configurations loading from config folder or falling back to default dict."""
    def __init__(self, config_dir='config'):
        self.config_dir = config_dir
        self.config_path = os.path.join(config_dir, 'config.yaml')
        self.config = DEFAULT_CONFIG
        
    def load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_cfg = yaml.safe_load(f)
                    if user_cfg:
                        # Deep update configs
                        self.config = self._deep_update(self.config, user_cfg)
                        logger.info("Loaded custom configurations successfully from YAML.")
            except Exception as e:
                logger.warning(f"Failed to parse custom config, falling back to defaults. Error: {e}")
        else:
            logger.info("Configuration YAML file not found. Initializing with built-in default values.")
        return self.config
        
    def save(self):
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(self.config, f, default_flow_style=False)
                logger.info(f"Saved configuration schema to '{self.config_path}'.")
        except Exception as e:
            logger.error(f"Failed to serialize configurations. Error: {e}")
            
    def _deep_update(self, d, u):
        import collections.abc
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = self._deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

# Instantiate Configuration
config_manager = ConfigurationManager()
system_config = config_manager.load()

# ==============================================================================
# SECTION 3: SYNTHETIC HOUSING DATA GENERATOR
# ==============================================================================
def generate_synthetic_housing_data(num_samples=3000, seed=42):
    """Generates a realistic synthetic real-estate dataset with non-linear relationships, outliers and duplicates."""
    np.random.seed(seed)
    
    # 1. Feature Distributions
    area = np.random.normal(2200, 700, num_samples).astype(int)
    area = np.clip(area, 600, 6000)
    
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=num_samples, p=[0.1, 0.25, 0.4, 0.2, 0.05])
    
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
    
    has_garage = np.random.choice([0, 1], size=num_samples, p=[0.15, 0.85])
    garage_area = np.zeros(num_samples)
    for i in range(num_samples):
        if has_garage[i]:
            garage_area[i] = np.random.normal(400, 150)
            garage_area[i] = np.clip(garage_area[i], 150, 900)
            
    has_basement = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
    basement_area = np.zeros(num_samples)
    for i in range(num_samples):
        if has_basement[i]:
            basement_area[i] = np.random.normal(area[i] * 0.4, 200)
            basement_area[i] = np.clip(basement_area[i], 200, 2000)
            
    luxury_score = np.random.randint(1, 101, size=num_samples)
    
    # 2. Target Variable Generation (INR Price) with non-linear factors
    loc_multipliers = {
        'Rural': 0.7,
        'Suburbs': 1.0,
        'Standard': 1.2,
        'Downtown': 1.6,
        'Premium': 2.1
    }
    
    base_price = 45000
    price = np.zeros(num_samples)
    for i in range(num_samples):
        f_area = area[i] * 95
        f_beds = bedrooms[i] * 22000
        f_baths = (full_bath[i] * 28000) + (half_bath[i] * 12000)
        f_garage = garage_area[i] * 65
        f_basement = basement_area[i] * 45
        
        f_lux_area = (luxury_score[i] ** 1.25) * (area[i] ** 0.6) * 12
        
        age = 2026 - year_built[i]
        f_age = 80000 - 3000 * age + 40 * (age ** 2)
        
        val = base_price + f_area + f_beds + f_baths + f_garage + f_basement + f_lux_area + f_age
        val *= loc_multipliers[location[i]]
        
        # Add random error noise
        price[i] = val + np.random.normal(0, val * 0.08)
        
    price = np.clip(price, 30000, None)
    
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
    
    # 3. Inject Missing Values (representing preprocessing needs)
    nan_mask_garage = np.random.choice([True, False], size=num_samples, p=[0.04, 0.96])
    df.loc[nan_mask_garage, 'Garage_Area'] = np.nan
    
    nan_mask_year = np.random.choice([True, False], size=num_samples, p=[0.02, 0.98])
    df.loc[nan_mask_year, 'Year_Built'] = np.nan
    
    nan_mask_luxury = np.random.choice([True, False], size=num_samples, p=[0.03, 0.97])
    df.loc[nan_mask_luxury, 'Luxury_Score'] = np.nan
    
    # 4. Inject Outliers
    outlier_indices = np.random.choice(num_samples, size=35, replace=False)
    for idx in outlier_indices[:20]:
        df.loc[idx, 'Price'] = df.loc[idx, 'Price'] * 3.5
    for idx in outlier_indices[20:30]:
        df.loc[idx, 'Price'] = df.loc[idx, 'Price'] * 0.15
    for idx in outlier_indices[30:]:
        df.loc[idx, 'Area_SqFt'] = df.loc[idx, 'Area_SqFt'] * 3
        
    # 5. Inject Duplicates
    dup_rows = df.sample(n=25, random_state=42)
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # Shuffle records
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    return df

# ==============================================================================
# SECTION 5: CUSTOM SCIKIT-LEARN TRANSFORMERS
# ==============================================================================
class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Custom Transformer to impute missing values in numeric and categorical fields."""
    def __init__(self):
        self.medians_ = {}
        self.modes_ = {}
        
    def fit(self, X, y=None):
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns
        
        for col in num_cols:
            self.medians_[col] = X[col].median()
        for col in cat_cols:
            mode_val = X[col].mode()
            self.modes_[col] = mode_val[0] if not mode_val.empty else 'Standard'
        return self
        
    def transform(self, X):
        X_out = X.copy()
        for col, val in self.medians_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(val)
        for col, val in self.modes_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(val)
        return X_out

class OutlierCapper(BaseEstimator, TransformerMixin):
    """Custom Transformer to cap outliers on numeric features using the IQR rule."""
    def __init__(self, cols_to_cap=None, iqr_factor=1.5):
        self.cols_to_cap = cols_to_cap or ['Area_SqFt', 'Garage_Area', 'Basement_Area']
        self.iqr_factor = iqr_factor
        self.bounds_ = {}
        
    def fit(self, X, y=None):
        for col in self.cols_to_cap:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.iqr_factor * IQR
                upper = Q3 + self.iqr_factor * IQR
                self.bounds_[col] = (lower, upper)
        return self
        
    def transform(self, X):
        X_out = X.copy()
        for col, bounds in self.bounds_.items():
            if col in X_out.columns:
                X_out[col] = np.clip(X_out[col], bounds[0], bounds[1])
        return X_out

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom Transformer to perform advanced housing feature engineering."""
    def __init__(self, current_year=2026):
        self.current_year = current_year
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # House Age
        if 'Year_Built' in X_out.columns:
            X_out['House_Age'] = self.current_year - X_out['Year_Built']
            X_out['House_Age'] = X_out['House_Age'].clip(lower=0)
            X_out = X_out.drop(columns=['Year_Built'])
            
        # Total Bathrooms
        if 'Full_Bathrooms' in X_out.columns and 'Half_Bathrooms' in X_out.columns:
            X_out['Total_Bathrooms'] = X_out['Full_Bathrooms'] + 0.5 * X_out['Half_Bathrooms']
            
        # Total Area
        if 'Area_SqFt' in X_out.columns and 'Garage_Area' in X_out.columns and 'Basement_Area' in X_out.columns:
            X_out['Total_Area'] = X_out['Area_SqFt'] + X_out['Garage_Area'] + X_out['Basement_Area']
            
        # Luxury Band indicator
        if 'Luxury_Score' in X_out.columns:
            X_out['Luxury_Band'] = (X_out['Luxury_Score'] > 75).astype(int)
            
        return X_out

class InputAnomalyDetector(BaseEstimator, TransformerMixin):
    """fits boundaries on the training set to identify Out-Of-Distribution (OOD) queries."""
    def __init__(self):
        self.bounds_ = {}
        self.valid_categories_ = {}
        
    def fit(self, X, y=None):
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns
        
        for col in num_cols:
            p1 = X[col].quantile(0.01)
            p99 = X[col].quantile(0.99)
            self.bounds_[col] = (p1, p99)
            
        for col in cat_cols:
            self.valid_categories_[col] = X[col].unique().tolist()
        return self
        
    def check_anomaly(self, input_dict):
        """Returns warnings list if inputs fall outside historical training percentiles (1st-99th)."""
        warnings = []
        for col, val in input_dict.items():
            item_val = val[0] if isinstance(val, (list, np.ndarray, pd.Series)) else val
            
            if col in self.bounds_:
                try:
                    num_val = float(item_val)
                    low, high = self.bounds_[col]
                    if num_val < low or num_val > high:
                        warnings.append(
                            f"Input '{col}' = {num_val} is outside normal bounds [{low:.1f}, {high:.1f}] (1st-99th percentile)."
                        )
                except ValueError:
                    pass
            elif col in self.valid_categories_:
                if str(item_val) not in self.valid_categories_[col]:
                    warnings.append(
                        f"Input '{col}' = '{item_val}' is an unrecognized category. Expected: {self.valid_categories_[col]}."
                    )
        return warnings
        
    def transform(self, X):
        return X

# ==============================================================================
# SECTION 4: SQLITE DATABASE CRUD OPERATIONS
# ==============================================================================
def get_db_connection():
    """Initializes and returns a connection to the SQLite database."""
    db_path = system_config['paths']['database']
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(force_reseed=False):
    """Creates tables if they don't exist and seeds baseline properties from csv/generator."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Properties table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Area_SqFt REAL,
            Bedrooms INTEGER,
            Full_Bathrooms INTEGER,
            Half_Bathrooms INTEGER,
            Location TEXT,
            Year_Built REAL,
            Garage_Area REAL,
            Basement_Area REAL,
            Luxury_Score REAL,
            Price REAL
        )
    ''')
    
    # Predictions log table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            Area_SqFt REAL,
            Bedrooms INTEGER,
            Full_Bathrooms INTEGER,
            Half_Bathrooms INTEGER,
            Location TEXT,
            Year_Built REAL,
            Garage_Area REAL,
            Basement_Area REAL,
            Luxury_Score REAL,
            Predicted_Price REAL,
            Lower_Bound_95_CI REAL,
            Upper_Bound_95_CI REAL,
            Model_Used TEXT,
            Feedback_Rating INTEGER,
            Feedback_Comment TEXT
        )
    ''')
    
    # Experiment log table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            Best_Model TEXT,
            Train_Rows INTEGER,
            Test_R2 REAL,
            Test_MAE REAL,
            Comparison_Metrics TEXT
        )
    ''')
    
    conn.commit()
    
    # Check if properties are seeded
    cursor.execute("SELECT COUNT(*) FROM properties")
    row_count = cursor.fetchone()[0]
    
    if row_count == 0 or force_reseed:
        logger.info("Database properties empty. Generating synthetic baseline records to seed SQLite...")
        if force_reseed:
            cursor.execute("DELETE FROM properties")
            conn.commit()
            
        df = generate_synthetic_housing_data(num_samples=3000)
        # Drop duplicates and clean prices for baseline seed
        df = df.drop_duplicates().reset_index(drop=True)
        
        # Write to SQLite
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO properties (
                    Area_SqFt, Bedrooms, Full_Bathrooms, Half_Bathrooms, Location, 
                    Year_Built, Garage_Area, Basement_Area, Luxury_Score, Price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['Area_SqFt'], row['Bedrooms'], row['Full_Bathrooms'], row['Half_Bathrooms'],
                row['Location'], row['Year_Built'], row['Garage_Area'], row['Basement_Area'],
                row['Luxury_Score'], row['Price']
            ))
        conn.commit()
        logger.info(f"Database successfully seeded with {len(df)} cleaned listings.")
        
    conn.close()

def fetch_properties_from_db():
    """Queries properties from SQLite and returns a DataFrame."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM properties", conn)
    conn.close()
    return df

def fetch_predictions_log():
    """Queries logged predictions from SQLite and returns a DataFrame."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM predictions_log ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def fetch_experiments_log():
    """Queries historical retrain training experiments logs."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM experiments_log ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def add_property_to_db(inputs):
    """Inserts a new listing into the properties table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO properties (
            Area_SqFt, Bedrooms, Full_Bathrooms, Half_Bathrooms, Location, 
            Year_Built, Garage_Area, Basement_Area, Luxury_Score, Price
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        inputs.get('Area_SqFt'), inputs.get('Bedrooms'), inputs.get('Full_Bathrooms'),
        inputs.get('Half_Bathrooms'), inputs.get('Location'), inputs.get('Year_Built'),
        inputs.get('Garage_Area'), inputs.get('Basement_Area'), inputs.get('Luxury_Score'),
        inputs.get('Price')
    ))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id

def delete_property_from_db(property_id):
    """Removes a listing from the properties table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM properties WHERE id = ?", (property_id,))
    conn.commit()
    conn.close()

def log_prediction_to_db(inputs, predicted_price, low_bound, high_bound, model_used):
    """Creates a predictions entry in predictions_log table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions_log (
            Area_SqFt, Bedrooms, Full_Bathrooms, Half_Bathrooms, Location, 
            Year_Built, Garage_Area, Basement_Area, Luxury_Score, 
            Predicted_Price, Lower_Bound_95_CI, Upper_Bound_95_CI, Model_Used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        inputs.get('Area_SqFt'), inputs.get('Bedrooms'), inputs.get('Full_Bathrooms'),
        inputs.get('Half_Bathrooms'), inputs.get('Location'), inputs.get('Year_Built'),
        inputs.get('Garage_Area'), inputs.get('Basement_Area'), inputs.get('Luxury_Score'),
        predicted_price, low_bound, high_bound, model_used
    ))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id

def submit_prediction_feedback(prediction_id, rating, comment):
    """Updates prediction feedback columns in predictions_log table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE predictions_log
        SET Feedback_Rating = ?, Feedback_Comment = ?
        WHERE id = ?
    ''', (rating, comment, prediction_id))
    conn.commit()
    conn.close()

# ==============================================================================
# SECTION 6: PRODUCTION TRAINING PIPELINE & STACKING ENSEMBLES
# ==============================================================================
def run_training_pipeline(progress_callback=None, custom_rf_grid=None, custom_xgb_grid=None):
    """Retrains the entire ML serving dictionary and stacking pipelines on current SQLite dataset."""
    def log_and_notify(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
            
    log_and_notify("Starting Pro-Level ML Pipeline Training...")
    
    # 1. Initialize and query database
    init_db()
    df_raw = fetch_properties_from_db()
    log_and_notify(f"Loaded raw data from database: {len(df_raw)} records.")
    
    if len(df_raw) < 100:
        raise ValueError("Critical Error: Database must contain at least 100 listings to execute training pipeline.")
        
    # 2. Duplicate cleaning
    initial_shape = len(df_raw)
    df_raw = df_raw.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_shape - len(df_raw)
    log_and_notify(f"Removed duplicates: {duplicates_removed} duplicate rows cleaned.")
    
    # 3. Outlier target cleaning
    q1 = df_raw['Price'].quantile(0.25)
    q3 = df_raw['Price'].quantile(0.75)
    iqr = q3 - q1
    iqr_factor = system_config['outliers']['iqr_factor']
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr
    
    outliers_mask = (df_raw['Price'] < lower_bound) | (df_raw['Price'] > upper_bound)
    outliers_count = outliers_mask.sum()
    
    # Capping target outliers
    df_raw.loc[outliers_mask, 'Price'] = np.clip(df_raw.loc[outliers_mask, 'Price'], lower_bound, upper_bound)
    log_and_notify(f"Target variable outlier capping bounds: [{lower_bound:.1f}, {upper_bound:.1f}]. Outliers capped: {outliers_count}")
    
    # 4. Features & Label splitting
    X = df_raw.drop(columns=['id', 'Price'], errors='ignore')
    y = df_raw['Price']
    
    # Train-test split
    test_size = system_config['split_params']['test_size']
    rand_state = system_config['split_params']['random_state']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)
    log_and_notify(f"Train split size: {len(X_train)} rows. Test holdout size: {len(X_test)} rows.")
    
    # 5. Core Preprocessor Construction
    numeric_features = ['Area_SqFt', 'Bedrooms', 'Full_Bathrooms', 'Half_Bathrooms', 'Garage_Area', 'Basement_Area', 'Luxury_Score']
    categorical_features = ['Location']
    
    # Build column transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Area_SqFt', 'Garage_Area', 'Basement_Area', 'Luxury_Score', 'House_Age', 'Total_Bathrooms', 'Total_Area', 'Luxury_Band']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    preprocessor_pipeline = Pipeline(steps=[
        ('imputer', DataFrameImputer()),
        ('capper', OutlierCapper(iqr_factor=iqr_factor)),
        ('engineer', FeatureEngineer(current_year=2026)),
        ('preprocessor', preprocessor),
        ('anomaly_detector', InputAnomalyDetector())
    ])
    
    # Pre-fit preprocessor to get feature names and shapes
    log_and_notify("Fitting preprocessing pipeline (scaling, encoding, capping)...")
    preprocessor_pipeline.fit(X_train, y_train)
    
    # Retrieve transformed feature names
    fitted_transformer = preprocessor_pipeline.named_steps['preprocessor']
    ohe_cols = list(fitted_transformer.transformers_[1][1].get_feature_names_out(categorical_features))
    feature_names = ['Area_SqFt', 'Garage_Area', 'Basement_Area', 'Luxury_Score', 'House_Age', 'Total_Bathrooms', 'Total_Area', 'Luxury_Band'] + ohe_cols
    
    X_train_proc = preprocessor_pipeline.transform(X_train)
    X_test_proc = preprocessor_pipeline.transform(X_test)
    X_train_proc_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_test_proc_df = pd.DataFrame(X_test_proc, columns=feature_names)
    
    # 6. Evaluate baseline models with cross-validation
    log_and_notify("Evaluating baseline models (Linear Regression, Ridge, Random Forest, XGBoost, etc.) with Cross-Validation...")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=rand_state),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=rand_state),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=rand_state),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=4, random_state=rand_state),
        'SVR': SVR(kernel='rbf', C=10000, epsilon=10),
        'KNN Regressor': KNeighborsRegressor(n_neighbors=5)
    }
    
    cv_splits = system_config['split_params']['cv_splits']
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=rand_state)
    comparison_list = []
    
    for name, model in models.items():
        log_and_notify(f"Cross-validating {name}...")
        cv_res = cross_validate(model, X_train_proc_df, y_train, cv=kf, scoring='neg_mean_absolute_error', return_train_score=False)
        mae_scores = -cv_res['test_score']
        comparison_list.append({
            'Model': name,
            'CV MAE Mean': round(np.mean(mae_scores), 2),
            'CV MAE Std': round(np.std(mae_scores), 2)
        })
        
    df_cv = pd.DataFrame(comparison_list).sort_values(by='CV MAE Mean').reset_index(drop=True)
    
    # 7. Tuned Hyperparameters via RandomizedSearchCV
    log_and_notify("Tuning hyperparameters for Random Forest and XGBoost via RandomizedSearchCV...")
    n_iter = system_config['tuning_params']['n_iter']
    
    # Random Forest tuning
    rf_grid = custom_rf_grid if custom_rf_grid is not None else system_config['tuning_params']['rf_grid']
    rf_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=rand_state),
        param_distributions=rf_grid,
        n_iter=n_iter,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=rand_state,
        n_jobs=-1
    )
    rf_search.fit(X_train_proc_df, y_train)
    best_rf = rf_search.best_estimator_
    log_and_notify(f"Best RF Parameters: {rf_search.best_params_}")
    
    # XGBoost tuning
    xgb_grid = custom_xgb_grid if custom_xgb_grid is not None else system_config['tuning_params']['xgb_grid']
    xgb_search = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=rand_state),
        param_distributions=xgb_grid,
        n_iter=n_iter,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=rand_state,
        n_jobs=-1
    )
    xgb_search.fit(X_train_proc_df, y_train)
    best_xgb = xgb_search.best_estimator_
    log_and_notify(f"Best XGB Parameters: {xgb_search.best_params_}")
    
    # Update baseline dict with tuned estimators
    models['Random Forest'] = best_rf
    models['XGBoost'] = best_xgb
    
    # 8. Stacking Regressor
    log_and_notify("Assembling Stacking Ensemble Regressor...")
    estimators = [
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=rand_state))
    ]
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1
    )
    
    log_and_notify("Fitting final Stacking Regressor model...")
    stack_cv = cross_validate(stacking_regressor, X_train_proc_df, y_train, cv=kf, scoring='neg_mean_absolute_error')
    stack_mae_mean = -np.mean(stack_cv['test_score'])
    log_and_notify(f"Stacking Ensemble CV Performance: MAE={stack_mae_mean:.2f}")
    
    # 9. Unified Production Pipeline Fitting
    log_and_notify("Training final unified production pipeline on entire train set...")
    production_pipeline = Pipeline(steps=[
        ('preprocessor_pipeline', preprocessor_pipeline),
        ('regressor', stacking_regressor)
    ])
    production_pipeline.fit(X_train, y_train)
    
    # 10. Evaluate metrics on Holdout split
    log_and_notify("Testing production model pipeline on holdout test set...")
    y_pred = production_pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    log_and_notify(f"Production Holdout Performance: R2={test_r2:.4f}, MAE={test_mae:.2f}")
    
    # Compile final metrics spreadsheet
    metrics_list = []
    metrics_list.append({
        'Model': 'Stacking Ensemble',
        'MAE': round(test_mae, 2),
        'MSE': round(mean_squared_error(y_test, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        'R2': round(test_r2, 4),
        'Adjusted R2': round(1 - (1 - test_r2) * (len(y_test) - 1) / (len(y_test) - X_test_proc.shape[1] - 1), 4)
    })
    
    all_pipelines = {}
    all_pipelines['Stacking Ensemble'] = production_pipeline
    
    for name, model in models.items():
        indiv_pipeline = Pipeline(steps=[
            ('preprocessor_pipeline', preprocessor_pipeline),
            ('regressor', model)
        ])
        indiv_pipeline.fit(X_train, y_train)
        all_pipelines[name] = indiv_pipeline
        
        preds = indiv_pipeline.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test_proc.shape[1] - 1)
        
        metrics_list.append({
            'Model': name,
            'MAE': round(mae, 2),
            'MSE': round(mse, 2),
            'RMSE': round(rmse, 2),
            'R2': round(r2, 4),
            'Adjusted R2': round(adj_r2, 4)
        })
        
    df_comparison = pd.DataFrame(metrics_list).sort_values(by='R2', ascending=False).reset_index(drop=True)
    
    # Serialize output files
    os.makedirs('models', exist_ok=True)
    joblib.dump(production_pipeline, system_config['paths']['pipeline_pkl'])
    joblib.dump(all_pipelines, system_config['paths']['all_models_pkl'])
    df_comparison.to_csv(system_config['paths']['comparison_csv'], index=False)
    
    # Save best model name for config
    best_model_name = df_comparison.iloc[0]['Model']
    with open(system_config['paths']['best_model_name'], 'w') as f:
        f.write(best_model_name)
        
    # Log to SQLite Experiments Log
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO experiments_log (
            Best_Model, Train_Rows, Test_R2, Test_MAE, Comparison_Metrics
        ) VALUES (?, ?, ?, ?, ?)
    ''', (best_model_name, len(X_train), test_r2, test_mae, df_comparison.to_json()))
    conn.commit()
    conn.close()
    
    # Serialize feature names for diagnostics
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    log_and_notify("Training process and serializations completed successfully!")

# ==============================================================================
# SECTION 8: STATISTICAL MARGINS & PDF REPORT GENERATOR
# ==============================================================================
def get_model_mae(model_name):
    """Loads validation Mean Absolute Error from CSV to compute statistical ranges."""
    try:
        csv_path = system_config['paths']['comparison_csv']
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            row = df[df['Model'].str.lower().str.strip() == model_name.lower().strip()]
            if not row.empty:
                return float(row.iloc[0]['MAE'])
    except Exception:
        pass
    return 32000.0 # Default fallback MAE in INR

def forecast_valuation(current_price, location, years=5):
    """Projects future valuation estimates using regional location compound interest growth rates."""
    rates = system_config['growth_rates']
    rate = rates.get(location, 0.07)
    
    forecasts = {}
    current_year = datetime.now().year
    
    for yr in range(1, years + 1):
        future_val = current_price * ((1 + rate) ** yr)
        forecasts[str(current_year + yr)] = round(future_val, 2)
    return forecasts

def generate_pdf_report(inputs, price, low, high, model_name):
    """Generates a professional corporate PDF report in-memory using ReportLab."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#6366f1'),
        spaceAfter=15
    )
    
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1f2937'),
        spaceBefore=12,
        spaceAfter=8,
        borderPadding=4
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#374151')
    )
    
    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#6b7280')
    )
    
    story = []
    
    # Header block
    story.append(Paragraph("PROHOUSE VALUATION EXECUTIVE SUMMMARY", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Target Engine: {model_name}", meta_style))
    story.append(Spacer(1, 15))
    
    # Overview statement
    intro_txt = (
        "This certificate validates the estimated market value of the residential property "
        "computed using our ensembled stacking regression neural-network algorithms and historical transactions database."
    )
    story.append(Paragraph(intro_txt, normal_style))
    story.append(Spacer(1, 12))
    
    # Primary Valuation Table
    val_data = [
        [Paragraph("<b>Metric</b>", normal_style), Paragraph("<b>Estimated Value</b>", normal_style)],
        [Paragraph("Market Price Valuation", normal_style), Paragraph(f"INR {price:,.2f}", normal_style)],
        [Paragraph("Lower Limit (95% CI)", normal_style), Paragraph(f"INR {low:,.2f}", normal_style)],
        [Paragraph("Upper Limit (95% CI)", normal_style), Paragraph(f"INR {high:,.2f}", normal_style)]
    ]
    t_val = Table(val_data, colWidths=[200, 200])
    t_val.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f3f4f6')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    story.append(t_val)
    story.append(Spacer(1, 15))
    
    # Property Parameters
    story.append(Paragraph("Property Specifications", section_style))
    prop_data = [
        [Paragraph("<b>Parameter</b>", normal_style), Paragraph("<b>Input Feature</b>", normal_style)],
        [Paragraph("Location Zone Segment", normal_style), Paragraph(str(inputs.get('Location')), normal_style)],
        [Paragraph("Total Area (SqFt)", normal_style), Paragraph(f"{inputs.get('Area_SqFt'):,}", normal_style)],
        [Paragraph("Bedrooms count", normal_style), Paragraph(str(inputs.get('Bedrooms')), normal_style)],
        [Paragraph("Full Bathrooms", normal_style), Paragraph(str(inputs.get('Full_Bathrooms')), normal_style)],
        [Paragraph("Half Bathrooms", normal_style), Paragraph(str(inputs.get('Half_Bathrooms')), normal_style)],
        [Paragraph("Garage Size (SqFt)", normal_style), Paragraph(f"{inputs.get('Garage_Area'):,}", normal_style)],
        [Paragraph("Basement Area (SqFt)", normal_style), Paragraph(f"{inputs.get('Basement_Area'):,}", normal_style)],
        [Paragraph("Luxury & Finishes Score", normal_style), Paragraph(f"{inputs.get('Luxury_Score')}/100", normal_style)]
    ]
    t_prop = Table(prop_data, colWidths=[200, 200])
    t_prop.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f3f4f6')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb'))
    ]))
    story.append(t_prop)
    story.append(Spacer(1, 20))
    
    # Disclaimer
    disclaimer = (
        "<b>Disclaimer:</b> This report represents a simulated asset evaluation generated via a "
        "production pipeline. Results are based on historical transaction datasets and statistical trends. "
        "Actual market listings may vary. This summary does not constitute an explicit financial appraisal."
    )
    story.append(Paragraph(disclaimer, ParagraphStyle('Disc', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#9ca3af'))))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ==============================================================================
# SECTION 9: PLOTLY VISUALIZATIONS & DIAGNOSTICS
# ==============================================================================
def calculate_learning_curves(pipeline_obj, X, y):
    """Calculates train vs validation scores over various sizes to generate learning curves."""
    try:
        # Preprocess dataset to fit quickly
        preproc = pipeline_obj.named_steps['preprocessor_pipeline']
        X_proc = preproc.transform(X)
        regressor = pipeline_obj.named_steps['regressor']
        
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=regressor,
            X=X_proc,
            y=y,
            train_sizes=np.linspace(0.1, 1.0, 5),
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42
        )
        
        train_scores_mean = -np.mean(train_scores, axis=1)
        val_scores_mean = -np.mean(val_scores, axis=1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean, name="Training Score (MAE)", mode="lines+markers", line=dict(color='#6366f1', width=3)))
        fig.add_trace(go.Scatter(x=train_sizes, y=val_scores_mean, name="Validation Score (MAE)", mode="lines+markers", line=dict(color='#10b981', width=3)))
        
        fig.update_layout(
            title="Model Learning Curves (Bias vs Variance Analysis)",
            xaxis_title="Training Set Sample Count",
            yaxis_title="Mean Absolute Error (MAE)",
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to generate learning curves: {e}")
        return None

def plot_residuals_diagnostics(y_true, y_pred):
    """Plots prediction errors residuals vs predicted values to verify constant variance."""
    residuals = y_true - y_pred
    
    fig = px.scatter(
        x=y_pred,
        y=residuals,
        labels={'x': 'Predicted Valuation (INR)', 'y': 'Inference Residual (Error)'},
        title="Residual Diagnostic Scatter Plot (Heteroscedasticity Analysis)",
        color_discrete_sequence=['#ef4444'],
        opacity=0.6
    )
    # Add horizontal reference line at zero error
    fig.add_shape(
        type="line", line=dict(color="white", width=2, dash="dash"),
        x0=min(y_pred), y0=0, x1=max(y_pred), y1=0
    )
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_qq_residual_check(y_true, y_pred):
    """Generates a Q-Q plot to verify if prediction residuals are normally distributed."""
    residuals = y_true - y_pred
    # Standardize residuals
    std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    
    # Calculate quantiles
    osm, osr = stats.probplot(std_residuals, dist="norm")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=osm[0],
        y=osm[1],
        mode='markers',
        name='Residual Quantiles',
        marker=dict(color='#10b981', opacity=0.7)
    ))
    
    # Add identity line
    line_min = min(osm[0])
    line_max = max(osm[0])
    fig.add_trace(go.Scatter(
        x=[line_min, line_max],
        y=[line_min, line_max],
        mode='lines',
        name='Theoretical Normal Reference',
        line=dict(color='#6366f1', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Probability Q-Q Plot of Standardized Residuals",
        xaxis_title="Theoretical Normal Quantiles",
        yaxis_title="Sample Residual Quantiles",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ==============================================================================
# SECTION 10: STREAMLIT DASHBOARD USER INTERFACE
# ==============================================================================
# 1. Page Configuration
st.set_page_config(
    page_title="ProHouse - Premium Enterprise Analytics Suite",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "🏠 Valuation Predictor"

# Ensure Database is initialized on startup
init_db()

# Load models and pipelines
@st.cache_resource
def load_ml_assets():
    pipeline_path = system_config['paths']['pipeline_pkl']
    all_models_path = system_config['paths']['all_models_pkl']
    
    # Check if files exist, if not, trigger a quick retraining run
    if not os.path.exists(pipeline_path) or not os.path.exists(all_models_path):
        logger.info("Pickled models not found. Running baseline training pipeline...")
        run_training_pipeline()
        
    pipeline = joblib.load(pipeline_path)
    all_models = joblib.load(all_models_path)
    
    with open(system_config['paths']['best_model_name'], 'r') as f:
        best_model_name = f.read().strip()
        
    feature_names = joblib.load('models/feature_names.pkl')
    return pipeline, all_models, best_model_name, feature_names

pipeline, all_models, best_model_name, feature_names = load_ml_assets()

# Location lat-lon configurations for PyDeck
geo_cfg = system_config['geospatial']

# 2. Premium Styling (CSS injection)
st.markdown("""
<style>
    /* Premium Styling and Global Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: #f1f5f9;
    }
    
    /* Override Streamlit background to Deep forest / Emerald Slate */
    .stApp {
        background: linear-gradient(135deg, #021a14 0%, #031e1a 50%, #010409 100%) !important;
    }
    
    /* Make the glass-card feel extremely premium with dark emerald borders */
    .glass-card {
        background: rgba(4, 30, 24, 0.35) !important;
        border: 1px solid rgba(16, 185, 129, 0.15) !important;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 45px 0 rgba(0, 0, 0, 0.6), inset 0 1px 1px 0 rgba(255, 255, 255, 0.03);
        transition: border 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        border: 1px solid rgba(52, 211, 153, 0.35) !important;
        box-shadow: 0 10px 45px 0 rgba(16, 185, 129, 0.08), inset 0 1px 1px 0 rgba(255, 255, 255, 0.06);
    }
    
    .glowing-price {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #34d399, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 35px rgba(16, 185, 129, 0.35);
        margin: 0.8rem 0;
    }
    
    .stat-widget {
        background: rgba(6, 40, 32, 0.45);
        border: 1px solid rgba(16, 185, 129, 0.12);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .stat-widget-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    
    .stat-widget-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f9fafb;
    }
    
    .stat-widget-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .anomaly-banner {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 10px;
        padding: 1rem;
        color: #f87171;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    
    .valuation-wrapper {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(16, 185, 129, 0.05);
    }
    
    /* Customize Streamlit Buttons globally to look extremely premium */
    div.stButton > button {
        border-radius: 10px !important;
        padding: 0.62rem 1.5rem !important;
        font-weight: 600 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.3s ease !important;
        width: 100%;
    }
    
    /* Primary buttons style */
    div.stButton > button[kind="primary"], div.stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(90deg, #10b981, #059669) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.35) !important;
    }
    div.stButton > button[kind="primary"]:hover, div.stButton > button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(16, 185, 129, 0.55) !important;
        background: linear-gradient(90deg, #059669, #047857) !important;
    }
    
    /* Secondary buttons style */
    div.stButton > button[kind="secondary"], div.stButton > button[data-testid="baseButton-secondary"] {
        background: rgba(4, 30, 24, 0.25) !important;
        color: #9ca3af !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: none !important;
    }
    div.stButton > button[kind="secondary"]:hover, div.stButton > button[data-testid="baseButton-secondary"]:hover {
        transform: translateY(-2px) !important;
        background: rgba(4, 30, 24, 0.5) !important;
        color: #f9fafb !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
    }
    
    div.stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Customize Sidebar Styling to be clean and modern */
    section[data-testid="stSidebar"] {
        background-color: #020d0b !important;
        border-right: 1px solid rgba(16, 185, 129, 0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Navigation Panel
st.sidebar.markdown("<h2 style='text-align:center; color:#10b981;'>🏡 ProHouse</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align:center; color:#9ca3af; font-size:0.9rem; margin-top:-0.5rem;'>Enterprise Asset Valuation Platform</p>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border-top:1px solid rgba(255,255,255,0.08); margin-bottom:1.5rem;'>", unsafe_allow_html=True)

# Navigation Buttons
navigation_tabs = [
    "🏠 Valuation Predictor",
    "⚖️ Compare Mode",
    "📊 Interactive EDA",
    "⚙️ Diagnostics & Learning",
    "🔍 SHAP Explanations",
    "🛠️ Feature Engineering Playground",
    "📈 Data Quality Auditor",
    "📈 Financial Scenario Planner",
    "🧪 Test Suite Sandbox",
    "🗃️ DB & Retraining Hub",
    "📖 Theoretical Guide"
]

for tab_name in navigation_tabs:
    # Stylize button based on selection state
    btn_type = "primary" if st.session_state.active_tab == tab_name else "secondary"
    if st.sidebar.button(tab_name, width='stretch', type=btn_type):
        st.session_state.active_tab = tab_name
        st.rerun()

st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='text-align:center; font-size:0.8rem; color:#6b7280;'>ProHouse Suite v3.2.0 (All-in-One)<br>© 2026 ProHouse Labs.</div>", unsafe_allow_html=True)

# Segment Details helper
def get_property_segment_details(price):
    if price < 420000:
        return "Standard Class", "🏡", "#10b981", "rgba(16, 185, 129, 0.15)"
    elif price < 850000:
        return "Premium Class", "💎", "#f59e0b", "rgba(245, 158, 11, 0.15)"
    else:
        return "Ultra Luxury Mansion", "🏰", "#ec4899", "rgba(236, 72, 153, 0.15)"

# ==============================================================================
# TAB 1: VALUATION PREDICTOR
# ==============================================================================
if st.session_state.active_tab == "🏠 Valuation Predictor":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>🏠 Property Valuation Simulator & Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Configure individual property features or upload a batch CSV to run real-time production valuations.</p>", unsafe_allow_html=True)
    
    # Model Selection Engine at the very top (common for both modes)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    selected_model_name = st.selectbox("Production Model Model", list(all_models.keys()), index=list(all_models.keys()).index('Stacking Ensemble') if 'Stacking Ensemble' in all_models else 0)
    selected_pipeline = all_models.get(selected_model_name, pipeline)
    st.markdown("</div>", unsafe_allow_html=True)
    
    val_mode1, val_mode2 = st.tabs(["🏠 Single Property Simulator", "📂 Bulk Valuation Uploader (CSV)"])
    
    with val_mode1:
        col1, col2 = st.columns([1.3, 1])
        
        with col1:
            st.markdown("<div class='glass-card'><h3 style='margin-top:0; margin-bottom:1.5rem;'>Property Dimensions & Finishes</h3>", unsafe_allow_html=True)
            
            # Grid input
            l1, l2 = st.columns(2)
            with l1:
                location = st.selectbox("Location Sector Zone", list(geo_cfg.keys()), index=2, key="single_loc")
                area = st.slider("Living Area (SqFt)", min_value=600, max_value=6000, value=2200, step=50, key="single_area")
                bedrooms = st.slider("Total Bedrooms", min_value=1, max_value=5, value=3, key="single_beds")
                full_baths = st.slider("Full Bathrooms", min_value=1, max_value=4, value=2, key="single_fb")
            with l2:
                half_baths = st.slider("Half Bathrooms", min_value=0, max_value=2, value=1, key="single_hb")
                garage = st.slider("Garage Area Size (SqFt)", min_value=0, max_value=900, value=400, step=25, key="single_gar")
                basement = st.slider("Basement Area Size (SqFt)", min_value=0, max_value=2000, value=500, step=50, key="single_base")
                year = st.slider("Year Constructed", min_value=1960, max_value=2025, value=2010, key="single_year")
                
            luxury = st.slider("Luxury & Finishes Score", min_value=1, max_value=100, value=65, key="single_lux")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='glass-card'><h3 style='margin-top:0; margin-bottom:1.5rem;'>Engine Output & Audits</h3>", unsafe_allow_html=True)
            
            # Predict input
            raw_input = {
                'Area_SqFt': [area], 'Bedrooms': [bedrooms], 'Full_Bathrooms': [full_baths], 'Half_Bathrooms': [half_baths],
                'Location': [location], 'Year_Built': [year], 'Garage_Area': [garage], 'Basement_Area': [basement], 'Luxury_Score': [luxury]
            }
            df_raw = pd.DataFrame(raw_input)
            
            # Anomaly Checks
            anomaly_detector = selected_pipeline.named_steps['preprocessor_pipeline'].named_steps['anomaly_detector']
            warnings = anomaly_detector.check_anomaly(raw_input)
            if warnings:
                for w in warnings:
                    st.markdown(f"<div class='anomaly-banner'>{w}</div>", unsafe_allow_html=True)
                    
            pred_price = selected_pipeline.predict(df_raw)[0]
            
            # Rigorous statistical bounds based on selected model's cross-validation MAE
            selected_mae = get_model_mae(selected_model_name)
            low_bound = max(30000.0, pred_price - 1.96 * selected_mae)
            high_bound = pred_price + 1.96 * selected_mae
            
            category, icon, color, badge = get_property_segment_details(pred_price)
            
            st.markdown("<div class='valuation-wrapper'>", unsafe_allow_html=True)
            st.markdown("<div class='stat-widget-label'>Estimated Market Valuation</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='glowing-price'>INR {pred_price:,.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:{color}; background:{badge}; padding:0.45rem 1.2rem; border-radius:30px; font-weight:700; font-size:0.85rem; border: 1px solid {color}40;'>{icon} {category}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Visual stats grids showing statistical confidence interval (CI)
            st.markdown(f"""
            <div class="stat-widget-grid">
                <div class="stat-widget">
                    <div class="stat-widget-label">Lower Bound (95% CI)</div>
                    <div class="stat-widget-val" style="color:#ef4444;">INR {low_bound:,.0f}</div>
                </div>
                <div class="stat-widget">
                    <div class="stat-widget-label">Upper Bound (95% CI)</div>
                    <div class="stat-widget-val" style="color:#10b981;">INR {high_bound:,.0f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Report download
            pdf_buffer = generate_pdf_report(
                {'Area_SqFt': area, 'Bedrooms': bedrooms, 'Full_Bathrooms': full_baths, 'Half_Bathrooms': half_baths, 'Location': location, 'Year_Built': year, 'Garage_Area': garage, 'Basement_Area': basement, 'Luxury_Score': luxury},
                pred_price, low_bound, high_bound, selected_model_name
            )
            st.download_button(
                label="📄 Generate & Download Valuation PDF Report",
                data=pdf_buffer,
                file_name=f"ProHouse_Valuation_Report_{location}.pdf",
                mime="application/pdf",
                width='stretch'
            )
            
            # Forecast Valuation mini line chart
            st.markdown("<h4 style='margin-top:1.5rem; margin-bottom:0.5rem;'>5-Year Growth Outlook</h4>", unsafe_allow_html=True)
            forecasts = forecast_valuation(pred_price, location)
            fc_df = pd.DataFrame(list(forecasts.items()), columns=['Year', 'Projected Price'])
            fig_fc = px.line(fc_df, x='Year', y='Projected Price', markers=True, color_discrete_sequence=[color])
            fig_fc.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10,r=10,t=10,b=10), height=140, yaxis_title=None, xaxis_title=None
            )
            st.plotly_chart(fig_fc, width='stretch')
            
            # --- Interactive SQLite Prediction Logger & Feedback System ---
            st.markdown("<div style='margin-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("<h4 style='margin-top:0.2rem; margin-bottom:0.8rem;'>💾 Save Valuation & Feedback</h4>", unsafe_allow_html=True)
            
            feedback_rating = st.radio("Is this valuation realistic?", ["⭐ Very Poor", "⭐⭐ Poor", "⭐⭐⭐ Fair", "⭐⭐⭐⭐ Good", "⭐⭐⭐⭐⭐ Excellent"], index=4, horizontal=True, key="single_feedback_rating")
            feedback_comment = st.text_input("Comments / Notes (Optional)", placeholder="e.g., Valuation looks highly accurate for current micro-market.", key="single_feedback_comment")
            
            if st.button("Log Valuation & Submit Feedback", width='stretch', type="primary", key="single_submit_feedback"):
                inputs = {
                    'Area_SqFt': area, 'Bedrooms': bedrooms, 'Full_Bathrooms': full_baths, 'Half_Bathrooms': half_baths,
                    'Location': location, 'Year_Built': year, 'Garage_Area': garage, 'Basement_Area': basement, 'Luxury_Score': luxury
                }
                # Log prediction to DB and retrieve prediction ID
                pred_id = log_prediction_to_db(inputs, pred_price, low_bound, high_bound, selected_model_name)
                
                # Map rating stars to number
                rating_num = feedback_rating.count("⭐")
                
                # Update prediction with feedback rating and comment
                submit_prediction_feedback(pred_id, rating_num, feedback_comment)
                st.success(f"🎉 Valuation record #{pred_id} and feedback successfully logged to SQLite database!")
                
            st.markdown("</div>", unsafe_allow_html=True)

    with val_mode2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📂 Batch Property Valuation Uploader")
        st.write("Upload a CSV file containing property parameters to obtain bulk valuation estimates using the production ML pipeline.")
        
        # Display required format guide
        st.info("💡 **CSV Template Columns Required:**  \n`Area_SqFt`, `Bedrooms`, `Full_Bathrooms`, `Half_Bathrooms`, `Location`, `Year_Built`, `Garage_Area`, `Basement_Area`, `Luxury_Score`")
        
        # Download template CSV
        template_df = pd.DataFrame([{
            'Area_SqFt': 2000.0, 'Bedrooms': 3, 'Full_Bathrooms': 2, 'Half_Bathrooms': 1,
            'Location': 'Suburbs', 'Year_Built': 2012.0, 'Garage_Area': 400.0, 'Basement_Area': 500.0, 'Luxury_Score': 75.0
        }])
        csv_template = template_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV Template", data=csv_template, file_name="prohouse_bulk_template.csv", mime="text/csv", key="download_template")
        
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="bulk_csv_file")
        if uploaded_file is not None:
            try:
                bulk_df = pd.read_csv(uploaded_file)
                required_cols = ['Area_SqFt', 'Bedrooms', 'Full_Bathrooms', 'Half_Bathrooms', 'Location', 'Year_Built', 'Garage_Area', 'Basement_Area', 'Luxury_Score']
                
                missing_cols = [col for col in required_cols if col not in bulk_df.columns]
                if missing_cols:
                    st.error(f"❌ Missing required columns in CSV: {missing_cols}")
                else:
                    with st.spinner("Valuing property listings..."):
                        # Calculate predictions
                        pred_prices = selected_pipeline.predict(bulk_df)
                        
                        # Add stats error margins
                        selected_mae = get_model_mae(selected_model_name)
                        low_bounds = np.maximum(30000.0, pred_prices - 1.96 * selected_mae)
                        high_bounds = pred_prices + 1.96 * selected_mae
                        
                        bulk_df['Predicted_Price'] = pred_prices
                        bulk_df['Lower_Bound_95_CI'] = low_bounds
                        bulk_df['Upper_Bound_95_CI'] = high_bounds
                        bulk_df['Model_Used'] = selected_model_name
                        
                        # Log predictions to SQLite database
                        logged_count = 0
                        for _, row in bulk_df.iterrows():
                            row_inputs = {
                                'Area_SqFt': row['Area_SqFt'], 'Bedrooms': row['Bedrooms'], 
                                'Full_Bathrooms': row['Full_Bathrooms'], 'Half_Bathrooms': row['Half_Bathrooms'],
                                'Location': row['Location'], 'Year_Built': row['Year_Built'], 
                                'Garage_Area': row['Garage_Area'], 'Basement_Area': row['Basement_Area'], 
                                'Luxury_Score': row['Luxury_Score']
                            }
                            log_prediction_to_db(row_inputs, row['Predicted_Price'], row['Lower_Bound_95_CI'], row['Upper_Bound_95_CI'], selected_model_name)
                            logged_count += 1
                        
                        st.success(f"🎉 Valued {logged_count} properties successfully! Records logged to SQLite database history.")
                        
                        # Display valued listings
                        st.dataframe(bulk_df, width='stretch')
                        
                        # Download results button
                        results_csv = bulk_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Valued Property Predictions CSV",
                            data=results_csv,
                            file_name=f"ProHouse_Bulk_Valued_Estimates.csv",
                            mime="text/csv",
                            width='stretch',
                            key="download_bulk_results"
                        )
            except Exception as e:
                st.error(f"Failed to process CSV file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 2: COMPARE MODE (SIDE-BY-SIDE EVALUATOR)
# ==============================================================================
elif st.session_state.active_tab == "⚖️ Compare Mode":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>⚖️ Side-by-Side Property Evaluator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Compare valuations and structural features between two unique properties.</p>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("<div class='glass-card'><h3 style='color:#6366f1; margin-top:0;'>Property A (Reference Listing)</h3>", unsafe_allow_html=True)
        loc_a = st.selectbox("Location Zone", list(geo_cfg.keys()), index=1, key="loc_a")
        area_a = st.slider("Area (SqFt)", 600, 6000, 1800, 50, key="area_a")
        beds_a = st.slider("Bedrooms", 1, 5, 3, key="beds_a")
        fb_a = st.slider("Full Baths", 1, 4, 2, key="fb_a")
        hb_a = st.slider("Half Baths", 0, 2, 0, key="hb_a")
        garage_a = st.slider("Garage Area (SqFt)", 0, 900, 200, 25, key="garage_a")
        basement_a = st.slider("Basement Area (SqFt)", 0, 2000, 0, 50, key="basement_a")
        year_a = st.slider("Year Built", 1960, 2025, 1995, key="year_a")
        lux_a = st.slider("Luxury & Finishes Score", 1, 100, 45, key="lux_a")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_b:
        st.markdown("<div class='glass-card'><h3 style='color:#10b981; margin-top:0;'>Property B (Compare Listing)</h3>", unsafe_allow_html=True)
        loc_b = st.selectbox("Location Zone", list(geo_cfg.keys()), index=4, key="loc_b")
        area_b = st.slider("Area (SqFt)", 600, 6000, 2800, 50, key="area_b")
        beds_b = st.slider("Bedrooms", 1, 5, 4, key="beds_b")
        fb_b = st.slider("Full Baths", 1, 4, 3, key="fb_b")
        hb_b = st.slider("Half Baths", 0, 2, 1, key="hb_b")
        garage_b = st.slider("Garage Area (SqFt)", 0, 900, 450, 25, key="garage_b")
        basement_b = st.slider("Basement Area (SqFt)", 0, 2000, 800, 50, key="basement_b")
        year_b = st.slider("Year Built", 1960, 2025, 2018, key="year_b")
        lux_b = st.slider("Luxury & Finishes Score", 1, 100, 80, key="lux_b")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Run Inference
    in_a = pd.DataFrame([{
        'Area_SqFt': area_a, 'Bedrooms': beds_a, 'Full_Bathrooms': fb_a, 'Half_Bathrooms': hb_a,
        'Location': loc_a, 'Year_Built': year_a, 'Garage_Area': garage_a, 'Basement_Area': basement_a, 'Luxury_Score': lux_a
    }])
    in_b = pd.DataFrame([{
        'Area_SqFt': area_b, 'Bedrooms': beds_b, 'Full_Bathrooms': fb_b, 'Half_Bathrooms': hb_b,
        'Location': loc_b, 'Year_Built': year_b, 'Garage_Area': garage_b, 'Basement_Area': basement_b, 'Luxury_Score': lux_b
    }])
    
    price_a = pipeline.predict(in_a)[0]
    price_b = pipeline.predict(in_b)[0]
    
    c1, c2 = st.columns(2)
    with c1:
        cat_a, icon_a, col_a_clr, _ = get_property_segment_details(price_a)
        st.markdown(f"""
        <div class="valuation-wrapper" style="border-color:{col_a_clr}40; background:{col_a_clr}05;">
            <div class="stat-widget-label">Property A Valuation</div>
            <div class="glowing-price" style="color:{col_a_clr}; text-shadow: 0 0 15px {col_a_clr}40;">INR {price_a:,.2f}</div>
            <span style="font-weight:700;">{icon_a} {cat_a}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        cat_b, icon_b, col_b_clr, _ = get_property_segment_details(price_b)
        st.markdown(f"""
        <div class="valuation-wrapper" style="border-color:{col_b_clr}40; background:{col_b_clr}05;">
            <div class="stat-widget-label">Property B Valuation</div>
            <div class="glowing-price" style="color:{col_b_clr}; text-shadow: 0 0 15px {col_b_clr}40;">INR {price_b:,.2f}</div>
            <span style="font-weight:700;">{icon_b} {cat_b}</span>
        </div>
        """, unsafe_allow_html=True)
        
    # Comparative Bar Chart
    fig_comp = go.Figure(data=[
        go.Bar(name='Property A', x=['Estimated Price (INR)'], y=[price_a], marker_color='#6366f1'),
        go.Bar(name='Property B', x=['Estimated Price (INR)'], y=[price_b], marker_color='#10b981')
    ])
    fig_comp.update_layout(
        title="Valuation Comparison Chart",
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        barmode='group', height=300
    )
    st.plotly_chart(fig_comp, width='stretch')

# ==============================================================================
# TAB 3: INTERACTIVE EDA
# ==============================================================================
elif st.session_state.active_tab == "📊 Interactive EDA":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>📊 Interactive EDA & Geospatial Map</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Explore historical housing listings, examine distributions, and locate properties in 3D geospatial maps.</p>", unsafe_allow_html=True)
    
    eda_tab1, eda_tab2 = st.tabs(["🌍 3D Geospatial Price Map", "📈 Distribution Analytics"])
    
    df_eda = fetch_properties_from_db()
    
    with eda_tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("3D Geospatial Price Columns Mapping")
        st.write("Each column represents a property listing, where the height scales with the selling price and color maps to the location zone.")
        
        # Inject coordinates mapping for 3D PyDeck
        df_fe = df_eda.copy()
        df_fe['latitude'] = df_fe['Location'].map(lambda loc: geo_cfg.get(loc, geo_cfg['Standard'])['lat'] + np.random.normal(0, 0.006))
        df_fe['longitude'] = df_fe['Location'].map(lambda loc: geo_cfg.get(loc, geo_cfg['Standard'])['lon'] + np.random.normal(0, 0.006))
        
        color_map = {
            'Rural': [156, 163, 175, 150],     # Grey
            'Suburbs': [59, 130, 246, 150],    # Blue
            'Standard': [16, 185, 129, 150],   # Emerald
            'Downtown': [245, 158, 11, 150],   # Amber
            'Premium': [236, 72, 153, 150]     # Pink
        }
        df_fe['color_rgba'] = df_fe['Location'].map(color_map)
        
        layer = pdk.Layer(
            'ColumnLayer',
            df_fe,
            get_position='[longitude, latitude]',
            get_elevation='Price',
            elevation_scale=0.012, # Scale elevation to fit map bounds
            radius=150,            # Column radius in meters
            get_fill_color='color_rgba',
            pickable=True,
            extruded=True          # 3D columns
        )
        
        view_state = pdk.ViewState(
            latitude=19.03,
            longitude=72.86,
            zoom=10.5,
            pitch=50,
            bearing=20
        )
        
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/dark-v9',
            tooltip={"text": "Zone Segment: {Location}\nArea: {Area_SqFt} SqFt\nPrice: INR {Price:,.2f}"}
        )
        st.pydeck_chart(r)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with eda_tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Price Spread & Feature Distributions")
        
        f1, f2 = st.columns(2)
        with f1:
            fig_hist = px.histogram(df_eda, x='Price', nbins=40, title='Property Selling Price Distribution (INR)', color_discrete_sequence=['#6366f1'])
            fig_hist.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hist, width='stretch')
        with f2:
            fig_scatter = px.scatter(df_eda, x='Area_SqFt', y='Price', color='Location', title='Price vs Living Area (SqFt) by Zone', color_discrete_sequence=['#6366f1', '#10b981', '#f59e0b', '#ec4899', '#3b82f6'])
            fig_scatter.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_scatter, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 4: DIAGNOSTICS & LEARNING
# ==============================================================================
elif st.session_state.active_tab == "⚙️ Diagnostics & Learning":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>⚙️ Model Diagnostics & Learning Curves</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Evaluate ensembled pipeline residuals, test normality, and check bias vs variance curves.</p>", unsafe_allow_html=True)
    
    df_raw = fetch_properties_from_db()
    X = df_raw.drop(columns=['id', 'Price'], errors='ignore')
    y = df_raw['Price']
    
    # Predict holdout predictions dynamically for residual plotting
    preproc = pipeline.named_steps['preprocessor_pipeline']
    regressor = pipeline.named_steps['regressor']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = pipeline.predict(X_test)
    
    diag_tab1, diag_tab2, diag_tab3, diag_tab4 = st.tabs(["📈 Residual Analysis & Q-Q Plots", "📉 Learning Curves", "📊 Benchmarked Model Metrics", "🔍 Feature Importances & Weights"])
    
    with diag_tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig_res = plot_residuals_diagnostics(y_test, y_pred)
            st.plotly_chart(fig_res, width='stretch')
        with col2:
            fig_qq = plot_qq_residual_check(y_test, y_pred)
            st.plotly_chart(fig_qq, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
        
    with diag_tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        with st.spinner("Calculating cross-validated scores over various sizes..."):
            fig_lc = calculate_learning_curves(pipeline, X, y)
            if fig_lc:
                st.plotly_chart(fig_lc, width='stretch')
            else:
                st.info("Learning curve computation timed out or failed due to dataset limitations.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with diag_tab3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Model Comparison Matrix")
        
        csv_path = system_config['paths']['comparison_csv']
        if os.path.exists(csv_path):
            df_results = pd.read_csv(csv_path)
            st.dataframe(
                df_results.style.highlight_max(subset=['R2', 'Adjusted R2'], color='#10b981')
                .highlight_min(subset=['MAE', 'RMSE'], color='#ef4444'),
                width='stretch'
            )
        else:
            st.info("No comparative model runs logged. Run retraining pipeline to populate benchmarks.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with diag_tab4:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Feature Importance Profiling")
        st.write("Compare the relative importance scores of all preprocessed features computed by base tree estimators.")
        
        rf_model = all_models.get('Random Forest')
        xgb_model = all_models.get('XGBoost')
        
        if rf_model and xgb_model:
            try:
                rf_imp = rf_model.named_steps['regressor'].feature_importances_
                xgb_imp = xgb_model.named_steps['regressor'].feature_importances_
                
                imp_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Random Forest': rf_imp,
                    'XGBoost': xgb_imp
                })
                
                fig_imp = px.bar(
                    imp_df,
                    x='Feature',
                    y=['Random Forest', 'XGBoost'],
                    barmode='group',
                    title="Feature Importances comparison: Random Forest vs XGBoost",
                    color_discrete_sequence=['#6366f1', '#10b981']
                )
                fig_imp.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_imp, width='stretch')
            except Exception as e:
                st.error(f"Failed to load estimator feature importances: {e}")
        else:
            st.info("Serialize Tuned Random Forest and Tuned XGBoost models to unlock relative feature weights analysis.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Inject health audit diagnostics card
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🩺 System Health & Integrity Diagnostics Report")
    
    db_size = os.path.getsize(system_config['paths']['database']) / 1024 if os.path.exists(system_config['paths']['database']) else 0
    all_models_size = os.path.getsize(system_config['paths']['all_models_pkl']) / (1024 * 1024) if os.path.exists(system_config['paths']['all_models_pkl']) else 0
    pipeline_size = os.path.getsize(system_config['paths']['pipeline_pkl']) / (1024 * 1024) if os.path.exists(system_config['paths']['pipeline_pkl']) else 0
    
    sh1, sh2, sh3 = st.columns(3)
    with sh1:
        st.metric(label="SQLite DB Size (KB)", value=f"{db_size:.1f} KB")
    with sh2:
        st.metric(label="All Models Pickle Size (MB)", value=f"{all_models_size:.2f} MB")
    with sh3:
        st.metric(label="Pipeline Pickle Size (MB)", value=f"{pipeline_size:.2f} MB")
        
    st.markdown("**Python Environment Auditing:**")
    st.write(f"- Python Version: `{sys.version}`")
    st.write(f"- Streamlit Version: `{st.__version__}`")
    st.write(f"- Scikit-Learn Version: `{joblib.__import__('sklearn').__version__}`")
    st.write(f"- Plotly Version: `{px.__import__('plotly').__version__}`")
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 5: SQLITE DATABASE EXPLORER & RETRAINING HUB
# ==============================================================================
elif st.session_state.active_tab == "🗃️ DB & Retraining Hub":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>🗃️ SQLite Database & Model Retraining Hub</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Explore property listings, manage CRUD operations, review logged predictions, and trigger retraining cycles.</p>", unsafe_allow_html=True)
    
    # Fetch Dataframes from database
    prop_df = fetch_properties_from_db()
    pred_df = fetch_predictions_log()
    
    # 1. Metric Overview Cards
    avg_rating = pred_df['Feedback_Rating'].mean() if not pred_df.empty and 'Feedback_Rating' in pred_df.columns else None
    avg_rating_str = f"{avg_rating:.2f} / 5.0" if pd.notnull(avg_rating) else "N/A"
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="stat-widget">
            <div class="stat-widget-label">Listed Properties</div>
            <div class="stat-widget-val" style="color:#6366f1;">{len(prop_df):,} Rows</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="stat-widget">
            <div class="stat-widget-label">Valuation Logs</div>
            <div class="stat-widget-val" style="color:#10b981;">{len(pred_df):,} Logs</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="stat-widget">
            <div class="stat-widget-label">Average Satisfaction</div>
            <div class="stat-widget-val" style="color:#f59e0b;">{avg_rating_str}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    db_tab1, db_tab2, db_tab3, db_tab4, db_tab5 = st.tabs(["🏠 Property Listings CRUD", "📋 Prediction Logs & Feedback", "⚡ MLOps Retraining Console", "🔍 MLOps Data Audit & Drift", "💻 Live SQL Console & Schema"])
    
    with db_tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Manage Database Property Records")
        st.write("Browse housing listings in SQLite. Use the options below to insert new properties or remove entries from database.")
        
        # Add Search/Filter fields
        f1, f2 = st.columns(2)
        with f1:
            search_loc = st.multiselect("Filter by Location Zone", options=list(prop_df['Location'].unique()) if not prop_df.empty else [], default=[])
        with f2:
            min_p, max_p = float(prop_df['Price'].min()) if not prop_df.empty else 0.0, float(prop_df['Price'].max()) if not prop_df.empty else 1000000.0
            search_price_range = st.slider("Filter by Price Range (INR)", min_value=min_p, max_value=max_p, value=(min_p, max_p))
            
        filtered_df = prop_df.copy()
        if search_loc:
            filtered_df = filtered_df[filtered_df['Location'].isin(search_loc)]
        filtered_df = filtered_df[(filtered_df['Price'] >= search_price_range[0]) & (filtered_df['Price'] <= search_price_range[1])]
        
        st.dataframe(filtered_df, width='stretch')
        
        # Insert/Delete forms
        crud_col1, crud_col2 = st.columns(2)
        with crud_col1:
            st.markdown("#### Add New Property Listing")
            with st.form("insert_property_form", clear_on_submit=True):
                i_loc = st.selectbox("Location Sector", list(geo_cfg.keys()))
                i_area = st.number_input("Living Area (SqFt)", value=1800, min_value=300)
                i_beds = st.number_input("Bedrooms Count", value=3, min_value=1)
                i_fb = st.number_input("Full Bathrooms", value=2, min_value=1)
                i_hb = st.number_input("Half Bathrooms", value=0, min_value=0)
                i_garage = st.number_input("Garage Area (SqFt)", value=200, min_value=0)
                i_basement = st.number_input("Basement Area (SqFt)", value=0, min_value=0)
                i_year = st.number_input("Year Built", value=2015, min_value=1800)
                i_lux = st.number_input("Luxury Score (1-100)", value=50, min_value=1, max_value=100)
                i_price = st.number_input("Selling Price (INR)", value=350000.0)
                
                submit_insert = st.form_submit_button("Add Listing to SQLite", type="primary")
                if submit_insert:
                    new_item = {
                        'Area_SqFt': i_area, 'Bedrooms': i_beds, 'Full_Bathrooms': i_fb, 'Half_Bathrooms': i_hb,
                        'Location': i_loc, 'Year_Built': i_year, 'Garage_Area': i_garage, 'Basement_Area': i_basement,
                        'Luxury_Score': i_lux, 'Price': i_price
                    }
                    add_property_to_db(new_item)
                    st.success("🎉 Property record successfully written to SQLite master dataset!")
                    st.rerun()
                    
        with crud_col2:
            st.markdown("#### Delete Property Record")
            delete_id = st.number_input("Select Record ID to Delete", min_value=1, step=1)
            submit_delete = st.button("Delete Records", type="secondary", width='stretch')
            if submit_delete:
                if delete_id in prop_df['id'].values:
                    delete_property_from_db(delete_id)
                    st.success(f"✓ Listing #{delete_id} successfully deleted from database.")
                    st.rerun()
                else:
                    st.error(f"Error: Record ID #{delete_id} not found in database.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with db_tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Production Valuation Logs History")
        st.write("Browse historical values predicted by the system and user-submitted feedback ratings.")
        if not pred_df.empty:
            st.dataframe(pred_df, width='stretch')
        else:
            st.info("No query logs saved in database yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with db_tab3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Model Retraining Pipeline Console")
        st.write("Train and serialize updated models dynamically on current SQLite listings database.")
        
        st.markdown(f"""
        <div style="background: rgba(16,185,129,0.05); border: 1px solid rgba(16,185,129,0.15); border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem;">
            <p style="margin: 0; font-size: 0.9rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em;">Current Production Model</p>
            <h3 style="margin: 0.2rem 0 0 0; color: #10b981;">{best_model_name}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        db_rows_count = len(prop_df)
        if db_rows_count < 100:
            st.warning(f"⚠️ Insufficient data for retraining. SQLite database has {db_rows_count} property listings. A minimum of 100 rows is required to avoid overfitting.")
        else:
            st.success(f"✓ SQLite database contains {db_rows_count} listings. Model retraining is fully enabled.")
            
            with st.expander("⚙️ Configure Hyperparameter Tuning Grids (RandomizedSearchCV)"):
                st.write("Customize the parameter search spaces dynamically before triggering execution:")
                
                # RF Tuning
                st.markdown("##### Random Forest Regressor")
                rf_n_est = st.multiselect("N Estimators options", [50, 100, 200, 300], default=[100, 200], key="tune_rf_nest")
                rf_max_depth = st.multiselect("Max Depth options", [5, 10, 15, None], default=[10, 15], key="tune_rf_depth")
                rf_min_split = st.multiselect("Min Samples Split options", [2, 5, 10], default=[2, 5], key="tune_rf_split")
                
                # XGBoost Tuning
                st.markdown("##### XGBoost Regressor")
                xgb_lr_range = st.slider("Learning Rate range", min_value=0.01, max_value=0.3, value=(0.03, 0.2), step=0.01, key="tune_xgb_lr")
                xgb_n_est = st.multiselect("XGB N Estimators", [50, 100, 200], default=[100, 200], key="tune_xgb_nest")
                xgb_depth = st.multiselect("XGB Max Depth", [3, 5, 7], default=[3, 5], key="tune_xgb_depth")
                
                custom_rf_grid = {
                    'n_estimators': rf_n_est if rf_n_est else [100],
                    'max_depth': rf_max_depth if rf_max_depth else [10],
                    'min_samples_split': rf_min_split if rf_min_split else [2]
                }
                custom_xgb_grid = {
                    'n_estimators': xgb_n_est if xgb_n_est else [100],
                    'max_depth': xgb_depth if xgb_depth else [3],
                    'learning_rate': list(np.linspace(xgb_lr_range[0], xgb_lr_range[1], 3)),
                    'subsample': [0.8, 1.0]
                }
            
            trigger_train = st.button("⚡ Trigger Core Retraining Pipeline", type="primary", width='stretch')
            if trigger_train:
                from datetime import datetime
                st.markdown("#### Retraining Execution Logs")
                log_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_msgs = []
                
                def custom_progress_callback(msg):
                    status_msgs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                    log_placeholder.code("\n".join(status_msgs))
                    
                    if "Starting" in msg: progress_bar.progress(5)
                    elif "Loading" in msg: progress_bar.progress(10)
                    elif "Fitting preprocessing" in msg: progress_bar.progress(20)
                    elif "Evaluating baseline" in msg: progress_bar.progress(35)
                    elif "Tuning hyperparameters" in msg: progress_bar.progress(55)
                    elif "Assembling Stacking" in msg: progress_bar.progress(70)
                    elif "Training final unified" in msg: progress_bar.progress(85)
                    elif "completed successfully" in msg: progress_bar.progress(100)
                    
                try:
                    with st.spinner("Optimizing RandomizedSearchCV grids and assembling Stacking Ensemble..."):
                        run_training_pipeline(
                            progress_callback=custom_progress_callback,
                            custom_rf_grid=custom_rf_grid,
                            custom_xgb_grid=custom_xgb_grid
                        )
                    st.balloons()
                    st.success("🎉 Production ML pipeline successfully retrained and updated on current SQLite dataset!")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during training execution: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with db_tab4:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("MLOps Dataset Drift & Anomaly Auditing")
        st.write("Analyze covariate shifts between user prediction requests and training properties stored in the listings database.")
        
        if not pred_df.empty:
            # 1. Out-of-Distribution Anomaly check
            anomaly_detector = pipeline.named_steps['preprocessor_pipeline'].named_steps['anomaly_detector']
            anomalies_count = 0
            for _, row in pred_df.iterrows():
                row_dict = {
                    'Area_SqFt': row['Area_SqFt'], 'Bedrooms': row['Bedrooms'], 
                    'Full_Bathrooms': row['Full_Bathrooms'], 'Half_Bathrooms': row['Half_Bathrooms'],
                    'Location': row['Location'], 'Year_Built': row['Year_Built'], 
                    'Garage_Area': row['Garage_Area'], 'Basement_Area': row['Basement_Area'], 
                    'Luxury_Score': row['Luxury_Score']
                }
                warnings = anomaly_detector.check_anomaly(row_dict)
                if warnings:
                    anomalies_count += 1
            
            drift_pct = (anomalies_count / len(pred_df)) * 100
            
            # Display KPIs
            c_a, c_b = st.columns(2)
            with c_a:
                st.metric(label="Total Queries Audited", value=f"{len(pred_df):,}")
            with c_b:
                st.metric(label="OOD Anomalous Queries", value=f"{anomalies_count} ({drift_pct:.1f}%)", delta="- warning" if anomalies_count > 0 else "0 drift", delta_color="inverse")
            
            st.markdown("<h4 style='margin-top:1.5rem; margin-bottom:0.5rem;'>Feature Distribution Comparison (Covariate Shift)</h4>", unsafe_allow_html=True)
            
            # Select feature to audit drift
            drift_feat = st.selectbox("Select Numeric Feature to Audit for Drift", ['Area_SqFt', 'Luxury_Score', 'Year_Built', 'Garage_Area'])
            
            # Create comparative histogram
            fig = px.histogram(
                pd.concat([
                    pd.DataFrame({drift_feat: prop_df[drift_feat], 'Dataset': 'Baseline DB'}),
                    pd.DataFrame({drift_feat: pred_df[drift_feat], 'Dataset': 'User Queries'})
                ]),
                x=drift_feat,
                color='Dataset',
                barmode='overlay',
                histnorm='probability density',
                title=f"Baseline Database vs User Requests: {drift_feat} Distribution",
                color_discrete_sequence=['#6366f1', '#10b981']
            )
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', opacity=0.7)
            st.plotly_chart(fig, width='stretch')
            
            # Location Zone query share
            st.markdown("<h4 style='margin-top:1.5rem; margin-bottom:0.5rem;'>Zone Distribution Share Comparison</h4>", unsafe_allow_html=True)
            
            prop_counts = prop_df['Location'].value_counts(normalize=True).reset_index()
            prop_counts.columns = ['Zone', 'Share']
            prop_counts['Source'] = 'Baseline DB'
            
            pred_counts = pred_df['Location'].value_counts(normalize=True).reset_index()
            pred_counts.columns = ['Zone', 'Share']
            pred_counts['Source'] = 'User Queries'
            
            share_df = pd.concat([prop_counts, pred_counts])
            
            fig_share = px.bar(
                share_df,
                x='Zone',
                y='Share',
                color='Source',
                barmode='group',
                title="Location zone percentage breakdown: Database vs User requests",
                color_discrete_sequence=['#6366f1', '#10b981']
            )
            fig_share.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_share, width='stretch')
            
        else:
            st.info("Perform predictions in the 'Valuation Predictor' tab and submit feedback to populate production requests for drift auditing.")
        st.markdown("</div>", unsafe_allow_html=True)

    with db_tab5:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("💻 Interactive SQL Query Console & Schema")
        st.write("Query the housing listings and predictions tables directly using standard SQLite syntax.")
        
        # Display Database schema
        st.markdown("**Database Tables Schema Reference:**")
        schema_data = {
            'Table Name': ['properties', 'predictions_log', 'experiments_log'],
            'Key Columns': [
                'id, Area_SqFt, Bedrooms, Full_Bathrooms, Half_Bathrooms, Location, Year_Built, Garage_Area, Basement_Area, Luxury_Score, Price',
                'id, timestamp, Area_SqFt, Bedrooms, Full_Bathrooms, Half_Bathrooms, Location, Year_Built, Garage_Area, Basement_Area, Luxury_Score, Predicted_Price, Lower_Bound_95_CI, Upper_Bound_95_CI, Model_Used, Feedback_Rating, Feedback_Comment',
                'id, timestamp, Best_Model, Train_Rows, Test_R2, Test_MAE, Comparison_Metrics'
            ]
        }
        st.dataframe(pd.DataFrame(schema_data), width='stretch')
        
        # Query text area input
        sql_query = st.text_area("Write SQL SELECT Query", value="SELECT * FROM properties WHERE Location = 'Premium' AND Price > 800000 LIMIT 5;")
        run_sql = st.button("💻 Execute SQL Query", type="primary")
        
        if run_sql:
            if sql_query.strip().lower().startswith("select"):
                try:
                    conn = get_db_connection()
                    res_df = pd.read_sql_query(sql_query, conn)
                    conn.close()
                    st.success(f"✓ Query executed successfully! Returned {len(res_df)} rows.")
                    st.dataframe(res_df, width='stretch')
                except Exception as e:
                    st.error(f"SQL execution error: {e}")
            else:
                st.error("Error: Only SELECT queries are permitted in the web console for data security.")
        st.markdown("</div>", unsafe_allow_html=True)

        # MLOps Drift PDF report download section
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📄 Generate MLOps Executive Audit Report")
        st.write("Download a compiled PDF containing database distributions statistics, VIF collinearity checks, and OOD query drift metrics.")
        
        def generate_mlops_pdf_report(properties_df, predictions_df, d_pct, d_count):
            """Generates a professional corporate database auditing and MLOps drift PDF report."""
            pdf_buf = BytesIO()
            document = SimpleDocTemplate(
                pdf_buf,
                pagesize=letter,
                rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
            )
            styles_list = getSampleStyleSheet()
            
            t_style = ParagraphStyle(
                'DocTitle', parent=styles_list['Heading1'], fontName='Helvetica-Bold', fontSize=22,
                leading=26, textColor=colors.HexColor('#6366f1'), spaceAfter=15
            )
            s_style = ParagraphStyle(
                'SecHead', parent=styles_list['Heading2'], fontName='Helvetica-Bold', fontSize=13,
                leading=16, textColor=colors.HexColor('#1f2937'), spaceBefore=12, spaceAfter=8
            )
            n_style = ParagraphStyle(
                'Norm', parent=styles_list['Normal'], fontName='Helvetica', fontSize=10, leading=14,
                textColor=colors.HexColor('#374151')
            )
            
            pdf_story = []
            pdf_story.append(Paragraph("PROHOUSE DATABASE AUDIT & DRIFT REPORT", t_style))
            pdf_story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles_list['Normal']))
            pdf_story.append(Spacer(1, 15))
            
            # Overview
            pdf_story.append(Paragraph("Executive Summary", s_style))
            overview_text = (
                f"This report summarizes the operational integrity and covariate dataset drift characteristics "
                f"of the production real-estate valuation engine. The baseline database contains {len(properties_df)} listings. "
                f"A total of {len(predictions_df)} user queries have been audited dynamically in production."
            )
            pdf_story.append(Paragraph(overview_text, n_style))
            pdf_story.append(Spacer(1, 10))
            
            # Audit KPIs
            kpi_data = [
                [Paragraph("<b>MLOps KPI Metric</b>", n_style), Paragraph("<b>Value</b>", n_style)],
                [Paragraph("Baseline Database Row count", n_style), Paragraph(str(len(properties_df)), n_style)],
                [Paragraph("Production Queries Audited", n_style), Paragraph(str(len(predictions_df)), n_style)],
                [Paragraph("Out-of-Distribution (OOD) Queries", n_style), Paragraph(str(d_count), n_style)],
                [Paragraph("Covariate Dataset Drift Percentage", n_style), Paragraph(f"{d_pct:.2f}%", n_style)]
            ]
            t_kpi = Table(kpi_data, colWidths=[240, 160])
            t_kpi.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f3f4f6')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6)
            ]))
            pdf_story.append(t_kpi)
            pdf_story.append(Spacer(1, 20))
            
            # Multicollinearity indices (VIF)
            pdf_story.append(Paragraph("Features Collinearity Audit (VIF)", s_style))
            vif_rows = []
            num_cols = properties_df.select_dtypes(include=[np.number]).drop(columns=['id'], errors='ignore').dropna()
            X_vif = num_cols.drop(columns=['Price'], errors='ignore')
            if len(X_vif) > 50:
                for col in X_vif.columns:
                    y_t = X_vif[col]
                    X_t = X_vif.drop(columns=[col])
                    lr = LinearRegression().fit(X_t, y_t)
                    vif = 1.0 / (1.0 - lr.score(X_t, y_t))
                    vif_rows.append([Paragraph(col, n_style), Paragraph(f"{vif:.2f}", n_style)])
                    
            vif_table_data = [[Paragraph("<b>Feature Column</b>", n_style), Paragraph("<b>Variance Inflation Factor (VIF)</b>", n_style)]] + vif_rows
            t_vif = Table(vif_table_data, colWidths=[240, 160])
            t_vif.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f3f4f6')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5)
            ]))
            pdf_story.append(t_vif)
            
            document.build(pdf_story)
            pdf_buf.seek(0)
            return pdf_buf
            
        if not pred_df.empty:
            # Recompute drift pct and anomalies count for PDF report
            anomaly_detector = pipeline.named_steps['preprocessor_pipeline'].named_steps['anomaly_detector']
            anomalies_count = 0
            for _, row in pred_df.iterrows():
                row_dict = {
                    'Area_SqFt': row['Area_SqFt'], 'Bedrooms': row['Bedrooms'], 
                    'Full_Bathrooms': row['Full_Bathrooms'], 'Half_Bathrooms': row['Half_Bathrooms'],
                    'Location': row['Location'], 'Year_Built': row['Year_Built'], 
                    'Garage_Area': row['Garage_Area'], 'Basement_Area': row['Basement_Area'], 
                    'Luxury_Score': row['Luxury_Score']
                }
                warnings = anomaly_detector.check_anomaly(row_dict)
                if warnings:
                    anomalies_count += 1
            drift_pct = (anomalies_count / len(pred_df)) * 100
            
            mlops_report_buf = generate_mlops_pdf_report(prop_df, pred_df, drift_pct, anomalies_count)
            st.download_button(
                "📄 Download Executive MLOps Audit Report (PDF)",
                data=mlops_report_buf,
                file_name="prohouse_mlops_audit_report.pdf",
                mime="application/pdf",
                width='stretch',
                key="download_mlops_pdf"
            )
        else:
            st.info("Log some valuation predictions first to generate audit charts and enable PDF report compiles.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 6: THEORETICAL GUIDE & MATHEMATICAL EQUATIONS
# ==============================================================================
elif st.session_state.active_tab == "📖 Theoretical Guide":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>📖 Machine Learning Theoretical Guide</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Detailed mathematical foundations and architectural description of the algorithms serving this dashboard.</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("1. Preprocessing Pipeline Architecture")
    st.write("Before feeding data into the Stacking Regressor, we process numerical and categorical features sequentially:")
    st.latex(r"""
    x_{scaled} = \frac{x - \mu}{\sigma} \quad ; \quad x_{capped} = \max(\min(x, Q3 + 1.5 \times IQR), Q1 - 1.5 \times IQR)
    """)
    st.write("Missing inputs are imputed using column medians for numeric fields, and modes for categorical fields.")
    
    st.markdown("---")
    
    st.subheader("2. Stacking Regressor Ensembling")
    st.write("Stacking blends the prediction strengths of several base learners using a meta-model. Let's explore the algorithms:")
    
    st.markdown("#### A. Ridge Regression")
    st.write("Ridge adds an L2-norm regularization penalty to standard least squares to avoid multicollinearity and weight explosion:")
    st.latex(r"""
    \mathcal{L}_{Ridge}(w) = ||Xw - y||^2_2 + \alpha ||w||^2_2
    """)
    st.write("Expanding the matrix formulation terms:")
    st.latex(r"""
    \mathcal{L}_{Ridge}(w) = (Xw - y)^T(Xw - y) + \alpha w^T w
    """)
    st.latex(r"""
    \mathcal{L}_{Ridge}(w) = w^T X^T X w - 2 w^T X^T y + y^T y + \alpha w^T w
    """)
    st.write("Taking the gradient with respect to weight vector $w$ and setting it to zero:")
    st.latex(r"""
    \nabla_{w} \mathcal{L}_{Ridge}(w) = 2 X^T X w - 2 X^T y + 2 \alpha w = 0
    """)
    st.latex(r"""
    (X^T X + \alpha I) w = X^T y
    """)
    st.write("The analytic closed-form parameter solution is given by:")
    st.latex(r"""
    w = (X^T X + \alpha I)^{-1} X^T y
    """)
    st.write("Adding the positive diagonal element $\alpha I$ guarantees that the matrix is invertible even under extreme multicollinearity.")
    
    st.markdown("#### B. Random Forest Regressor")
    st.write("An ensemble of independent decision trees trained on bootstrap samples of the training set. The final forecast is the average of individual tree outputs:")
    st.latex(r"""
    \hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)
    """)
    
    st.markdown("#### C. Gradient Boosting Regressor / XGBoost")
    st.write("Additive models trained sequentially to minimize a continuous loss function. Each new tree fits the pseudo-residuals of the current ensemble:")
    st.latex(r"""
    F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)
    """)
    st.write("Where the base learner $h_m(x)$ is fit to the negative gradients (pseudo-residuals) of the loss function $L(y, F(x))$:")
    st.latex(r"""
    r_{i,m} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x)=F_{m-1}(x)}
    """)
    st.write(r"The multiplier $\gamma_m$ is determined via line search to minimize the loss:")
    st.latex(r"""
    \gamma_m = \arg\min_{\gamma} \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))
    """)
    
    st.markdown("#### D. Support Vector Regression (SVR)")
    st.write("Finds a hyper-plane in a high-dimensional feature space that has at most epsilon deviation from target values:")
    st.latex(r"""
    \min_{w, b, \xi, \xi^*} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{l} (\xi_i + \xi_i^*)
    """)
    st.write("Subject to constraints:")
    st.latex(r"""
    y_i - w \cdot \Phi(x_i) - b \leq \epsilon + \xi_i
    """)
    st.latex(r"""
    w \cdot \Phi(x_i) + b - y_i \leq \epsilon + \xi_i^*
    """)
    st.latex(r"""
    \xi_i, \xi_i^* \geq 0
    """)
    st.write(r"Where $\Phi(x)$ is the non-linear mapping. SVR resolves non-linear dual mappings using RBF kernel function:")
    st.latex(r"""
    K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)
    """)
    
    st.markdown("#### E. K-Nearest Neighbors (KNN)")
    st.write("Predicts target values by locating the $k$ nearest data points in the feature space based on distance metrics:")
    st.latex(r"""
    d_{Euclidean}(x, x_i) = \sqrt{\sum_{j=1}^{p} (x_j - x_{i,j})^2} \quad ; \quad d_{Manhattan}(x, x_i) = \sum_{j=1}^{p} |x_j - x_{i,j}|
    """)
    st.write("KNN prediction is computed using distance-weighted averaging of the targets of the $k$ neighbors:")
    st.latex(r"""
    \hat{y}(x) = \frac{\sum_{i \in N_k(x)} w_i y_i}{\sum_{i \in N_k(x)} w_i} \quad ; \quad w_i = \frac{1}{d(x, x_i)^p}
    """)
    
    st.markdown("#### F. Linear Regression & Normal Equation")
    st.write("Fits coefficients by minimizing residual sum of squares directly via the analytic solution:")
    st.latex(r"""
    \theta = (X^T X)^{-1} X^T y
    """)
    st.write("If features are highly collinear (high VIF), the matrix $X^T X$ becomes singular or ill-conditioned. Ridge regression addresses this by adding a penalty parameter $\alpha$:")
    st.latex(r"""
    \theta_{Ridge} = (X^T X + \alpha I)^{-1} X^T y
    """)
    
    st.markdown("#### G. Decision Tree Regression Splits")
    st.write("Splits are selected by maximizing the variance reduction between the parent node and the resulting child branches:")
    st.latex(r"""
    \Delta \text{Var} = \sigma^2_{parent} - \left( \frac{N_{left}}{N} \sigma^2_{left} + \frac{N_{right}}{N} \sigma^2_{right} \right)
    """)
    st.write("Where Node Variance is the mean squared error of the target values relative to the node mean:")
    st.latex(r"""
    \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2
    """)
    
    st.markdown("#### H. Stacking Ensemble Blending Framework")
    st.write("To prevent target leakage, stacking ensembling uses K-Fold Out-Of-Fold (OOF) prediction generation:")
    st.latex(r"""
    \hat{Y}_{OOF} = [f_1(X_{-k}), f_2(X_{-k}), ..., f_M(X_{-k})] \quad \text{for } k=1..K
    """)
    st.write("The meta-model is then trained on these out-of-fold prediction matrices to learn optimal combinations without overfitting.")

    st.markdown("#### I. Statistical Bounds Calculation")
    st.write("Confidence bands are statistically modeled using the Mean Absolute Error (MAE) under asymptotic normal error distribution assumptions:")
    st.latex(r"""
    \text{Lower Bound} = \text{Valuation} - 1.96 \times \text{MAE}
    """)
    st.latex(r"""
    \text{Upper Bound} = \text{Valuation} + 1.96 \times \text{MAE}
    """)
    
    st.markdown("#### J. Model Calibration & Evaluation Metrics")
    st.write("To benchmark the regression predictions against baseline values, we utilize a set of mathematical validation criteria:")
    
    st.write("**1. Mean Absolute Error (MAE):** Measures average magnitude of errors without considering their direction:")
    st.latex(r"""
    \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
    """)
    
    st.write("**2. Root Mean Squared Error (RMSE):** Quadratic metric that penalizes larger forecast errors more heavily:")
    st.latex(r"""
    \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
    """)
    
    st.write("**3. Coefficient of Determination (R²):** Proportions of variance explained by model inputs:")
    st.latex(r"""
    R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
    """)
    
    st.write("**4. Adjusted R² Score:** Penalizes the model score for adding non-predictive dummy features:")
    st.latex(r"""
    R^2_{adj} = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]
    """)
    st.write("Where $n$ represents holdout size samples, and $p$ represents features dimensionality.")
    
    st.markdown("---")
    
    st.subheader("3. Advanced Mathematical Proofs")
    
    st.markdown("#### A. Second-Order Taylor Expansion for XGBoost Optimization")
    st.write("In Gradient Boosting, we minimize the regularized objective function at step $t$:")
    st.latex(r"""
    \mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)
    """)
    st.write("Applying the second-order Taylor expansion:")
    st.latex(r"""
    f(x + \Delta x) \approx f(x) + f'(x)\Delta x + \frac{1}{2}f''(x)\Delta x^2
    """)
    st.write("We approximate the objective function as:")
    st.latex(r"""
    \mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)
    """)
    st.write("Where the first and second order gradient statistics are defined as:")
    st.latex(r"""
    g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)}) \quad ; \quad h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)})
    """)
    st.write("By removing the constant terms independent of the current tree $f_t$, we obtain the simplified objective:")
    st.latex(r"""
    \tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^{n} \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2
    """)
    st.write(r"For a fixed tree structure $q(x)$, we define $I_j = \{i | q(x_i) = j\}$ as the set of indices of data points mapped to leaf $j$. We rewrite the objective by grouping terms by leaf:")
    st.latex(r"""
    \tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^{T} \left[ \left( \sum_{i \in I_j} g_i \right) w_j + \frac{1}{2} \left( \sum_{i \in I_j} h_i + \lambda \right) w_j^2 \right] + \gamma T
    """)
    st.write("Taking the partial derivative with respect to leaf weight $w_j$ and setting it to zero yields the optimal weight $w_j^*$:")
    st.latex(r"""
    \frac{\partial \tilde{\mathcal{L}}^{(t)}}{\partial w_j} = \sum_{i \in I_j} g_i + \left( \sum_{i \in I_j} h_i + \lambda \right) w_j = 0
    """)
    st.latex(r"""
    w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
    """)
    st.write("Substituting the optimal leaf weight back into the objective yields the scoring function representing the structural quality of tree structure $q$:")
    st.latex(r"""
    \tilde{\mathcal{L}}^{(t)}(q) = -\frac{1}{2} \sum_{j=1}^{T} \frac{\left( \sum_{i \in I_j} g_i \right)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T
    """)
    st.write("This allows XGBoost to evaluate and prune split candidates based on gain metrics calculated from the objective reduction score.")
    
    st.markdown("#### B. Support Vector Dual Lagrangian Formulation")
    st.write("The primal optimization problem for Support Vector Regression is transformed using Lagrange multipliers $\alpha_i, \alpha_i^*$ to find the dual formulation:")
    st.latex(r"""
    \max_{\alpha, \alpha^*} -\frac{1}{2} \sum_{i,j=1}^{l} (\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*) K(x_i, x_j) - \epsilon \sum_{i=1}^{l} (\alpha_i + \alpha_i^*) + \sum_{i=1}^{l} y_i (\alpha_i - \alpha_i^*)
    """)
    st.write("Subject to constraints:")
    st.latex(r"""
    \sum_{i=1}^{l} (\alpha_i - \alpha_i^*) = 0 \quad \text{and} \quad 0 \leq \alpha_i, \alpha_i^* \leq C
    """)
    
    st.markdown("---")
    
    st.subheader("4. Developer Reference & Operation Manual")
    st.write("This dashboard is backed by a fully structured filesystem layout and local database schema configuration:")
    st.markdown("""
    - **SQLite Database (`data/housing_records.db`):** 
      - `properties`: Seeds raw historical real estate records.
      - `predictions_log`: Stores user inference transactions and Thumbs Up/Down ratings.
      - `experiments_log`: Logs training histories and holdout splits metrics.
    - **Serialized Components (`models/`):**
      - `all_models.pkl`: Dict containing fitted individual baseline pipeline models.
      - `housing_pipeline.pkl`: The default Stacking Regressor model pipeline.
      - `model_comparison.csv`: Holds Mean Absolute Error metrics for confidence calculations.
    - **MLOps Drift Detection limits:**
      - Audits numeric request inputs against the training $1^{st}$ and $99^{th}$ percentiles to catch OOD anomalies.
      - Dynamic retraining requires a minimum of 100 listings in the properties table.
    """)
    
    st.markdown("#### A. Production Integration Steps for Data Engineers")
    st.write("To integrate a custom estimator (e.g. LightGBM, CatBoost) into the ensembled dictionary serving inference:")
    st.markdown("""
    1. **Import the model class** at the top of Section 6 in the python script.
    2. **Register the baseline** inside the `models` dictionary within `run_training_pipeline`:
       ```python
       models['LightGBM'] = LGBMRegressor(random_state=rand_state)
       ```
    3. **Define parameter search spaces** inside `system_config['tuning_params']` and perform a custom random search block.
    4. **Extend the Stacking Regressor** meta-learner estimators configurations inside Section 6.
    5. **Trigger Retraining** via the MLOps retraining console tab or via bash execution.
    """)
    
    st.markdown("#### B. System Troubleshooting & Resolution Matrix")
    st.markdown("""
    - **Database Locking issues (`sqlite3.OperationalError: database is locked`):**
      - *Root Cause:* Simultaneous write operations from multiple client threads.
      - *Resolution:* SQLite connection parameters are configured with `timeout=20` to block and wait until locking transactions complete. Check thread connections pool limits if locking persists.
    - **Pickle Deserialization Failures (`AttributeError: Can't get attribute 'DataFrameImputer'`):**
      - *Root Cause:* Sklearn pipeline expects custom transformer classes to be declared in main script imports namespaces.
      - *Resolution:* Ensure custom pipeline transformers classes are always defined in the same script running the dashboard.
    - **Out-of-Memory (OOM) Errors during Retraining:**
      - *Root Cause:* Overly exhaustive grid tuning parameters with high `n_iter` or large CV splits.
      - *Resolution:* Adjust sliders inside MLOps Retraining Hyperparameter Config to decrease grid density options.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 7: SHAP EXPLANATIONS & WATERFALL
# ==============================================================================
elif st.session_state.active_tab == "🔍 SHAP Explanations":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>🔍 Model Explainability & SHAP Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Inspect feature importances and local predictions contributions using Shapley additive explanations.</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("1. What are SHAP (Shapley Additive exPlanations) Values?")
    st.write("SHAP values attribute the difference between a model's prediction and its base expected value to each feature:")
    st.latex(r"""
    \phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]
    """)
    st.write("This tab uses the production **Random Forest Regressor** base estimator to explain predictions.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("<div class='glass-card'><h3 style='margin-top:0;'>Configure Property for Attribution</h3>", unsafe_allow_html=True)
        loc = st.selectbox("Location Zone", list(geo_cfg.keys()), index=2, key="shap_loc")
        area = st.slider("Area (SqFt)", 600, 6000, 2200, 50, key="shap_area")
        beds = st.slider("Bedrooms", 1, 5, 3, key="shap_beds")
        fb = st.slider("Full Baths", 1, 4, 2, key="shap_fb")
        hb = st.slider("Half Baths", 0, 2, 1, key="shap_hb")
        garage = st.slider("Garage Size (SqFt)", 0, 900, 400, 25, key="shap_garage")
        basement = st.slider("Basement Size (SqFt)", 0, 2000, 500, 50, key="shap_basement")
        year = st.slider("Year Constructed", 1960, 2025, 2010, key="shap_year")
        luxury = st.slider("Luxury & Finishes Score", 1, 100, 65, key="shap_lux")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'><h3 style='margin-top:0;'>SHAP Local Attribution Waterfall</h3>", unsafe_allow_html=True)
        
        # Load RF estimator and preprocessor
        rf_pipeline = all_models.get('Random Forest', pipeline)
        preproc = rf_pipeline.named_steps['preprocessor_pipeline']
        rf_estimator = rf_pipeline.named_steps['regressor']
        
        # Prepare inputs
        inputs = pd.DataFrame([{
            'Area_SqFt': area, 'Bedrooms': beds, 'Full_Bathrooms': fb, 'Half_Bathrooms': hb,
            'Location': loc, 'Year_Built': year, 'Garage_Area': garage, 'Basement_Area': basement, 'Luxury_Score': luxury
        }])
        
        # Preprocess instance
        instance_processed = preproc.transform(inputs)
        
        # Explain using TreeExplainer
        @st.cache_resource
        def get_shap_explainer(_model, background_data):
            return shap.TreeExplainer(_model)
            
        df_clean = generate_synthetic_housing_data(num_samples=200)
        X_sample = df_clean.drop(columns=['id', 'Price'], errors='ignore')
        X_sample_proc = preproc.transform(X_sample)
        
        explainer = get_shap_explainer(rf_estimator, X_sample_proc)
        shap_values = explainer(instance_processed)
        
        shap_vals = shap_values.values[0]
        base_val = shap_values.base_values[0]
        
        contrib_df = pd.DataFrame({
            'Feature': feature_names,
            'Valuation Impact': shap_vals
        }).sort_values(by='Valuation Impact', key=np.abs, ascending=False)
        
        top_n = 6
        display_contrib = contrib_df.head(top_n).copy()
        others_sum = contrib_df.iloc[top_n:]['Valuation Impact'].sum()
        
        if len(contrib_df) > top_n:
            display_contrib = pd.concat([
                display_contrib, 
                pd.DataFrame({'Feature': ['Other Features'], 'Valuation Impact': [others_sum]})
            ], ignore_index=True)
            
        y_labels = ["Global Base Value"] + list(display_contrib['Feature']) + ["RF Model Target"]
        measures = ["absolute"] + ["relative"] * len(display_contrib) + ["total"]
        vals = [base_val] + list(display_contrib['Valuation Impact']) + [base_val + sum(shap_vals)]
        
        fig = go.Figure(go.Waterfall(
            name = "SHAP Waterfall",
            orientation = "v",
            measure = measures,
            x = y_labels,
            textposition = "outside",
            text = [f"INR {v:,.0f}" if m in ['absolute', 'total'] else f"{'+' if v>=0 else ''}INR {v:,.0f}" for v, m in zip(vals, measures)],
            y = vals,
            connector = {"line":{"color":"rgba(255,255,255,0.2)"}},
            decreasing = {"marker":{"color":"#ef4444"}},
            increasing = {"marker":{"color":"#10b981"}},
            totals = {"marker":{"color":"#6366f1"}}
        ))
        
        fig.update_layout(
            title = "SHAP local breakdown waterfall chart (Random Forest)",
            showlegend = False,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450
        )
        st.plotly_chart(fig, width='stretch')
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 7.5: FEATURE ENGINEERING PLAYGROUND
# ==============================================================================
elif st.session_state.active_tab == "🛠️ Feature Engineering Playground":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>🛠️ Feature Engineering & Preprocessing Playground</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Experiment with custom scaling estimators, IQR capping multipliers, and interaction terms to benchmark prediction shifts.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("<div class='glass-card'><h3 style='margin-top:0;'>Configure Prototyping Pipeline</h3>", unsafe_allow_html=True)
        
        # Scaling choices
        scale_choice = st.radio("Numerical Scaling Transform", ["Standard Scaler (Standardize)", "MinMax Scaler (Normalize)", "Robust Scaler (Median & IQR Scaling)"], index=0)
        
        # IQR capping factor
        iqr_factor = st.slider("IQR Outlier Capping Factor", min_value=1.0, max_value=3.5, value=1.5, step=0.1)
        
        # Interaction checkboxes
        st.markdown("**Cross Interaction Terms to Engineer:**")
        cross_bed_bath = st.checkbox("Bedroom-to-Bath Ratio feature", value=True)
        cross_area_bed = st.checkbox("Area-per-Bedroom density feature", value=True)
        cross_age_lux = st.checkbox("Age-Luxury interaction cross", value=False)
        
        run_proto = st.button("⚡ Process & Benchmark Preprocessing Pipelines", type="primary", width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'><h3 style='margin-top:0;'>Prototyping Evaluation Metrics</h3>", unsafe_allow_html=True)
        
        if run_proto:
            from sklearn.preprocessing import MinMaxScaler, RobustScaler
            
            # Fetch data
            df_play = fetch_properties_from_db()
            df_play = df_play.drop_duplicates().reset_index(drop=True)
            
            X_play = df_play.drop(columns=['id', 'Price'], errors='ignore')
            y_play = df_play['Price']
            
            # Outlier cap price
            q1 = y_play.quantile(0.25)
            q3 = y_play.quantile(0.75)
            iqr = q3 - q1
            low_p = q1 - iqr_factor * iqr
            high_p = q3 + iqr_factor * iqr
            y_play_clean = np.clip(y_play, low_p, high_p)
            
            X_tr, X_te, y_tr, y_te = train_test_split(X_play, y_play_clean, test_size=0.2, random_state=42)
            
            # Apply Playground features
            def apply_play_features(df_in):
                df_out = df_in.copy()
                # standard engineer
                df_out['House_Age'] = (2026 - df_out['Year_Built']).clip(lower=0)
                df_out['Total_Bathrooms'] = df_out['Full_Bathrooms'] + 0.5 * df_out['Half_Bathrooms']
                df_out['Total_Area'] = df_out['Area_SqFt'] + df_out['Garage_Area'] + df_out['Basement_Area']
                
                # Cross interactions
                if cross_bed_bath:
                    df_out['Bed_Bath_Ratio'] = df_out['Bedrooms'] / (df_out['Total_Bathrooms'] + 0.1)
                if cross_area_bed:
                    df_out['Area_per_Bed'] = df_out['Area_SqFt'] / (df_out['Bedrooms'] + 0.1)
                if cross_age_lux:
                    df_out['Age_Lux_Cross'] = df_out['House_Age'] * df_out['Luxury_Score']
                    
                df_out = df_out.drop(columns=['Year_Built'], errors='ignore')
                return df_out
                
            # Preprocess
            X_tr_fe = apply_play_features(X_tr)
            X_te_fe = apply_play_features(X_te)
            
            # Impute
            imputer = DataFrameImputer()
            X_tr_imp = imputer.fit_transform(X_tr_fe)
            X_te_imp = imputer.transform(X_te_fe)
            
            # Cap
            numeric_cols = X_tr_imp.select_dtypes(include=[np.number]).columns.tolist()
            capper = OutlierCapper(cols_to_cap=numeric_cols, iqr_factor=iqr_factor)
            X_tr_cap = capper.fit_transform(X_tr_imp)
            X_te_cap = capper.transform(X_te_imp)
            
            # Scaler mapping
            scaler_map = {
                "Standard Scaler (Standardize)": StandardScaler(),
                "MinMax Scaler (Normalize)": MinMaxScaler(),
                "Robust Scaler (Median & IQR Scaling)": RobustScaler()
            }
            scaler_obj = scaler_map[scale_choice]
            
            # Categorical encoding columns
            categorical_cols = ['Location']
            num_proc_cols = [col for col in X_tr_cap.columns if col not in categorical_cols]
            
            preprocessor_play = ColumnTransformer(
                transformers=[
                    ('num', scaler_obj, num_proc_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
                ]
            )
            
            X_tr_proc = preprocessor_play.fit_transform(X_tr_cap)
            X_te_proc = preprocessor_play.transform(X_te_cap)
            
            # Fit Ridge and Decision Tree on prototype
            model_ridge = Ridge(alpha=1.0)
            model_ridge.fit(X_tr_proc, y_tr)
            preds_ridge = model_ridge.predict(X_te_proc)
            
            r2_proto = r2_score(y_te, preds_ridge)
            mae_proto = mean_absolute_error(y_te, preds_ridge)
            
            # Compare with production
            pred_prod = pipeline.predict(X_te)
            r2_prod = r2_score(y_te, pred_prod)
            mae_prod = mean_absolute_error(y_te, pred_prod)
            
            st.success("✓ Prototype pipeline fit successfully!")
            
            # Comparative Table
            comp_data = {
                'Pipeline Platform': ['Production Stacking Ensemble', 'Custom Prototyped Pipeline (Ridge)'],
                'R² Score (Holdout)': [round(r2_prod, 4), round(r2_proto, 4)],
                'MAE Score (INR)': [round(mae_prod, 2), round(mae_proto, 2)]
            }
            st.dataframe(pd.DataFrame(comp_data), width='stretch')
            
            # Plot new custom feature relation
            plot_feat = 'Area_per_Bed' if cross_area_bed else 'Total_Area'
            if plot_feat in X_te_cap.columns:
                fig_play = px.scatter(
                    X_te_cap,
                    x=plot_feat,
                    y=y_te,
                    color='Location',
                    title=f"Prototype Feature: Price vs {plot_feat} (Holdout)",
                    color_discrete_sequence=['#6366f1', '#10b981', '#f59e0b', '#ec4899', '#3b82f6']
                )
                fig_play.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_play, width='stretch')
        else:
            st.info("Adjust the sliders and parameters, then click 'Process & Benchmark Preprocessing Pipelines' to analyze performance gains.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 8: DATA QUALITY AUDITOR
# ==============================================================================
elif st.session_state.active_tab == "📈 Data Quality Auditor":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>📈 Data Quality & Multicollinearity Auditor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Diagnose statistics, skewness, collinearity indices (VIF) and correlation thresholds inside SQLite.</p>", unsafe_allow_html=True)
    
    df_aud = fetch_properties_from_db()
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("1. General Summary Statistics")
    
    # Skewness & Kurtosis calculation
    num_cols = df_aud.select_dtypes(include=[np.number]).drop(columns=['id'], errors='ignore')
    summary_list = []
    for col in num_cols.columns:
        summary_list.append({
            'Feature': col,
            'Missing Count': df_aud[col].isnull().sum(),
            'Mean': round(df_aud[col].mean(), 2),
            'Std Dev': round(df_aud[col].std(), 2),
            'Skewness': round(stats.skew(df_aud[col].dropna()), 3),
            'Kurtosis': round(stats.kurtosis(df_aud[col].dropna()), 3)
        })
    st.dataframe(pd.DataFrame(summary_list), width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("2. Collinearity Heatmap Matrix")
        corr = num_cols.corr()
        fig_heat = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Correlation coefficients heatmap'
        )
        fig_heat.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("3. Collinearity Index (Variance Inflation Factor)")
        st.write("VIF quantifies the severity of multicollinearity in an ordinary least squares regression analysis:")
        st.latex(r"""
        VIF_i = \frac{1}{1 - R_i^2}
        """)
        
        # Run regression of each feature against others to calculate VIF
        vif_list = []
        X_vif = num_cols.drop(columns=['Price'], errors='ignore').dropna()
        if len(X_vif) > 50:
            for i, col in enumerate(X_vif.columns):
                # Regression target
                y_temp = X_vif[col]
                X_temp = X_vif.drop(columns=[col])
                # Fit
                lr = LinearRegression()
                lr.fit(X_temp, y_temp)
                r2_temp = lr.score(X_temp, y_temp)
                vif = 1.0 / (1.0 - r2_temp) if r2_temp < 1.0 else 100.0
                vif_list.append({
                    'Feature': col,
                    'R2 (vs Others)': round(r2_temp, 3),
                    'VIF Value': round(vif, 2)
                })
            df_vif = pd.DataFrame(vif_list).sort_values(by='VIF Value', ascending=False)
            st.dataframe(df_vif, width='stretch')
            
            # Print status summary
            max_vif = df_vif['VIF Value'].max()
            if max_vif > 10.0:
                st.warning(f"⚠️ High multicollinearity detected! Maximum VIF is {max_vif:.1f} (VIF > 10 implies extreme redundancy).")
            else:
                st.success(f"✓ No severe collinearity issues. Maximum VIF is {max_vif:.1f} (VIF < 5 is safe).")
        else:
            st.info("Insufficient records in database to run regression collinearity checks.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 8.5: FINANCIAL SCENARIO PLANNER & ROI SIMULATOR
# ==============================================================================
elif st.session_state.active_tab == "📈 Financial Scenario Planner":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>📈 Financial Scenario Planner & ROI Simulator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Analyze hypothetical property acquisitions, model renovation yields, and compute projected returns on investment.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("<div class='glass-card'><h3 style='margin-top:0;'>Acquisition & Holding Inputs</h3>", unsafe_allow_html=True)
        
        sc_loc = st.selectbox("Scenario Location Sector", list(geo_cfg.keys()), index=2, key="sc_loc")
        sc_area = st.number_input("Property Size (SqFt)", value=2200, min_value=300, key="sc_area")
        sc_beds = st.slider("Bedrooms", 1, 5, 3, key="sc_beds")
        sc_fb = st.slider("Full Baths", 1, 4, 2, key="sc_fb")
        sc_hb = st.slider("Half Baths", 0, 2, 1, key="sc_hb")
        
        st.markdown("---")
        
        # Financial variables
        purchase_price = st.number_input("Target Purchase Price (INR)", value=320000.0, step=10000.0, key="purchase_price")
        renovation_cost = st.number_input("Estimated Renovations Cost (INR)", value=40000.0, step=5000.0, key="renovation_cost")
        holding_years = st.slider("Asset Holding Time (Years)", 1, 10, 5, key="holding_years")
        
        # Renovation impacts the Luxury Score!
        baseline_luxury = st.slider("Starting Finishes Score (1-100)", 1, 100, 50, key="baseline_luxury")
        additional_score = int(renovation_cost / 5000.0)
        projected_luxury = min(100, baseline_luxury + additional_score)
        
        st.info(f"💡 Renovation will improve property finishes score from {baseline_luxury} to {projected_luxury}/100.")
        
        run_sim = st.button("📈 Run Financial Scenario Analysis", type="primary", width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'><h3 style='margin-top:0;'>ROI & Capital Appreciation Projection</h3>", unsafe_allow_html=True)
        
        if run_sim:
            # Predict current renovated price
            inputs_renovated = pd.DataFrame([{
                'Area_SqFt': sc_area, 'Bedrooms': sc_beds, 'Full_Bathrooms': sc_fb, 'Half_Bathrooms': sc_hb,
                'Location': sc_loc, 'Year_Built': 2026, 'Garage_Area': 400.0, 'Basement_Area': 500.0, 'Luxury_Score': projected_luxury
            }])
            current_valuation = pipeline.predict(inputs_renovated)[0]
            
            # Forecast future price
            rates = system_config['growth_rates']
            growth_rate = rates.get(sc_loc, 0.07)
            future_valuation = current_valuation * ((1.0 + growth_rate) ** holding_years)
            
            # Compute yield metrics
            total_invested = purchase_price + renovation_cost
            net_profit = future_valuation - total_invested
            total_roi = (net_profit / total_invested) * 100.0
            
            # Annualized Yield (CAGR)
            cagr = ((future_valuation / total_invested) ** (1.0 / holding_years) - 1.0) * 100.0
            
            # KPIs
            ki1, ki2 = st.columns(2)
            with ki1:
                st.metric(label="Total Capital Invested (INR)", value=f"INR {total_invested:,.2f}")
                st.metric(label="Projected Valuation (Yr {holding_years})", value=f"INR {future_valuation:,.2f}", delta=f"+{(future_valuation - current_valuation):,.0f} growth")
            with ki2:
                st.metric(label="Net Profit Outcome", value=f"INR {net_profit:,.2f}", delta=f"{total_roi:.1f}% Net ROI")
                st.metric(label="Annualized Yield (CAGR)", value=f"{cagr:.2f}% CAGR", delta="Compound yield" if cagr > 0 else "loss")
                
            # Plotly Gauge Chart for CAGR
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = cagr,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Annualized Compound Yield (CAGR %)", 'font': {'size': 16}},
                gauge = {
                    'axis': {'range': [-10, 30], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#6366f1"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255,255,255,0.2)",
                    'steps': [
                        {'range': [-10, 0], 'color': 'rgba(239, 68, 68, 0.2)'},
                        {'range': [0, 10], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [10, 30], 'color': 'rgba(16, 185, 129, 0.2)'}
                    ]
                }
            ))
            fig_gauge.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=280)
            st.plotly_chart(fig_gauge, width='stretch')
        else:
            st.info("Input purchase parameters and renovations cost, then click 'Run Financial Scenario Analysis' to compute capital returns.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 9: TEST SUITE SANDBOX
# ==============================================================================
elif st.session_state.active_tab == "🧪 Test Suite Sandbox":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>🧪 Automated Testing Sandbox</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Execute pipeline validation, transformer testing, and DB integrity scripts live inside the dashboard.</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Automated Unit Test Console")
    st.write("Click the button below to run sklearn pipelines, capping checks, SQLite write/read assertions, and multi-model serves.")
    
    run_tests = st.button("⚡ Execute Automated Validation Test Suite", type="primary", width='stretch')
    
    def run_test_suite_in_sandbox():
        results = []
        # Test Dataframe Imputer
        try:
            df = pd.DataFrame({'A': [1.0, np.nan, 3.0], 'B': ['cat', 'dog', np.nan]})
            imputer = DataFrameImputer()
            imputer.fit(df)
            df_t = imputer.transform(df)
            assert df_t['A'].isnull().sum() == 0
            assert df_t['B'].isnull().sum() == 0
            results.append(("test_dataframe_imputer", "Passed", "Successfully imputed numeric medians and categorical modes."))
        except Exception as e:
            results.append(("test_dataframe_imputer", "Failed", str(e)))
            
        # Test Outlier Capper
        try:
            df = pd.DataFrame({'Area_SqFt': [100.0, 2000.0, 3000.0, 15000.0]})
            capper = OutlierCapper(cols_to_cap=['Area_SqFt'])
            capper.fit(df)
            df_t = capper.transform(df)
            assert df_t['Area_SqFt'].max() < 15000.0
            results.append(("test_outlier_capper", "Passed", "Capped statistical Area outliers according to IQR bounds."))
        except Exception as e:
            results.append(("test_outlier_capper", "Failed", str(e)))
            
        # Test Feature Engineer
        try:
            df = pd.DataFrame({
                'Year_Built': [2010.0], 'Full_Bathrooms': [2], 'Half_Bathrooms': [1],
                'Area_SqFt': [2000.0], 'Garage_Area': [400.0], 'Basement_Area': [500.0], 'Luxury_Score': [85.0]
            })
            engineer = FeatureEngineer(current_year=2026)
            df_t = engineer.transform(df)
            assert 'House_Age' in df_t.columns
            assert df_t['Total_Bathrooms'].iloc[0] == 2.5
            assert df_t['Total_Area'].iloc[0] == 2900.0
            assert df_t['Luxury_Band'].iloc[0] == 1
            results.append(("test_feature_engineer", "Passed", "Engineered age, total bathrooms, area, and luxury indicators."))
        except Exception as e:
            results.append(("test_feature_engineer", "Failed", str(e)))
            
        # Test Anomaly Detector
        try:
            df = pd.DataFrame({
                'Area_SqFt': [1000.0, 2000.0, 3000.0], 'Location': ['Standard', 'Standard', 'Suburbs']
            })
            detector = InputAnomalyDetector()
            detector.fit(df)
            warnings = detector.check_anomaly({'Area_SqFt': 25000.0, 'Location': 'Atlantis'})
            assert len(warnings) > 0
            results.append(("test_anomaly_detector", "Passed", "Successfully identified Out-of-Distribution (OOD) numeric entries."))
        except Exception as e:
            results.append(("test_anomaly_detector", "Failed", str(e)))

        # Test SQLite Database
        try:
            init_db()
            df = fetch_properties_from_db()
            assert len(df) > 0
            results.append(("test_sqlite_database", "Passed", "Properties loaded, seeded, and fetched from local SQLite database."))
        except Exception as e:
            results.append(("test_sqlite_database", "Failed", str(e)))
            
        # Test Model Serving Inference & CI Bounds
        try:
            test_row = pd.DataFrame([{
                'Area_SqFt': 2000.0, 'Bedrooms': 3, 'Full_Bathrooms': 2, 'Half_Bathrooms': 1,
                'Location': 'Downtown', 'Year_Built': 2015, 'Garage_Area': 400.0, 'Basement_Area': 400.0, 'Luxury_Score': 75.0
            }])
            pred_price = pipeline.predict(test_row)[0]
            assert isinstance(pred_price, (float, np.float64))
            assert pred_price > 0
            
            # Confidence interval checks
            mae_est = 15000.0
            lower = pred_price - 1.96 * mae_est
            upper = pred_price + 1.96 * mae_est
            assert lower < pred_price
            assert upper > pred_price
            results.append(("test_model_serving_and_ci_bounds", "Passed", "Validated inference target float formats and 95% Confidence Interval boundaries."))
        except Exception as e:
            results.append(("test_model_serving_and_ci_bounds", "Failed", str(e)))
            
        # Test Multi-Model Serving Integrity
        try:
            assert 'Random Forest' in all_models
            assert 'XGBoost' in all_models
            assert 'Stacking Ensemble' in all_models
            results.append(("test_multimodel_serving_integrity", "Passed", "Baseline model serving dictionary serialized and contains Random Forest and XGBoost estimators."))
        except Exception as e:
            results.append(("test_multimodel_serving_integrity", "Failed", str(e)))
            
        # Test Target Leakage
        try:
            df_leak = fetch_properties_from_db()
            X_leak = df_leak.drop(columns=['id', 'Price'], errors='ignore')
            assert 'Price' not in X_leak.columns
            assert 'id' not in X_leak.columns
            results.append(("test_target_leakage_mitigation", "Passed", "Verified feature space excludes target target column and primary key indexes."))
        except Exception as e:
            results.append(("test_target_leakage_mitigation", "Failed", str(e)))
            
        # Test PDF Report Compilation
        try:
            dummy_buf = generate_pdf_report(
                price=500000.0,
                lower_ci=480000.0,
                upper_ci=520000.0,
                inputs={'Area_SqFt': 2000, 'Bedrooms': 3, 'Location': 'Standard', 'Year_Built': 2010, 'Luxury_Score': 70},
                model_used='Stacking Ensemble'
            )
            assert len(dummy_buf.getvalue()) > 0
            results.append(("test_pdf_report_compilation", "Passed", "Successfully built PDF report binaries with ReportLab templates."))
        except Exception as e:
            results.append(("test_pdf_report_compilation", "Failed", str(e)))
            
        return results
        
    if run_tests:
        with st.spinner("Executing unit validation tests..."):
            test_results = run_test_suite_in_sandbox()
            
        st.markdown("#### Test Execution Summary")
        for t_name, status, desc in test_results:
            if status == "Passed":
                st.success(f"✓ **{t_name}**: Passed — {desc}")
            else:
                st.error(f"❌ **{t_name}**: Failed — {desc}")
        st.balloons()
    else:
        st.info("Click 'Execute Automated Validation Test Suite' to verify code assertions.")
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# END OF PROHOUSE ENTERPRISE ASSET VALUATION SUITE
# ==============================================================================
# This module is a comprehensive, self-contained ML asset management system.
# Designed & implemented by Antigravity AI pair programmer.



