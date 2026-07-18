import os
import json
import yaml
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from generate_data import generate_synthetic_data

def load_config():
    """Dynamically resolves and loads the YAML configuration relative to utils.py location."""
    base_dir = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(base_dir, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# ==========================================
# ENTERPRISE EXPERIMENT TRACKER
# ==========================================

class ExperimentTracker:
    """
    Tracks and logs training runs and ensembling results to models/experiment_log.json.
    Provides metadata records for the dashboard to trace version performance.
    """
    def __init__(self, log_path=None):
        config = load_config()
        self.log_path = log_path or config['paths']['experiment_json']
        # Resolve absolute log path
        base_dir = os.path.abspath(os.path.dirname(__file__))
        self.log_path = os.path.join(base_dir, self.log_path)
        
    def log_run(self, model_name, r2_mean, r2_std, mae_mean, mae_std, parameters=None):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        logs = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    logs = json.load(f)
            except Exception:
                logs = []
                
        run_log = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': model_name,
            'cv_r2_mean': float(r2_mean),
            'cv_r2_std': float(r2_std),
            'cv_mae_mean': float(mae_mean),
            'cv_mae_std': float(mae_std),
            'parameters': parameters or {}
        }
        logs.append(run_log)
        
        with open(self.log_path, 'w') as f:
            json.dump(logs, f, indent=4)
            
    def get_logs(self):
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

# ==========================================
# ANOMALY & DRIFT DETECTION
# ==========================================

class InputAnomalyDetector(BaseEstimator, TransformerMixin):
    """
    Fits boundaries on the training set to identify Out-Of-Distribution (OOD) queries.
    Determines if user slider input falls outside 1st - 99th percentile ranges.
    """
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
        """
        Validates dictionary of properties inputs.
        Returns warning messages if values fall outside historical bounds.
        """
        warnings = []
        for col, val in input_dict.items():
            # Handle list structures if passed (e.g. dataframe inputs)
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
        # Passthrough
        return X

# ==========================================
# FUTURE VALUATION PROJECTIONS
# ==========================================

def forecast_valuation(base_price, location, years=5):
    """Computes compound interest growth forecasts based on YAML growth factors."""
    config = load_config()
    rates = config['growth_rates']
    growth_rate = rates.get(location, 0.08) # 8% default fallback
    
    forecasts = {}
    current_year = 2026
    current_val = base_price
    
    for i in range(1, years + 1):
        year = current_year + i
        current_val *= (1 + growth_rate)
        forecasts[year] = round(current_val, 2)
        
    return forecasts

# ==========================================
# DATA LOADING & CLEANING
# ==========================================

def load_data(filepath=None):
    """Loads the dataset from the SQLite database. Auto-seeds it if empty."""
    init_db()
    df = fetch_properties_from_db()
    # Drop SQLite metadata columns for standard downstream pipelines compatibility
    df = df.drop(columns=['id', 'created_at'], errors='ignore')
    return df

def clean_data(df):
    """Cleans raw training dataframe by removing duplicates and capping the target variable 'Price'."""
    df_cleaned = df.copy()
    config = load_config()
    iqr_factor = config['outliers']['iqr_factor']
    
    # 1. Remove duplicates
    initial_shape = df_cleaned.shape
    df_cleaned = df_cleaned.drop_duplicates()
    duplicates_removed = initial_shape[0] - df_cleaned.shape[0]
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows.")
        
    # 2. Cap outliers in Target (Price) using IQR
    if 'Price' in df_cleaned.columns:
        Q1 = df_cleaned['Price'].quantile(0.25)
        Q3 = df_cleaned['Price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        
        outliers_count = len(df_cleaned[(df_cleaned['Price'] < lower_bound) | (df_cleaned['Price'] > upper_bound)])
        df_cleaned['Price'] = np.clip(df_cleaned['Price'], lower_bound, upper_bound)
        print(f"Target variable 'Price': capped {outliers_count} outliers. Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
    return df_cleaned


# ==========================================
# PRODUCTION PIPELINE CUSTOM TRANSFORMERS
# ==========================================

class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Custom Transformer to impute missing values in pandas DataFrames."""
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
    """Custom Transformer to cap outliers on numeric columns using IQR."""
    def __init__(self, cols_to_cap=None):
        self.cols_to_cap = cols_to_cap or ['Area_SqFt', 'Garage_Area', 'Basement_Area']
        self.bounds_ = {}
        
    def fit(self, X, y=None):
        config = load_config()
        iqr_factor = config['outliers']['iqr_factor']
        for col in self.cols_to_cap:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_factor * IQR
                upper_bound = Q3 + iqr_factor * IQR
                self.bounds_[col] = (lower_bound, upper_bound)
        return self
        
    def transform(self, X):
        X_out = X.copy()
        for col, bounds in self.bounds_.items():
            if col in X_out.columns:
                X_out[col] = np.clip(X_out[col], bounds[0], bounds[1])
        return X_out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom Transformer for feature engineering calculations."""
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
            
        # Luxury Band
        if 'Luxury_Score' in X_out.columns:
            X_out['Luxury_Band'] = (X_out['Luxury_Score'] > 75).astype(int)
            
        return X_out


def get_preprocessing_pipeline():
    """Constructs the complete preprocessing and validation Pipeline."""
    num_cols = ['Area_SqFt', 'Bedrooms', 'Full_Bathrooms', 'Half_Bathrooms', 
                'Garage_Area', 'Basement_Area', 'Luxury_Score', 'House_Age', 
                'Total_Bathrooms', 'Total_Area', 'Luxury_Band']
    cat_cols = ['Location']
    
    # Preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
    ])
    
    # Production Pipeline
    pipeline = Pipeline(steps=[
        ('imputer', DataFrameImputer()),
        ('anomaly_detector', InputAnomalyDetector()), # Tracks percentiles bounds
        ('capper', OutlierCapper(cols_to_cap=['Area_SqFt', 'Garage_Area', 'Basement_Area'])),
        ('engineer', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline


# ==========================================
# BACKWARD COMPATIBLE legacy code
# ==========================================

def feature_engineering(df):
    """Legacy feature engineering helper function for backward compatibility."""
    df_fe = df.copy()
    if 'Year_Built' in df_fe.columns:
        df_fe['House_Age'] = 2026 - df_fe['Year_Built']
        df_fe['House_Age'] = df_fe['House_Age'].clip(lower=0)
    if 'Full_Bathrooms' in df_fe.columns and 'Half_Bathrooms' in df_fe.columns:
        df_fe['Total_Bathrooms'] = df_fe['Full_Bathrooms'] + 0.5 * df_fe['Half_Bathrooms']
    if 'Area_SqFt' in df_fe.columns and 'Garage_Area' in df_fe.columns and 'Basement_Area' in df_fe.columns:
        df_fe['Total_Area'] = df_fe['Area_SqFt'] + df_fe['Garage_Area'] + df_fe['Basement_Area']
    if 'Price' in df_fe.columns:
        df_fe['Price_Per_SqFt'] = df_fe['Price'] / df_fe['Area_SqFt']
    if 'Luxury_Score' in df_fe.columns:
        df_fe['Luxury_Band'] = (df_fe['Luxury_Score'] > 75).astype(int)
    return df_fe

def preprocess_pipeline(df, is_training=True, scaler=None, encoder=None):
    """Legacy preprocessing pipeline helper function."""
    df_fe = feature_engineering(df)
    target_col = 'Price'
    drop_cols = [target_col, 'Price_Per_SqFt', 'Year_Built']
    y = df_fe[target_col] if target_col in df_fe.columns else None
    X = df_fe.drop(columns=[col for col in drop_cols if col in df_fe.columns])
    
    num_cols = ['Area_SqFt', 'Bedrooms', 'Full_Bathrooms', 'Half_Bathrooms', 
                'Garage_Area', 'Basement_Area', 'Luxury_Score', 'House_Age', 
                'Total_Bathrooms', 'Total_Area', 'Luxury_Band']
    cat_cols = ['Location']
    
    if is_training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(X[cat_cols])
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(X[num_cols])
    else:
        cat_encoded = encoder.transform(X[cat_cols])
        num_scaled = scaler.transform(X[num_cols])
        
    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    df_cat = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=X.index)
    df_num = pd.DataFrame(num_scaled, columns=num_cols, index=X.index)
    X_processed = pd.concat([df_num, df_cat], axis=1)
    
    return X_processed, y, scaler, encoder, X_processed.columns.tolist()

# ==========================================
# SQLITE DATABASE HUB & INTERACTION LAYER
# ==========================================

def get_db_connection(db_path=None):
    """Establishes connection to the SQLite database, creating parent directories if needed."""
    if db_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, 'data', 'housing_records.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)

def init_db(db_path=None):
    """Initializes the database schema and seeds properties from the raw CSV dataset if empty."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # 1. Properties Listings Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Area_SqFt REAL NOT NULL,
            Bedrooms INTEGER NOT NULL,
            Full_Bathrooms INTEGER NOT NULL,
            Half_Bathrooms INTEGER NOT NULL,
            Location TEXT NOT NULL,
            Year_Built REAL,
            Garage_Area REAL,
            Basement_Area REAL,
            Luxury_Score REAL,
            Price REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 2. Prediction Logs & User Feedback Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Area_SqFt REAL NOT NULL,
            Bedrooms INTEGER NOT NULL,
            Full_Bathrooms INTEGER NOT NULL,
            Half_Bathrooms INTEGER NOT NULL,
            Location TEXT NOT NULL,
            Year_Built REAL NOT NULL,
            Garage_Area REAL NOT NULL,
            Basement_Area REAL NOT NULL,
            Luxury_Score REAL NOT NULL,
            Predicted_Price REAL NOT NULL,
            Lower_Bound REAL NOT NULL,
            Upper_Bound REAL NOT NULL,
            Model_Used TEXT NOT NULL,
            Feedback_Rating INTEGER,
            Feedback_Comment TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    
    # Seed Listings if empty
    cursor.execute("SELECT COUNT(*) FROM properties")
    count = cursor.fetchone()[0]
    
    if count == 0:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, 'data', 'housing.csv')
        # Generate raw csv if it doesn't exist to seed
        if not os.path.exists(csv_path):
            print("CSV not found for seeding database. Creating housing.csv...")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df = generate_synthetic_data(num_samples=3000)
            df.to_csv(csv_path, index=False)
        else:
            df = pd.read_csv(csv_path)
            
        print(f"Seeding properties table from '{csv_path}' ({len(df)} rows)...")
        df_seed = df.replace({np.nan: None})
        
        insert_query = """
            INSERT INTO properties (
                Area_SqFt, Bedrooms, Full_Bathrooms, Half_Bathrooms, Location, 
                Year_Built, Garage_Area, Basement_Area, Luxury_Score, Price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        records = df_seed[[
            'Area_SqFt', 'Bedrooms', 'Full_Bathrooms', 'Half_Bathrooms', 'Location',
            'Year_Built', 'Garage_Area', 'Basement_Area', 'Luxury_Score', 'Price'
        ]].values.tolist()
        
        cursor.executemany(insert_query, records)
        conn.commit()
        print("Database seeded successfully.")
        
    conn.close()

def fetch_properties_from_db(db_path=None):
    """Retrieves all property listings from SQLite as a Pandas DataFrame."""
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM properties ORDER BY id DESC", conn)
    conn.close()
    return df

def add_property_to_db(property_dict, db_path=None):
    """Inserts a single custom property listing into SQLite database properties table."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    insert_query = """
        INSERT INTO properties (
            Area_SqFt, Bedrooms, Full_Bathrooms, Half_Bathrooms, Location, 
            Year_Built, Garage_Area, Basement_Area, Luxury_Score, Price
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.execute(insert_query, (
        float(property_dict['Area_SqFt']),
        int(property_dict['Bedrooms']),
        int(property_dict['Full_Bathrooms']),
        int(property_dict['Half_Bathrooms']),
        str(property_dict['Location']),
        float(property_dict['Year_Built']) if property_dict.get('Year_Built') is not None else None,
        float(property_dict['Garage_Area']) if property_dict.get('Garage_Area') is not None else None,
        float(property_dict['Basement_Area']) if property_dict.get('Basement_Area') is not None else None,
        float(property_dict['Luxury_Score']) if property_dict.get('Luxury_Score') is not None else None,
        float(property_dict['Price'])
    ))
    conn.commit()
    conn.close()

def delete_property_from_db(property_id, db_path=None):
    """Deletes a property listing from SQLite by its row ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM properties WHERE id = ?", (int(property_id),))
    conn.commit()
    conn.close()

def log_prediction_to_db(inputs, pred_price, low_bound, high_bound, model_used, db_path=None):
    """Logs a single model valuation query to database, returning the generated record row ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    insert_query = """
        INSERT INTO predictions_log (
            Area_SqFt, Bedrooms, Full_Bathrooms, Half_Bathrooms, Location,
            Year_Built, Garage_Area, Basement_Area, Luxury_Score, 
            Predicted_Price, Lower_Bound, Upper_Bound, Model_Used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.execute(insert_query, (
        float(inputs['Area_SqFt']),
        int(inputs['Bedrooms']),
        int(inputs['Full_Bathrooms']),
        int(inputs['Half_Bathrooms']),
        str(inputs['Location']),
        float(inputs['Year_Built']),
        float(inputs['Garage_Area']),
        float(inputs['Basement_Area']),
        float(inputs['Luxury_Score']),
        float(pred_price),
        float(low_bound),
        float(high_bound),
        str(model_used)
    ))
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id

def submit_prediction_feedback(prediction_id, rating, comment, db_path=None):
    """Updates the user feedback rating and text comment on a logged prediction record."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE predictions_log
        SET Feedback_Rating = ?, Feedback_Comment = ?
        WHERE id = ?
    """, (
        int(rating) if rating is not None else None,
        str(comment) if comment else None,
        int(prediction_id)
    ))
    conn.commit()
    conn.close()

def fetch_predictions_log(db_path=None):
    """Retrieves all logged property valuation predictions and feedback records."""
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM predictions_log ORDER BY id DESC", conn)
    conn.close()
    return df

def generate_pdf_report(inputs, pred_price, low_bound, high_bound, best_model_name):
    """Generates a professional PDF valuation report using ReportLab and returns a BytesIO buffer."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from io import BytesIO
    from datetime import datetime

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1e1b4b'),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4b5563'),
        spaceAfter=25
    )
    
    heading_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#312e81'),
        spaceBefore=15,
        spaceAfter=10
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1f2937')
    )
    
    price_style = ParagraphStyle(
        'Price',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#10b981')
    )

    # Title
    story.append(Paragraph("PROHOUSE VALUATION REPORT", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Production Model: {best_model_name}", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Section: Valuation Summary
    story.append(Paragraph("1. Valuation Estimate Summary", heading_style))
    story.append(Spacer(1, 5))
    
    summary_data = [
        [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value (INR)</b>", body_style)],
        [Paragraph("Estimated Market Valuation", body_style), Paragraph(f"<b>INR {pred_price:,.2f}</b>", price_style)],
        [Paragraph("Lower Limit (94% confidence)", body_style), Paragraph(f"INR {low_bound:,.2f}", body_style)],
        [Paragraph("Upper Limit (106% confidence)", body_style), Paragraph(f"INR {high_bound:,.2f}", body_style)]
    ]
    
    summary_table = Table(summary_data, colWidths=[250, 200])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f3f4f6')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1f2937')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9fafb')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Section: Property Specifications
    story.append(Paragraph("2. Property Inputs & Parameters", heading_style))
    story.append(Spacer(1, 5))
    
    spec_data = [
        [Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Input Value</b>", body_style)],
        [Paragraph("Location Sector Zone", body_style), Paragraph(str(inputs.get('Location', 'Standard')), body_style)],
        [Paragraph("Living Area (SqFt)", body_style), Paragraph(f"{inputs.get('Area_SqFt', 0):,.0f} SqFt", body_style)],
        [Paragraph("Bedrooms", body_style), Paragraph(str(inputs.get('Bedrooms', 0)), body_style)],
        [Paragraph("Bathrooms (Full / Half)", body_style), Paragraph(f"{inputs.get('Full_Bathrooms', 0)} Full / {inputs.get('Half_Bathrooms', 0)} Half", body_style)],
        [Paragraph("Garage Area (SqFt)", body_style), Paragraph(f"{inputs.get('Garage_Area', 0):,.0f} SqFt", body_style)],
        [Paragraph("Basement Area (SqFt)", body_style), Paragraph(f"{inputs.get('Basement_Area', 0):,.0f} SqFt", body_style)],
        [Paragraph("Year Constructed", body_style), Paragraph(str(int(inputs.get('Year_Built', 2010))), body_style)],
        [Paragraph("Luxury & Finishes Score", body_style), Paragraph(f"{inputs.get('Luxury_Score', 0)} / 100", body_style)]
    ]
    
    spec_table = Table(spec_data, colWidths=[250, 200])
    spec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f3f4f6')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1f2937')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9fafb')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    story.append(spec_table)
    story.append(Spacer(1, 30))
    
    # Disclaimer
    disclaimer_text = (
        "<b>Disclaimer:</b> This report represents a simulated property valuation calculated via a production-grade machine "
        "learning pipeline (Stacking Ensemble including Tuned Random Forest, Tuned XGBoost, and Gradient Boosting Regressors). "
        "The estimate is based on historical cross-sectional real-estate datasets and should be used for analytical and "
        "research purposes only. Actual market conditions may vary."
    )
    story.append(Paragraph(disclaimer_text, ParagraphStyle('Disclaimer', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8, leading=11, textColor=colors.HexColor('#9ca3af'))))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def get_model_mae(model_name):
    """Loads the Mean Absolute Error (MAE) of a model from validation statistics to compute error bands."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, 'models', 'model_comparison.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            row = df[df['Model'].str.lower().str.strip() == model_name.lower().strip()]
            if not row.empty:
                return float(row.iloc[0]['MAE'])
    except Exception as e:
        pass
    return 30000.0 # Default fallback MAE (30,000 INR)
