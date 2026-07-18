import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import logging

import utils

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_models_with_cv(X_train, y_train, config):
    """
    Evaluates multiple regression models using K-Fold Cross-Validation.
    Logs each baseline model in the ExperimentTracker.
    """
    logger.info("Evaluating baseline models using Cross-Validation...")
    cv_splits = config['split_params']['cv_splits']
    random_state = config['split_params']['random_state']
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=random_state),
        'Random Forest': RandomForestRegressor(random_state=random_state, n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
        'XGBoost': XGBRegressor(random_state=random_state, n_estimators=100),
        'SVR': SVR(kernel='rbf', C=100000, epsilon=0.1),
        'KNN Regressor': KNeighborsRegressor(n_neighbors=5)
    }
    
    cv_results = []
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    tracker = utils.ExperimentTracker()
    
    for name, model in models.items():
        logger.info(f"Cross-validating {name}...")
        scores = cross_validate(
            model, X_train, y_train, 
            cv=kf, 
            scoring=['r2', 'neg_mean_absolute_error'], 
            n_jobs=-1
        )
        
        r2_mean = np.mean(scores['test_r2'])
        r2_std = np.std(scores['test_r2'])
        mae_mean = -np.mean(scores['test_neg_mean_absolute_error'])
        mae_std = np.std(scores['test_neg_mean_absolute_error'])
        
        cv_results.append({
            'Model': name,
            'CV R2 Mean': round(r2_mean, 4),
            'CV R2 Std': round(r2_std, 4),
            'CV MAE Mean': round(mae_mean, 2),
            'CV MAE Std': round(mae_std, 2)
        })
        
        # Log to tracker
        tracker.log_run(
            model_name=name,
            r2_mean=r2_mean,
            r2_std=r2_std,
            mae_mean=mae_mean,
            mae_std=mae_std,
            parameters={"status": "baseline_cv"}
        )
        
    df_cv = pd.DataFrame(cv_results)
    df_cv = df_cv.sort_values(by='CV R2 Mean', ascending=False).reset_index(drop=True)
    return df_cv, models

def tune_base_models(X_train, y_train, config):
    """
    Performs RandomizedSearchCV to tune base estimators based on configurations.
    """
    logger.info("Tuning base estimators for ensembling...")
    random_state = config['split_params']['random_state']
    tuning_cfg = config['tuning_params']
    
    # 1. Tune Random Forest
    rf = RandomForestRegressor(random_state=random_state)
    rf_grid = tuning_cfg['rf_grid']
    logger.info("Optimizing Random Forest hyperparameters...")
    rf_search = RandomizedSearchCV(
        rf, param_distributions=rf_grid, 
        n_iter=tuning_cfg['n_iter'], cv=tuning_cfg['cv'], 
        scoring='r2', random_state=random_state, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    tuned_rf = rf_search.best_estimator_
    logger.info(f"Best RF Parameters: {rf_search.best_params_}")
    
    # 2. Tune XGBoost
    xgb = XGBRegressor(random_state=random_state)
    xgb_grid = tuning_cfg['xgb_grid']
    logger.info("Optimizing XGBoost hyperparameters...")
    xgb_search = RandomizedSearchCV(
        xgb, param_distributions=xgb_grid, 
        n_iter=tuning_cfg['n_iter'], cv=tuning_cfg['cv'], 
        scoring='r2', random_state=random_state, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    tuned_xgb = xgb_search.best_estimator_
    logger.info(f"Best XGB Parameters: {xgb_search.best_params_}")
    
    return tuned_rf, tuned_xgb

def run_training_pipeline(progress_callback=None):
    def log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    log("Starting Pro-Level ML Pipeline Training...")
    
    # 0. Load Configuration
    config = utils.load_config()
    random_state = config['split_params']['random_state']
    test_size = config['split_params']['test_size']
    cv_splits = config['split_params']['cv_splits']
    
    # 1. Load Data
    log("Loading data from database...")
    df = utils.load_data()
    
    # 2. Clean Data (Duplicates & Target Outliers)
    log(f"Initial shape: {df.shape}. Cleaning duplicates and outlier prices...")
    df_clean = utils.clean_data(df)
    
    # 3. Split features and target
    X = df_clean.drop(columns=['Price', 'Price_Per_SqFt'], errors='ignore')
    y = df_clean['Price']
    
    # 4. Train-Test Split (Raw data level to prevent any leakages!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    log(f"Splitting dataset: Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 5. Fit Preprocessing Pipeline on Training Data
    preprocessor_pipeline = utils.get_preprocessing_pipeline()
    
    # Fit preprocessor on X_train and transform both splits
    log("Fitting preprocessing pipeline (scaling, encoding, capping)...")
    X_train_proc = preprocessor_pipeline.fit_transform(X_train)
    X_test_proc = preprocessor_pipeline.transform(X_test)
    
    # Extract feature names from preprocessing transformers
    fitted_preprocessor = preprocessor_pipeline.named_steps['preprocessor']
    num_features = fitted_preprocessor.transformers_[0][2]
    cat_encoder = fitted_preprocessor.transformers_[1][1]
    cat_features = cat_encoder.get_feature_names_out(['Location']).tolist()
    feature_names = num_features + cat_features
    
    X_train_proc_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_test_proc_df = pd.DataFrame(X_test_proc, columns=feature_names)
    
    # 6. Run Cross-Validation on Baseline Models
    log("Evaluating baseline models (Linear Regression, Ridge, Random Forest, XGBoost, etc.) with Cross-Validation...")
    df_cv, models = evaluate_models_with_cv(X_train_proc_df, y_train, config)
    log(f"Cross-Validation complete. Top models evaluated.\n{df_cv.to_string()}")
    
    # 7. Hyperparameter Tuning of Base Models
    log("Tuning hyperparameters for Random Forest and XGBoost via RandomizedSearchCV (this might take a few seconds)...")
    tuned_rf, tuned_xgb = tune_base_models(X_train_proc_df, y_train, config)
    
    # 8. Build Stacking Regressor
    log("Assembling Stacking Ensemble Regressor...")
    estimators = [
        ('rf', tuned_rf),
        ('xgb', tuned_xgb),
        ('gbr', GradientBoostingRegressor(random_state=random_state))
    ]
    
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1
    )
    
    # Fit Stacking Model
    log("Fitting final Stacking Regressor model...")
    stacking_regressor.fit(X_train_proc_df, y_train)
    
    # 9. Evaluate Stacking Regressor with CV
    log("Evaluating Stacking Regressor with cross-validation...")
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    stack_cv = cross_validate(
        stacking_regressor, X_train_proc_df, y_train,
        cv=kf, scoring=['r2', 'neg_mean_absolute_error'], n_jobs=-1
    )
    stack_r2 = np.mean(stack_cv['test_r2'])
    stack_mae = -np.mean(stack_cv['test_neg_mean_absolute_error'])
    
    log(f"Stacking Ensemble CV Performance: R2={stack_r2:.4f}, MAE={stack_mae:.2f}")
    
    # Log Stacking to Tracker
    tracker = utils.ExperimentTracker()
    tracker.log_run(
        model_name="Stacking Ensemble",
        r2_mean=stack_r2,
        r2_std=np.std(stack_cv['test_r2']),
        mae_mean=stack_mae,
        mae_std=np.std(stack_cv['test_neg_mean_absolute_error']),
        parameters={
            "estimators": ["Tuned RF", "Tuned XGBoost", "GradientBoosting"],
            "final_estimator": "Ridge"
        }
    )
    
    # Add Stacking to Comparison Table
    stack_row = pd.DataFrame([{
        'Model': 'Stacking Ensemble',
        'CV R2 Mean': round(stack_r2, 4),
        'CV R2 Std': round(np.std(stack_cv['test_r2']), 4),
        'CV MAE Mean': round(stack_mae, 2),
        'CV MAE Std': round(np.std(stack_cv['test_neg_mean_absolute_error']), 2)
    }])
    df_cv = pd.concat([stack_row, df_cv], ignore_index=True)
    
    # 10. Fit and Serialize the Final Production Unified Pipeline
    log("Training final unified production pipeline on entire train set...")
    production_pipeline = Pipeline(steps=[
        ('preprocessor_pipeline', preprocessor_pipeline),
        ('regressor', stacking_regressor)
    ])
    
    production_pipeline.fit(X_train, y_train)
    
    # Test inference on holdout test set
    log("Testing production model pipeline on holdout test set...")
    y_pred = production_pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    log(f"Production Holdout Performance: R2={test_r2:.4f}, MAE={test_mae:.2f}")
    
    # Save Pipeline and Metrics
    pipeline_path = config['paths']['pipeline_pkl']
    comparison_path = config['paths']['comparison_csv']
    best_model_name_path = config['paths']['best_model_name']
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(production_pipeline, pipeline_path)
    log(f"Saved unified production pipeline to '{pipeline_path}'")
    
    # Save all individual pipelines in a dictionary
    all_pipelines = {}
    all_pipelines['Stacking Ensemble'] = production_pipeline
    
    # Create CSV comparison
    comparison_list = []
    comparison_list.append({
        'Model': 'Stacking Ensemble',
        'MAE': round(mean_absolute_error(y_test, y_pred), 2),
        'MSE': round(mean_squared_error(y_test, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        'R2': round(test_r2, 4),
        'Adjusted R2': round(1 - (1 - test_r2) * (len(y_test) - 1) / (len(y_test) - X_test_proc.shape[1] - 1), 4)
    })
    
    log("Computing comparisons across all 8 models on holdout split...")
    for name, model in models.items():
        # Build individual pipeline using the fitted preprocessor
        individual_pipeline = Pipeline(steps=[
            ('preprocessor_pipeline', preprocessor_pipeline),
            ('regressor', model)
        ])
        individual_pipeline.fit(X_train, y_train)
        all_pipelines[name] = individual_pipeline
        
        preds = individual_pipeline.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test_proc.shape[1] - 1)
        
        comparison_list.append({
            'Model': name,
            'MAE': round(mae, 2),
            'MSE': round(mse, 2),
            'RMSE': round(rmse, 2),
            'R2': round(r2, 4),
            'Adjusted R2': round(adj_r2, 4)
        })
        
    joblib.dump(all_pipelines, 'models/all_models.pkl')
    log("Saved all model pipelines dictionary to 'models/all_models.pkl'")
    
    df_comparison = pd.DataFrame(comparison_list).sort_values(by='R2', ascending=False).reset_index(drop=True)
    df_comparison.to_csv(comparison_path, index=False)
    log(f"Saved comparison model metrics table to '{comparison_path}'")
    
    # Serialize legacy files
    fitted_preproc = production_pipeline.named_steps['preprocessor_pipeline']
    fitted_scaler = fitted_preproc.named_steps['preprocessor'].transformers_[0][1]
    fitted_encoder = fitted_preproc.named_steps['preprocessor'].transformers_[1][1]
    
    joblib.dump(production_pipeline.named_steps['regressor'], 'models/best_model.pkl')
    joblib.dump(fitted_scaler, 'models/scaler.pkl')
    joblib.dump(fitted_encoder, 'models/encoder.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    with open(best_model_name_path, 'w') as f:
        f.write("Stacking Ensemble")
        
    log("Training process and serializations completed successfully!")
    return df_comparison

def main():
    run_training_pipeline()

if __name__ == '__main__':
    main()
