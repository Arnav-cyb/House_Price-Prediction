import os
import json

def create_notebook_file():
    notebook_content = {
     "cells": [
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "# House Price Prediction - Placement & Final Year Project\n",
        "\n",
        "This notebook contains the complete end-to-end Machine Learning pipeline for predicting house prices. It covers:\n",
        "1. **Data Preprocessing & Cleaning** (Missing values, duplicates, outliers handling)\n",
        "2. **Exploratory Data Analysis (EDA)** (Distributions, box plots, correlation heatmap, scatter plots)\n",
        "3. **Feature Engineering** (House Age, Total Bathrooms, Total Area, Price per SqFt, Luxury score analysis)\n",
        "4. **Feature Selection** (SelectKBest & Recursive Feature Elimination)\n",
        "5. **Model Comparison** (Linear Regression, Decision Trees, Random Forest, XGBoost, SVR, KNN)\n",
        "6. **Hyperparameter Tuning** (GridSearchCV / RandomizedSearchCV)\n",
        "7. **Explainable AI (SHAP)** (Feature contribution visualization)"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## Step 1: Libraries Setup & Data Loading"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# Install required packages if not already installed\n",
        "# !pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap plotly joblib\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Set style for plots\n",
        "sns.set_theme(style='whitegrid')\n",
        "plt.rcParams['figure.figsize'] = (10, 6)\n",
        "plt.rcParams['font.size'] = 12"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# Generate or Load synthetic data\n",
        "import sys\n",
        "sys.path.append('..')  # Add parent directory to path\n",
        "import utils\n",
        "\n",
        "# This will automatically generate housing.csv if not present\n",
        "df = utils.load_data('../data/housing.csv')\n",
        "print(f\"Data Loaded successfully. Shape: {df.shape}\")\n",
        "df.head()"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## Step 2: Data Preprocessing & Cleaning\n",
        "\n",
        "We handle duplicates, impute missing values, and handle outliers using the IQR method."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# Show duplicates and missing values\n",
        "print(\"Duplicate rows:\", df.duplicated().sum())\n",
        "print(\"\\nMissing values:\")\n",
        "print(df.isnull().sum())\n",
        "\n",
        "# Clean the data using our pipeline in utils\n",
        "df_clean = utils.clean_data(df)\n",
        "print(\"\\nCleaned data duplicate rows:\", df_clean.duplicated().sum())\n",
        "print(\"Missing values after cleaning:\")\n",
        "print(df_clean.isnull().sum())"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## Step 3: Exploratory Data Analysis (EDA)\n",
        "\n",
        "Let's visualize the distributions and correlations."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# 1. Price Distribution\n",
        "plt.figure(figsize=(10, 5))\n",
        "sns.histplot(df_clean['Price'], kde=True, color='royalblue', bins=30)\n",
        "plt.title('House Price Distribution')\n",
        "plt.xlabel('Price (INR)')\n",
        "plt.ylabel('Frequency')\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# 2. Location-wise Average Price\n",
        "plt.figure(figsize=(10, 5))\n",
        "order = df_clean.groupby('Location')['Price'].mean().sort_values().index\n",
        "sns.barplot(x='Location', y='Price', data=df_clean, order=order, palette='viridis')\n",
        "plt.title('Average House Price by Location')\n",
        "plt.ylabel('Average Price (INR)')\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# 3. Area vs Price Scatter Plot\n",
        "plt.figure(figsize=(10, 5))\n",
        "sns.scatterplot(x='Area_SqFt', y='Price', hue='Location', data=df_clean, alpha=0.7)\n",
        "plt.title('Area (SqFt) vs Price')\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# 4. Boxplot for Outliers in Price\n",
        "plt.figure(figsize=(8, 4))\n",
        "sns.boxplot(x=df['Price'], color='salmon')\n",
        "plt.title('Outliers in Original Price Data (Before Capping)')\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## Step 4: Feature Engineering\n",
        "\n",
        "We'll create the new features: `House_Age`, `Total_Bathrooms`, `Total_Area`, `Price_Per_SqFt`, `Luxury_Band`."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "df_fe = utils.feature_engineering(df_clean)\n",
        "print(\"Engineered Features:\")\n",
        "df_fe[['Year_Built', 'House_Age', 'Full_Bathrooms', 'Half_Bathrooms', 'Total_Bathrooms', 'Area_SqFt', 'Garage_Area', 'Basement_Area', 'Total_Area', 'Price_Per_SqFt']].head()"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# 5. Correlation Heatmap\n",
        "plt.figure(figsize=(12, 10))\n",
        "numerical_cols = df_fe.select_dtypes(include=[np.number]).columns\n",
        "corr_matrix = df_fe[numerical_cols].corr()\n",
        "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)\n",
        "plt.title('Correlation Matrix of Features')\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## Step 5: Feature Selection & Preprocessing\n",
        "\n",
        "We apply scaling and encoding and use `SelectKBest` and `RFE` to determine feature importance."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "# Split target and features\n",
        "X_proc, y, scaler, encoder, feature_names = utils.preprocess_pipeline(df_fe, is_training=True)\n",
        "print(f\"Preprocessed feature space shape: {X_proc.shape}\")\n",
        "\n",
        "# Feature Selection using SelectKBest\n",
        "from sklearn.feature_selection import SelectKBest, f_regression\n",
        "selector = SelectKBest(score_func=f_regression, k=8)\n",
        "selector.fit(X_proc, y)\n",
        "scores = pd.DataFrame({'Feature': X_proc.columns, 'F-Score': selector.scores_}).sort_values(by='F-Score', ascending=False)\n",
        "print(\"\\nSelectKBest Top Features:\")\n",
        "print(scores)"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## Step 6: Model Training & Comparison\n",
        "\n",
        "Let's split the dataset (80-20) and evaluate multiple models."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)\n",
        "\n",
        "# Let's import the train script evaluation logic\n",
        "import train\n",
        "comparison_table, trained_models = train.evaluate_models(X_train, X_test, y_train, y_test)\n",
        "comparison_table"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## Step 7: Hyperparameter Tuning\n",
        "\n",
        "We perform hyperparameter tuning on the best model."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "best_model_name = comparison_table.iloc[0]['Model']\n",
        "best_model = trained_models[best_model_name]\n",
        "\n",
        "best_tuned_model = train.tune_hyperparameters(best_model_name, best_model, X_train, y_train)\n",
        "\n",
        "# Evaluate\n",
        "from sklearn.metrics import r2_score, mean_absolute_error\n",
        "preds = best_tuned_model.predict(X_test)\n",
        "print(f\"Tuned {best_model_name} Test R2: {r2_score(y_test, preds):.4f}\")\n",
        "print(f\"Tuned {best_model_name} Test MAE: {mean_absolute_error(y_test, preds):.2f}\")"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## Step 8: SHAP Explainability\n",
        "\n",
        "Let's understand feature importance using SHAP explainers on our best model (Random Forest / XGBoost)."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "import shap\n",
        "\n",
        "# Initialize JavaScript for plots\n",
        "shap.initjs()\n",
        "\n",
        "# Compute SHAP values\n",
        "explainer = shap.Explainer(best_tuned_model, X_train)\n",
        "shap_values = explainer(X_test[:100])  # Compute for first 100 test samples to run fast\n",
        "\n",
        "# SHAP Summary Plot\n",
        "plt.figure(figsize=(10, 6))\n",
        "shap.summary_plot(shap_values, X_test[:100], show=False)\n",
        "plt.title('SHAP Feature Importance Summary', fontsize=16)\n",
        "plt.show()"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# Force plot or Waterfall plot for a single house prediction explanation\n",
        "plt.figure(figsize=(10, 4))\n",
        "shap.plots.waterfall(shap_values[0], show=False)\n",
        "plt.title('SHAP Explanation for a Single Prediction', fontsize=14)\n",
        "plt.show()"
       ]
      }
     ],
     "metadata": {
      "kernelspec": {
       "display_name": "Python 3 (ipykernel)",
       "language": "python",
       "name": "python3"
      },
      "language_info": {
       "name": "python"
      }
     },
     "nbformat": 4,
     "nbformat_minor": 2
    }
    
    os.makedirs('notebook', exist_ok=True)
    with open('notebook/HousePricePrediction.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=1)
    print("Notebook compiled and saved to notebook/HousePricePrediction.ipynb")

if __name__ == '__main__':
    create_notebook_file()
