# ProHouse: House Price Prediction & Analytics Hub
A Placement-Level / Final Year Machine Learning & Analytics Project

---

## 📌 Project Overview
**ProHouse** is an end-to-end Machine Learning and interactive dashboard application designed to predict residential property valuations. Rather than using basic linear regression, this project incorporates an industry-grade machine learning pipeline—complete with data cleaning, outlier capping, advanced feature engineering, automatic feature selection, multi-model evaluation, randomized hyperparameter search, explainable AI (SHAP), and dynamic PDF report generation.

The front-end is powered by a sleek **Streamlit** multi-page application, allowing users to value properties in real-time, view interactive EDA charts, inspect model evaluation reports, and understand the model's decision-making process.

---

## 🚀 Key Features

### 1. Robust Data Preprocessing
- **Duplicate Removal:** Automatically screens and deletes repeated entries.
- **Imputation:** Handles missing data columns (e.g., Garage Area, Year Built, Luxury Score) using median values.
- **Outlier Mitigation:** Detects skewed features (Price & Area) using the Interquartile Range (IQR) method and caps them at $1.5 \times \text{IQR}$ to protect data distribution integrity.
- **Preprocessing Pipeline:** Integrated using scikit-learn's `StandardScaler` for numerical scaling and `OneHotEncoder` for handling categorical locations.

### 2. Feature Engineering & Selection
- **Engineered Features:**
  - `House_Age` = $\text{Current Year (2026)} - \text{Year Built}$
  - `Total_Bathrooms` = $\text{Full Bathrooms} + 0.5 \times \text{Half Bathrooms}$
  - `Total_Area` = $\text{Area\_SqFt} + \text{Garage\_Area} + \text{Basement\_Area}$
  - `Luxury_Band` = binary indicator ($1$ if `Luxury_Score` $> 75$, else $0$).
- **Data Leakage Defense:** Excludes `Price_Per_SqFt` from model training inputs to prevent feedback leakage, using it solely for EDA.
- **Feature Selection:** Highlights key contributors via `SelectKBest` (F-Regression scores) and **Recursive Feature Elimination (RFE)** with a Random Forest estimator.

### 3. Model Training & Comparison
Compares 7 distinct machine learning algorithms:
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**
5. **XGBoost Regressor**
6. **Support Vector Regression (SVR)**
7. **K-Nearest Neighbors (KNN)**

A comparison table compiles **MAE**, **MSE**, **RMSE**, **$R^2$ Score**, and **Adjusted $R^2$ Score** metrics, automatically selecting the top model for final tuning.

### 4. Hyperparameter Tuning
- Optimized the best model (typically Random Forest or XGBoost) using **RandomizedSearchCV** to locate the best combination of parameters (estimators, depth, learning rates, subsamples).

### 5. Interactive Dashboard Web App
- **Tab 1: Valuation Predictor:** Input sliders for area, rooms, location, age, and luxury score. Displays predicted price (INR) with confidence ranges and a premium interactive dial. Generates downloadable PDF valuation reports.
- **Tab 2: Exploratory Data Analysis:** Price histograms, boxplots for locations, scatter-plots with trendlines, and a correlation matrix heatmap.
- **Tab 3: Model Performance:** Displays training scores, feature importances, Actual vs. Predicted scatter-plots, and Residual plots.
- **Tab 4: Explainable AI (SHAP):** Visualizes feature attribution contributions using a Waterfall plot, explaining how each input parameter shifts prediction from the average baseline.

---

## 📂 Project Folder Structure
```
House-Price-Prediction/
│
├── data/
│   ├── housing.csv               # Raw synthetic housing data
│   └── housing_processed.csv     # Preprocessed data with engineered features
│
├── notebook/
│   └── HousePricePrediction.ipynb # Jupyter notebook documenting EDA & pipeline
│
├── models/
│   ├── best_model.pkl            # Serialized tuned best model
│   ├── scaler.pkl                # Serialized StandardScaler
│   ├── encoder.pkl               # Serialized OneHotEncoder
│   ├── feature_names.pkl         # Feature column alignment list
│   ├── best_model_name.txt       # text file referencing best model
│   └── model_comparison.csv      # CSV output containing comparative stats
│
├── app.py                        # Streamlit Web App script (Multi-tab layout)
├── requirements.txt              # Project packages list
├── README.md                     # Documentation
├── utils.py                      # Data Cleaning, Feature Engineering & Pipelines
├── train.py                      # Selection, Evaluation, Tuning, Serialization
├── predict.py                    # Command-line prediction script
├── generate_data.py              # Synthetic dataset generator
└── create_notebook.py            # Code helper compiling the Jupyter notebook
```

---

## 🛠️ Setup & Installation Guide

1. **Clone or Open the workspace directory:**
   ```bash
   cd c:\Users\GURU\OneDrive\Desktop\arn
   ```

2. **Install dependencies:**
   Make sure you have python installed. Run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models (Optional):**
   *Note: If you skip this, the Streamlit app will automatically run data generation and model training upon launching.*
   ```bash
   python train.py
   ```

4. **Launch the Streamlit Web Application:**
   ```bash
   streamlit run app.py
   ```

---

## 🖥️ Command-Line Inference API
You can run predictions directly from your terminal using `predict.py`:
```bash
python predict.py --area 2500 --bedrooms 3 --full-bath 2 --half-bath 1 --location Premium --year-built 2015 --luxury 85
```

Output:
```
==========================================
 PREDICTED PRICE:  INR 834,142.15
 Confidence Range: INR 784,093.62 to INR 884,190.68
==========================================
```

---

## 🛡️ Self-Healing Mechanism
The project includes a self-healing setup. If any of the required datasets, model artifacts, or notebooks are missing, running `app.py` or `predict.py` will automatically trigger:
1. `generate_data.py` (synthetic data setup)
2. `create_notebook.py` (compile `.ipynb` notebook)
3. `train.py` (model preprocessing, selection, tuning, and pickle serialization)
This assures a completely error-free setup for recruiters or interviewers looking to review the project code.
