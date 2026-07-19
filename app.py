import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from io import BytesIO
import shap
import utils

# --- Load Serials ---
@st.cache_resource
def load_ml_components():
    pipeline = joblib.load('models/housing_pipeline.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    with open('models/best_model_name.txt', 'r') as f:
        best_model_name = f.read().strip()
    config = utils.load_config()
    
    # Load all models dictionary if available for serving
    all_models = {}
    if os.path.exists('models/all_models.pkl'):
        try:
            all_models = joblib.load('models/all_models.pkl')
        except Exception:
            all_models = {best_model_name: pipeline}
    else:
        all_models = {best_model_name: pipeline}
        
    return pipeline, feature_names, best_model_name, config, all_models

pipeline, feature_names, best_model_name, config, all_models = load_ml_components()

# --- Page Configurations ---
st.set_page_config(
    page_title="ProHouse - Premium Valuation & Analytics Hub",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed to highlight custom horizontal navbar!
)

# --- Session State navigation setup ---
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "🏠 Valuation Predictor"

# --- Custom Styling (SaaS Design System) ---
st.markdown("""
<style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #030712;
        color: #f3f4f6;
    }
    
    /* Remove default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Horizontal Navbar Design */
    .nav-container {
        display: flex;
        justify-content: center;
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 0.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2.2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(25px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.45);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
    }
    
    /* Valuation Box and Prices */
    .valuation-wrapper {
        background: radial-gradient(circle at top left, rgba(99, 102, 241, 0.18), rgba(0,0,0,0));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.15);
    }
    
    .glowing-price {
        font-size: 3.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.04em;
        margin: 0.6rem 0;
        font-family: 'Outfit', sans-serif;
        filter: drop-shadow(0px 0px 25px rgba(16, 185, 129, 0.3));
    }
    
    /* Custom Stats Grid Widget */
    .stat-widget-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    
    .stat-widget {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    .stat-widget:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(99, 102, 241, 0.2);
    }
    .stat-widget-label {
        font-size: 0.72rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    .stat-widget-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f3f4f6;
        margin-top: 0.25rem;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Anomaly Warning Styling */
    .anomaly-banner {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1rem;
        color: #fca5a5;
        font-size: 0.88rem;
    }
    
    /* Sidebar Navigation Style (Fallback) */
    .stRadio [role="radiogroup"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Custom Navigation Bar Render ---
tabs = ["🏠 Valuation Predictor", "⚔️ Compare Mode", "📊 Interactive EDA", "🧠 SHAP Explainability", "📈 Model Performance", "🗃️ DB & Retraining Hub"]
cols = st.columns(len(tabs))
for idx, tab_name in enumerate(tabs):
    with cols[idx]:
        is_active = st.session_state.active_tab == tab_name
        button_type = "primary" if is_active else "secondary"
        if st.button(tab_name, width='stretch', key=f"nav_tab_{idx}", type=button_type):
            st.session_state.active_tab = tab_name
            st.rerun()

# Run database auto-initialization to ensure SQLite file and tables are ready
utils.init_db()

# Load processed dataset from database and run feature engineering for live updates
df_raw_db = utils.load_data()
df_fe = utils.feature_engineering(df_raw_db)
geo_cfg = config['geospatial']

np.random.seed(42)
lats, lons = [], []
for loc in df_fe['Location']:
    center = geo_cfg.get(loc, {'lat': 19.03, 'lon': 72.86})
    lats.append(center['lat'] + np.random.normal(0, 0.012))
    lons.append(center['lon'] + np.random.normal(0, 0.012))
df_fe['latitude'] = lats
df_fe['longitude'] = lons

# Pre-define segment labels mapping
def get_segment_details(price):
    if price < 250000:
        return "Budget Class", "💸", "#3b82f6", "rgba(59, 130, 246, 0.15)"
    elif price < 500000:
        return "Standard Class", "🏡", "#10b981", "rgba(16, 185, 129, 0.15)"
    elif price < 850000:
        return "Premium Class", "💎", "#f59e0b", "rgba(245, 158, 11, 0.15)"
    else:
        return "Ultra Luxury Mansion", "🏰", "#ec4899", "rgba(236, 72, 153, 0.15)"


# ==========================================
# TAB 1: SINGLE VALUATION PREDICTOR
# ==========================================
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
                location = st.selectbox("Location Sector Zone", list(geo_cfg.keys()), index=2)
                area = st.slider("Living Area (SqFt)", min_value=600, max_value=6000, value=2200, step=50)
                bedrooms = st.slider("Total Bedrooms", min_value=1, max_value=5, value=3)
                full_baths = st.slider("Full Bathrooms", min_value=1, max_value=4, value=2)
            with l2:
                half_baths = st.slider("Half Bathrooms", min_value=0, max_value=2, value=1)
                garage = st.slider("Garage Area Size (SqFt)", min_value=0, max_value=900, value=400, step=25)
                basement = st.slider("Basement Area Size (SqFt)", min_value=0, max_value=2000, value=500, step=50)
                year = st.slider("Year Constructed", min_value=1960, max_value=2025, value=2010)
                
            luxury = st.slider("Luxury & Finishes Score", min_value=1, max_value=100, value=65)
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
            selected_mae = utils.get_model_mae(selected_model_name)
            low_bound = max(30000.0, pred_price - 1.96 * selected_mae)
            high_bound = pred_price + 1.96 * selected_mae
            
            category, icon, color, badge = get_segment_details(pred_price)
            
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
            pdf_buffer = utils.generate_pdf_report(
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
            forecasts = utils.forecast_valuation(pred_price, location)
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
                pred_id = utils.log_prediction_to_db(inputs, pred_price, low_bound, high_bound, selected_model_name)
                
                # Map rating stars to number
                rating_num = feedback_rating.count("⭐")
                
                # Update prediction with feedback rating and comment
                utils.submit_prediction_feedback(pred_id, rating_num, feedback_comment)
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
        st.download_button("📥 Download CSV Template", data=csv_template, file_name="prohouse_bulk_template.csv", mime="text/csv")
        
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
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
                        selected_mae = utils.get_model_mae(selected_model_name)
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
                            utils.log_prediction_to_db(row_inputs, row['Predicted_Price'], row['Lower_Bound_95_CI'], row['Upper_Bound_95_CI'], selected_model_name)
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
                            width='stretch'
                        )
            except Exception as e:
                st.error(f"Failed to process CSV file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# TAB 2: COMPARE MODE (SIDE-BY-SIDE EVALUATOR)
# ==========================================
elif st.session_state.active_tab == "⚔️ Compare Mode":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>⚔️ Side-by-Side Scenario Evaluator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Compare valuations, metric scores, and future outlooks for two property scenarios simultaneously.</p>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; color:#6366f1;'>Property Scenario A</h3>", unsafe_allow_html=True)
        loc_a = st.selectbox("Location (A)", list(geo_cfg.keys()), index=2, key="loc_a")
        area_a = st.slider("Area SqFt (A)", min_value=600, max_value=6000, value=2000, step=50, key="area_a")
        beds_a = st.slider("Bedrooms (A)", min_value=1, max_value=5, value=3, key="beds_a")
        baths_a = st.slider("Full Baths (A)", min_value=1, max_value=4, value=2, key="baths_a")
        garage_a = st.slider("Garage Area (A)", min_value=0, max_value=900, value=400, key="garage_a")
        basement_a = st.slider("Basement Area (A)", min_value=0, max_value=2000, value=500, key="basement_a")
        year_a = st.slider("Year Constructed (A)", min_value=1960, max_value=2025, value=2005, key="year_a")
        lux_a = st.slider("Luxury Finishes (A)", min_value=1, max_value=100, value=50, key="lux_a")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_b:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; color:#ec4899;'>Property Scenario B</h3>", unsafe_allow_html=True)
        loc_b = st.selectbox("Location (B)", list(geo_cfg.keys()), index=4, key="loc_b")
        area_b = st.slider("Area SqFt (B)", min_value=600, max_value=6000, value=2600, step=50, key="area_b")
        beds_b = st.slider("Bedrooms (B)", min_value=1, max_value=5, value=4, key="beds_b")
        baths_b = st.slider("Full Baths (B)", min_value=1, max_value=4, value=3, key="baths_b")
        garage_b = st.slider("Garage Area (B)", min_value=0, max_value=900, value=400, key="garage_b")
        basement_b = st.slider("Basement Area (B)", min_value=0, max_value=2000, value=500, key="basement_b")
        year_b = st.slider("Year Constructed (B)", min_value=1960, max_value=2025, value=2015, key="year_b")
        lux_b = st.slider("Luxury Finishes (B)", min_value=1, max_value=100, value=80, key="lux_b")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Evaluate
    df_raw_a = pd.DataFrame({'Area_SqFt':[area_a],'Bedrooms':[beds_a],'Full_Bathrooms':[baths_a],'Half_Bathrooms':[1],'Location':[loc_a],'Year_Built':[year_a],'Garage_Area':[garage_a],'Basement_Area':[basement_a],'Luxury_Score':[lux_a]})
    df_raw_b = pd.DataFrame({'Area_SqFt':[area_b],'Bedrooms':[beds_b],'Full_Bathrooms':[baths_b],'Half_Bathrooms':[1],'Location':[loc_b],'Year_Built':[year_b],'Garage_Area':[garage_b],'Basement_Area':[basement_b],'Luxury_Score':[lux_b]})
    
    pred_a = pipeline.predict(df_raw_a)[0]
    pred_b = pipeline.predict(df_raw_b)[0]
    
    # Comparison metrics block
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Valuation Comparisons")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        cat_a, icon_a, color_a, _ = get_segment_details(pred_a)
        st.markdown(f"<div class='stat-widget'><div class='stat-widget-label'>Scenario A Value</div><div class='stat-val' style='color:#6366f1; font-size:1.8rem;'>INR {pred_a:,.2f}</div><span style='color:{color_a}; font-size:0.8rem;'>{icon_a} {cat_a}</span></div>", unsafe_allow_html=True)
    with c2:
        cat_b, icon_b, color_b, _ = get_segment_details(pred_b)
        st.markdown(f"<div class='stat-widget'><div class='stat-widget-label'>Scenario B Value</div><div class='stat-val' style='color:#ec4899; font-size:1.8rem;'>INR {pred_b:,.2f}</div><span style='color:{color_b}; font-size:0.8rem;'>{icon_b} {cat_b}</span></div>", unsafe_allow_html=True)
    with c3:
        diff = pred_b - pred_a
        pct_diff = (diff / pred_a) * 100
        diff_color = "#10b981" if diff >= 0 else "#ef4444"
        st.markdown(f"<div class='stat-widget'><div class='stat-widget-label'>Difference (B - A)</div><div class='stat-val' style='color:{diff_color}; font-size:1.8rem;'>INR {abs(diff):,.2f}</div><span style='color:{diff_color}; font-size:0.8rem;'>{'+' if diff>=0 else '-'}{abs(pct_diff):.1f}% Valuation Margin</span></div>", unsafe_allow_html=True)
        
    # Comparative Future Valuation Chart
    st.markdown("<h4 style='margin-top:2rem;'>5-Year Projected Valuation Trajectories</h4>", unsafe_allow_html=True)
    fc_a = utils.forecast_valuation(pred_a, loc_a)
    fc_b = utils.forecast_valuation(pred_b, loc_b)
    
    compare_forecast_df = pd.DataFrame({
        'Year': list(fc_a.keys()) + list(fc_b.keys()),
        'Projected Valuation': list(fc_a.values()) + list(fc_b.values()),
        'Scenario': ['Scenario A'] * len(fc_a) + ['Scenario B'] * len(fc_b)
    })
    
    fig_comp = px.line(compare_forecast_df, x='Year', y='Projected Valuation', color='Scenario', markers=True,
                        color_discrete_map={'Scenario A': '#6366f1', 'Scenario B': '#ec4899'})
    fig_comp.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'family':'Plus Jakarta Sans'})
    st.plotly_chart(fig_comp, width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# TAB 3: INTERACTIVE EDA & 3D MAP
# ==========================================
elif st.session_state.active_tab == "📊 Interactive EDA":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>📊 Database Insights & 3D Mapping</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Examine property distribution densities, regression fits, and explore listing densities in 3D Space.</p>", unsafe_allow_html=True)
    
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs(["🗺️ 3D Geospatial Map", "💰 Price Analysis", "📐 Feature Distributions", "🔗 Heatmaps"])
    plotly_template = 'plotly_dark'
    
    with eda_tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("3D Geospatial Property Density Map")
        st.markdown("Properties clustered by Location zone. Colored by segment grade (Premium: Pink, Standard: Emerald, Suburbs: Blue, Rural: Grey). Rotate, zoom, and pitch map area to view density projections.")
        
        # Color Map mapping
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
        st.subheader("Price Spread & Location Outliers")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_fe, x='Price', title='Valuation Spread Frequency', color_discrete_sequence=['#6366f1'])
            fig.update_layout(template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch')
        with col2:
            fig = px.box(df_fe, x='Location', y='Price', color='Location', title='Capped Price Ranges by Zone Segment', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
        
    with eda_tab3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Explore Numeric Relationships")
        feat = st.selectbox("Select Dimension parameter to plot against Price", ['Area_SqFt', 'Total_Area', 'House_Age', 'Luxury_Score', 'Garage_Area', 'Basement_Area'])
        fig = px.scatter(df_fe, x=feat, y='Price', color='Location', title=f"{feat} vs Price", color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
        
    with eda_tab4:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Linear Feature Correlation coefficients Matrix")
        num_cols = df_fe.select_dtypes(include=[np.number]).columns
        corr = df_fe[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        fig.update_layout(template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=550)
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# TAB 4: SHAP EXPLAINABILITY
# ==========================================
elif st.session_state.active_tab == "🧠 SHAP Explainability":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>🧠 Explainable AI - SHAP Local Attributions</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Inspect feature-level attribution weights driving the valuation using Shapley Tree Explainer.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("<div class='glass-card'><h3 style='margin-top:0; margin-bottom:1.5rem;'>Choose Cases to Explain</h3>", unsafe_allow_html=True)
        loc_shap = st.selectbox("Location Segment (SHAP)", list(geo_cfg.keys()), index=2, key="shap_loc")
        area_shap = st.slider("Living Area (SHAP)", min_value=600, max_value=6000, value=2200, step=50, key="shap_area")
        beds_shap = st.slider("Bedrooms count (SHAP)", min_value=1, max_value=5, value=3, key="shap_beds")
        luxury_shap = st.slider("Luxury Score (SHAP)", min_value=1, max_value=100, value=65, key="shap_lux")
        year_shap = st.slider("Year Built (SHAP)", min_value=1960, max_value=2025, value=2010, key="shap_year")
        garage_shap = st.slider("Garage Area (SHAP)", min_value=0, max_value=900, value=400, step=50, key="shap_garage")
        basement_shap = st.slider("Basement Area (SHAP)", min_value=0, max_value=2000, value=500, step=50, key="shap_basement")
        st.markdown("</div>", unsafe_allow_html=True)
        
        raw_sh = {
            'Area_SqFt': [area_shap], 'Bedrooms': [beds_shap], 'Full_Bathrooms': [2], 'Half_Bathrooms': [1],
            'Location': [loc_shap], 'Year_Built': [year_shap], 'Garage_Area': [garage_shap], 'Basement_Area': [basement_shap], 'Luxury_Score': [luxury_shap]
        }
        df_sh = pd.DataFrame(raw_sh)
        pred_sh = pipeline.predict(df_sh)[0]
        
    with col2:
        st.markdown("<div class='glass-card'><h3 style='margin-top:0; margin-bottom:1.5rem;'>Shapley Attributions Decomposition</h3>", unsafe_allow_html=True)
        
        stacking_model = pipeline.named_steps['regressor']
        rf_estimator = stacking_model.estimators_[0] # RF base model
        
        preproc = pipeline.named_steps['preprocessor_pipeline']
        instance_processed = preproc.transform(df_sh)
        
        @st.cache_resource
        def get_shap_explainer(_model, background_data):
            return shap.TreeExplainer(_model)
            
        df_clean = utils.clean_data(utils.load_data())
        X_sample = df_clean.drop(columns=['Price', 'Price_Per_SqFt'], errors='ignore')
        X_sample_proc = preproc.transform(X_sample[:200])
        
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
            decreasing = {"marker":{"color":"#ef4444"}}, # Red
            increasing = {"marker":{"color":"#10b981"}}, # Green
            totals = {"marker":{"color":"#6366f1"}}     # Indigo
        ))
        
        fig.update_layout(
            title = "SHAP local breakdown waterfall chart",
            showlegend = False,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': "Plus Jakarta Sans"},
            height = 500,
            margin=dict(l=20,r=20,t=40,b=20)
        )
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("### Interpretations of SHAP Indicators:")
    st.markdown("""
    - **Global Base Value**: Represents the average baseline target house price prediction across the dataset (~INR 500,000).
    - **Emerald Bars (+)**: Characteristics that inflate the property value relative to the base average.
    - **Ruby Bars (-)**: Characteristics that depreciate the property value relative to the base average.
    - **RF Model Target**: The final calculated target prediction of the base Random Forest model, matching the sum of inputs.
    """)


# ==========================================
# TAB 5: MODEL AUDIT & EXPERIMENT LOGS
# ==========================================
elif st.session_state.active_tab == "📈 Model Performance":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>📈 Model Benchmark & Experiment History</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Detailed benchmark matrices and MLflow-style logs of models trained in production.</p>", unsafe_allow_html=True)
    
    df_results = pd.read_csv(config['paths']['comparison_csv'])
    
    col1, col2 = st.columns([1.1, 1.2])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Cross-Validated Test Metrics")
        st.dataframe(df_results.style.highlight_max(subset=['R2', 'Adjusted R2'], color='#10b981').highlight_min(subset=['MAE', 'RMSE'], color='#ef4444'), width='stretch')
        
        # Bar chart comparison
        fig = px.bar(df_results, x='Model', y='R2', text_auto=".3f", title="R² Score Comparison (Higher is Better)", color='R2', color_continuous_scale="viridis")
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_range=[0, 1], height=320, font={'family':'Plus Jakarta Sans'})
        st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Tuned Base Model diagnostics (Holdout Split)")
        
        from sklearn.model_selection import train_test_split
        df_clean = utils.clean_data(utils.load_data())
        X = df_clean.drop(columns=['Price', 'Price_Per_SqFt'], errors='ignore')
        y = df_clean['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        test_preds = pipeline.predict(X_test)
        
        perf_tab1, perf_tab2 = st.tabs(["Actual vs Predicted", "Residual Analysis"])
        with perf_tab1:
            val_df = pd.DataFrame({'Actual': y_test, 'Predicted': test_preds})
            fig = px.scatter(val_df, x='Actual', y='Predicted', title="Actual vs Predicted Prices", labels={'Actual':'Actual Price (INR)', 'Predicted':'Predicted Price (INR)'})
            fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="Red", dash="dash"))
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch')
        with perf_tab2:
            residuals = y_test - test_preds
            res_df = pd.DataFrame({'Predicted': test_preds, 'Residuals': residuals})
            fig = px.scatter(res_df, x='Predicted', y='Residuals', title="Residuals vs Predicted (Error Spread)", labels={'Predicted':'Predicted Value', 'Residuals':'Prediction Error'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
            
    # MLflow-style Experiment history logging
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Training Experiment Logs database")
    st.markdown("Logs of past cross-validated model tuning runs loaded dynamically from `models/experiment_log.json`.")
    tracker = utils.ExperimentTracker()
    logs = tracker.get_logs()
    
    if logs:
        log_df = pd.DataFrame(logs)[['timestamp', 'model_name', 'cv_r2_mean', 'cv_r2_std', 'cv_mae_mean', 'cv_mae_std', 'parameters']]
        st.dataframe(log_df.style.format({
            'cv_r2_mean': '{:.4f}', 'cv_r2_std': '{:.4f}', 
            'cv_mae_mean': 'INR {:,.2f}', 'cv_mae_std': 'INR {:,.2f}'
        }), width='stretch')
    else:
        st.info("No training runs logged in experiment database yet.")
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# TAB 6: SQLITE DATABASE EXPLORER & RETRAINING HUB
# ==========================================
elif st.session_state.active_tab == "🗃️ DB & Retraining Hub":
    st.markdown("<h1 style='font-size:2.4rem; margin-bottom: 0.2rem; text-align:center;'>🗃️ SQLite Database & Model Retraining Hub</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; font-size:1.1rem; margin-bottom:2rem; text-align:center;'>Explore property listings, manage CRUD operations, review logged predictions, and trigger retraining cycles.</p>", unsafe_allow_html=True)
    
    # Fetch Dataframes from database
    prop_df = utils.fetch_properties_from_db()
    pred_df = utils.fetch_predictions_log()
    
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
    
    db_tab1, db_tab2, db_tab3, db_tab4 = st.tabs(["🏠 Property Listings CRUD", "📋 Prediction Logs & Feedback", "⚡ MLOps Retraining Console", "🔍 MLOps Data Audit & Drift"])
    
    with db_tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Manage Database Property Records")
        st.write("Browse housing listings in SQLite. Use the options below to insert new properties or remove entries from database.")
        
        # Add Search/Filter fields
        f1, f2 = st.columns(2)
        with f1:
            search_loc = st.multiselect("Filter by Location Zone", options=list(prop_df['Location'].unique()) if not prop_df.empty else [], default=[])
        with f2:
            max_price_val = int(prop_df['Price'].max()) if not prop_df.empty else 1000000
            price_range = st.slider("Filter by Price Range (INR)", 0, max_price_val, (0, max_price_val))
            
        filtered_df = prop_df.copy()
        if search_loc:
            filtered_df = filtered_df[filtered_df['Location'].isin(search_loc)]
        if not filtered_df.empty:
            filtered_df = filtered_df[(filtered_df['Price'] >= price_range[0]) & (filtered_df['Price'] <= price_range[1])]
            
        st.dataframe(filtered_df, width='stretch')
        
        # CRUD operations: Create and Delete
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            with st.expander("➕ Add New Property Record", expanded=False):
                with st.form("add_property_form", clear_on_submit=True):
                    new_loc = st.selectbox("Location Sector Zone", ['Rural', 'Suburbs', 'Standard', 'Downtown', 'Premium'], index=2)
                    new_area = st.number_input("Area Size (SqFt)", min_value=100.0, max_value=20000.0, value=1500.0)
                    new_bedrooms = st.slider("Bedrooms count", 1, 10, 3)
                    new_full_bath = st.slider("Full Bathrooms count", 1, 8, 2)
                    new_half_bath = st.slider("Half Bathrooms count", 0, 4, 1)
                    new_garage = st.number_input("Garage Area Size (SqFt)", min_value=0.0, max_value=5000.0, value=300.0)
                    new_basement = st.number_input("Basement Area Size (SqFt)", min_value=0.0, max_value=10000.0, value=0.0)
                    new_year = st.number_input("Year Built", min_value=1800, max_value=2026, value=2015)
                    new_luxury = st.slider("Luxury & Finishes Score", 1, 100, 50)
                    new_price = st.number_input("Price (INR)", min_value=1000.0, value=350000.0, step=10000.0)
                    
                    submit_add = st.form_submit_button("Insert Property Listing")
                    if submit_add:
                        new_prop = {
                            'Area_SqFt': new_area, 'Bedrooms': new_bedrooms, 'Full_Bathrooms': new_full_bath, 'Half_Bathrooms': new_half_bath,
                            'Location': new_loc, 'Year_Built': new_year, 'Garage_Area': new_garage, 'Basement_Area': new_basement, 'Luxury_Score': new_luxury,
                            'Price': new_price
                        }
                        utils.add_property_to_db(new_prop)
                        st.success("🎉 Property inserted successfully into the SQLite listings database!")
                        st.rerun()
                        
        with col_c2:
            with st.expander("❌ Delete Listings by ID", expanded=False):
                st.write("Enter the database record IDs you want to delete.")
                delete_id_input = st.text_input("Property IDs (comma-separated, e.g. 3001, 3002)")
                submit_delete = st.button("Delete Records", type="secondary", width='stretch')
                if submit_delete and delete_id_input:
                    try:
                        ids_to_delete = [int(x.strip()) for x in delete_id_input.split(",") if x.strip().isdigit()]
                        if ids_to_delete:
                            for idx in ids_to_delete:
                                utils.delete_property_from_db(idx)
                            st.success(f"🗑️ Successfully deleted listings: {ids_to_delete}")
                            st.rerun()
                        else:
                            st.warning("Please enter valid numeric IDs.")
                    except Exception as e:
                        st.error(f"Deletion failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with db_tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Property Valuation Prediction History")
        st.write("Logs of predictions requested by users, including ratings and review feedback comments.")
        if not pred_df.empty:
            st.dataframe(pred_df, width='stretch')
        else:
            st.info("No predictions logged in the database yet. Perform calculations in the 'Valuation Predictor' tab and submit feedback to see entries.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with db_tab3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("MLOps Core Model Retraining Center")
        st.write("Trigger a model retraining run over all listings currently saved in the SQLite properties database.")
        
        # Display current best model
        st.markdown(f"""
        <div style="background: rgba(99,102,241,0.05); border: 1px solid rgba(99,102,241,0.15); border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem;">
            <p style="margin: 0; font-size: 0.9rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em;">Current Production Model</p>
            <h3 style="margin: 0.2rem 0 0 0; color: #6366f1;">{best_model_name}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        db_rows_count = len(prop_df)
        if db_rows_count < 500:
            st.warning(f"⚠️ Insufficient data for retraining. SQLite database has {db_rows_count} property listings. A minimum of 500 rows is required to avoid overfitting.")
        else:
            st.success(f"✓ SQLite database contains {db_rows_count} listings. Model retraining is fully enabled.")
            
            trigger_train = st.button("⚡ Trigger Core Retraining Pipeline", type="primary", width='stretch')
            if trigger_train:
                import train
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
                        train.run_training_pipeline(progress_callback=custom_progress_callback)
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
