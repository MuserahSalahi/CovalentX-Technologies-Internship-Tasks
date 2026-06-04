import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# Page Configuration & Professional Design
# ==========================================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Custom CSS styling for premium look
st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 16px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-card {
        background-color: #F0FDF4;
        border: 1px solid #DCFCE7;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        color: #166534;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏠 Real Estate Valuation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter the property dimensions and features below to forecast house price using Linear Regression.</div>', unsafe_allow_html=True)

# ==========================================
# Backend Model Training 
# ==========================================
@st.cache_data
def train_ml_model():
    df = pd.read_csv('house_price_prediction.csv')
    X = df[['Area_SqFt', 'Bedrooms', 'Bathrooms']]
    y = df['Price_USD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, df

try:
    model, df = train_ml_model()
except FileNotFoundError:
    st.error("Error: 'house_price_prediction.csv' file isn't found! Please check your CSV file must be in the same folder as your app.py file.")
    st.stop()

# ==========================================
# User Interface (UI) - Secured with st.form (No Max Limits)
# ==========================================
st.subheader("Input Property Specifications")

# Inputs ko Form ke andar wrap kiya hai taake value clear karte waqt app tang na kare
with st.form("property_input_form", clear_on_submit=False):
    
    col1, col2, col3 = st.columns(3)

    with col1:
        # min_value aur max_value ko hata diya hai taake koi limit na rahe
        area_input = st.number_input(
            "Total Area (SqFt)", 
            value=None,
            placeholder="e.g. 2500",
            step=50
        )

    with col2:
        bedrooms_input = st.number_input(
            "Bedrooms Count", 
            value=None,
            placeholder="e.g. 3",
            step=1
        )

    with col3:
        bathrooms_input = st.number_input(
            "Bathrooms Count", 
            value=None,
            placeholder="e.g. 2",
            step=1
        )

    submit_button = st.form_submit_button("Price Prediction", type="primary", use_container_width=True)

# ==========================================
# Form Execution Logic
# ==========================================
if submit_button:
    # Check 1: Agar koi field khali chhor di hai
    if area_input is None or bedrooms_input is None or bathrooms_input is None:
        st.error("⚠️ Please fill in all the input fields before generating the prediction.")
    
    # Check 2: Negative values ko rokne ke liye simple validation
    elif area_input <= 0 or bedrooms_input <= 0 or bathrooms_input <= 0:
        st.error("⚠️ Please enter valid positive numbers for property features.")
        
    else:
        with st.spinner("Processing features through Linear Regression model..."):
            features = np.array([[area_input, bedrooms_input, bathrooms_input]])
            predicted_value = model.predict(features)[0]
            
            # Agar prediction galti se negative chali jaye (unrealistic inputs par), toh usey 0 par clip kar dein
            if predicted_value < 0:
                predicted_value = 0
                
            st.markdown(f"""
                <div class="result-card">
                    Estimated Market Valuation: ${predicted_value:,.2f}
                </div>
            """, unsafe_allow_html=True)

# Sidebar for Professional Insights
st.sidebar.header("Dataset Overview")
st.sidebar.write(f"**Total Observations:** {df.shape[0]} rows")
st.sidebar.write(f"**Target Feature:** `Price_USD`")














