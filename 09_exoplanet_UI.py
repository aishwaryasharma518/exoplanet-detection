# 09_exoplanet_ui.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from scipy import stats

st.set_page_config(page_title="Exoplanet Detector", page_icon="🪐", layout="wide")

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .planet {
        background-color: #1e3c72;
        color: white;
    }
    .no-planet {
        background-color: #2e7d32;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1 style="color: white;">🪐 Exoplanet Detection System</h1><p style="color: #ddd;">Using Machine Learning on NASA Kepler Light Curve Data</p></div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load('random_forest_model.pkl')
    except:
        return None

model = load_model()

if model is None:
    st.error("❌ Model not found! Run 05_train_model.py first")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📁 Data Source")
    uploaded_file = st.file_uploader("Upload a light curve CSV file", type=['csv'])
    
    st.markdown("---")
    st.header("📊 Sample Data")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🪐 Planet Star", use_container_width=True):
            uploaded_file = "star_0_planet.csv"
    with col2:
        if st.button("⭐ Non-Planet Star", use_container_width=True):
            uploaded_file = "star_10_noplanet.csv"
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("**Model:** Random Forest Classifier\n**Data:** NASA Kepler Mission")

def extract_features(flux):
    threshold = 0.995
    dip_points = flux[flux < threshold]
    window = min(100, len(flux) // 10)
    if window < 2:
        window = 2
    smoothed = np.convolve(flux, np.ones(window)/window, mode='same')
    noise_std = np.std(flux - smoothed)
    
    return np.array([[
        (1 - flux.min()) * 100,
        len(dip_points),
        (1 - flux.min()) / noise_std if noise_std > 0 else 0,
        1 - flux.min(),
        len(dip_points) / len(flux) * 100,
        len(dip_points),
        flux.min(),
        np.std(flux),
        stats.skew(flux),
        stats.kurtosis(flux),
        np.percentile(flux, 75) - np.percentile(flux, 25)
    ]])

def plot_light_curve(time, flux, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    display_points = min(5000, len(time))
    ax.plot(time[:display_points], flux[:display_points], 'b-', linewidth=0.5)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(title)
    ax.set_ylim(0.996, 1.002)
    ax.axhline(y=0.998, color='r', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    return fig

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Light Curve Visualization")
with col2:
    st.subheader("🔮 Prediction Result")

if uploaded_file is not None:
    try:
        if isinstance(uploaded_file, str):
            df = pd.read_csv(uploaded_file)
            file_name = uploaded_file
        else:
            df = pd.read_csv(uploaded_file)
            file_name = uploaded_file.name
        
        features = extract_features(df['flux'].values)
        prediction = model.predict(features)[0]
        
        min_flux = df['flux'].min()
        transit_depth = (1 - min_flux) * 100
        
        fig = plot_light_curve(df['time'].values, df['flux'].values, f"Light Curve: {file_name}")
        
        with col1:
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            if prediction == 1:
                st.markdown("""
                <div class="prediction-box planet">
                    <h2>🪐 EXOPLANET DETECTED!</h2>
                    <p>A transit dip was found in the light curve</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box no-planet">
                    <h2>⭐ NO PLANET DETECTED</h2>
                    <p>No significant transit dip found</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.metric("Transit Depth", f"{transit_depth:.3f}%")
            st.metric("Minimum Flux", f"{min_flux:.6f}")
            
    except Exception as e:
        st.error(f"Error: {e}")

else:
    with col1:
        st.info("👈 Upload a light curve CSV file or click a sample button")
    with col2:
        st.info("🪐 Prediction will appear here")

st.markdown("---")
st.markdown("<p style='text-align: center;'>🚀 Exoplanet Detection System | Model: Random Forest | Data: NASA Kepler Mission</p>", unsafe_allow_html=True)