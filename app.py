import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Spaceship Titanic Kaggle Competition Benchmark & Stacking Ensemble",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 Spaceship Titanic Kaggle Competition Benchmark & Stacking Ensemble")
st.markdown("**Domain**: `Machine Learning / Competitive ML` | **Tech Stack**: `Stacking Classifier, LightGBM, XGBoost, Streamlit`")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **GitHub**: [Portfolio Repositories](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("⚙️ Domain Input Telemetry")
    cryo_sleep_flag = st.slider("Cryo Sleep Flag", int(0), int(1), int(0))
    passenger_age = st.slider("Passenger Age", float(0.0), float(80.0), float(28.0))
    room_service_spend = st.slider("Room Service Spend", float(0.0), float(9900.0), float(220.0))
    food_court_spend = st.slider("Food Court Spend", float(0.0), float(18000.0), float(450.0))
    shopping_mall_spend = st.slider("Shopping Mall Spend", float(0.0), float(8500.0), float(170.0))
    spa_amenities_spend = st.slider("Spa Amenities Spend", float(0.0), float(15000.0), float(310.0))
    vr_deck_spend = st.slider("Vr Deck Spend", float(0.0), float(14500.0), float(300.0))
    cabin_deck_tier = st.slider("Cabin Deck Tier", int(1), int(8), int(4))

with col2:
    st.subheader("🔮 Predictive Model Inference")
    model_path = os.path.join(os.path.dirname(__file__), "models/model_pipeline.joblib")
    scaler_path = os.path.join(os.path.dirname(__file__), "models/scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_df = pd.DataFrame([{"cryo_sleep_flag": cryo_sleep_flag, "passenger_age": passenger_age, "room_service_spend": room_service_spend, "food_court_spend": food_court_spend, "shopping_mall_spend": shopping_mall_spend, "spa_amenities_spend": spa_amenities_spend, "vr_deck_spend": vr_deck_spend, "cabin_deck_tier": cabin_deck_tier}])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        
        st.markdown("#### Real-Time Prediction Output")
        st.info(f"Predicted `transported_or_survived`: **{pred}**")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            st.progress(float(probs[1]) if len(probs) > 1 else float(probs[0]))
            st.caption(f"Confidence Probability Score: **{np.max(probs):.2%}**")
    else:
        st.warning("Model or Scaler artifact not found in models/ directory.")

st.markdown("---")
st.markdown("### 📊 Benchmark Metrics")
metrics_path = os.path.join(os.path.dirname(__file__), "reports/metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    st.json(metrics_data)
