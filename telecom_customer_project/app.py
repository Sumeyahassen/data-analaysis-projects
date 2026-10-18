import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Company A - Churn Prediction", page_icon="📞", layout="centered")

@st.cache_resource
def load_model_and_structure():
    # Load the saved model
    model_path = 'models/best_churn_model.pkl'
    if not os.path.exists(model_path):
        model_path = '../models/best_churn_model.pkl'  # fallback if running from src/
        
    model = joblib.load(model_path)
    
    # Load a small sample of the processed data to get column structure
    data_path = 'data/processed/merged_telecom_data.csv'
    if not os.path.exists(data_path):
        data_path = '../data/processed/merged_telecom_data.csv'
        
    df = pd.read_csv(data_path, nrows=5)
    return model, df

model, sample_df = load_model_and_structure()

st.title("📞 Company A - Customer Churn Prediction App")
st.write("Enter customer details below to test real-time predictions using the trained Random Forest model.")

st.markdown("---")

# User inputs for key numerical features
col1, col2 = st.columns(2)

with col1:
    eqpdays = st.number_input("Equipment Days (eqpdays)", min_value=0, max_value=2000, value=300, help="Age of current equipment in days")
    months = st.number_input("Account Tenure (Months)", min_value=0, max_value=120, value=12, help="Total months subscribed")

with col2:
    mou_Mean = st.number_input("Average Monthly Minutes (mou_Mean)", min_value=0.0, max_value=10000.0, value=500.0)
    totrev = st.number_input("Total Revenue (totrev)", min_value=0.0, max_value=20000.0, value=400.0)

st.markdown("---")

if st.button("Predict Churn Risk", type="primary"):
    # Build input dataframe matching the exact training columns expected by the model
    if hasattr(model, "feature_names_in_"):
        # If scikit-learn model stores feature names
        final_input = pd.DataFrame(0, index=[0], columns=model.feature_names_in_)
    else:
        # Fallback based on processed dataframe structure
        processed_sample = pd.get_dummies(sample_df.drop(columns=['Customer_ID', 'churn'], errors='ignore'), drop_first=True)
        final_input = pd.DataFrame(0, index=[0], columns=processed_sample.columns)

    # Assign user values to corresponding columns if they exist in the model features
    for col, val in [('eqpdays', eqpdays), ('months', months), ('mou_Mean', mou_Mean), ('totrev', totrev)]:
        if col in final_input.columns:
            final_input[col] = val

    # Make prediction
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1] * 100

    # Display results
    st.subheader("Prediction Results:")
    
    if prediction == 1:
        st.error(f"⚠️ **At-Risk (Churn Predicted)**")
    else:
        st.success(f"✅ **Safe (Retained Customer)**")
        
    st.metric(label="Calculated Churn Probability", value=f"{probability:.2f}%")