import pandas as pd
import joblib

def batch_predict_all():
    print("Loading full dataset for bulk prediction...")
    telecom_customer_df = pd.read_csv('../data/processed/merged_telecom_data.csv')
    
    # Preprocessing & encoding steps to match training pipeline
    telecom_customer_df = telecom_customer_df.fillna(0)
    telecom_customer_encoded_df = pd.get_dummies(telecom_customer_df, drop_first=True)
    telecom_customer_encoded_df = telecom_customer_encoded_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Extract Customer ID if it exists
    customer_ids = telecom_customer_encoded_df['Customer_ID'] if 'Customer_ID' in telecom_customer_encoded_df.columns else telecom_customer_encoded_df.index
    
    # Prepare features (X) by dropping ID and target column
    X = telecom_customer_encoded_df.drop(columns=['Customer_ID', 'churn'], errors='ignore')

    # ==========================================
    # Step 1: Load the Saved Best Model
    # ==========================================
    model_file_path = '../models/best_churn_model.pkl'
    print(f"Loading model from {model_file_path}...")
    loaded_model = joblib.load(model_file_path)
    print("Model loaded successfully!")

    # ==========================================
    # Step 2: Make Predictions for All Customers
    # ==========================================
    print("Running predictions for all customers (this may take a few seconds)...")
    predictions = loaded_model.predict(X)
    probabilities = loaded_model.predict_proba(X)[:, 1]

    # ==========================================
    # Step 3: Create Output DataFrame & Save to CSV
    # ==========================================
    output_df = pd.DataFrame({
        'Customer_ID': customer_ids,
        'Predicted_Churn_Status': ['Churn' if p == 1 else 'Retained' for p in predictions],
        'Churn_Probability_%': (probabilities * 100).round(2)
    })

    output_path = '../data/processed/all_customers_churn_predictions.csv'
    output_df.to_csv(output_path, index=False)
    print(f"Batch predictions successfully saved to: {output_path}")

    # Display summary statistics
    churn_count = (output_df['Predicted_Churn_Status'] == 'Churn').sum()
    retained_count = (output_df['Predicted_Churn_Status'] == 'Retained').sum()
    print(f"\n--- Prediction Summary ---")
    print(f"Total Customers Analyzed: {len(output_df)}")
    print(f"Predicted to Churn (At-Risk): {churn_count}")
    print(f"Predicted to Retain (Safe): {retained_count}")

if __name__ == "__main__":
    batch_predict_all()