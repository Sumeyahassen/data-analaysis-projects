## Exploratory Data Analysis (EDA) - Report 01: Target Variable Analysis

### 1. Objective
The goal of this initial analysis is to examine the distribution of the target variable (`churn`) after merging the `client` and `record` datasets using `Customer_ID`.

### 2. Findings: Customer Churn Distribution
* **Stayed (Churn = 0):** ~50.44% of the total customers.
* **Churned (Churn = 1):** ~49.56% of the total customers.
### 3. Key Insights
* **Balanced Dataset:** Unlike many real-world telecom datasets where churn rates are heavily skewed (e.g., only 10% churn), this dataset exhibits a remarkably balanced distribution (~50/50 split).
* **Modeling Implication:** Because the classes are well-balanced, standard classification metrics (like Accuracy and ROC-AUC) will be reliable, and we may not need complex resampling techniques (such as SMOTE) to handle class imbalance during model training.
# Exploratory Data Analysis & Preprocessing Report 02

**Project:** Telecom Customer Churn Prediction (100,000 Clients)  
**Institution:** University of Tokyo - AI & Data Analytics Project  
**Date:** October 2026  

---

## 1. Executive Summary
This report documents the second phase of the Telecom Churn Prediction Pipeline: **Data Preprocessing, Missing Value Handling, and Categorical Encoding**. Following the target variable distribution analysis in EDA 01, the primary goal of this stage was to ensure dataset completeness, clean null records, transform non-numeric strings into machine-readable numeric features, and prepare the dataset for predictive machine learning modeling.

---

## 2. Key Achievements & Workflow

### 2.1. Dataset Merging & Verification
- The core datasets (`client.csv` and `record.csv`) were merged using `Customer_ID` as the primary key via an inner join.
- The consolidated dataset was saved as `merged_telecom_data.csv` under `data/processed/`.
- Total Records: **100,000 customers**.

### 2.2. Comprehensive Missing Value Handling (Null Imputation)
Initially, a total of **342,969 null values** were detected across numerical and categorical features. To avoid data loss associated with row deletion, an imputation strategy was executed:
1. **Numerical Features:** Imputed using column-wise **Median** values to maintain robustness against outliers.
2. **Categorical/Text Features:** Imputed with the placeholder string `'Unknown'`.
3. **Verification:** Executed iterative checks until `telecom_customer_df.isnull().sum().sum()` returned **0**.

```python
# Verification Status
Currently null value amount: 0
2.3. Feature Encoding & Expansion (One-Hot Encoding)Machine learning algorithms require numeric matrices for mathematical operations. We applied pd.get_dummies() to convert text columns into binary indicators:Initial Feature Count: ~100 original columns.Post-Encoding Feature Count: 229 columns (5 rows × 229 columns matrix shape).Technical Explanation: The feature expansion from 100 to 229 columns occurred because multi-class categorical variables (e.g., demographic flags, credit ratings, contract categories like kid3_5, creditcd) were decomposed into individual binary (True/False or 1/0) feature columns. This preserves categorical information without imposing false ordinal rankings on the model.3. Data Architecture SummaryMetric / StageStatus / ValueMerged Dataset Filedata/processed/merged_telecom_data.csvTotal Rows100,000Initial Null Count342,969Post-Cleaning Null Count0Initial Columns~100Encoded Columns229Data TypesNumeric (Float64, Int64) & Boolean Indicators4. Next Steps & Action PlanWith a clean, zero-null, fully encoded dataset, the project transitions into model development:Feature-Target Splitting (Train-Test Split):Separate predictors ($X$, 227 features after dropping Customer_ID and churn) from the target variable ($y$, churn).Partition data into 80% Training Set (80,000 samples) and 20% Testing Set (20,000 samples) using random_state=42 for reproducibility.Baseline Model Training (Random Forest Classifier):Train an initial Ensemble Tree-based model (RandomForestClassifier) to establish baseline performance metrics.Evaluate model accuracy, precision, recall, and ROC-AUC score on the unseen 20% test partition.Feature Importance & Insights Analysis:Extract top predictive features influencing telecom churn (e.g., monthly charges, call duration, tenure) to provide actionable recommendations for Company A's retention team.
# Final Model Evaluation & Business Insights Report 04

**Project:** Telecom Customer Churn Prediction (100,000 Clients)  
**Institution:** University of Tokyo - AI & Data Analytics Project  
**Date:** October 2026  

---

## 1. Executive Summary
This report concludes the machine learning pipeline for Company A’s customer churn prediction project. Following robust data preprocessing (handling 342,969 null values and One-Hot Encoding to 227 features), we implemented a baseline Random Forest Classifier and successfully executed **Hyperparameter Tuning (RandomizedSearchCV)** to optimize predictive performance.

---

## 2. Model Performance Summary

### 2.1. Baseline vs. Tuned Model Accuracy
- **Baseline Random Forest Accuracy:** `61.92%`
- **Tuned Random Forest Accuracy:** **`63.24%`**
- **Optimal Hyperparameters Found:**
  - `n_estimators`: 300
  - `max_depth`: 30
  - `min_samples_split`: 5
  - `min_samples_leaf`: 4

### 2.2. Classification Metrics (Tuned Model)
- **Class 0 (Retained Customers):** Precision = 0.64, Recall = 0.62, F1-Score = 0.63
- **Class 1 (Churned Customers):** Precision = 0.63, Recall = 0.65, F1-Score = 0.64
- **Overall Accuracy:** **63%** across 20,000 test samples.

---

## 3. Key Feature Importance Insights (Business Intelligence)
Based on the trained Random Forest model, the top features influencing customer churn are:
1. **`eqpdays` (Equipment Age):** Older equipment correlates heavily with customer churn. Offering phone upgrades can significantly reduce churn rates.
2. **`change_mou` & `change_rev`:** Sudden drops or fluctuations in call minutes and monthly revenue signal at-risk customers.
3. **`months` (Tenure):** Customer loyalty over time plays a critical role in retention strategy.

---

## 4. Conclusion & Recommendations for Company A
1. **Targeted Retention Campaigns:** Focus promotional offers on customers whose equipment (`eqpdays`) is outdated.
2. **Early Warning System:** Monitor sudden changes in `change_mou` to intervene before customers officially churn.
3. **Future Enhancements:** Future iterations can explore advanced algorithms like XGBoost or LightGBM alongside SMOTE to handle class boundaries more aggressively.