import pandas as pd
from sklearn.preprocessing import LabelEncoder
# to lode data from raw folder
def load_data(file_path):
    return pd.read_csv(file_path)
# to clean data and to remove unimportant colums
def clean_and_preprocess(df):
    if 'LoanID' in df.columns:
        df = df.drop(columns=['LoanID'])
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df