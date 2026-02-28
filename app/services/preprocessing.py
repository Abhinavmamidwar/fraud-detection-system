import pandas as pd

def preprocess_input(data: dict, scaler):
    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)
    return df_scaled
