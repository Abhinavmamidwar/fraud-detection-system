import pandas as pd
import numpy as np


def create_features(df, feature_list):
    features = pd.DataFrame(index=df.index)

    features["step"] = df["step"]
    features["amount"] = df["amount"]
    features["oldbalanceOrg"] = df["oldbalanceOrg"]

    type_dummies = pd.get_dummies(df["type"], prefix="type")

    expected_type_cols = [
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    ]

    for col in expected_type_cols:
        if col not in type_dummies.columns:
            type_dummies[col] = 0

    for col in expected_type_cols:
        features[col] = type_dummies[col].values

    features["sender_is_customer"] = (df["nameOrig"].str[0] == "C").astype(int)
    features["sender_is_merchant"] = (df["nameOrig"].str[0] == "M").astype(int)
    features["receiver_is_customer"] = (df["nameDest"].str[0] == "C").astype(int)
    features["receiver_is_merchant"] = (df["nameDest"].str[0] == "M").astype(int)

    features["amount_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    features["sender_balance_zero"] = (df["oldbalanceOrg"] == 0).astype(int)
    features["is_large_tx"] = (df["amount"] > 200000).astype(int)
    features["is_small_tx"] = (df["amount"] < 1000).astype(int)
    features["same_initial"] = (
        df["nameOrig"].str[0] == df["nameDest"].str[0]
    ).astype(int)

    features["amount_rounded"] = (
        df["amount"] == df["amount"].round()
    ).astype(int)

    features["log_amount"] = np.log1p(df["amount"])
    features["log_balance"] = np.log1p(df["oldbalanceOrg"])

    # Ensure model feature alignment
    for col in feature_list:
        if col not in features.columns:
            features[col] = 0

    return features[feature_list]
