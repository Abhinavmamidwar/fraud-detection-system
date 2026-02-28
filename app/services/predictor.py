import pandas as pd
from app.core.config import COLUMNS_TO_SCALE
from app.services.feature_engineering import create_features
from app.services.risk_engine import get_risk_decision


def predict_single(model, scaler, feature_list, transaction_dict):
    df = pd.DataFrame([transaction_dict])

    features = create_features(df, feature_list)

    features_scaled = features.copy()

    for col in COLUMNS_TO_SCALE:
        if col in features.columns:
            try:
                features_scaled[[col]] = scaler.transform(features[[col]])
            except:
                features_scaled[col] = features[col]

    proba = model.predict_proba(features_scaled)[0, 1]
    return get_risk_decision(proba)


def predict_batch(model, scaler, feature_list, transactions):
    df = pd.DataFrame(transactions)

    features = create_features(df, feature_list)
    features_scaled = features.copy()

    for col in COLUMNS_TO_SCALE:
        if col in features.columns:
            try:
                features_scaled[[col]] = scaler.transform(features[[col]])
            except:
                features_scaled[col] = features[col]

    probabilities = model.predict_proba(features_scaled)[:, 1]

    return [get_risk_decision(p) for p in probabilities]
