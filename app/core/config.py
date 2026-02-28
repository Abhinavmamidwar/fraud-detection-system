THRESHOLDS = {
    "BLOCK": 0.20,
    "FLAG": 0.11,
    "WATCH": 0.09,
}

MODEL_PATH = "models/production_fraud_model.pkl"
SCALER_PATH = "models/production_scaler.pkl"
FEATURE_PATH = "models/production_features.pkl"

COLUMNS_TO_SCALE = [
    "amount",
    "oldbalanceOrg",
    "amount_ratio",
    "log_amount",
    "log_balance",
]
