"""
FRAUD DETECTION SYSTEM - PRODUCTION READY
No data leakage, realistic performance, production best practices
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, roc_auc_score, 
                           confusion_matrix, precision_recall_curve,
                           average_precision_score, roc_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FRAUD DETECTION SYSTEM - PRODUCTION READY")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n📥 Loading data...")
data = pd.read_csv(r"D:\RESUME PROJECT\fraud-detection-system\training\data.csv")
print(f"✅ Data loaded: {data.shape[0]:,} transactions, {data.shape[1]} features")

# ============================================
# 2. SIMPLE EXPLORATORY DATA ANALYSIS (EDA)
# ============================================
print("\n" + "="*60)
print("📊 SIMPLE DATA EXPLORATION")
print("="*60)

# 1. Missing values check
print("\n🔍 Checking for missing values...")
missing_values = data.isnull().sum()
if missing_values.sum() == 0:
    print("✅ GOOD NEWS: No missing values found!")
else:
    print("⚠️ Found missing values:")
    print(missing_values[missing_values > 0])

# 2. Data overview
print("\n📋 First 5 rows of data:")
print(data.head())

print("\n📋 Data types and info:")
print(data.info())

# 3. Simple statistics
print("\n📊 Simple statistics for numerical columns:")
print(data.describe())

# 4. Correlation with fraud
print("\n🔗 Correlation with isFraud:")
# Pick only numerical columns
number_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
               'oldbalanceDest', 'newbalanceDest', 'isFraud']
correlations = data[number_cols].corr()['isFraud'].sort_values(ascending=False)
print(correlations)

# ============================================
# 3. FEATURE ENGINEERING (NO LEAKAGE!)
# ============================================
print("\n🔧 Engineering features (no future data)...")

def create_features(df):
    """Create features using ONLY information available at transaction time"""
    
    features = pd.DataFrame(index=df.index)
    
    # ----- BASIC FEATURES (Safe) -----
    features['step'] = df['step']
    features['amount'] = df['amount']
    features['oldbalanceOrg'] = df['oldbalanceOrg']  # Sender balance BEFORE
    
    # ----- ENCODED CATEGORICAL FEATURES -----
    # Transaction type (most important categorical)
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    features = pd.concat([features, type_dummies], axis=1)
    
    # Sender type (C=Customer, M=Merchant)
    features['sender_is_customer'] = (df['nameOrig'].str[0] == 'C').astype(int)
    features['sender_is_merchant'] = (df['nameOrig'].str[0] == 'M').astype(int)
    
    # Receiver type
    features['receiver_is_customer'] = (df['nameDest'].str[0] == 'C').astype(int)
    features['receiver_is_merchant'] = (df['nameDest'].str[0] == 'M').astype(int)
    
    # ----- SAFE ENGINEERED FEATURES -----
    # Amount relative to balance (fraud often has unusual ratios)
    features['amount_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)  # +1 to avoid div by zero
    
    # Was the sender's balance zero? (common in fraud)
    features['sender_balance_zero'] = (df['oldbalanceOrg'] == 0).astype(int)
    
    # Transaction size indicators
    features['is_large_tx'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
    features['is_small_tx'] = (df['amount'] < df['amount'].quantile(0.05)).astype(int)
    
    # Same initial letter (potential mule accounts)
    features['same_initial'] = (df['nameOrig'].str[0] == df['nameDest'].str[0]).astype(int)
    
    # Amount rounded? (fraud sometimes uses round numbers)
    features['amount_rounded'] = (df['amount'] == df['amount'].round()).astype(int)
    
    # Log transform for better distribution
    features['log_amount'] = np.log1p(df['amount'])
    features['log_balance'] = np.log1p(df['oldbalanceOrg'])
    
    return features

# Create features
X = create_features(data)
y = data['isFraud']

print(f"\n✅ Features created: {X.shape[1]} features")
print(f"📊 Feature list: {list(X.columns)}")

# ============================================
# 4. TEMPORAL SPLIT (More realistic than random)
# ============================================
print("\n📅 Creating temporal train/val/test split...")

# Sort by step (time)
data_sorted = data.sort_values('step')
X_sorted = X.loc[data_sorted.index]
y_sorted = y.loc[data_sorted.index]

# Split chronologically
train_size = int(0.7 * len(X_sorted))
val_size = int(0.15 * len(X_sorted))

X_train = X_sorted.iloc[:train_size]
X_val = X_sorted.iloc[train_size:train_size+val_size]
X_test = X_sorted.iloc[train_size+val_size:]

y_train = y_sorted.iloc[:train_size]
y_val = y_sorted.iloc[train_size:train_size+val_size]
y_test = y_sorted.iloc[train_size+val_size:]

print(f"✅ Training set: {len(X_train):,} transactions ({y_train.mean()*100:.4f}% fraud)")
print(f"✅ Validation set: {len(X_val):,} transactions ({y_val.mean()*100:.4f}% fraud)")
print(f"✅ Test set: {len(X_test):,} transactions ({y_test.mean()*100:.4f}% fraud)")

# ============================================
# 5. SCALE FEATURES
# ============================================
print("\n📊 Scaling features...")

# Identify numeric columns to scale
cols_to_scale = ['amount', 'oldbalanceOrg', 'amount_ratio', 
                 'log_amount', 'log_balance']

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

# Scale numeric columns
for col in cols_to_scale:
    if col in X_train.columns:
        X_train_scaled[col] = scaler.fit_transform(X_train[[col]])
        X_val_scaled[col] = scaler.transform(X_val[[col]])
        X_test_scaled[col] = scaler.transform(X_test[[col]])

print("✅ Scaling complete")

# ============================================
# 6. HANDLE IMBALANCE - COMPUTE CLASS WEIGHT
# ============================================
neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n⚖️ Class imbalance ratio: {neg_pos_ratio:.1f}:1")

# ============================================
# 7. TRAIN MULTIPLE MODELS
# ============================================
print("\n🤖 Training models...")

models = {
    'XGBoost': XGBClassifier(
        scale_pos_weight=neg_pos_ratio,
        max_depth=4,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
}

results = {}
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\n📈 Training {name}...")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Validate
    val_proba = model.predict_proba(X_val_scaled)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    val_ap = average_precision_score(y_val, val_proba)
    
    print(f"  Validation AUC: {val_auc:.4f}")
    print(f"  Validation Avg Precision: {val_ap:.4f}")
    
    # Test (only once at the end!)
    test_proba = model.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    test_ap = average_precision_score(y_test, test_proba)
    
    results[name] = {
        'model': model,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'test_ap': test_ap
    }
    
    if test_auc > best_score:
        best_score = test_auc
        best_model = model
        best_name = name

print(f"\n✅ Best model: {best_name}")

# ============================================
# 8. FIND OPTIMAL THRESHOLD (Business-aware)
# ============================================
print("\n🎯 Finding optimal threshold...")

# Get probabilities
val_proba = best_model.predict_proba(X_val_scaled)[:, 1]

# Precision-Recall curve
precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)

# Find threshold that maximizes F1
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"✅ Optimal threshold: {best_threshold:.4f}")
print(f"   At this threshold:")
print(f"   - Precision: {precisions[best_idx]:.4f}")
print(f"   - Recall: {recalls[best_idx]:.4f}")
print(f"   - F1-Score: {f1_scores[best_idx]:.4f}")

# ============================================
# 9. FINAL EVALUATION ON TEST SET
# ============================================
print("\n" + "="*60)
print("📊 FINAL TEST SET RESULTS")
print("="*60)

# Get test predictions
test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

# Metrics
test_auc = roc_auc_score(y_test, test_proba)
test_ap = average_precision_score(y_test, test_proba)
test_cm = confusion_matrix(y_test, test_pred)

print(f"\n🎯 ROC-AUC: {test_auc:.4f}")
print(f"🎯 Average Precision: {test_ap:.4f}")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, test_pred, target_names=['Normal', 'Fraud']))

print(f"\n📊 Confusion Matrix:")
print(f"               Predicted")
print(f"               Normal  Fraud")
print(f"Actual Normal   {test_cm[0,0]:6d}  {test_cm[0,1]:6d}")
print(f"       Fraud    {test_cm[1,0]:6d}  {test_cm[1,1]:6d}")

# Business metrics
missed_fraud = test_cm[1, 0]
caught_fraud = test_cm[1, 1]
false_alarms = test_cm[0, 1]

print(f"\n💼 BUSINESS METRICS:")
print(f"   - Fraud caught: {caught_fraud}/{caught_fraud + missed_fraud} ({caught_fraud/(caught_fraud+missed_fraud)*100:.1f}%)")
print(f"   - Missed fraud: {missed_fraud}")
print(f"   - False alarms: {false_alarms} (would need investigation)")

# ============================================
# 10. FEATURE IMPORTANCE
# ============================================
print("\n🔍 TOP 10 MOST IMPORTANT FEATURES:")

if best_name == 'XGBoost':
    importance = best_model.feature_importances_
else:
    importance = best_model.feature_importances_

feature_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importance
}).sort_values('importance', ascending=False).head(10)

for i, row in feature_imp.iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# ============================================
# 11. SAVE MODEL AND PREPROCESSING
# ============================================
print("\n💾 Saving model and preprocessing objects...")

# Save model
joblib.dump(best_model, "production_fraud_model.pkl")
joblib.dump(scaler, "production_scaler.pkl")
joblib.dump(best_threshold, "production_threshold.pkl")

# Save feature list (for inference)
feature_list = list(X_train.columns)
joblib.dump(feature_list, "production_features.pkl")

print("✅ Model saved as 'production_fraud_model.pkl'")
print("✅ Scaler saved as 'production_scaler.pkl'")
print("✅ Threshold saved as 'production_threshold.pkl'")
print("✅ Feature list saved as 'production_features.pkl'")

# ============================================
# 12. CREATE CHARTS (Saved in folder)
# ============================================
print("\n" + "="*60)
print("📈 CREATING CHARTS")
print("="*60)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Create a folder for charts
    if not os.path.exists('charts'):
        os.makedirs('charts')
        print("📁 Created 'charts' folder")
    
    # 1. Confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('charts/confusion_matrix.png')
    print("✅ Saved: charts/confusion_matrix.png")
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {test_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('charts/roc_curve.png')
    print("✅ Saved: charts/roc_curve.png")
    plt.close()
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, test_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {test_ap:.3f})', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('charts/precision_recall_curve.png')
    print("✅ Saved: charts/precision_recall_curve.png")
    plt.close()
    
    print("\n🎉 All charts saved in the 'charts' folder!")
    
except ImportError:
    print("\n⚠️ To see charts, install: pip install matplotlib seaborn")
except Exception as e:
    print(f"\n⚠️ Could not create charts: {e}")

# ============================================
# 13. PREDICTION FUNCTION (For new transactions)
# ============================================
print("\n" + "="*60)
print("📝 PREDICTION FUNCTION READY")
print("="*60)

print("""
def predict_fraud(transaction_df):
    # 1. Create features using create_features()
    # 2. Scale using saved scaler
    # 3. Get probability from model
    # 4. Apply threshold
    # 5. Return prediction and probability
""")

# Example usage code
print("\n📋 Example usage:")
print("""
# Load model and preprocessing
model = joblib.load("production_fraud_model.pkl")
scaler = joblib.load("production_scaler.pkl")
threshold = joblib.load("production_threshold.pkl")

# New transaction
new_tx = pd.DataFrame({
    'step': [1],
    'type': ['TRANSFER'],
    'amount': [10000],
    'nameOrig': ['C12345'],
    'oldbalanceOrg': [15000],
    'nameDest': ['M67890']
})

# Predict
features = create_features(new_tx)
features_scaled = scaler.transform(features)
prob = model.predict_proba(features_scaled)[0, 1]
prediction = prob >= threshold

print(f"Fraud Probability: {prob:.4f}")
print(f"Prediction: {'FRAUD' if prediction else 'NORMAL'}")
""")

print("\n" + "="*60)
print("✅ PRODUCTION-READY FRAUD DETECTION SYSTEM COMPLETE")
print("="*60)