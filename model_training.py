"""
Sepsis Risk Assessment - Model Training Script
Trains ensemble ML models on synthetic sepsis data (replace with real dataset)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
# import lightgbm as lgb
import joblib
import json
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_NAMES = [
    'age', 'gender', 'temperature', 'heart_rate', 'respiratory_rate',
    'systolic_bp', 'diastolic_bp', 'spo2', 'blood_sugar', 'wbc_count',
    'platelet_count', 'lactate_level', 'creatinine_level', 'urine_output',
    'mental_status_num', 'fever', 'chills', 'confusion', 'rapid_breathing',
    'low_bp', 'fatigue', 'vomiting', 'organ_dysfunction'
]

def generate_synthetic_data(n_samples=5000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 90, n_samples),
        'gender': np.random.randint(0, 2, n_samples),
        'temperature': np.random.normal(37.5, 1.5, n_samples),
        'heart_rate': np.random.normal(90, 25, n_samples),
        'respiratory_rate': np.random.normal(18, 6, n_samples),
        'systolic_bp': np.random.normal(110, 25, n_samples),
        'diastolic_bp': np.random.normal(70, 15, n_samples),
        'spo2': np.random.normal(96, 4, n_samples),
        'blood_sugar': np.random.normal(130, 50, n_samples),
        'wbc_count': np.random.normal(11, 6, n_samples),
        'platelet_count': np.random.normal(200, 80, n_samples),
        'lactate_level': np.random.exponential(2, n_samples),
        'creatinine_level': np.random.exponential(1.2, n_samples),
        'urine_output': np.random.normal(50, 20, n_samples),
        'mental_status_num': np.random.randint(0, 4, n_samples),
        'fever': np.random.randint(0, 2, n_samples),
        'chills': np.random.randint(0, 2, n_samples),
        'confusion': np.random.randint(0, 2, n_samples),
        'rapid_breathing': np.random.randint(0, 2, n_samples),
        'low_bp': np.random.randint(0, 2, n_samples),
        'fatigue': np.random.randint(0, 2, n_samples),
        'vomiting': np.random.randint(0, 2, n_samples),
        'organ_dysfunction': np.random.randint(0, 2, n_samples),
    }
    df = pd.DataFrame(data)

    # Sepsis scoring rule
    score = (
        (df['temperature'] > 38.3).astype(int) * 2 +
        (df['heart_rate'] > 90).astype(int) * 2 +
        (df['respiratory_rate'] > 20).astype(int) * 2 +
        (df['systolic_bp'] < 90).astype(int) * 3 +
        (df['spo2'] < 94).astype(int) * 2 +
        (df['wbc_count'] > 12).astype(int) * 2 +
        (df['lactate_level'] > 2).astype(int) * 3 +
        (df['creatinine_level'] > 1.5).astype(int) * 2 +
        df['fever'] + df['chills'] + df['confusion'] * 2 +
        df['organ_dysfunction'] * 3 +
        (df['age'] > 65).astype(int)
    )
    labels = np.select(
        [score < 3, score < 6, score < 10, score < 14, score >= 14],
        [0, 1, 2, 3, 4]
    )
    df['label'] = labels
    return df

def train_models():
    print("Generating training data...")
    df = generate_synthetic_data(5000)
    X = df[FEATURE_NAMES].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

    # XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                    use_label_encoder=False, eval_metric='mlogloss',
                                    random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_s, y_train)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_model.pkl'))

    # LightGBM
    # print("Training LightGBM...")
    # lgb_model = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
    #                                  random_state=42, n_jobs=-1, verbose=-1)
    # lgb_model.fit(X_train_s, y_train)
    # joblib.dump(lgb_model, os.path.join(MODELS_DIR, 'lgb_model.pkl'))

    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_s, y_train)
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'rf_model.pkl'))

    # Evaluate
    for name, model in [('XGBoost', xgb_model), ('RandomForest', rf_model)]:
        acc = model.score(X_test_s, y_test)
        print(f"{name} Accuracy: {acc:.4f}")

    # Save feature names
    with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'w') as f:
        json.dump(FEATURE_NAMES, f)

    print("\nAll models trained and saved successfully!")

if __name__ == '__main__':
    train_models()
