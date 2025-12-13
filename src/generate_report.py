
"""
Generate Model Performance Report
Creates ROC curves, confusion matrices, and feature importance plots.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from train_model import load_and_split_data, TRAIN_CUTOFF, DATA_PATH

# --- Definitions needed for unpickling ---
def add_engineered_features(df):
    """Add ratio and interaction features."""
    df = df.copy()
    if 'debt' in df.columns and 'income' in df.columns:
        df['debt_to_income'] = df['debt'] / (df['income'] + 1e-6)
    if 'credit_used' in df.columns and 'credit_limit' in df.columns:
        df['credit_utilization'] = df['credit_used'] / (df['credit_limit'] + 1e-6)
    if 'age' in df.columns and 'income' in df.columns:
        df['income_per_age'] = df['income'] / (df['age'] + 1e-6)
    if 'years_employed' in df.columns and 'age' in df.columns:
        df['employment_ratio'] = df['years_employed'] / (df['age'] + 1e-6)
    return df

class EngineerAndProcessNumeric(BaseEstimator, TransformerMixin):
    def __init__(self, engineer=None):
        self.engineer = engineer
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.numeric_cols_ = None
    
    def fit(self, X, y=None):
        if self.engineer:
            X_eng = self.engineer.transform(X)
        else:
            X_eng = X
        if isinstance(X_eng, pd.DataFrame):
            self.numeric_cols_ = X_eng.select_dtypes(include=[np.number]).columns.tolist()
            X_numeric = X_eng[self.numeric_cols_].values
        else:
            X_numeric = X_eng
        self.imputer.fit(X_numeric)
        X_imputed = self.imputer.transform(X_numeric)
        self.scaler.fit(X_imputed)
        return self
    
    def transform(self, X):
        if self.engineer:
            X_eng = self.engineer.transform(X)
        else:
            X_eng = X
        if isinstance(X_eng, pd.DataFrame):
            if self.numeric_cols_:
                try:
                    X_numeric = X_eng[self.numeric_cols_].values
                except KeyError:
                     X_numeric = X_eng.select_dtypes(include=[np.number]).values
            else:
                 X_numeric = X_eng.select_dtypes(include=[np.number]).values
        else:
            X_numeric = X_eng
        X_imputed = self.imputer.transform(X_numeric)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled
# ----------------------------------------

def generate_report():
    print("Generating Model Performance Report...")
    
    # Create reports directory
    if not os.path.exists('reports'):
        os.makedirs('reports')
        
    # Load model
    print("Loading model...")
    try:
        model_data = joblib.load('models/credit_card_approval_model.pkl')
        model = model_data['model']
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load test data
    print("Loading data...")
    _, X_test, _, y_test, _ = load_and_split_data(DATA_PATH, TRAIN_CUTOFF)
    
    # Predictions
    print("Making predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # 1. ROC Curve
    print("Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('reports/roc_curve.png')
    plt.close()
    
    # 2. Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('reports/confusion_matrix.png')
    plt.close()
    
    # 3. Distribution of Probabilities
    print("Generating Probability Distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred_proba[y_test==0], color='red', label='Denied (0)', kde=True, alpha=0.5)
    sns.histplot(y_pred_proba[y_test==1], color='green', label='Approved (1)', kde=True, alpha=0.5)
    plt.xlabel('Predicted Probability of Approval')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('reports/prob_distribution.png')
    plt.close()

    # Save metrics text
    print("Saving metrics report...")
    report = classification_report(y_test, y_pred)
    with open('reports/performance_metrics.txt', 'w') as f:
        f.write("Model Performance Metrics\n")
        f.write("=========================\n\n")
        f.write(f"Test AUC: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
        
    print("Report generation complete. Files saved in 'reports/' directory.")

if __name__ == "__main__":
    generate_report()
