"""
Flask API for Credit Card Approval Detection
Provides REST endpoint for real-time predictions with monitoring hooks.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model
MODEL_PATH = 'models/credit_card_approval_model.pkl'

def add_engineered_features(df):
    """Add ratio and interaction features."""
    df = df.copy()
    
    # Debt-to-income ratio
    if 'debt' in df.columns and 'income' in df.columns:
        df['debt_to_income'] = df['debt'] / (df['income'] + 1e-6)
    
    # Credit utilization (if available)
    if 'credit_used' in df.columns and 'credit_limit' in df.columns:
        df['credit_utilization'] = df['credit_used'] / (df['credit_limit'] + 1e-6)
    
    # Age-to-income ratio (if available)
    if 'age' in df.columns and 'income' in df.columns:
        df['income_per_age'] = df['income'] / (df['age'] + 1e-6)
    
    # Years employed ratio
    if 'years_employed' in df.columns and 'age' in df.columns:
        df['employment_ratio'] = df['years_employed'] / (df['age'] + 1e-6)
    
    return df

class EngineerAndProcessNumeric(BaseEstimator, TransformerMixin):
    """Apply feature engineering then process all numeric columns."""
    def __init__(self, engineer=None):
        self.engineer = engineer
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.numeric_cols_ = None
    
    def fit(self, X, y=None):
        # X is DataFrame with original numeric features
        if self.engineer:
            X_eng = self.engineer.transform(X)
        else:
            X_eng = X
            
        # After engineering, identify all numeric columns
        if isinstance(X_eng, pd.DataFrame):
            self.numeric_cols_ = X_eng.select_dtypes(include=[np.number]).columns.tolist()
            X_numeric = X_eng[self.numeric_cols_].values
        else:
            # If array, assume all are numeric after engineering
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
            # Ensure we use the same columns as found during fit
            if self.numeric_cols_:
                # Check if columns exist
                missing = set(self.numeric_cols_) - set(X_eng.columns)
                if missing:
                    pass
                try:
                    X_numeric = X_eng[self.numeric_cols_].values
                except KeyError:
                     # Fallback if columns missing
                     X_numeric = X_eng.select_dtypes(include=[np.number]).values
            else:
                 X_numeric = X_eng.select_dtypes(include=[np.number]).values
        else:
            X_numeric = X_eng
        
        X_imputed = self.imputer.transform(X_numeric)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    features = model_data.get('features', [])
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    logger.info(f"Model features: {len(features)}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    features = []

# Monitoring data storage (in production, use a database)
request_log = []

def record_request_stats(payload, probability, decision, response_time_ms):
    """
    Record request statistics for monitoring and drift detection.
    
    In production, this should write to a database or monitoring service.
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'payload': payload,
        'probability': float(probability),
        'decision': int(decision),
        'response_time_ms': response_time_ms
    }
    request_log.append(log_entry)
    
    # Keep only last 1000 entries in memory
    if len(request_log) > 1000:
        request_log.pop(0)
    
    logger.info(f"Request logged: decision={decision}, prob={probability:.4f}")

def detect_data_drift(new_data, reference_data=None):
    """
    Simple drift detection using statistical tests.
    
    In production, implement more sophisticated drift detection
    (e.g., Kolmogorov-Smirnov test, PSI, etc.)
    """
    # Placeholder for drift detection
    # This would compare new_data distribution to reference_data
    return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 503
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'features_count': len(features)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict credit card approval probability.
    
    Expected JSON payload:
    {
        "age": 25,
        "income": 50000,
        "debt": 10000,
        ...
    }
    
    Returns:
    {
        "approval_probability": 0.75,
        "approved": 1,
        "explanation": "..." (optional)
    }
    """
    start_time = datetime.now()
    
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 503
    
    try:
        # Get JSON payload
        payload = request.get_json()
        
        if not payload:
            return jsonify({'error': 'No JSON payload provided'}), 400
        
        # Convert to DataFrame
        df_in = pd.DataFrame([payload])
        
        # Check for missing required features
        missing_features = set(features) - set(df_in.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Fill missing features with defaults
            for feat in missing_features:
                df_in[feat] = 0  # Default value
        
        # Ensure correct column order
        df_in = df_in.reindex(columns=features, fill_value=0)
        
        # Make prediction
        try:
            proba = model.predict_proba(df_in)[:, 1][0]
            decision = int(proba >= 0.5)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({
                'error': f'Prediction failed: {str(e)}'
            }), 500
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Record request
        record_request_stats(payload, proba, decision, response_time)
        
        # Detect drift (placeholder)
        drift_detected = detect_data_drift(df_in)
        
        response = {
            'approval_probability': float(proba),
            'approved': decision,
            'response_time_ms': round(response_time, 2)
        }
        
        if drift_detected:
            response['warning'] = 'Data drift detected'
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """
    Get request statistics (for monitoring).
    
    In production, this should query a proper database.
    """
    if not request_log:
        return jsonify({
            'total_requests': 0,
            'message': 'No requests logged yet'
        })
    
    total = len(request_log)
    approved = sum(1 for entry in request_log if entry['decision'] == 1)
    denied = total - approved
    
    avg_prob = np.mean([entry['probability'] for entry in request_log])
    avg_response_time = np.mean([entry['response_time_ms'] for entry in request_log])
    
    return jsonify({
        'total_requests': total,
        'approved': approved,
        'denied': denied,
        'approval_rate': round(approved / total, 4) if total > 0 else 0,
        'average_probability': round(avg_prob, 4),
        'average_response_time_ms': round(avg_response_time, 2)
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        model_data = joblib.load(MODEL_PATH)
        return jsonify({
            'features': model_data.get('features', []),
            'test_auc': model_data.get('test_auc'),
            'best_params': model_data.get('best_params', {}),
            'trained_date': model_data.get('trained_date')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting Flask API on {host}:{port}")
    app.run(host=host, port=port, debug=False)

