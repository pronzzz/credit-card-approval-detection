"""
Credit Card Approval Detection - Training Script
Implements robust pipeline with temporal splitting, advanced resampling,
feature engineering, Bayesian optimization, and explainability.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

import optuna
from optuna.samplers import TPESampler

import shap

# Configuration
DATA_PATH = 'data/credit_card_applications.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'credit_card_approval_model.pkl')
TRAIN_CUTOFF = datetime(2023, 1, 1)

def ensure_directories():
    """Create necessary directories if they don't exist."""
    Path(MODEL_DIR).mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)

def load_and_split_data(data_path, train_cutoff):
    """
    Load data and perform temporal split to avoid data leakage.
    
    Args:
        data_path: Path to CSV file
        train_cutoff: Datetime cutoff for train/test split
        
    Returns:
        X_train, X_test, y_train, y_test, features
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Parse dates if application_date column exists
    if 'application_date' in df.columns:
        df['application_date'] = pd.to_datetime(df['application_date'])
        df.sort_values('application_date', inplace=True)
        train_df = df[df['application_date'] < train_cutoff].copy()
        test_df = df[df['application_date'] >= train_cutoff].copy()
        print(f"Temporal split: {len(train_df)} train, {len(test_df)} test")
    else:
        # Fallback to random split if no date column
        print("No application_date column found, using random split...")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['approved'])
    
    target = 'approved'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")
    
    # Identify features (exclude target and date columns)
    exclude_cols = [target, 'application_date']
    features = [c for c in df.columns if c not in exclude_cols]
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    print(f"Features: {len(features)}")
    print(f"Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test, features

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

def create_feature_engineering_pipeline(features):
    """
    Create feature engineering pipeline with ratio features and interactions.
    
    Args:
        features: List of feature column names
        
    Returns:
        Feature engineering transformer
    """
    return FunctionTransformer(add_engineered_features, validate=False)

def build_preprocessing_pipeline(X_train, features):
    """
    Build preprocessing pipeline with separate numeric and categorical handling.
    
    Args:
        X_train: Training features DataFrame
        features: List of all feature names
        
    Returns:
        Preprocessing pipeline
    """
    # Identify numeric and categorical features
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(cat_features)}")
    
    # Feature engineering step - will add new numeric columns
    feature_engineering = create_feature_engineering_pipeline(features)
    
    # Numeric pipeline (for all numeric features including engineered ones)
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Build preprocessing pipeline:
    # Step 1: Apply feature engineering to create new columns
    # Step 2: After engineering, identify all numeric columns (original + engineered)
    # Step 3: Process all numeric together, and categorical separately
    
    # We'll use a custom approach: feature engineering first, then process all numeric
    # Since we need to know which columns are numeric after engineering, we'll
    # create a transformer that does this dynamically
    
    class DynamicNumericSelector:
        """Select all numeric columns after feature engineering."""
        def fit(self, X, y=None):
            # X is already a DataFrame after feature engineering
            self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
            return self
        
        def transform(self, X):
            return X[self.numeric_cols_]
    
    
    # Build transformers - after feature engineering, all numeric (including engineered)
    # will be processed together
    transformers = []
    
    if cat_features:
        transformers.append(('cat', categorical_pipeline, cat_features))
    
    # For numeric: we'll process them after feature engineering
    # Create a pipeline that: engineers -> selects numeric -> processes numeric
    # Note: explicit numeric_with_engineering pipeline removed in favor of EngineerAndProcessNumeric approach below
    
    # For categorical: process original categorical features
    if cat_features:
        cat_pipeline = categorical_pipeline
    else:
        cat_pipeline = 'drop'
    
    # Main preprocessor: numeric (with engineering) and categorical
    if numeric_features:
        # Use numeric features as input, but pipeline will engineer first
        full_preprocessor = ColumnTransformer([
            ('numeric', 'passthrough', numeric_features + [f for f in features if f not in numeric_features and f not in cat_features]), # Placeholder, actual logic handled below
            ('categorical', cat_pipeline, cat_features) if cat_features else ('drop', 'drop', [])
        ], remainder='drop')
    else:
        full_preprocessor = ColumnTransformer([
            ('categorical', cat_pipeline, cat_features) if cat_features else ('drop', 'drop', [])
        ], remainder='drop')
    
    # Build transformers
    transformers = []
    if numeric_features:
        transformers.append(('num', EngineerAndProcessNumeric(engineer=feature_engineering), numeric_features))
    if cat_features:
        transformers.append(('cat', categorical_pipeline, cat_features))
    
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    
    return preprocessor, numeric_features, cat_features

# Moved to global scope for pickling
from sklearn.base import BaseEstimator, TransformerMixin

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

def create_objective_function(X_train, y_train, preprocessor):
    """
    Create Optuna objective function for hyperparameter optimization.
    
    Args:
        X_train: Training features
        y_train: Training target
        preprocessor: Preprocessing pipeline
        
    Returns:
        Objective function for Optuna
    """
    def objective(trial):
        # Hyperparameters to optimize
        C = trial.suggest_float('C', 1e-4, 1e2, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9) if penalty == 'elasticnet' else None
        sampler_type = trial.suggest_categorical('sampler', ['smote_enn', 'smote', 'adasyn', 'none'])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
        
        # Select solver based on penalty
        if penalty == 'l1':
            solver = 'liblinear'
        elif penalty == 'elasticnet':
            solver = 'saga'
        else:
            solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])
        
        # Create resampling strategy
        if sampler_type == 'smote_enn':
            sampler = SMOTEENN(random_state=42)
        elif sampler_type == 'smote':
            sampler = SMOTE(random_state=42)
        elif sampler_type == 'adasyn':
            sampler = ADASYN(random_state=42)
        else:
            sampler = None
        
        # Build model
        lr_params = {
            'C': C,
            'penalty': penalty,
            'solver': solver,
            'class_weight': class_weight,
            'random_state': 42,
            'max_iter': 1000
        }
        
        if penalty == 'elasticnet':
            lr_params['l1_ratio'] = l1_ratio
        
        base_clf = LogisticRegression(**lr_params)
        
        # Build pipeline
        if sampler:
            pipe = ImbPipeline([
                ('pre', preprocessor),
                ('sampler', sampler),
                ('clf', base_clf)
            ])
        else:
            pipe = Pipeline([
                ('pre', preprocessor),
                ('clf', base_clf)
            ])
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            try:
                pipe.fit(X_t, y_t)
                preds = pipe.predict_proba(X_v)[:, 1]
                auc = roc_auc_score(y_v, preds)
                aucs.append(auc)
            except Exception as e:
                print(f"Trial failed: {e}")
                return -1.0  # Return bad score if trial fails
        
        return np.mean(aucs)
    
    return objective

def train_best_model(X_train, y_train, X_test, y_test, preprocessor, study):
    """
    Train final model with best hyperparameters and evaluate.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        preprocessor: Preprocessing pipeline
        study: Optuna study with best parameters
        
    Returns:
        Trained pipeline and evaluation metrics
    """
    best_params = study.best_params
    print("\n" + "="*50)
    print("Best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("="*50 + "\n")
    
    # Rebuild pipeline with best parameters
    sampler_type = best_params['sampler']
    
    if sampler_type == 'smote_enn':
        sampler = SMOTEENN(random_state=42)
    elif sampler_type == 'smote':
        sampler = SMOTE(random_state=42)
    elif sampler_type == 'adasyn':
        sampler = ADASYN(random_state=42)
    else:
        sampler = None
    
    # Build logistic regression with best params
    lr_params = {
        'C': best_params['C'],
        'penalty': best_params['penalty'],
        'class_weight': best_params['class_weight'],
        'random_state': 42,
        'max_iter': 1000
    }
    
    if best_params['penalty'] == 'l1':
        lr_params['solver'] = 'liblinear'
    elif best_params['penalty'] == 'elasticnet':
        lr_params['solver'] = 'saga'
        lr_params['l1_ratio'] = best_params.get('l1_ratio', 0.5)
    else:
        lr_params['solver'] = best_params.get('solver', 'lbfgs')
    
    best_clf = LogisticRegression(**lr_params)
    
    # Build final pipeline
    if sampler:
        final_pipe = ImbPipeline([
            ('pre', preprocessor),
            ('sampler', sampler),
            ('clf', best_clf)
        ])
    else:
        final_pipe = Pipeline([
            ('pre', preprocessor),
            ('clf', best_clf)
        ])
    
    # Train
    print("Training final model...")
    final_pipe.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = final_pipe.predict(X_test)
    y_proba = final_pipe.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    test_auc = roc_auc_score(y_test, y_proba)
    print(f"\nTest ROC AUC: {test_auc:.4f}")
    
    return final_pipe, test_auc

def create_shap_explainer(pipeline, X_train_sample, X_test_sample):
    """
    Create SHAP explainer for model interpretability.
    
    Args:
        pipeline: Trained pipeline
        X_train_sample: Sample of training data for background
        X_test_sample: Sample of test data to explain
        
    Returns:
        SHAP values
    """
    print("\nGenerating SHAP explanations...")
    
    try:
        # Transform data through preprocessing (before any resampling)
        X_train_transformed = pipeline.named_steps['pre'].transform(X_train_sample)
        X_test_transformed = pipeline.named_steps['pre'].transform(X_test_sample)
        
        # Get the classifier (it's always the last step)
        clf = pipeline.named_steps['clf']
        
        # For logistic regression, use LinearExplainer
        # Use a smaller background sample for performance
        background_size = min(100, len(X_train_transformed))
        background_sample = X_train_transformed[:background_size] if hasattr(X_train_transformed, '__getitem__') else X_train_transformed
        
        explainer = shap.LinearExplainer(clf, background_sample)
        shap_values = explainer.shap_values(X_test_transformed)
        
        print("SHAP values computed successfully")
        print(f"  Background samples: {background_size}")
        print(f"  Explained samples: {len(X_test_transformed) if hasattr(X_test_transformed, '__len__') else 'N/A'}")
        
        return shap_values, explainer
        
    except Exception as e:
        print(f"Warning: Could not generate SHAP explanations: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main training function."""
    print("="*60)
    print("Credit Card Approval Detection - Model Training")
    print("="*60)
    
    # Ensure directories exist
    ensure_directories()
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"\nError: Data file not found at {DATA_PATH}")
        print("Please run generate_sample_data.py first to create sample data.")
        return
    
    # Load and split data
    X_train, X_test, y_train, y_test, features = load_and_split_data(DATA_PATH, TRAIN_CUTOFF)
    
    # Build preprocessing pipeline
    preprocessor, numeric_features, cat_features = build_preprocessing_pipeline(X_train, features)
    
    # Hyperparameter optimization
    print("\nStarting hyperparameter optimization with Optuna...")
    objective = create_objective_function(X_train, y_train, preprocessor)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='credit_approval_optimization'
    )
    
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print(f"\nBest CV AUC: {study.best_value:.4f}")
    
    # Train final model
    final_pipe, test_auc = train_best_model(X_train, y_train, X_test, y_test, preprocessor, study)
    
    # Generate SHAP explanations (on sample for performance)
    sample_size = min(100, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=42)
    X_train_sample = X_train.sample(n=min(500, len(X_train)), random_state=42)
    
    shap_values, explainer = create_shap_explainer(final_pipe, X_train_sample, X_test_sample)
    
    # Save model
    print(f"\nSaving model to {MODEL_PATH}...")
    model_metadata = {
        'model': final_pipe,
        'features': features,
        'numeric_features': numeric_features,
        'cat_features': cat_features,
        'test_auc': test_auc,
        'best_params': study.best_params,
        'trained_date': datetime.now().isoformat()
    }
    
    joblib.dump(model_metadata, MODEL_PATH)
    print("Model saved successfully!")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

if __name__ == '__main__':
    main()

