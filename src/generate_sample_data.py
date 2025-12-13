"""
Generate sample credit card application data for testing.
Creates a realistic dataset with temporal structure and class imbalance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(n_samples=10000, output_path='data/credit_card_applications.csv'):
    """
    Generate synthetic credit card application data.
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save CSV file
    """
    np.random.seed(42)
    
    # Create date range (2 years of data)
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 730)) for _ in range(n_samples)]
    dates.sort()
    
    # Generate features
    data = {
        'application_date': dates,
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10.5, 0.5, n_samples).astype(int),
        'debt': np.random.lognormal(8, 0.8, n_samples).astype(int),
        'credit_score': np.random.normal(650, 100, n_samples).astype(int),
        'years_employed': np.random.exponential(5, n_samples).astype(int),
        'credit_used': np.random.lognormal(7, 0.7, n_samples).astype(int),
        'credit_limit': np.random.lognormal(8.5, 0.6, n_samples).astype(int),
        'num_credit_cards': np.random.poisson(2, n_samples),
        'num_loans': np.random.poisson(1, n_samples),
        'employment_status': np.random.choice(
            ['employed', 'self-employed', 'unemployed', 'retired'],
            n_samples,
            p=[0.6, 0.2, 0.15, 0.05]
        ),
        'education': np.random.choice(
            ['high_school', 'bachelor', 'master', 'phd'],
            n_samples,
            p=[0.3, 0.4, 0.25, 0.05]
        ),
        'marital_status': np.random.choice(
            ['single', 'married', 'divorced'],
            n_samples,
            p=[0.4, 0.5, 0.1]
        ),
        'residence_type': np.random.choice(
            ['own', 'rent', 'other'],
            n_samples,
            p=[0.5, 0.4, 0.1]
        )
    }
    
    df = pd.DataFrame(data)
    
    # Clip values to reasonable ranges
    df['age'] = df['age'].clip(18, 80)
    df['income'] = df['income'].clip(10000, 500000)
    df['debt'] = df['debt'].clip(0, 200000)
    df['credit_score'] = df['credit_score'].clip(300, 850)
    df['years_employed'] = df['years_employed'].clip(0, 50)
    df['credit_used'] = df['credit_used'].clip(0, df['credit_limit'])
    
    # Generate target variable with realistic logic (creating class imbalance)
    # Higher income, lower debt, higher credit score = more likely approved
    debt_to_income = df['debt'] / (df['income'] + 1)
    credit_utilization = df['credit_used'] / (df['credit_limit'] + 1)
    
    # Approval probability based on features
    approval_prob = (
        0.3 +  # Base probability
        0.3 * (df['credit_score'] - 300) / 550 +  # Credit score effect
        0.2 * (1 - debt_to_income.clip(0, 1)) +  # Debt-to-income effect
        0.1 * (1 - credit_utilization.clip(0, 1)) +  # Credit utilization effect
        0.05 * (df['years_employed'] / 20).clip(0, 1) +  # Employment stability
        -0.1 * (df['num_loans'] > 3).astype(int) +  # Too many loans
        -0.1 * (df['employment_status'] == 'unemployed').astype(int)  # Unemployment
    )
    
    approval_prob = approval_prob.clip(0, 1)
    
    # Add some noise
    approval_prob += np.random.normal(0, 0.1, n_samples)
    approval_prob = approval_prob.clip(0, 1)
    
    # Create binary target (this creates imbalance - more approvals)
    df['approved'] = (approval_prob > 0.5).astype(int)
    
    # Ensure we have some class imbalance (70% approved, 30% denied)
    # Adjust to make it more realistic
    approved_count = df['approved'].sum()
    if approved_count / len(df) > 0.75:
        # Too many approvals, adjust threshold
        threshold = np.percentile(approval_prob, 30)
        df['approved'] = (approval_prob > threshold).astype(int)
    elif approved_count / len(df) < 0.6:
        # Too many denials, adjust threshold
        threshold = np.percentile(approval_prob, 40)
        df['approved'] = (approval_prob > threshold).astype(int)
    
    # Add some missing values (5% missing in some columns)
    missing_cols = ['debt', 'credit_used', 'years_employed']
    for col in missing_cols:
        missing_idx = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated {n_samples} samples")
    print(f"Saved to {output_path}")
    print(f"\nClass distribution:")
    print(df['approved'].value_counts())
    print(f"\nApproval rate: {df['approved'].mean():.2%}")
    print(f"\nData shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    
    return df

if __name__ == '__main__':
    generate_sample_data(n_samples=10000)

