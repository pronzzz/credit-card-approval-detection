# Quick Start Guide

Get up and running with the Credit Card Approval Detection system in 5 minutes.

## Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt
```

## Step 2: Generate Sample Data

```bash
python src/generate_sample_data.py
```

This creates `data/credit_card_applications.csv` with 10,000 synthetic credit applications.

## Step 3: Train the Model

```bash
python src/train_model.py
```

**What happens:**
- Data is loaded and split temporally (train before 2023, test after)
- Features are engineered (ratios, interactions)
- Hyperparameters are optimized using Optuna (50 trials, ~10-30 min)
- Best model is trained and saved
- SHAP explanations are generated
- Model saved to `models/credit_card_approval_model.pkl`

**Expected output:**
```
Credit Card Approval Detection - Model Training
============================================================
Loading data from data/credit_card_applications.csv...
Temporal split: 7000 train, 3000 test
Features: 13
Train class distribution: {1: 4900, 0: 2100}
...

Best hyperparameters:
  C: 0.123
  penalty: l2
  sampler: smote_enn
  ...

Test ROC AUC: 0.8523
```

## Step 4: Start the API

In a new terminal:

```bash
python src/api.py
```

You should see:
```
INFO - Model loaded successfully from models/credit_card_approval_model.pkl
INFO - Starting Flask API on 0.0.0.0:5000
```

## Step 5: Test the API

In another terminal (or use the test script):

```bash
# Option 1: Use the test script
python src/test_api.py

# Option 2: Manual curl commands
curl http://localhost:5000/health

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 75000,
    "debt": 15000,
    "credit_score": 720,
    "years_employed": 8,
    "credit_used": 5000,
    "credit_limit": 20000,
    "num_credit_cards": 2,
    "num_loans": 1,
    "employment_status": "employed",
    "education": "bachelor",
    "marital_status": "married",
    "residence_type": "own"
  }'
```

## Expected API Response

```json
{
  "approval_probability": 0.8234,
  "approved": 1,
  "response_time_ms": 12.45
}
```

## Troubleshooting

### "Model not loaded" error
- Make sure you ran `train_model.py` successfully
- Check that `models/credit_card_approval_model.pkl` exists

### "Data file not found" error
- Run `generate_sample_data.py` first
- Or provide your own CSV at `data/credit_card_applications.csv`

### Port already in use
- Change the port: `set PORT=5001` (Windows) or `export PORT=5001` (Linux/Mac)
- Or kill the process using port 5000

### Import errors
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Customize features in `src/train_model.py`
- Add your own data format
- Deploy to production (see README for recommendations)

## Using Your Own Data

Your CSV should have:
- An `approved` column (0 or 1)
- An `application_date` column (for temporal splitting)
- All the feature columns (see README for full list)

Update `DATA_PATH` in `src/train_model.py` if your file is in a different location.

