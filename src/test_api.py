"""
Simple test script for the API.
Run this after starting the API server to test endpoints.
"""

import requests
import json

BASE_URL = "http://localhost:5001"

def test_health():
    """Test health check endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_model_info():
    """Test model info endpoint."""
    print("Testing /model_info endpoint...")
    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_predict():
    """Test prediction endpoint."""
    print("Testing /predict endpoint...")
    
    sample_request = {
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
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=sample_request,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Request: {json.dumps(sample_request, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_stats():
    """Test statistics endpoint."""
    print("Testing /stats endpoint...")
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("="*60)
    print("API Test Suite")
    print("="*60)
    print()
    
    try:
        test_health()
        test_model_info()
        test_predict()
        test_stats()
        
        print("="*60)
        print("All tests completed!")
        print("="*60)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API.")
        print("Make sure the API server is running:")
        print("  python src/api.py")
    except Exception as e:
        print(f"Error: {e}")

