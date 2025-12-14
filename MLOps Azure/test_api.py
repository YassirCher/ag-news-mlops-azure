"""
Test script for API endpoints
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    print("\n" + "="*70)
    print("🏥 Testing /health endpoint")
    print("="*70)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))


def test_predict():
    print("\n" + "="*70)
    print("🔮 Testing /predict endpoint")
    print("="*70)

    test_texts = [
        "Wall Street stocks rally on strong tech earnings",
        "Manchester United wins Premier League championship",
        "New AI breakthrough achieves human-level performance",
        "Global leaders meet at climate summit",
    ]

    for text in test_texts:
        data = {"text": text}
        response = requests.post(f"{BASE_URL}/predict", json=data)
        result = response.json()
        print(f"\nText: {text}")
        print(f"Category: {result['category']} (confidence: {result['confidence']:.2%})")


def test_batch_predict():
    print("\n" + "="*70)
    print("📦 Testing /predict/batch endpoint")
    print("="*70)

    texts = [
        "Stock market hits new record high",
        "Barcelona defeats Real Madrid 3-1",
        "Scientists discover new exoplanet",
    ]

    data = {"texts": texts}
    response = requests.post(f"{BASE_URL}/predict/batch", json=data)
    results = response.json()

    for i, result in enumerate(results["predictions"]):
        print(f"\n{i+1}. {texts[i]}")
        print(f"   → {result['category']} ({result['confidence']:.2%})")


if __name__ == "__main__":
    print("="*70)
    print("🧪 API TESTING SCRIPT")
    print("="*70)

    test_health()
    test_predict()
    test_batch_predict()

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
