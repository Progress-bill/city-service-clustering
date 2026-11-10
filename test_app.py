"""
Unit and Integration Tests for app.py (Flask Web Application)
"""
from app import app
import json

def test_flask_routes():
    client = app.test_client()

    # 1. Test GET /
    res = client.get("/")
    assert res.status_code == 200
    assert b"City-Service Review Profiles" in res.data
    assert b"2D PCA Latent Space Projection" in res.data
    print("Test 1 Passed: GET / served index.html successfully.")

    # 2. Test GET /sample-data/reviews
    res_sample = client.get("/sample-data/reviews")
    assert res_sample.status_code == 200
    assert b"gmap_id" in res_sample.data
    print("Test 2 Passed: GET /sample-data/reviews served sample reviews CSV.")

    # 3. Test GET /sample-data/aggregated
    res_agg = client.get("/sample-data/aggregated")
    assert res_agg.status_code == 200
    assert b"business_id" in res_agg.data
    print("Test 3 Passed: GET /sample-data/aggregated served aggregated CSV.")

    # 4. Test GET /settings and /engine-settings
    res_set = client.get("/settings")
    assert res_set.status_code == 200
    assert b"Google Colab Engine Settings" in res_set.data
    res_eng = client.get("/engine-settings")
    assert res_eng.status_code == 200
    print("Test 4 Passed: GET /settings and /engine-settings served settings page.")

    # 5. Test POST /api/train with sample data (Zero-downtime clustering test)
    res_train = client.post("/api/train", data={"k": "4", "use_sample": "true", "sample_type": "reviews"})
    assert res_train.status_code == 200
    train_data = res_train.get_json()
    assert train_data["status"] == "success"
    assert train_data["k_clusters"] == 4
    assert "metrics" in train_data
    assert "silhouette_score" in train_data["metrics"]
    assert len(train_data["scatter_points"]) == 100
    print("Test 5 Passed: POST /api/train executed clustering pipeline successfully!")

    print("\nAll Flask Web App route and clustering tests passed successfully!")

if __name__ == "__main__":
    test_flask_routes()
