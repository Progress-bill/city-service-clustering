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

    print("\nAll Flask Web App route tests passed successfully!")

if __name__ == "__main__":
    test_flask_routes()
