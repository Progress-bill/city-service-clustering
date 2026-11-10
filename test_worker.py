"""
Unit and Integration Tests for colab_worker.py
"""
from fastapi.testclient import TestClient
from colab_worker import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["worker"] == "colab-ml-engine"

def test_train_clustering_case_a():
    with open("sample_data/sample_reviews.csv", "rb") as f:
        response = client.post(
            "/train-clustering",
            files={"file": ("sample_reviews.csv", f, "text/csv")},
            data={"k": 4}
        )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "success"
    assert data["k_clusters"] == 4
    assert data["sample_size"] == 100  # 100 businesses
    assert "metrics" in data
    assert "silhouette_score" in data["metrics"]
    assert "davies_bouldin_index" in data["metrics"]
    assert "inertia" in data["metrics"]
    assert "pca_variance_ratio" in data["metrics"]
    assert len(data["metrics"]["pca_variance_ratio"]) == 2
    assert len(data["profiles"]) == 4
    assert len(data["scatter_points"]) == 100
    assert "id" in data["scatter_points"][0]
    assert "pca_x" in data["scatter_points"][0]
    assert "pca_y" in data["scatter_points"][0]
    assert "cluster" in data["scatter_points"][0]
    print("\nCase A Test Passed successfully! Metrics:", data["metrics"])

def test_train_clustering_case_b():
    with open("sample_data/sample_aggregated.csv", "rb") as f:
        response = client.post(
            "/train-clustering",
            files={"file": ("sample_aggregated.csv", f, "text/csv")},
            data={"k": 3}
        )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "success"
    assert data["k_clusters"] == 3
    assert data["sample_size"] == 100
    assert len(data["profiles"]) == 3
    print("\nCase B Test Passed successfully! Schema:", data["schema_detected"])

if __name__ == "__main__":
    test_health_check()
    test_train_clustering_case_a()
    test_train_clustering_case_b()
    print("\nAll Colab Worker ML endpoints verified successfully!")
