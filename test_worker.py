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
    assert data["sample_size"] == 100
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
    assert "strategic_decisions" in data
    assert len(data["strategic_decisions"]) == 4
    assert "market_trends" in data
    assert "market_health_score" in data["market_trends"]
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

def test_train_clustering_json_lines():
    import io
    sample_jsonl = (
        b'{"name": "Biz 1", "gmap_id": "0x1", "avg_rating": 4.5, "num_of_reviews": 20, "latitude": 43.5, "longitude": -96.7}\n'
        b'{"name": "Biz 2", "gmap_id": "0x2", "avg_rating": 3.8, "num_of_reviews": 5, "latitude": 44.1, "longitude": -97.2}\n'
        b'{"name": "Biz 3", "gmap_id": "0x3", "avg_rating": 4.9, "num_of_reviews": 150, "latitude": 43.8, "longitude": -96.9}\n'
    )
    response = client.post(
        "/train-clustering",
        files={"file": ("meta-South_Dakota.json", io.BytesIO(sample_jsonl), "application/json")},
        data={"k": 2}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "success"
    assert data["k_clusters"] == 2
    assert data["sample_size"] == 3
    assert "metrics" in data
    print("\nJSON Lines (meta-South_Dakota.json) Test Passed in colab_worker! Schema:", data["schema_detected"])

if __name__ == "__main__":
    test_health_check()
    test_train_clustering_case_a()
    test_train_clustering_case_b()
    test_train_clustering_json_lines()
    print("\nAll Colab Worker ML endpoints verified successfully!")
