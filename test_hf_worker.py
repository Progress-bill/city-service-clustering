"""
Unit and Integration Tests for Hugging Face Spaces ML Engine (main.py)
"""
from fastapi.testclient import TestClient
from hf_space.main import app

client = TestClient(app)

def test_hf_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["worker"] == "hf-spaces-ml-engine"
    print("HF Spaces Health Check: OK")

def test_hf_train_clustering_case_a():
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
    print("HF Spaces Case A Raw Reviews Training: OK -> Silhouette:", data["metrics"]["silhouette_score"])

def test_hf_train_clustering_case_b():
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
    print("HF Spaces Case B Tabular Training: OK")

if __name__ == "__main__":
    test_hf_health_check()
    test_hf_train_clustering_case_a()
    test_hf_train_clustering_case_b()
    print("\nAll Hugging Face Spaces ML Worker tests passed successfully!")
