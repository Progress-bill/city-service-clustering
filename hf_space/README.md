---
title: City Service Clustering ML Engine
emoji: 🏙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# City-Service Review Clustering ML Engine (Hugging Face Space)

A 24/7 containerized Docker microservice powered by **FastAPI**, **Scikit-Learn**, **Pandas**, and **NumPy** for unsupervised review pattern segmentation.

## 🚀 Specifications
- **Runtime:** Hugging Face Spaces Docker (`CPU basic: 2 vCPU, 16GB RAM`)
- **Port:** `7860`
- **Endpoints:**
  - `GET /health` — Health check endpoint for external dashboard pings
  - `POST /train-clustering` — Review aggregation, standardisation, K-Means ($K=2$ to $10$), PCA 2D reduction, and metrics calculation

## 🧪 Testing Locally
```bash
docker build -t city-service-ml-worker .
docker run -p 7860:7860 city-service-ml-worker
```
Then visit:
`http://localhost:7860/docs`
