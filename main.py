"""
Hugging Face Spaces ML Worker: Containerized FastAPI Clustering Engine
======================================================================
Production microservice for City-Service Review Profiles & Pattern Segmentation.
Listens on port 7860 (Hugging Face Spaces Docker runtime default).
"""
from hf_space.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
