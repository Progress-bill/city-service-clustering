"""
Hugging Face Spaces ML Worker: Containerized FastAPI Clustering Engine
======================================================================
Production microservice for City-Service Review Profiles & Pattern Segmentation.
Listens on port 7860 (Hugging Face Spaces Docker runtime default).

Endpoints:
- GET  /         : Service status & metadata
- GET  /health   : Health probe for dashboard connectivity
- POST /train-clustering : Feature aggregation, standardisation, KMeans & PCA
"""

import io
import sys
import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("hf-spaces-ml-engine")

app = FastAPI(
    title="City-Service Clustering ML Engine",
    description="24/7 Containerized FastAPI Microservice on Hugging Face Spaces for Unsupervised Clustering and PCA Projection",
    version="2.0.0"
)

# Enable CORS for Render dashboard and direct browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "City-Service Clustering ML Engine",
        "runtime": "Hugging Face Spaces Docker (2 vCPU, 16GB RAM)",
        "status": "online",
        "worker": "hf-spaces-ml-engine",
        "docs_url": "/docs",
        "endpoints": ["/health", "/train-clustering"]
    }


@app.get("/health")
def health():
    """Health probe endpoint used by the Render Flask dashboard."""
    return {
        "status": "online",
        "worker": "hf-spaces-ml-engine"
    }


def _detect_and_process_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Defensively detect schema:
    - Case A: Raw Reviews (contains gmap_id and rating) -> aggregates to business level
    - Case B: Aggregated/Tabular Data -> extracts numeric features
    Returns: (features_df, id_series, schema_case)
    """
    # Normalize column names for case-insensitive matching
    col_map = {str(c).strip().lower(): c for c in df.columns}

    # Check Case A: Raw Reviews
    is_case_a = ("gmap_id" in col_map) and ("rating" in col_map)

    if is_case_a:
        logger.info("Detected Case A: Raw review dataset with 'gmap_id' and 'rating'.")
        gmap_col = col_map["gmap_id"]
        rating_col = col_map["rating"]

        # Ensure ratings are numeric
        df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
        df_valid = df.dropna(subset=[gmap_col, rating_col]).copy()

        if df_valid.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Raw review dataset contains no valid rows with non-null gmap_id and numeric rating."
            )

        # Detect text column for review length calculation
        text_candidates = ["text", "review_text", "review", "comment", "caption", "body", "content"]
        text_col = None
        for cand in text_candidates:
            if cand in col_map:
                text_col = col_map[cand]
                break

        # Compute review character length
        if text_col is not None:
            df_valid["_review_length"] = df_valid[text_col].fillna("").astype(str).str.len()
        else:
            df_valid["_review_length"] = 0.0

        # Perform groupby aggregation per business
        grouped = df_valid.groupby(gmap_col)

        # 1. mean_rating
        mean_rating = grouped[rating_col].mean().rename("mean_rating")

        # 2. std_rating (impute 0.0 for 1-review businesses where std is NaN)
        std_rating = grouped[rating_col].std(ddof=0).fillna(0.0).rename("std_rating")

        # 3. log_review_count: log(1 + count)
        counts = grouped.size()
        log_review_count = np.log1p(counts).rename("log_review_count")

        # 4. avg_review_length
        avg_review_length = grouped["_review_length"].mean().rename("avg_review_length")

        agg_df = pd.concat([mean_rating, std_rating, log_review_count, avg_review_length], axis=1).reset_index()
        biz_ids = agg_df[gmap_col].astype(str)
        features = agg_df[["mean_rating", "std_rating", "log_review_count", "avg_review_length"]].copy()

        return features, biz_ids, "Case A: Raw Reviews Aggregation"

    else:
        logger.info("Detected Case B: Pre-aggregated / generic tabular dataset.")
        # Find business ID candidate column if present
        id_candidates = ["gmap_id", "business_id", "biz_id", "id", "name", "business_name", "store_id"]
        id_col = None
        for cand in id_candidates:
            if cand in col_map:
                id_col = col_map[cand]
                break

        if id_col is not None:
            biz_ids = df[id_col].astype(str)
            numeric_df = df.drop(columns=[id_col]).select_dtypes(include=[np.number])
        else:
            biz_ids = pd.Series([f"item_{i+1}" for i in range(len(df))], index=df.index)
            numeric_df = df.select_dtypes(include=[np.number])

        # Require at least 2 numeric dimensions
        if numeric_df.shape[1] < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset must have at least 2 numeric feature columns for clustering. Found only {numeric_df.shape[1]} numeric column(s)."
            )

        # Impute missing values with column median
        features = numeric_df.copy()
        features = features.fillna(features.median(numeric_only=True))

        features = features.dropna(axis=1)
        if features.shape[1] < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Fewer than 2 valid numeric features remain after cleaning missing values."
            )

        return features, biz_ids.loc[features.index], "Case B: Tabular Numeric Features"


@app.post("/train-clustering")
async def train_clustering(
    file: UploadFile = File(...),
    k: Optional[int] = Form(None),
    k_query: Optional[int] = Query(None, alias="k")
):
    """
    Main clustering training endpoint:
    - Ingests CSV/JSON dataset
    - Runs defensive schema detection (Case A vs Case B)
    - Normalizes features via StandardScaler
    - Executes KMeans(n_clusters=k)
    - Computes Silhouette Score, Davies-Bouldin Index, Inertia, PCA 2D coordinates & explained variance
    """
    # Resolve k from Form or Query, default to 4
    resolved_k = 4
    if k is not None:
        resolved_k = k
    elif k_query is not None:
        resolved_k = k_query

    # Validate k bounds (2 to 10)
    if resolved_k < 2:
        resolved_k = 2
    elif resolved_k > 10:
        resolved_k = 10

    # Ingest uploaded file
    try:
        contents = await file.read()
        filename = (file.filename or "").lower()

        if filename.endswith(".json"):
            df = pd.read_json(io.BytesIO(contents))
        elif filename.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            try:
                df = pd.read_csv(io.BytesIO(contents))
            except Exception:
                df = pd.read_csv(io.BytesIO(contents), sep=None, engine="python")
    except Exception as e:
        logger.error(f"Failed to parse uploaded dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse uploaded file into a DataFrame: {str(e)}"
        )

    if df.empty or len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded dataset is empty."
        )

    # Process features defensively
    features_df, id_series, schema_case = _detect_and_process_dataframe(df)
    n_samples = len(features_df)

    if n_samples < resolved_k:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Sample size ({n_samples}) is smaller than requested clusters k={resolved_k}."
        )

    logger.info(f"Processing {n_samples} items with {features_df.shape[1]} features under '{schema_case}', k={resolved_k}.")

    # 1. Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df.values)

    # 2. Partition Clustering (KMeans)
    kmeans = KMeans(n_clusters=resolved_k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    # 3. Evaluation Metrics
    if len(np.unique(labels)) > 1 and n_samples > resolved_k:
        try:
            sil_score = float(silhouette_score(X_scaled, labels))
        except Exception as e:
            logger.warning(f"Silhouette computation warning: {e}")
            sil_score = 0.0
    else:
        sil_score = 0.0

    try:
        db_index = float(davies_bouldin_score(X_scaled, labels))
    except Exception as e:
        logger.warning(f"Davies-Bouldin computation warning: {e}")
        db_index = 0.0

    inertia_val = float(kmeans.inertia_)

    # 4. Dimensionality Reduction (PCA 2D)
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    explained_var = [float(round(v, 4)) for v in pca.explained_variance_ratio_]

    # 5. Cluster Profiles Aggregation
    features_with_cluster = features_df.copy()
    features_with_cluster["cluster"] = labels

    profiles = []
    for cluster_id in range(resolved_k):
        c_df = features_with_cluster[features_with_cluster["cluster"] == cluster_id]
        c_size = len(c_df)
        c_pct = round((c_size / n_samples) * 100.0, 2)

        profile_dict: Dict[str, Any] = {
            "cluster": int(cluster_id),
            "size": int(c_size),
            "percentage": float(c_pct)
        }

        for col in features_df.columns:
            mean_val = float(c_df[col].mean()) if c_size > 0 else 0.0
            profile_dict[col] = round(mean_val, 4)

        profiles.append(profile_dict)

    # 6. Scatter Points Generation (with max cap for smooth browser rendering)
    MAX_SCATTER_POINTS = 5000
    if n_samples > MAX_SCATTER_POINTS:
        sample_indices = np.linspace(0, n_samples - 1, MAX_SCATTER_POINTS, dtype=int)
    else:
        sample_indices = range(n_samples)

    id_list = id_series.tolist()
    scatter_points = []
    for idx in sample_indices:
        scatter_points.append({
            "id": str(id_list[idx]),
            "pca_x": float(round(pca_coords[idx, 0], 4)),
            "pca_y": float(round(pca_coords[idx, 1], 4)),
            "cluster": int(labels[idx])
        })

    # Cluster centroids in 2D PCA space
    centroids_scaled = kmeans.cluster_centers_
    centroids_pca = pca.transform(centroids_scaled)
    centroid_points = []
    for c_id in range(resolved_k):
        centroid_points.append({
            "cluster": int(c_id),
            "pca_x": float(round(centroids_pca[c_id, 0], 4)),
            "pca_y": float(round(centroids_pca[c_id, 1], 4))
        })

    # Response payload
    return {
        "status": "success",
        "sample_size": int(n_samples),
        "k_clusters": int(resolved_k),
        "schema_detected": schema_case,
        "feature_names": list(features_df.columns),
        "metrics": {
            "silhouette_score": round(sil_score, 4),
            "davies_bouldin_index": round(db_index, 4),
            "inertia": round(inertia_val, 2),
            "pca_variance_ratio": explained_var
        },
        "profiles": profiles,
        "scatter_points": scatter_points,
        "centroids": centroid_points
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
