"""
Colab ML Worker: High-Performance FastAPI Engine for Review Clustering
======================================================================
Pre-configured for Google Colab with Static Ngrok Domain:
Domain: doctrine-wages-hubcap.ngrok-free.dev
Port:   8000
"""

import os
import io
import sys
import logging
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

import nest_asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("colab-worker")

# User Credentials
NGROK_AUTHTOKEN = "3IXML8C4lGPfsP7ehgRNGLoE5OC_4mb2Hu7PwsmABx1EbtvQ1"
STATIC_DOMAIN = "doctrine-wages-hubcap.ngrok-free.dev"
PORT = 8000

app = FastAPI(
    title="Google Colab ML Clustering Engine",
    description="Remote compute engine for City-Service Review Profiles & Business Pattern Segmentation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "online",
        "worker": "colab-ml-engine",
        "domain": STATIC_DOMAIN
    }


def _detect_and_process_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    col_map = {str(c).strip().lower(): c for c in df.columns}
    is_case_a = ("gmap_id" in col_map) and ("rating" in col_map)

    if is_case_a:
        logger.info("Detected Case A: Raw review dataset with 'gmap_id' and 'rating'.")
        gmap_col = col_map["gmap_id"]
        rating_col = col_map["rating"]

        df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
        df_valid = df.dropna(subset=[gmap_col, rating_col]).copy()

        if df_valid.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Raw review dataset contains no valid rows with non-null gmap_id and numeric rating."
            )

        text_candidates = ["text", "review_text", "review", "comment", "caption", "body", "content"]
        text_col = next((col_map[c] for c in text_candidates if c in col_map), None)
        df_valid["_review_length"] = df_valid[text_col].fillna("").astype(str).str.len() if text_col else 0.0

        grouped = df_valid.groupby(gmap_col)
        mean_rating = grouped[rating_col].mean().rename("mean_rating")
        std_rating = grouped[rating_col].std(ddof=0).fillna(0.0).rename("std_rating")
        log_review_count = np.log1p(grouped.size()).rename("log_review_count")
        avg_review_length = grouped["_review_length"].mean().rename("avg_review_length")

        agg_df = pd.concat([mean_rating, std_rating, log_review_count, avg_review_length], axis=1).reset_index()
        biz_ids = agg_df[gmap_col].astype(str)
        features = agg_df[["mean_rating", "std_rating", "log_review_count", "avg_review_length"]].copy()

        return features, biz_ids, "Case A: Raw Reviews Aggregation"
    else:
        logger.info("Detected Case B: Pre-aggregated / generic tabular dataset.")
        id_candidates = ["gmap_id", "business_id", "biz_id", "id", "name", "business_name", "store_id"]
        id_col = next((col_map[c] for c in id_candidates if c in col_map), None)

        if id_col is not None:
            biz_ids = df[id_col].astype(str)
            numeric_df = df.drop(columns=[id_col]).select_dtypes(include=[np.number])
        else:
            biz_ids = pd.Series([f"item_{i+1}" for i in range(len(df))], index=df.index)
            numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset must have at least 2 numeric feature columns. Found only {numeric_df.shape[1]}."
            )

        features = numeric_df.fillna(numeric_df.median(numeric_only=True)).dropna(axis=1)
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
    resolved_k = max(2, min(10, k or k_query or 4))

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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not parse uploaded file: {str(e)}")

    if df.empty or len(df) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded dataset is empty.")

    features_df, id_series, schema_case = _detect_and_process_dataframe(df)
    n_samples = len(features_df)

    if n_samples < resolved_k:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Sample size ({n_samples}) is smaller than requested clusters k={resolved_k}."
        )

    logger.info(f"Processing {n_samples} items with {features_df.shape[1]} features under '{schema_case}', k={resolved_k}.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df.values)

    kmeans = KMeans(n_clusters=resolved_k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    if len(np.unique(labels)) > 1 and n_samples > resolved_k:
        try:
            sil_score = float(silhouette_score(X_scaled, labels))
        except Exception:
            sil_score = 0.0
    else:
        sil_score = 0.0

    try:
        db_index = float(davies_bouldin_score(X_scaled, labels))
    except Exception:
        db_index = 0.0

    inertia_val = float(kmeans.inertia_)

    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    explained_var = [float(round(v, 4)) for v in pca.explained_variance_ratio_]

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
            profile_dict[col] = round(float(c_df[col].mean()) if c_size > 0 else 0.0, 4)
        profiles.append(profile_dict)

    MAX_SCATTER_POINTS = 5000
    sample_indices = np.linspace(0, n_samples - 1, MAX_SCATTER_POINTS, dtype=int) if n_samples > MAX_SCATTER_POINTS else range(n_samples)
    id_list = id_series.tolist()

    scatter_points = [{
        "id": str(id_list[idx]),
        "pca_x": float(round(pca_coords[idx, 0], 4)),
        "pca_y": float(round(pca_coords[idx, 1], 4)),
        "cluster": int(labels[idx])
    } for idx in sample_indices]

    centroids_pca = pca.transform(kmeans.cluster_centers_)
    centroids = [{
        "cluster": int(c_id),
        "pca_x": float(round(centroids_pca[c_id, 0], 4)),
        "pca_y": float(round(centroids_pca[c_id, 1], 4))
    } for c_id in range(resolved_k)]

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
        "centroids": centroids
    }


def start_server(token: str = NGROK_AUTHTOKEN, domain: str = STATIC_DOMAIN, port: int = PORT):
    nest_asyncio.apply()
    ngrok.set_auth_token(token)

    tunnel = None
    try:
        tunnel = ngrok.connect(addr=port, domain=domain)
    except Exception as e:
        logger.warning(f"Static domain '{domain}' connect error: {e}. Trying fallback...")
        try:
            tunnel = ngrok.connect(addr=port, hostname=domain)
        except Exception:
            tunnel = ngrok.connect(addr=port)

    public_url = tunnel.public_url.replace("http://", "https://")
    print("\n" + "=" * 70)
    print("🚀 GOOGLE COLAB ML TRAINING ENGINE IS LIVE!")
    print("=" * 70)
    print(f"👉 STATIC PUBLIC URL:  {public_url}")
    print(f"👉 HEALTH PROBE URL:   {public_url}/health")
    print(f"👉 SWAGGER API DOCS:   {public_url}/docs")
    print("=" * 70)
    print("✅ Your Render Web Dashboard is pre-configured for this domain.")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    start_server()
