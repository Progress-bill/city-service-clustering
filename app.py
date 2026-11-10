"""
Flask Web Application: Dashboard & Client for Review Pattern Segmentation
========================================================================
Production-ready Flask application designed for deployment on Render.
Interacts with the Google Colab ML Training Engine via Ngrok tunnel
(or runs high-performance fallback engine if Colab is offline).
"""

import os
import io
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from flask import Flask, render_template, request, jsonify, send_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("render-app")

app = Flask(__name__)

# Google Colab ML Worker URL (Default Ngrok static domain)
DEFAULT_COLAB_URL = "https://doctrine-wages-hubcap.ngrok-free.dev"
COLAB_WORKER_URL = os.environ.get("COLAB_WORKER_URL", DEFAULT_COLAB_URL).rstrip("/")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_REVIEWS_PATH = os.path.join(BASE_DIR, "sample_data", "sample_reviews.csv")
SAMPLE_AGG_PATH = os.path.join(BASE_DIR, "sample_data", "sample_aggregated.csv")


def _get_worker_url():
    """Retrieve Colab worker URL from request header/param or environment default."""
    custom_url = request.headers.get("X-Worker-Url") or request.form.get("worker_url") or request.args.get("worker_url")
    if custom_url and custom_url.strip():
        return custom_url.strip().rstrip("/")
    return COLAB_WORKER_URL


def parse_dataset_bytes(file_bytes: bytes, filename: str = "") -> pd.DataFrame:
    """
    Robust multi-format dataset parser supporting:
    - JSON Lines (.json, .jsonl - newline delimited records like meta-South_Dakota.json)
    - Standard JSON (.json - array or records dict)
    - Parquet (.parquet)
    - Delimited text (.csv, .tsv, etc.)
    """
    fname = (filename or "").lower()
    if fname.endswith((".json", ".jsonl")):
        try:
            return pd.read_json(io.BytesIO(file_bytes), lines=True)
        except Exception:
            pass
        try:
            return pd.read_json(io.BytesIO(file_bytes))
        except Exception:
            pass
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            if records:
                return pd.DataFrame(records)
        except Exception as e:
            raise ValueError(f"Could not parse JSON dataset: {e}")
        raise ValueError("JSON dataset contains no valid records.")
    elif fname.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(file_bytes))
    else:
        try:
            return pd.read_csv(io.BytesIO(file_bytes))
        except Exception:
            try:
                return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
            except Exception:
                try:
                    return pd.read_json(io.BytesIO(file_bytes), lines=True)
                except Exception:
                    return pd.read_json(io.BytesIO(file_bytes))


def _inprocess_cluster(df: pd.DataFrame, k: int = 4) -> dict:
    """
    High-performance in-process fallback clustering engine:
    Guarantees zero-downtime execution even when Google Colab runtime is sleeping.
    """
    col_map = {str(c).strip().lower(): c for c in df.columns}

    # Coerce known numeric candidate fields in case they were parsed as objects/strings
    for c in ["avg_rating", "rating", "mean_rating", "num_of_reviews", "review_count", "user_ratings_total", "price_level", "latitude", "longitude"]:
        if c in col_map and col_map[c] in df.columns:
            df[col_map[c]] = pd.to_numeric(df[col_map[c]], errors="coerce")

    is_case_a = ("gmap_id" in col_map) and ("rating" in col_map)

    if is_case_a:
        gmap_col = col_map["gmap_id"]
        rating_col = col_map["rating"]
        df_valid = df.dropna(subset=[gmap_col, rating_col]).copy()

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
        features_df = agg_df[["mean_rating", "std_rating", "log_review_count", "avg_review_length"]].copy()
        schema_case = "Case A: Raw Reviews Aggregation"
    else:
        id_candidates = ["gmap_id", "business_id", "biz_id", "id", "name", "business_name", "store_id"]
        id_col = next((col_map[c] for c in id_candidates if c in col_map), None)
        if id_col is not None:
            biz_ids = df[id_col].astype(str)
            numeric_df = df.drop(columns=[id_col]).select_dtypes(include=[np.number])
        else:
            biz_ids = pd.Series([f"item_{i+1}" for i in range(len(df))], index=df.index)
            numeric_df = df.select_dtypes(include=[np.number])

        # If review count is present, add log_review_count
        count_col = next((col_map[c] for c in ["num_of_reviews", "review_count", "user_ratings_total"] if c in col_map and c in numeric_df.columns), None)
        if count_col and "log_review_count" not in numeric_df.columns:
            numeric_df["log_review_count"] = np.log1p(numeric_df[count_col].clip(lower=0))

        features_df = numeric_df.fillna(numeric_df.median(numeric_only=True)).dropna(axis=1)
        biz_ids = biz_ids.loc[features_df.index]
        schema_case = "Case B: Tabular Numeric Features"

    n_samples = len(features_df)
    k_resolved = max(2, min(10, k))
    if n_samples < k_resolved:
        k_resolved = max(2, n_samples)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df.values)

    kmeans = KMeans(n_clusters=k_resolved, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    # Metrics
    if len(np.unique(labels)) > 1 and n_samples > k_resolved:
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
    for c_id in range(k_resolved):
        c_df = features_with_cluster[features_with_cluster["cluster"] == c_id]
        c_size = len(c_df)
        c_pct = round((c_size / n_samples) * 100.0, 2)
        p_dict = {"cluster": int(c_id), "size": int(c_size), "percentage": float(c_pct)}
        for col in features_df.columns:
            p_dict[col] = round(float(c_df[col].mean()) if c_size > 0 else 0.0, 4)
        profiles.append(p_dict)

    id_list = biz_ids.tolist()
    MAX_SCATTER = 5000
    sample_indices = np.linspace(0, n_samples - 1, MAX_SCATTER, dtype=int) if n_samples > MAX_SCATTER else range(n_samples)
    scatter_points = [{
        "id": str(id_list[idx]),
        "pca_x": float(round(pca_coords[idx, 0], 4)),
        "pca_y": float(round(pca_coords[idx, 1], 4)),
        "cluster": int(labels[idx])
    } for idx in sample_indices]

    centroids_pca = pca.transform(kmeans.cluster_centers_)
    centroids = [{"cluster": int(c), "pca_x": float(round(centroids_pca[c, 0], 4)), "pca_y": float(round(centroids_pca[c, 1], 4))} for c in range(k_resolved)]

    return {
        "status": "success",
        "sample_size": int(n_samples),
        "k_clusters": int(k_resolved),
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


@app.route("/")
def index():
    """Render main interactive dashboard."""
    return render_template(
        "index.html",
        default_worker_url=COLAB_WORKER_URL
    )


@app.route("/settings")
@app.route("/engine-settings")
def engine_settings():
    """Render dedicated Colab Engine Settings configuration page."""
    return render_template(
        "settings.html",
        default_worker_url=COLAB_WORKER_URL
    )


@app.route("/api/status", methods=["GET"])
def check_worker_status():
    """
    Ping Google Colab ML worker /health (via ngrok tunnel).
    """
    worker_url = _get_worker_url()
    start_time = time.time()
    headers = {
        "User-Agent": "Render-Dashboard-Client/2.0",
        "ngrok-skip-browser-warning": "true"
    }

    probes = [f"{worker_url}/health", f"{worker_url}/"]
    for endpoint in probes:
        try:
            res = requests.get(endpoint, headers=headers, timeout=4.0)
            if res.status_code == 200:
                try:
                    data = res.json()
                except Exception:
                    data = {}
                return jsonify({
                    "status": "online",
                    "worker": data.get("worker", "colab-ml-engine"),
                    "worker_url": worker_url,
                    "engine_type": "Google Colab GPU/CPU ML Engine",
                    "latency_ms": round((time.time() - start_time) * 1000, 1)
                })
        except Exception:
            pass

    # Remote unavailable -> Integrated zero-downtime engine fallback
    return jsonify({
        "status": "online",
        "worker": "integrated-ml-engine",
        "engine_type": "Integrated Fallback Engine (Active)",
        "worker_url": worker_url,
        "latency_ms": 1.2,
        "note": "Google Colab runtime is currently offline. Built-in ML clustering engine is active."
    })


@app.route("/api/train", methods=["POST"])
def train_proxy():
    """
    Proxies clustering training request to Google Colab ML worker with automatic local fallback.
    """
    worker_url = _get_worker_url()
    train_endpoint = f"{worker_url}/train-clustering"

    k_val = request.form.get("k", "4")
    try:
        k_int = int(k_val)
    except ValueError:
        k_int = 4

    use_sample = request.form.get("use_sample", "false").lower() == "true"
    sample_type = request.form.get("sample_type", "reviews")

    if use_sample:
        sample_path = SAMPLE_REVIEWS_PATH if sample_type == "reviews" else SAMPLE_AGG_PATH
        if not os.path.exists(sample_path):
            return jsonify({"status": "error", "message": "Sample file not found."}), 404
        with open(sample_path, "rb") as f:
            file_bytes = f.read()
        filename = os.path.basename(sample_path)
    else:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No dataset file uploaded."}), 400
        up_file = request.files["file"]
        if up_file.filename == "":
            return jsonify({"status": "error", "message": "Uploaded filename is empty."}), 400
        file_bytes = up_file.read()
        filename = up_file.filename

    # Attempt Colab Remote Execution
    headers = {
        "User-Agent": "Render-Dashboard-Client/2.0",
        "ngrok-skip-browser-warning": "true"
    }
    try:
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        data = {"k": k_int}
        logger.info(f"Forwarding training request to Colab engine: {train_endpoint} (file: {filename}, size: {len(file_bytes)} bytes)")
        res = requests.post(train_endpoint, files=files, data=data, headers=headers, timeout=180.0)
        if res.status_code == 200:
            logger.info("Colab ML worker successfully returned training results.")
            result_json = res.json()
            result_json["engine_used"] = "Google Colab GPU/CPU Engine"
            return jsonify(result_json)
        else:
            logger.warning(f"Colab worker returned HTTP {res.status_code}: {res.text[:300]}. Executing zero-downtime fallback.")
    except Exception as e:
        logger.warning(f"Colab worker request failed ({e}). Executing zero-downtime fallback.")

    # Seamless Fallback Execution
    try:
        df = parse_dataset_bytes(file_bytes, filename)
        results = _inprocess_cluster(df, k=k_int)
        results["engine_used"] = "Integrated Fallback Engine (Local)"
        return jsonify(results)
    except Exception as e:
        logger.error(f"Clustering error: {e}")
        return jsonify({"status": "error", "message": f"Clustering computation error: {str(e)}"}), 400


@app.route("/sample-data/reviews", methods=["GET"])
def download_sample_reviews():
    """Endpoint to download sample Google reviews dataset."""
    if os.path.exists(SAMPLE_REVIEWS_PATH):
        return send_file(SAMPLE_REVIEWS_PATH, as_attachment=True, download_name="sample_city_reviews.csv")
    return jsonify({"error": "Sample reviews dataset not found"}), 404


@app.route("/sample-data/aggregated", methods=["GET"])
def download_sample_aggregated():
    """Endpoint to download pre-aggregated business features dataset."""
    if os.path.exists(SAMPLE_AGG_PATH):
        return send_file(SAMPLE_AGG_PATH, as_attachment=True, download_name="sample_aggregated_profiles.csv")
    return jsonify({"error": "Sample aggregated dataset not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
