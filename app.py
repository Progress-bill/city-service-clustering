"""
Flask Web Application: Dashboard & Client for Review Pattern Segmentation
========================================================================
Production-ready Flask application designed for deployment on Render.
Interacts with the containerized Hugging Face Spaces ML engine (Docker / FastAPI)
via persistent HTTPS endpoints.
"""

import os
import time
import requests
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# Hugging Face Spaces ML Worker persistent HTTPS URL (Docker container endpoint)
DEFAULT_HF_URL = "https://progressmurau-city-business-clustering.hf.space"
HF_WORKER_URL = os.environ.get("HF_WORKER_URL", DEFAULT_HF_URL).rstrip("/")
if ".static.hf.space" in HF_WORKER_URL:
    HF_WORKER_URL = HF_WORKER_URL.replace(".static.hf.space", ".hf.space")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_REVIEWS_PATH = os.path.join(BASE_DIR, "sample_data", "sample_reviews.csv")
SAMPLE_AGG_PATH = os.path.join(BASE_DIR, "sample_data", "sample_aggregated.csv")


def _get_worker_url():
    """Retrieve worker URL from request header/param or environment default, auto-normalizing."""
    custom_url = request.headers.get("X-Worker-Url") or request.form.get("worker_url") or request.args.get("worker_url")
    if custom_url and custom_url.strip():
        url = custom_url.strip().rstrip("/")
        # Auto-correct common .static.hf.space typo to containerized .hf.space domain
        if ".static.hf.space" in url:
            url = url.replace(".static.hf.space", ".hf.space")
        return url
    return HF_WORKER_URL


@app.route("/")
def index():
    """Render main interactive dashboard."""
    return render_template(
        "index.html",
        default_worker_url=HF_WORKER_URL
    )


@app.route("/settings")
@app.route("/engine-settings")
def engine_settings():
    """Render dedicated Engine Settings configuration page."""
    return render_template(
        "settings.html",
        default_worker_url=HF_WORKER_URL
    )


@app.route("/api/status", methods=["GET"])
def check_worker_status():
    """
    Ping Hugging Face Spaces ML worker /health (or root /) and measure round-trip latency.
    """
    worker_url = _get_worker_url()
    start_time = time.time()
    headers = {
        "User-Agent": "Render-Dashboard-Client/2.0"
    }

    # Defensive probing: try /health first, fallback to /
    probes = [f"{worker_url}/health", f"{worker_url}/"]
    last_res = None
    last_err = None

    for endpoint in probes:
        try:
            res = requests.get(endpoint, headers=headers, timeout=8.0)
            latency = round((time.time() - start_time) * 1000, 1)

            if res.status_code == 200:
                try:
                    data = res.json()
                except Exception:
                    data = {}
                return jsonify({
                    "status": "online",
                    "worker": data.get("worker") or data.get("engine") or "hf-spaces-ml-engine",
                    "worker_url": worker_url,
                    "endpoint_reached": endpoint,
                    "latency_ms": latency
                })
            else:
                last_res = res
        except requests.exceptions.Timeout:
            last_err = "Connection timed out. The Hugging Face Space may be waking up from sleep or building."
        except requests.exceptions.RequestException as e:
            last_err = str(e)

    if last_res is not None:
        return jsonify({
            "status": "offline",
            "worker_url": worker_url,
            "error": f"HTTP {last_res.status_code}: {last_res.text[:140]}"
        }), 502
    else:
        return jsonify({
            "status": "offline",
            "worker_url": worker_url,
            "error": last_err or "Worker unreachable"
        }), 504


@app.route("/api/train", methods=["POST"])
def train_proxy():
    """
    Proxies clustering training request to the Hugging Face Spaces ML compute engine.
    Supports either an uploaded file or a preset sample dataset.
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

    headers = {
        "User-Agent": "Render-Dashboard-Client/2.0"
    }

    try:
        if use_sample:
            # Use embedded sample dataset
            sample_file_path = SAMPLE_REVIEWS_PATH if sample_type == "reviews" else SAMPLE_AGG_PATH
            if not os.path.exists(sample_file_path):
                return jsonify({"status": "error", "message": "Sample file not found on server."}), 404

            filename = os.path.basename(sample_file_path)
            with open(sample_file_path, "rb") as sf:
                files = {"file": (filename, sf.read(), "text/csv")}
                data = {"k": k_int}
                response = requests.post(train_endpoint, files=files, data=data, headers=headers, timeout=120.0)
        else:
            if "file" not in request.files:
                return jsonify({"status": "error", "message": "No dataset file uploaded."}), 400

            uploaded_file = request.files["file"]
            if uploaded_file.filename == "":
                return jsonify({"status": "error", "message": "Uploaded filename is empty."}), 400

            file_content = uploaded_file.read()
            files = {"file": (uploaded_file.filename, file_content, uploaded_file.content_type or "text/csv")}
            data = {"k": k_int}
            response = requests.post(train_endpoint, files=files, data=data, headers=headers, timeout=120.0)

        # Forward ML worker response
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            try:
                err_json = response.json()
                msg = err_json.get("detail", response.text[:200])
            except Exception:
                msg = response.text[:200]
            return jsonify({"status": "error", "message": f"ML Engine Error ({response.status_code}): {msg}"}), response.status_code

    except requests.exceptions.Timeout:
        return jsonify({
            "status": "error",
            "message": "Compute request timed out. The Hugging Face Space may be waking up or processing an exceptionally large dataset."
        }), 504
    except requests.exceptions.RequestException as e:
        return jsonify({
            "status": "error",
            "message": f"Unable to reach Hugging Face ML Engine at {worker_url}. Ensure the Space is active: {str(e)}"
        }), 502


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
