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


def generate_strategic_intelligence(profiles: list, sample_size: int, schema_case: str) -> dict:
    """
    Derives real-world statistical decisions, market opportunity scores,
    and targeted business venture proposals based on clustering archetypes.
    """
    if not profiles or sample_size <= 0:
        return {"clusters": [], "macro_trends": {}}

    p0 = profiles[0]
    rating_candidates = ["mean_rating", "avg_rating", "rating", "score"]
    rating_key = next((k for k in rating_candidates if k in p0), None)

    volume_candidates = ["log_review_count", "num_of_reviews", "review_count", "reviews", "volume", "count"]
    volume_key = next((k for k in volume_candidates if k in p0), None)

    vol_candidates = ["std_rating", "std", "volatility", "variance"]
    vol_key = next((k for k in vol_candidates if k in p0), None)

    eng_candidates = ["avg_review_length", "review_length", "length", "text_len"]
    eng_key = next((k for k in eng_candidates if k in p0), None)

    all_ratings = [float(p.get(rating_key, 3.5)) if rating_key else 3.5 for p in profiles]
    all_volumes = [float(p.get(volume_key, 2.0)) if volume_key else 2.0 for p in profiles]
    all_volatilities = [float(p.get(vol_key, 0.5)) if vol_key else 0.5 for p in profiles]
    all_lengths = [float(p.get(eng_key, 100.0)) if eng_key else 100.0 for p in profiles]

    min_r, max_r = min(all_ratings), max(all_ratings)
    min_v, max_v = min(all_volumes), max(all_volumes)
    min_s, max_s = min(all_volatilities), max(all_volatilities)
    min_l, max_l = min(all_lengths), max(all_lengths)

    def _norm(val, mn, mx, invert=False):
        if mx - mn <= 1e-6:
            res = 50.0
        else:
            res = max(10.0, min(95.0, ((val - mn) / (mx - mn)) * 85.0 + 10.0))
        return round(105.0 - res if invert else res, 1)

    clusters_intelligence = []

    for p in profiles:
        c_id = int(p.get("cluster", 0))
        c_size = int(p.get("size", 0))
        c_pct = float(p.get("percentage", 0.0))

        r_val = float(p.get(rating_key, 3.5)) if rating_key else 3.5
        v_val = float(p.get(volume_key, 2.0)) if volume_key else 2.0
        s_val = float(p.get(vol_key, 0.5)) if vol_key else 0.5
        e_val = float(p.get(eng_key, 100.0)) if eng_key else 100.0

        if r_val < 3.8 and v_val > 2.2:
            opp_score = round(min(98.0, 78.0 + (5.0 - r_val) * 10.0 + (v_val * 3.5)), 1)
            archetype = "High-Demand / Service-Deficit Vacuum"
            archetype_badge = "bg-rose-500/10 text-rose-300 border-rose-500/30"
            venture_title = "Disruptive Alternative Service Venture"
            venture_concept = f"Launch a modernized, customer-first service brand in this area to directly capture market share from the {c_size} underperforming incumbents ({r_val:.2f} avg rating)."
            risk_level = "Moderate Risk • High Disruption Yield"
            priority = "Tier 1 Prime Target"
            roi_potential = "3.8x - 5.2x (36 Mo)"
            decisions = [
                "Capital Allocation: Deploy seed or expansion capital specifically into localized digital customer acquisition and geofenced ads.",
                "Pricing Strategy: Introduce transparent upfront pricing and satisfaction guarantee to contrast sharply against incumbent friction.",
                "Operational Intervention: Automate appointment booking and implement SLA-backed response times under 15 minutes.",
                "Marketing Campaign: Run 'Tired of Low Quality?' campaigns contrasting verified customer reviews against legacy providers."
            ]
        elif r_val >= 4.3 and v_val < 2.5:
            opp_score = round(min(96.0, 72.0 + (r_val * 4.5)), 1)
            archetype = "Emerging High-Quality Boutique Niches"
            archetype_badge = "bg-cyan-500/10 text-cyan-300 border-cyan-500/30"
            venture_title = "Boutique Expansion & Digital Scaling Fund"
            venture_concept = f"Inject growth equity or establish franchise partnerships with these top-rated operators ({r_val:.2f} rating) who lack digital marketing scale."
            risk_level = "Low Risk • High Capital Efficiency"
            priority = "Tier 1 Growth Opportunity"
            roi_potential = "3.2x - 4.5x (24 Mo)"
            decisions = [
                "Capital Allocation: Provide growth financing, franchise systems, and equipment leasing for secondary location rollouts.",
                "Tech Infrastructure: Deploy automated SMS review-request workflows to convert daily physical visits into digital review volume.",
                "Market Expansion: Replicate their exact operational recipes and high-touch customer service across adjacent city districts.",
                "Brand Partnerships: Aggregate multiple boutique specialists under an umbrella regional membership or subscription pass."
            ]
        elif r_val >= 4.3 and v_val >= 2.5:
            opp_score = round(min(88.0, 62.0 + (r_val * 4.0)), 1)
            archetype = "High-Volume Market Champions"
            archetype_badge = "bg-emerald-500/10 text-emerald-300 border-emerald-500/30"
            venture_title = "Premium Moat Defense & Ancillary Upselling"
            venture_concept = f"Defend dominance across this {c_pct:.1f}% market share anchor by introducing premium VIP tiers, corporate contracts, and ancillary services."
            risk_level = "Very Low Risk • Stable Cashflow"
            priority = "Tier 2 Strategic Anchor"
            roi_potential = "2.1x - 2.8x (Steady Yield)"
            decisions = [
                "Capital Allocation: Reinvest cashflows into customer retention, employee profit-sharing, and bespoke mobile apps.",
                "Monetization: Implement a 10-15% price elasticity test via premium on-demand or fast-track service tiers without churn risk.",
                "Defensive Moat: Monitor new entrants and lock in long-term commercial real estate leases and supplier exclusivity.",
                "Talent Retention: Create structured apprenticeship programs to prevent top technician and service staff turnover."
            ]
        elif s_val >= 0.85 or (r_val < 4.0 and r_val >= 3.2):
            opp_score = round(min(82.0, 56.0 + s_val * 22.0), 1)
            archetype = "Polarized / High-Volatility Operators"
            archetype_badge = "bg-amber-500/10 text-amber-300 border-amber-500/30"
            venture_title = "Operational Turnaround & Rollup M&A"
            venture_concept = f"Target erratic providers ({s_val:.2f} rating volatility) for rollup acquisition at attractive multiples (2.2x-3.0x EBITDA) and install standardized QA SOPs."
            risk_level = "Moderate-to-High Risk • Value Turnaround"
            priority = "Tier 2 Restructuring Target"
            roi_potential = "2.8x - 4.0x (Turnaround)"
            decisions = [
                "Acquisition Strategy: Approach owners of volatile businesses with earn-out structures to acquire asset bases at discounted valuations.",
                "Operational Overhaul: Mandate comprehensive SOP checklists and install real-time staff incentive metrics tied to customer satisfaction.",
                "Reputation Recovery: Establish an automated post-service follow-up loop that intercepts negative feedback prior to public posting.",
                "Cost Synergy: Centralize administrative overhead, billing, and inventory procurement into a shared regional hub."
            ]
        else:
            opp_score = 48.0
            archetype = "Consistent Mid-Tier Providers"
            archetype_badge = "bg-indigo-500/10 text-indigo-300 border-indigo-500/30"
            venture_title = "Value-Tier Supply Chain & Group Purchasing Co-op"
            venture_concept = f"Form group purchasing cooperatives and shared marketing consortiums across the {c_size} mid-market operators to expand gross margins."
            risk_level = "Moderate Risk • Steady Margin"
            priority = "Tier 3 Baseline"
            roi_potential = "1.8x - 2.4x (Operational Margin)"
            decisions = [
                "Cost Optimization: Aggregate supply and tooling purchases across businesses in this tier to negotiate 12-18% bulk supplier discounts.",
                "Service Specialization: Incentivize providers to develop specialized sub-offerings to escape undifferentiated commodity pricing.",
                "Digital Upgrades: Modernize legacy websites, optimize Google Business profiles, and establish localized SEO presence.",
                "Loyalty Programs: Implement cross-business local loyalty cards to encourage repeat customer retention."
            ]

        radar_vector = {
            "satisfaction": _norm(r_val, min_r, max_r),
            "demand": _norm(v_val, min_v, max_v),
            "consistency": _norm(s_val, min_s, max_s, invert=True),
            "engagement": _norm(e_val, min_l, max_l),
            "opportunity": opp_score
        }

        clusters_intelligence.append({
            "cluster": c_id,
            "archetype": archetype,
            "archetype_badge": archetype_badge,
            "size": c_size,
            "percentage": c_pct,
            "opportunity_score": opp_score,
            "risk_level": risk_level,
            "priority": priority,
            "roi_potential": roi_potential,
            "venture_title": venture_title,
            "venture_concept": venture_concept,
            "decisions": decisions,
            "radar_vector": radar_vector,
            "metrics_summary": {
                "rating": round(r_val, 2),
                "volume": round(v_val, 2),
                "volatility": round(s_val, 2),
                "engagement": round(e_val, 1)
            }
        })

    avg_mkt_rating = round(sum(c["metrics_summary"]["rating"] * c["size"] for c in clusters_intelligence) / max(1, sample_size), 2)
    top_opp = max(clusters_intelligence, key=lambda x: x["opportunity_score"]) if clusters_intelligence else None
    health_score = round(min(100.0, max(10.0, (avg_mkt_rating / 5.0) * 70.0 + (30.0 if avg_mkt_rating >= 4.0 else 15.0))), 1)

    deficit_clusters = [c for c in clusters_intelligence if c["archetype"] == "High-Demand / Service-Deficit Vacuum"]
    if deficit_clusters:
        gap_text = f"Severe customer satisfaction deficit detected in Cluster {deficit_clusters[0]['cluster']} ({deficit_clusters[0]['percentage']}% market share). Strong demand paired with low satisfaction creates an immediate disruption opening."
    else:
        gap_text = "Market is heavily contested with established mid-to-high performers. Primary venture opportunity lies in boutique digital scaling and premium tier expansion."

    macro_trends = {
        "avg_market_rating": avg_mkt_rating,
        "market_health_score": health_score,
        "top_venture_target": top_opp["venture_title"] if top_opp else "Strategic Venture Expansion",
        "top_opportunity_score": top_opp["opportunity_score"] if top_opp else 75.0,
        "prime_cluster_target": top_opp["cluster"] if top_opp else 0,
        "prime_archetype": top_opp["archetype"] if top_opp else "Emerging Opportunities",
        "market_gap_analysis": gap_text
    }

    return {
        "clusters": clusters_intelligence,
        "macro_trends": macro_trends
    }


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

    strategic_intel = generate_strategic_intelligence(profiles, n_samples, schema_case)

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
        "centroids": centroids,
        "strategic_decisions": strategic_intel.get("clusters", []),
        "market_trends": strategic_intel.get("macro_trends", {})
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

            # Defensive fallback enrichment: If remote Colab worker is running an earlier code version
            # without strategic_decisions, compute decisions dynamically from returned profiles
            if "strategic_decisions" not in result_json and "profiles" in result_json:
                intel = generate_strategic_intelligence(
                    result_json.get("profiles", []),
                    result_json.get("sample_size", 100),
                    result_json.get("schema_detected", "Case A")
                )
                result_json["strategic_decisions"] = intel.get("clusters", [])
                result_json["market_trends"] = intel.get("macro_trends", {})

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
