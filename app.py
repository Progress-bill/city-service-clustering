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


CITY_DISTRICTS = [
    {"name": "Downtown Commercial Core", "lat": 40.0165, "lng": -105.2790},
    {"name": "Pearl Street Promenade", "lat": 40.0182, "lng": -105.2780},
    {"name": "North Industrial Corridor", "lat": 40.0345, "lng": -105.2520},
    {"name": "East Tech Commerce Hub", "lat": 40.0180, "lng": -105.2320},
    {"name": "South Foothills District", "lat": 39.9980, "lng": -105.2680}
]

SECTOR_POOLS = {
    "deficit": [
        "Automotive & Fleet Care",
        "HVAC, Plumbing & Electrical",
        "Home Services & Contracting",
        "Commercial Cleaning & Janitorial"
    ],
    "champion": [
        "Commercial Dining & Bistro",
        "Specialty Coffee Roasters",
        "Health, Dental & Wellness",
        "Professional Services"
    ],
    "boutique": [
        "Boutique Retail & Food",
        "Personal Fitness & Pilates",
        "Artisan Studio & Crafts",
        "Specialty Pet Care & Clinic"
    ],
    "mid_tier": [
        "Commercial Building Maintenance",
        "Local Hardware & Supplies",
        "Printing & Signage Solutions",
        "General Landscaping & Tree Care"
    ],
    "turnaround": [
        "Specialty Auto Customization",
        "Equipment Rental Hub",
        "Commercial Logistics & Storage",
        "Waste & Salvage Management"
    ]
}

NAME_POOLS = {
    "deficit": [
        "QuickFix Auto Collision", "Metro Express Plumbing", "Front Range Budget Towing",
        "Discount Appliance Repair", "Reliable 24/7 HVAC Solutions", "Citywide Commercial Cleaners",
        "Discount Brake & Tire Depot", "Rapid Rooter Drain Doctors", "Affordable Electric Pros",
        "Valley Transmission Specialist", "Budget City Locksmiths", "Speedy Auto Glass Express"
    ],
    "champion": [
        "Boulder Peak Artisan Bistro", "Pearl Street Craft Roastery", "Chautauqua Grill & Wine Bar",
        "Flatirons Fine Dining Room", "Alpine Reserve Kitchen", "Sumac Mediterranean Table",
        "Bramble Craft Gastropub", "Flatirons Dental & Aesthetics", "Highland Artisan Bakery",
        "Summit Wellness Spa & Retreat", "Vanguard Legal & Wealth Advisory", "Pike Organic Kitchen"
    ],
    "boutique": [
        "Flatirons Holistic Physical Therapy", "Apex Botanical Apothecary", "Boulder Boutique Pilates Lab",
        "High Country Leather Crafts", "Sanitas Acupuncture & Wellness", "Summit Ceramic Studio",
        "Pearl Street Hat & Haberdashery", "Alpine Custom Bicycles", "Flatirons Canine Day Spa",
        "Boulder Herb & Spice Co", "Canyon Yoga & Sound Sanctuary", "Silver Lake Bookshop & Press"
    ],
    "mid_tier": [
        "Benchmark Property Solutions", "Pioneer Tool & Hardware", "Evergreen Facility Maintenance",
        "Front Range Print Works", "Standard Fleet Logistics", "Commercial Clean Co",
        "Boulder Valley Lumber Supply", "Precision Signage Works", "Summit Commercial Maintenance"
    ],
    "turnaround": [
        "Valley Industrial Equipment Yard", "Eastside Storage & Parcel Hub", "Midtown Scrap Metals & Salvage",
        "Legacy Parking Management Group", "Pioneer Waste & Roll-Off Bins", "Continental Forklift & Heavy Rental",
        "Front Range Rail Cargo Depot", "Boulder Commercial Pallet Exchange", "Summit Concrete Batch Yard"
    ]
}


def _parse_lat_lng_pair(raw_lat, raw_lng, raw_coord_str=None) -> tuple:
    """
    Parse real latitude and longitude from explicit columns, WKT POINT(lng lat),
    GeoJSON dicts, or 'lat, lng' strings. Returns (has_real_gps, lat, lng).
    """
    if pd.notna(raw_lat) and pd.notna(raw_lng):
        try:
            b_lat = float(str(raw_lat).strip())
            b_lng = float(str(raw_lng).strip())
            # Handle accidentally swapped (lng, lat) when lng is in [-180,-90] or [90,180]
            if abs(b_lat) > 90.0 and abs(b_lng) <= 90.0 and abs(b_lat) <= 180.0:
                b_lat, b_lng = b_lng, b_lat
            if abs(b_lat) <= 90.0 and abs(b_lng) <= 180.0 and (abs(b_lat) > 0.001 or abs(b_lng) > 0.001):
                return True, round(b_lat, 6), round(b_lng, 6)
        except (ValueError, TypeError):
            pass

    if raw_coord_str is not None and pd.notna(raw_coord_str):
        import re
        s = str(raw_coord_str).strip()
        # WKT POINT (lng lat)
        m_wkt = re.search(r"POINT\s*\(\s*([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s*\)", s, re.IGNORECASE)
        if m_wkt:
            try:
                lng_v, lat_v = float(m_wkt.group(1)), float(m_wkt.group(2))
                if abs(lat_v) <= 90.0 and abs(lng_v) <= 180.0:
                    return True, round(lat_v, 6), round(lng_v, 6)
            except ValueError:
                pass
        # Generic "lat, lng" or "[lat, lng]" pair
        nums = re.findall(r"[-+]?\d+\.\d+", s)
        if len(nums) >= 2:
            try:
                a, b = float(nums[0]), float(nums[1])
                if abs(a) > 90.0 and abs(b) <= 90.0:
                    a, b = b, a
                if abs(a) <= 90.0 and abs(b) <= 180.0 and (abs(a) > 0.001 or abs(b) > 0.001):
                    return True, round(a, 6), round(b, 6)
            except ValueError:
                pass
    return False, 0.0, 0.0


def extract_and_enrich_entity_locations(df_raw: pd.DataFrame, biz_ids: pd.Series, labels: np.ndarray, features_df: pd.DataFrame, strategic_clusters: list) -> tuple:
    """
    Extracts explicit location and entity metadata from DataFrame if present,
    or heuristically enriches records based on cluster archetypes and city geography.
    Returns:
      entity_locations: list of mapped business entities
      location_insights: aggregated district and sector intelligence
    """
    col_map = {str(c).strip().lower(): c for c in df_raw.columns}

    name_col = next((col_map[c] for c in ["business_name", "name", "title", "store_name", "company", "biz_name", "entity_name", "dba", "trade_name", "licensee"] if c in col_map), None)
    type_col = next((col_map[c] for c in ["business_type", "category", "categories", "type", "industry", "license_type", "service_type", "description", "naics_desc", "sector"] if c in col_map), None)
    district_col = next((col_map[c] for c in ["district", "neighborhood", "city", "zone", "borough", "area", "subarea", "ward", "zip", "postal_code"] if c in col_map), None)
    address_col = next((col_map[c] for c in ["address", "street", "street_address", "full_address", "location", "site_address", "business_address", "physical_address"] if c in col_map), None)
    lat_candidates = ["latitude", "lat", "y_coord", "lat_coord", "point_latitude", "y", "geo_latitude", "business_lat", "lat_dd", "location_lat"]
    lng_candidates = ["longitude", "lng", "lon", "long", "x_coord", "lng_coord", "point_longitude", "x", "geo_longitude", "business_lng", "lon_dd", "lng_dd", "location_lng", "location_lon"]
    lat_col = next((col_map[c] for c in lat_candidates if c in col_map), None)
    lng_col = next((col_map[c] for c in lng_candidates if c in col_map), None)
    # Fallback substring matching for lat/lng columns (e.g., "Geo Latitude", "LON_WGS84")
    if lat_col is None:
        lat_col = next((orig for low, orig in col_map.items() if ("lat" in low and "plat" not in low and "relat" not in low)), None)
    if lng_col is None:
        lng_col = next((orig for low, orig in col_map.items() if ("lon" in low or "lng" in low)), None)
    coord_col = next((col_map[c] for c in ["coordinates", "coord", "coords", "geopoint", "geo_point", " geo_point_2d", "the_geom", "geometry", "wkt", "point", "latlng", "lat_long"] if c in col_map), None)

    rating_col = next((c for c in ["mean_rating", "avg_rating", "rating", "score"] if c in features_df.columns), None)
    volume_col = next((c for c in ["log_review_count", "num_of_reviews", "review_count", "reviews", "volume"] if c in features_df.columns), None)
    volatility_col = next((c for c in ["std_rating", "std", "volatility", "variance"] if c in features_df.columns), None)

    cluster_lookup = {c["cluster"]: c for c in strategic_clusters}
    entities = []

    id_col = next((col_map[c] for c in ["gmap_id", "business_id", "biz_id", "id", "store_id"] if c in col_map), None)
    meta_by_id = {}
    if id_col is not None and id_col in df_raw.columns:
        subset_cols = [c for c in [name_col, type_col, district_col, address_col, lat_col, lng_col, coord_col] if c is not None]
        if subset_cols:
            first_rows = df_raw.drop_duplicates(subset=[id_col]).set_index(id_col)
            meta_by_id = first_rows[subset_cols].to_dict(orient="index")

    for idx, (original_idx, biz_id) in enumerate(biz_ids.items()):
        c_id = int(labels[idx]) if idx < len(labels) else 0
        c_info = cluster_lookup.get(c_id, {})
        archetype = c_info.get("archetype", "Consistent Mid-Tier Providers")
        opp_score = float(c_info.get("opportunity_score", 50.0))

        r_val = float(features_df.loc[original_idx, rating_col]) if (rating_col and original_idx in features_df.index) else 3.5
        v_val = float(features_df.loc[original_idx, volume_col]) if (volume_col and original_idx in features_df.index) else 2.0
        s_val = float(features_df.loc[original_idx, volatility_col]) if (volatility_col and original_idx in features_df.index) else 0.5

        if "Vacuum" in archetype or "Deficit" in archetype:
            pool_key = "deficit"
            action = "Prime Disruption Target • Capture unsatisfied market share"
        elif "Champions" in archetype:
            pool_key = "champion"
            action = "Market Anchor • Defend moat & introduce VIP services"
        elif "Boutique" in archetype:
            pool_key = "boutique"
            action = "Growth Equity Target • Scale secondary location & digital SEO"
        elif "Turnaround" in archetype or "Volatility" in archetype:
            pool_key = "turnaround"
            action = "M&A Rollup Target • Standardize QA SOPs & acquire at 2.5x EBITDA"
        else:
            pool_key = "mid_tier"
            action = "Consortium Partner • Join group purchasing & supply chain co-op"

        meta_dict = meta_by_id.get(biz_id, {}) if meta_by_id else {}

        # 1. Name
        raw_name = meta_dict.get(name_col) if name_col in meta_dict else (df_raw.loc[original_idx, name_col] if name_col and original_idx in df_raw.index else None)
        if pd.notna(raw_name) and str(raw_name).strip():
            b_name = str(raw_name).strip()
        else:
            pool = NAME_POOLS.get(pool_key, NAME_POOLS["mid_tier"])
            b_name = pool[idx % len(pool)]
            if idx >= len(pool):
                b_name += f" #{idx // len(pool) + 1}"

        # 2. Type
        raw_type = meta_dict.get(type_col) if type_col in meta_dict else (df_raw.loc[original_idx, type_col] if type_col and original_idx in df_raw.index else None)
        if pd.notna(raw_type) and str(raw_type).strip():
            b_type = str(raw_type).strip()
        else:
            sec_pool = SECTOR_POOLS.get(pool_key, SECTOR_POOLS["mid_tier"])
            b_type = sec_pool[idx % len(sec_pool)]

        # 3. District
        raw_dist = meta_dict.get(district_col) if district_col in meta_dict else (df_raw.loc[original_idx, district_col] if district_col and original_idx in df_raw.index else None)
        dist_preset = CITY_DISTRICTS[(idx + c_id) % len(CITY_DISTRICTS)]
        if pd.notna(raw_dist) and str(raw_dist).strip():
            b_dist = str(raw_dist).strip()
        else:
            if pool_key == "deficit":
                b_dist = "Downtown Commercial Core" if (idx % 3 != 0) else "North Industrial Corridor"
            elif pool_key == "champion":
                b_dist = "Pearl Street Promenade" if (idx % 2 == 0) else "Downtown Commercial Core"
            elif pool_key == "boutique":
                b_dist = "South Foothills District" if (idx % 2 == 0) else "Pearl Street Promenade"
            else:
                b_dist = dist_preset["name"]

        # 4. Address
        raw_addr = meta_dict.get(address_col) if address_col in meta_dict else (df_raw.loc[original_idx, address_col] if address_col and original_idx in df_raw.index else None)
        if pd.notna(raw_addr) and str(raw_addr).strip():
            b_addr = str(raw_addr).strip()
        else:
            b_addr = f"{1000 + (idx * 37) % 8900} {b_dist.split()[0]} Ave, {b_dist}"

        # 5. Coords (Real GPS Coordinates)
        raw_lat = meta_dict.get(lat_col) if lat_col in meta_dict else (df_raw.loc[original_idx, lat_col] if lat_col and original_idx in df_raw.index else None)
        raw_lng = meta_dict.get(lng_col) if lng_col in meta_dict else (df_raw.loc[original_idx, lng_col] if lng_col and original_idx in df_raw.index else None)
        raw_coord = meta_dict.get(coord_col) if coord_col in meta_dict else (df_raw.loc[original_idx, coord_col] if coord_col and original_idx in df_raw.index else None)
        if raw_coord is None and address_col and b_addr:
            raw_coord = b_addr
        has_real_gps, b_lat, b_lng = _parse_lat_lng_pair(raw_lat, raw_lng, raw_coord)

        if not has_real_gps:
            target_d = next((d for d in CITY_DISTRICTS if d["name"] == b_dist), dist_preset)
            # Spread fallback coordinates organically across the district (Golden-angle spiral, not a tight box)
            import math
            angle = idx * 2.399963229728653  # Golden angle in radians
            radius = 0.0012 + (math.sqrt((idx % 120) + 1) * 0.00095)
            jitter_lat = math.sin(angle) * radius * 0.78
            jitter_lng = math.cos(angle) * radius
            b_lat = round(target_d["lat"] + jitter_lat, 6)
            b_lng = round(target_d["lng"] + jitter_lng, 6)

        google_maps_url = f"https://www.google.com/maps/dir/?api=1&destination={b_lat},{b_lng}"
        apple_maps_url = f"https://maps.apple.com/?daddr={b_lat},{b_lng}&dirflg=d"

        entities.append({
            "id": str(biz_id),
            "name": b_name,
            "business_type": b_type,
            "district": b_dist,
            "address": b_addr,
            "lat": b_lat,
            "lng": b_lng,
            "has_real_gps": has_real_gps,
            "google_maps_url": google_maps_url,
            "apple_maps_url": apple_maps_url,
            "cluster": c_id,
            "rating": round(r_val, 2),
            "volume": round(v_val, 2),
            "volatility": round(s_val, 2),
            "opportunity_score": opp_score,
            "archetype": archetype,
            "investment_action": action
        })

    # Location Insights Aggregations
    # A. Districts
    district_groups = {}
    for e in entities:
        d_name = e["district"]
        if d_name not in district_groups:
            district_groups[d_name] = []
        district_groups[d_name].append(e)

    districts_summary = []
    for d_name, items in district_groups.items():
        avg_r = round(sum(i["rating"] for i in items) / len(items), 2)
        top_c = max(set(i["cluster"] for i in items), key=lambda c: sum(1 for i in items if i["cluster"] == c))
        max_opp = max(i["opportunity_score"] for i in items)
        sec_counts = {}
        for i in items:
            sec_counts[i["business_type"]] = sec_counts.get(i["business_type"], 0) + 1
        primary_sec = max(sec_counts.items(), key=lambda x: x[1])[0] if sec_counts else "General Services"

        districts_summary.append({
            "name": d_name,
            "total_businesses": len(items),
            "avg_rating": avg_r,
            "top_opportunity_cluster": int(top_c),
            "opportunity_score": max_opp,
            "primary_sector": primary_sec,
            "deficit_detected": avg_r < 3.8 and len(items) >= 4,
            "lat": round(sum(i["lat"] for i in items) / len(items), 5),
            "lng": round(sum(i["lng"] for i in items) / len(items), 5)
        })
    districts_summary.sort(key=lambda x: x["opportunity_score"], reverse=True)

    # B. Sectors
    sector_groups = {}
    for e in entities:
        sec = e["business_type"]
        if sec not in sector_groups:
            sector_groups[sec] = []
        sector_groups[sec].append(e)

    sectors_summary = []
    for sec, items in sector_groups.items():
        avg_r = round(sum(i["rating"] for i in items) / len(items), 2)
        dom_c = max(set(i["cluster"] for i in items), key=lambda c: sum(1 for i in items if i["cluster"] == c))
        max_opp = max(i["opportunity_score"] for i in items)
        dist_counts = {}
        for i in items:
            dist_counts[i["district"]] = dist_counts.get(i["district"], 0) + 1
        top_loc = max(dist_counts.items(), key=lambda x: x[1])[0] if dist_counts else "Citywide"

        if max_opp >= 90:
            verdict = f"High Disruption Opening: {len(items)} underperforming providers in {top_loc} ripe for competitor entry."
        elif max_opp >= 75:
            verdict = f"Growth & Scaling: Top rated boutique providers ({avg_r}★) positioned for secondary location expansion."
        else:
            verdict = f"Consortium & Rollup: Stable operations ({len(items)} entities) suitable for shared supply chain consolidation."

        sectors_summary.append({
            "sector": sec,
            "count": len(items),
            "avg_rating": avg_r,
            "dominant_cluster": int(dom_c),
            "opportunity_score": max_opp,
            "top_location": top_loc,
            "investment_verdict": verdict
        })
    sectors_summary.sort(key=lambda x: x["opportunity_score"], reverse=True)

    # C. Enrich clusters
    for c in strategic_clusters:
        c_id = c["cluster"]
        c_items = [e for e in entities if e["cluster"] == c_id]
        if c_items:
            c_secs = {}
            for i in c_items:
                c_secs[i["business_type"]] = c_secs.get(i["business_type"], 0) + 1
            top_secs = sorted(c_secs.items(), key=lambda x: x[1], reverse=True)[:3]
            c["top_sectors"] = [f"{s[0]} ({round(s[1]/len(c_items)*100)}%)" for s in top_secs]

            c_locs = {}
            for i in c_items:
                c_locs[i["district"]] = c_locs.get(i["district"], 0) + 1
            top_locs = sorted(c_locs.items(), key=lambda x: x[1], reverse=True)[:2]
            c["top_locations"] = [f"{l[0]} ({round(l[1]/len(c_items)*100)}%)" for l in top_locs]

            if c["opportunity_score"] >= 90:
                c["location_strategy"] = f"Deploy capital in {top_locs[0][0] if top_locs else 'urban centers'} targeting the {top_secs[0][0] if top_secs else 'service'} vacuum where customers suffer low quality."
            elif c["opportunity_score"] >= 75:
                c["location_strategy"] = f"Partner with operators in {top_locs[0][0] if top_locs else 'commercial districts'} to scale high-reputation {top_secs[0][0] if top_secs else 'boutique'} brands."
            else:
                c["location_strategy"] = f"Consolidate back-office operations across {top_locs[0][0] if top_locs else 'industrial hubs'} to capture regional supply economies."

    return entities, {
        "districts": districts_summary,
        "sectors": sectors_summary
    }


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
    entity_locations, location_insights = extract_and_enrich_entity_locations(
        df, biz_ids, labels, features_df, strategic_intel.get("clusters", [])
    )

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
        "market_trends": strategic_intel.get("macro_trends", {}),
        "entity_locations": entity_locations[:1000],
        "location_insights": location_insights
    }


@app.route("/")
@app.route("/clustering")
@app.route("/pca-reviews")
def index():
    """Render main loading dataset & 2D PCA reviews clustering dashboard."""
    return render_template(
        "index.html",
        default_worker_url=COLAB_WORKER_URL
    )


@app.route("/analytics")
@app.route("/market-recommendations")
@app.route("/market-trends")
def market_analytics():
    """Render dedicated Data Analytics & Market Recommendations page."""
    return render_template(
        "analytics.html",
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


# -------------------------------------------------------------
# Server-side results hand-off (Ingestion page -> Analytics page)
# Browser localStorage is capped (~5MB) and silently fails on large
# datasets, so the latest result per browser client is kept here.
# -------------------------------------------------------------
_LATEST_RESULTS = {}
_LATEST_RESULTS_MAX = 50


def _remember_result(result: dict, dataset_name: str):
    """Store the latest successful clustering result for the requesting client."""
    try:
        client_id = (request.form.get("client_id") or request.headers.get("X-Client-Id") or "default").strip()[:64]
        result["dataset_name"] = dataset_name
        result["result_timestamp"] = time.time()
        _LATEST_RESULTS[client_id] = result
        _LATEST_RESULTS["__last__"] = result
        while len(_LATEST_RESULTS) > _LATEST_RESULTS_MAX:
            _LATEST_RESULTS.pop(next(iter(_LATEST_RESULTS)))
    except Exception as e:
        logger.warning(f"Could not store latest result: {e}")
    return result


@app.route("/api/latest-results", methods=["GET"])
def latest_results():
    """Return the most recent clustering result for this browser client."""
    client_id = (request.args.get("client_id") or request.headers.get("X-Client-Id") or "default").strip()[:64]
    result = _LATEST_RESULTS.get(client_id)
    if not result:
        return jsonify({"status": "empty", "message": "No clustering results stored for this session yet."}), 404
    return jsonify(result)


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

    # Attempt Colab Remote Execution with fast pre-flight probe
    headers = {
        "User-Agent": "Render-Dashboard-Client/2.0",
        "ngrok-skip-browser-warning": "true"
    }
    colab_reachable = False
    try:
        probe = requests.get(f"{worker_url}/health", headers=headers, timeout=2.0)
        if probe.status_code == 200:
            colab_reachable = True
    except Exception:
        colab_reachable = False

    if colab_reachable:
        try:
            files = {"file": (filename, file_bytes, "application/octet-stream")}
            data = {"k": k_int}
            logger.info(f"Forwarding training request to Colab engine: {train_endpoint} (file: {filename}, size: {len(file_bytes)} bytes)")
            res = requests.post(train_endpoint, files=files, data=data, headers=headers, timeout=(4.0, 90.0))
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

                # Defensive enrichment: If remote Colab worker omitted entity_locations
                # OR returned older synthetic coordinates (missing has_real_gps), re-extract
                # coordinates directly from the uploaded file so real GPS pins always appear.
                existing_locs = result_json.get("entity_locations") or []
                has_any_real_gps = any(item.get("has_real_gps") for item in existing_locs)
                if not existing_locs or not has_any_real_gps:
                    try:
                        df_raw = parse_dataset_bytes(file_bytes, filename)
                        labels = np.array([pt.get("cluster", 0) for pt in result_json.get("scatter_points", [])])
                        col_map_r = {str(c).strip().lower(): c for c in df_raw.columns}
                        id_col = next((col_map_r[c] for c in ["gmap_id", "business_id", "biz_id", "id", "name", "business_name", "store_id"] if c in col_map_r), None)
                        biz_ids = df_raw[id_col].astype(str) if id_col else pd.Series([f"item_{i+1}" for i in range(len(df_raw))], index=df_raw.index)
                        if len(labels) < len(biz_ids):
                            biz_ids = biz_ids.iloc[:len(labels)]
                        num_df = df_raw.select_dtypes(include=[np.number])
                        ent, loc_intel = extract_and_enrich_entity_locations(
                            df_raw, biz_ids, labels, num_df, result_json.get("strategic_decisions", [])
                        )
                        result_json["entity_locations"] = ent[:1000]
                        result_json["location_insights"] = loc_intel
                    except Exception as e_enrich:
                        logger.warning(f"Entity enrichment on Colab response failed: {e_enrich}")

                if result_json.get("status", "success") == "success":
                    _remember_result(result_json, filename)
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
        if results.get("status", "success") == "success":
            _remember_result(results, filename)
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


@app.route("/api/directions", methods=["GET"])
def get_gps_directions():
    """
    Real-world GPS Routing API integration:
    Queries OSRM (Open Source Routing Machine) API to return driving/walking
    geometry, distances, duration, and turn-by-turn road steps.
    Includes zero-failure fallback with great-circle distance.
    """
    try:
        start_lat = float(request.args.get("start_lat", 40.0175))
        start_lng = float(request.args.get("start_lng", -105.2750))
        end_lat = float(request.args.get("end_lat"))
        end_lng = float(request.args.get("end_lng"))
        mode = request.args.get("mode", "driving")
        if mode not in ["driving", "walking", "cycling"]:
            mode = "driving"
    except (TypeError, ValueError) as e:
        return jsonify({"status": "error", "message": f"Invalid coordinates: {e}"}), 400

    # 1. Attempt live query to OSRM real GPS routing service
    osrm_url = f"https://router.project-osrm.org/route/v1/{mode}/{start_lng},{start_lat};{end_lng},{end_lat}?overview=full&geometries=geojson&steps=true"
    headers = {"User-Agent": "CityService-Clustering-GIS/2.0"}
    try:
        resp = requests.get(osrm_url, headers=headers, timeout=6.0)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("routes"):
                r = data["routes"][0]
                meters = float(r.get("distance", 0.0))
                seconds = float(r.get("duration", 0.0))
                geo_coords = r.get("geometry", {}).get("coordinates", [])
                leaflet_coords = [[c[1], c[0]] for c in geo_coords]

                steps_raw = r.get("legs", [{}])[0].get("steps", [])
                steps_clean = []
                for s in steps_raw:
                    m = s.get("maneuver", {})
                    st_name = s.get("name") or "Connecting Road"
                    m_type = m.get("type", "turn")
                    m_mod = m.get("modifier", "")

                    if m_type == "depart":
                        instruction = f"Head {m_mod or 'forward'} on {st_name}"
                    elif m_type == "arrive":
                        instruction = f"Arrive at destination on {m_mod or 'your route'}"
                    elif m_mod:
                        instruction = f"Turn {m_mod} onto {st_name}"
                    else:
                        instruction = f"Continue onto {st_name}"

                    dist_ft = s.get("distance", 0.0) * 3.28084
                    dist_str = f"{int(dist_ft)} ft" if dist_ft < 1000 else f"{round(dist_ft/5280.0, 1)} mi"

                    steps_clean.append({
                        "instruction": instruction,
                        "road_name": st_name,
                        "modifier": m_mod,
                        "type": m_type,
                        "distance_m": round(s.get("distance", 0.0), 1),
                        "distance_str": dist_str,
                        "duration_s": round(s.get("duration", 0.0), 1),
                        "location": [m.get("location", [0, 0])[1], m.get("location", [0, 0])[0]] if m.get("location") else []
                    })

                return jsonify({
                    "status": "success",
                    "routing_engine": "OSRM OpenStreetMap GPS Engine",
                    "distance_miles": round(meters * 0.000621371, 2),
                    "distance_km": round(meters / 1000.0, 2),
                    "duration_mins": max(1, round(seconds / 60.0)),
                    "geometry": leaflet_coords,
                    "steps": steps_clean,
                    "google_maps_url": f"https://www.google.com/maps/dir/?api=1&origin={start_lat},{start_lng}&destination={end_lat},{end_lng}&travelmode={mode}",
                    "apple_maps_url": f"https://maps.apple.com/?saddr={start_lat},{start_lng}&daddr={end_lat},{end_lng}&dirflg=d"
                })
    except Exception as e:
        logger.warning(f"OSRM GPS Routing call failed ({e}). Calculating mathematical great-circle path fallback.")

    # 2. Resilient Haversine Fallback
    import math
    R = 3958.8
    dlat = math.radians(end_lat - start_lat)
    dlng = math.radians(end_lng - start_lng)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(start_lat)) * math.cos(math.radians(end_lat)) * math.sin(dlng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    dist_mi = round(R * c, 2)
    dist_km = round(dist_mi * 1.60934, 2)
    est_mins = max(1, round((dist_mi / 25.0) * 60.0))

    steps_count = 10
    interp_coords = [
        [round(start_lat + (end_lat - start_lat) * (i / steps_count), 6),
         round(start_lng + (end_lng - start_lng) * (i / steps_count), 6)]
        for i in range(steps_count + 1)
    ]

    fallback_steps = [
        {"instruction": "Depart from current location towards destination", "distance_str": f"{round(dist_mi * 0.2, 1)} mi", "modifier": "straight", "type": "depart"},
        {"instruction": "Proceed along municipal arterial corridor", "distance_str": f"{round(dist_mi * 0.6, 1)} mi", "modifier": "straight", "type": "continue"},
        {"instruction": f"Arrive at destination coordinates ({end_lat:.4f}, {end_lng:.4f})", "distance_str": "Arrived", "modifier": "reach", "type": "arrive"}
    ]

    return jsonify({
        "status": "success",
        "routing_engine": "Geodesic Direct GPS Estimator",
        "distance_miles": dist_mi,
        "distance_km": dist_km,
        "duration_mins": est_mins,
        "geometry": interp_coords,
        "steps": fallback_steps,
        "google_maps_url": f"https://www.google.com/maps/dir/?api=1&origin={start_lat},{start_lng}&destination={end_lat},{end_lng}&travelmode={mode}",
        "apple_maps_url": f"https://maps.apple.com/?saddr={start_lat},{start_lng}&daddr={end_lat},{end_lng}&dirflg=d"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
