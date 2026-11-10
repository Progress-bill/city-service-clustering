# City-Service Review Profiles: Business Pattern Segmentation

A production-ready, two-tier machine learning web system designed for unsupervised clustering of Google Maps city-service reviews. Heavy computational tasks (feature aggregation, standardisation, K-Means clustering, PCA reduction, and metric calculations) are containerized as a 24/7 microservice on **Hugging Face Spaces (Docker runtime)**, while the web dashboard runs on **Render (Flask / Chart.js)** for interactive visualization, parameter tuning, and pattern segmentation.

---

## 🏛 System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│     Tier 1: Render Web Application & Dashboard (Flask)          │
│  - Interactive Parameter Tuning (K = 2 to 10)                   │
│  - Dataset Upload & Instant Preset Sample Ingestion             │
│  - Real-time Remote Engine Health Monitoring & Latency Tracking │
│  - Interactive Chart.js Visualizations:                         │
│      * 2D PCA Latent Space Scatter + Centroids                  │
│      * Cluster Feature Attribute Profiles                       │
│      * Market Share Doughnut Distribution                       │
│      * Business Pattern Archetype Classification Table          │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                   HTTPS (Persistent Cloud API)
         https://<your-username>-city-service-ml-engine.hf.space
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│     Tier 2: Hugging Face Spaces ML Compute Engine (Docker)      │
│  - Hardware: CPU Basic (2 vCPU, 16GB RAM, 24/7 Availability)    │
│  - Framework: FastAPI + Uvicorn listening on port 7860          │
│  - Non-root user: UID 1000                                      │
│  - Schema Detection:                                            │
│      * Case A: Raw Reviews (mean, std, log-count, text length)  │
│      * Case B: Pre-aggregated Tabular Features                  │
│  - StandardScaler Normalization                                 │
│  - K-Means Clustering (random_state=42, n_init='auto')          │
│  - Unsupervised Validation Metrics:                             │
│      * Silhouette Score                                         │
│      * Davies-Bouldin Index                                     │
│      * Inertia (Within-cluster sum of squares)                  │
│  - PCA 2D Projection & Explained Variance Ratio                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```text
.
├── Dockerfile                   # Docker specification for Hugging Face Spaces (UID 1000, port 7860)
├── main.py                      # FastAPI entry point for container execution
├── hf_space/                    # Standalone Hugging Face Space package
│   ├── Dockerfile               # Non-root Dockerfile for HF Space
│   ├── main.py                  # Full ML clustering FastAPI microservice
│   ├── requirements.txt         # FastAPI, Uvicorn, Scikit-Learn, Pandas, NumPy
│   └── README.md                # HF Spaces YAML metadata (sdk: docker, app_port: 7860)
├── app.py                       # Render Web Application backend (Flask)
├── requirements.txt             # Production web dependencies for Render
├── Procfile                     # Render process manager command
├── render.yaml                  # Render Infrastructure-as-Code Blueprint
├── sample_data/
│   ├── sample_reviews.csv       # Realistic raw reviews dataset (Case A)
│   ├── sample_aggregated.csv    # Pre-aggregated business features (Case B)
│   └── generate_samples.py      # Synthetic review data generator
├── templates/
│   └── index.html               # Responsive dashboard UI (Chart.js 4.x)
├── static/
│   ├── css/
│   │   └── styles.css           # Custom styling and glassmorphic cards
│   └── js/
│       └── dashboard.js         # Chart.js rendering, AJAX training, live ping
└── README.md                    # Comprehensive system documentation
```

---

## 🐳 Step 1: Deploy ML Compute Engine to Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space):
   - **Space Name:** `city-service-ml-engine`
   - **License:** `mit`
   - **SDK:** `Docker` $\rightarrow$ `Blank`
   - **Hardware:** `CPU basic • 2 vCPU • 16GB RAM • Free`
2. Push the contents of the `hf_space/` directory (or the root repository) to your Hugging Face Space git repository:
   ```bash
   git remote add hf https://huggingface.co/spaces/<YOUR_USERNAME>/city-service-ml-engine
   git push hf main
   ```
3. Hugging Face Spaces automatically builds the Docker image and starts the server on port `7860`.
4. Your persistent API endpoint will be:
   ```
   https://<YOUR_USERNAME>-city-service-ml-engine.hf.space
   ```
5. Test connectivity:
   ```bash
   curl https://<YOUR_USERNAME>-city-service-ml-engine.hf.space/health
   ```
   Response:
   ```json
   {
     "status": "online",
     "worker": "hf-spaces-ml-engine"
   }
   ```

---

## 🌐 Step 2: Deploy Dashboard to Render

### Option A: Render Blueprint (`render.yaml`)
1. In the [Render Dashboard](https://dashboard.render.com), click **New** $\rightarrow$ **Blueprint**.
2. Connect your repository. Render automatically reads `render.yaml`.
3. Set the environment variable `ML_WORKER_URL` to your Hugging Face Space URL.

### Option B: Manual Web Service
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
- **Environment Variables:**
  - `ML_WORKER_URL`: `https://<YOUR_USERNAME>-city-service-ml-engine.hf.space`
  - `FLASK_ENV`: `production`

---

## 🔬 Machine Learning Pipeline & Defensive Schema Detection

### 1. Ingestion Modes
- **Case A (Raw Reviews):** When `gmap_id` and `rating` are detected:
  - `mean_rating`: Average rating per business.
  - `std_rating`: Standard deviation ($ddof=0$), imputing `0.0` for 1-review businesses.
  - `log_review_count`: $\log(1 + \text{review\_count})$ to normalize power-law distribution.
  - `avg_review_length`: Average character length of textual customer reviews.
- **Case B (Pre-Aggregated Tabular Data):** When generic numeric columns are supplied:
  - Auto-identifies numeric features (requires $\ge 2$ dimensions).
  - Imputes missing values with column medians.

### 2. Standardization & Partition Clustering
- `StandardScaler`: Normalizes feature variances to unit scale.
- `KMeans(n_clusters=k, random_state=42, n_init='auto')`: Fits $K \in [2, 10]$ clusters.

### 3. Evaluation & Dimensionality Reduction
- **Silhouette Score:** Evaluates average silhouette coefficient.
- **Davies-Bouldin Index:** Ratio of within-cluster scatter to separation (lower is better).
- **Inertia:** Within-cluster sum of squares (WCSS).
- **PCA 2D Projection:** Projects dimensions to 2 principal components with explained variance ratios.

---

## 🧪 Testing Locally

To run the complete system locally:

1. **Terminal 1: Start ML Engine**
   ```bash
   uvicorn hf_space.main:app --host 0.0.0.0 --port 7860
   ```
2. **Terminal 2: Start Flask Dashboard**
   ```bash
   set ML_WORKER_URL=http://localhost:7860
   python app.py
   ```
3. Navigate to `http://localhost:5000` in your browser. Click **"Load Sample Reviews (Case A)"** to execute clustering immediately!
