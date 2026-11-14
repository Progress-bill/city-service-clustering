/**
 * Dashboard & Visualization Logic for City-Service Review Clustering
 * Chart.js 4.x Implementation with Remote Colab ML Engine Integration
 */

// Stable per-browser ID so the server can hand results to the Analytics page
function getCityServiceClientId() {
  let id = localStorage.getItem('city_service_client_id');
  if (!id) {
    id = 'c_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 10);
    localStorage.setItem('city_service_client_id', id);
  }
  return id;
}

// Palette of distinct modern neon colors for clusters K=0 to K=9
const CLUSTER_COLORS = [
  { fill: 'rgba(99, 102, 241, 0.75)', border: '#6366f1', glow: 'rgba(99, 102, 241, 0.4)' },  // Indigo
  { fill: 'rgba(6, 182, 212, 0.75)',  border: '#06b6d4', glow: 'rgba(6, 182, 212, 0.4)' },   // Cyan
  { fill: 'rgba(245, 158, 11, 0.75)', border: '#f59e0b', glow: 'rgba(245, 158, 11, 0.4)' },  // Amber
  { fill: 'rgba(16, 185, 129, 0.75)', border: '#10b981', glow: 'rgba(16, 185, 129, 0.4)' },  // Emerald
  { fill: 'rgba(236, 72, 153, 0.75)', border: '#ec4899', glow: 'rgba(236, 72, 153, 0.4)' },  // Pink
  { fill: 'rgba(168, 85, 247, 0.75)', border: '#a855f7', glow: 'rgba(168, 85, 247, 0.4)' },  // Purple
  { fill: 'rgba(239, 68, 68, 0.75)',  border: '#ef4444', glow: 'rgba(239, 68, 68, 0.4)' },   // Red
  { fill: 'rgba(59, 130, 246, 0.75)', border: '#3b82f6', glow: 'rgba(59, 130, 246, 0.4)' },  // Blue
  { fill: 'rgba(20, 184, 166, 0.75)', border: '#14b8a6', glow: 'rgba(20, 184, 166, 0.4)' },  // Teal
  { fill: 'rgba(249, 115, 22, 0.75)', border: '#f97316', glow: 'rgba(249, 115, 22, 0.4)' },  // Orange
];

// Global Chart instances
let pcaScatterChart = null;
let clusterProfilesChart = null;
let clusterPieChart = null;
let marketMatrixChart = null;
let ventureRadarChart = null;
let showCentroids = true;
let currentClusteringData = null;

// DOM Elements
const workerStatusBadge = document.getElementById('worker-status-badge');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const latencyTag = document.getElementById('latency-tag');
const btnToggleConfig = document.getElementById('btn-toggle-config');
const workerConfigDrawer = document.getElementById('worker-config-drawer');
const inputWorkerUrl = document.getElementById('input-worker-url');
const btnSaveWorkerUrl = document.getElementById('btn-save-worker-url');
const btnResetWorkerUrl = document.getElementById('btn-reset-worker-url');

const form = document.getElementById('clustering-form');
const kSlider = document.getElementById('k-slider');
const kBadge = document.getElementById('k-value-badge');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('dataset-file');
const fileInfo = document.getElementById('file-info');
const btnSubmit = document.getElementById('btn-submit');
const btnSubmitText = document.getElementById('btn-submit-text');
const btnSpinner = document.getElementById('btn-spinner');
const alertBox = document.getElementById('alert-box');

const btnQuickSampleReviews = document.getElementById('btn-quick-sample-reviews');
const btnQuickSampleAgg = document.getElementById('btn-quick-sample-agg');
const btnToggleCentroids = document.getElementById('btn-toggle-centroids');

// Metric elements
const metricSil = document.getElementById('metric-silhouette');
const metricSilLabel = document.getElementById('metric-silhouette-label');
const metricDb = document.getElementById('metric-db');
const metricInertia = document.getElementById('metric-inertia');
const metricPcaVar = document.getElementById('metric-pca-var');
const metricPcaVarSub = document.getElementById('metric-pca-var-sub');
const scatterMeta = document.getElementById('scatter-meta');
const chartPlaceholder = document.getElementById('chart-placeholder');
const detectedSchemaPill = document.getElementById('detected-schema-pill');
const profilesTableBody = document.getElementById('profiles-table-body');

// Global toggle for Engine Settings Modal
window.toggleSettingsModal = function(e) {
  if (e && e.preventDefault) e.preventDefault();
  const modal = document.getElementById('settings-modal');
  if (modal) {
    modal.classList.toggle('hidden');
    // Sync current URL into modal input
    const modalInput = document.getElementById('modal-worker-url');
    if (modalInput && inputWorkerUrl) {
      modalInput.value = inputWorkerUrl.value;
    }
  }
};

window.closeSettingsModal = function() {
  const modal = document.getElementById('settings-modal');
  if (modal) {
    modal.classList.add('hidden');
  }
};

// -------------------------------------------------------------
// Initialization & Worker Connection Management
// -------------------------------------------------------------
function bootstrapApp() {
  initUIListeners();
  checkWorkerHealth();
  setInterval(checkWorkerHealth, 30000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootstrapApp);
} else {
  bootstrapApp();
}

function initUIListeners() {
  // Slider update
  if (kSlider) {
    kSlider.addEventListener('input', (e) => {
      kBadge.textContent = `K = ${e.target.value}`;
    });
  }

  // Config drawer / modal toggle
  if (btnToggleConfig) {
    btnToggleConfig.addEventListener('click', (e) => {
      window.toggleSettingsModal(e);
    });
  }

  // Close modal when clicking backdrop
  const settingsModal = document.getElementById('settings-modal');
  if (settingsModal) {
    settingsModal.addEventListener('click', (e) => {
      if (e.target === settingsModal) {
        window.closeSettingsModal();
      }
    });
  }

  // Close on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      window.closeSettingsModal();
    }
  });

  // Modal elements
  const modalInput = document.getElementById('modal-worker-url');
  const modalBtnPing = document.getElementById('modal-btn-ping');
  const modalBtnSave = document.getElementById('modal-btn-save');
  const modalPresetHf = document.getElementById('modal-preset-hf');
  const modalPresetLocal = document.getElementById('modal-preset-local');

  if (modalPresetHf && modalInput) {
    modalPresetHf.addEventListener('click', () => {
      modalInput.value = 'https://doctrine-wages-hubcap.ngrok-free.dev';
    });
  }

  if (modalPresetLocal && modalInput) {
    modalPresetLocal.addEventListener('click', () => {
      modalInput.value = 'http://localhost:8000';
    });
  }

  if (modalBtnSave && modalInput) {
    modalBtnSave.addEventListener('click', () => {
      const url = modalInput.value.trim();
      if (url) {
        localStorage.setItem('hf_worker_url', url);
        localStorage.setItem('colab_worker_url', url);
        if (inputWorkerUrl) inputWorkerUrl.value = url;
        checkWorkerHealth();
        window.closeSettingsModal();
        showAlert('success', `Updated and connected to ML Engine at: ${url}`);
      }
    });
  }

  if (modalBtnPing && modalInput) {
    modalBtnPing.addEventListener('click', () => {
      const url = modalInput.value.trim();
      if (url && inputWorkerUrl) inputWorkerUrl.value = url;
      checkWorkerHealth();
    });
  }

  // Save / Update Worker URL from quick drawer
  if (btnSaveWorkerUrl) {
    btnSaveWorkerUrl.addEventListener('click', () => {
      const url = inputWorkerUrl.value.trim();
      if (url) {
        localStorage.setItem('hf_worker_url', url);
        localStorage.setItem('colab_worker_url', url);
        if (modalInput) modalInput.value = url;
        checkWorkerHealth();
      }
    });
  }

  // Reset Worker URL
  if (btnResetWorkerUrl) {
    btnResetWorkerUrl.addEventListener('click', () => {
      localStorage.removeItem('hf_worker_url');
      localStorage.removeItem('colab_worker_url');
      const defUrl = 'https://doctrine-wages-hubcap.ngrok-free.dev';
      inputWorkerUrl.value = defUrl;
      if (modalInput) modalInput.value = defUrl;
      checkWorkerHealth();
    });
  }

  // Load saved worker URL if present
  let savedUrl = localStorage.getItem('hf_worker_url') || localStorage.getItem('colab_worker_url');
  if (savedUrl && savedUrl.includes('.static.hf.space')) {
    savedUrl = savedUrl.replace('.static.hf.space', '.hf.space');
    localStorage.setItem('hf_worker_url', savedUrl);
  }
  if (savedUrl) {
    inputWorkerUrl.value = savedUrl;
    if (modalInput) modalInput.value = savedUrl;
  }

  // Dropzone file handling
  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', handleFileSelected);

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('border-indigo-500', 'bg-indigo-950/30');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('border-indigo-500', 'bg-indigo-950/30');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('border-indigo-500', 'bg-indigo-950/30');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelected();
    }
  });

  // Form submit
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!fileInput.files || fileInput.files.length === 0) {
      showAlert('warning', 'Please select or upload a dataset file first, or click one of the sample buttons.');
      return;
    }
    executeClustering({ useSample: false });
  });

  // Quick preset sample buttons
  btnQuickSampleReviews.addEventListener('click', () => {
    fileInfo.classList.remove('hidden');
    fileInfo.innerHTML = '<i class="fa-solid fa-file-csv mr-1"></i> Loaded: Preset Sample Reviews (Case A)';
    executeClustering({ useSample: true, sampleType: 'reviews' });
  });

  btnQuickSampleAgg.addEventListener('click', () => {
    fileInfo.classList.remove('hidden');
    fileInfo.innerHTML = '<i class="fa-solid fa-table mr-1"></i> Loaded: Preset Aggregated Data (Case B)';
    executeClustering({ useSample: true, sampleType: 'aggregated' });
  });

  // Toggle Centroids
  btnToggleCentroids.addEventListener('click', () => {
    showCentroids = !showCentroids;
    btnToggleCentroids.classList.toggle('bg-indigo-600', showCentroids);
    btnToggleCentroids.classList.toggle('text-white', showCentroids);
    if (currentClusteringData) {
      renderPcaScatterPlot(currentClusteringData);
    }
  });
}

function handleFileSelected() {
  if (fileInput.files && fileInput.files[0]) {
    const file = fileInput.files[0];
    fileInfo.classList.remove('hidden');
    fileInfo.innerHTML = `<i class="fa-solid fa-file-lines mr-1 text-indigo-400"></i> ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  }
}

// -------------------------------------------------------------
// Health Check & Worker Status
// -------------------------------------------------------------
async function checkWorkerHealth() {
  const currentWorkerUrl = (inputWorkerUrl ? inputWorkerUrl.value.trim() : '') || 'https://doctrine-wages-hubcap.ngrok-free.dev';
  if (statusDot) statusDot.className = 'w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse';
  if (statusText) statusText.textContent = 'Pinging ML Engine...';
  if (latencyTag) latencyTag.classList.add('hidden');

  const modalDot = document.getElementById('modal-status-dot');
  const modalText = document.getElementById('modal-status-text');
  const modalLat = document.getElementById('modal-latency-text');
  if (modalDot) modalDot.className = 'w-2 h-2 rounded-full bg-amber-400 animate-pulse';
  if (modalText) {
    modalText.textContent = 'Pinging...';
    modalText.className = 'text-xs font-mono font-semibold text-amber-300';
  }

  try {
    const res = await fetch(`/api/status?worker_url=${encodeURIComponent(currentWorkerUrl)}`);
    const data = await res.json();

    if (res.ok && data.status === 'online') {
      if (statusDot) statusDot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-lg shadow-emerald-500/50';
      if (statusText) {
        statusText.textContent = 'Colab ML Engine: Online';
        statusText.className = 'text-emerald-300 font-semibold text-xs';
      }
      if (data.latency_ms && latencyTag) {
        latencyTag.textContent = `${data.latency_ms} ms`;
        latencyTag.classList.remove('hidden');
      }

      if (modalDot) modalDot.className = 'w-2 h-2 rounded-full bg-emerald-400';
      if (modalText) {
        modalText.textContent = 'ONLINE (200 OK)';
        modalText.className = 'text-xs font-mono font-semibold text-emerald-400';
      }
      if (modalLat && data.latency_ms) modalLat.textContent = `${data.latency_ms} ms`;
    } else {
      markWorkerOffline(data.error || 'Worker unavailable');
    }
  } catch (err) {
    markWorkerOffline(err.message);
  }
}

function markWorkerOffline(errMsg) {
  if (statusDot) statusDot.className = 'w-2.5 h-2.5 rounded-full bg-rose-500 shadow-lg shadow-rose-500/50';
  if (statusText) {
    statusText.textContent = 'Colab ML Engine: Offline';
    statusText.className = 'text-rose-400 font-medium text-xs';
  }
  if (latencyTag) latencyTag.classList.add('hidden');

  const modalDot = document.getElementById('modal-status-dot');
  const modalText = document.getElementById('modal-status-text');
  const modalLat = document.getElementById('modal-latency-text');
  if (modalDot) modalDot.className = 'w-2 h-2 rounded-full bg-rose-500';
  if (modalText) {
    modalText.textContent = 'OFFLINE';
    modalText.className = 'text-xs font-mono font-semibold text-rose-400';
  }
  if (modalLat) modalLat.textContent = 'N/A';
}

// -------------------------------------------------------------
// Clustering Execution API Call
// -------------------------------------------------------------
async function executeClustering(options = {}) {
  const { useSample = false, sampleType = 'reviews' } = options;
  const currentWorkerUrl = inputWorkerUrl.value.trim();
  const kVal = kSlider.value;

  hideAlert();
  setLoadingState(true);

  const formData = new FormData();
  formData.append('k', kVal);
  formData.append('worker_url', currentWorkerUrl);
  formData.append('client_id', getCityServiceClientId());

  if (useSample) {
    formData.append('use_sample', 'true');
    formData.append('sample_type', sampleType);
  } else {
    formData.append('file', fileInput.files[0]);
  }

  try {
    const response = await fetch('/api/train', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (!response.ok || data.status === 'error') {
      const msg = data.message || `Clustering failed with status ${response.status}`;
      showAlert('error', msg);
      return;
    }

    // Success: store and visualize
    currentClusteringData = data;
    renderDashboard(data);
    showAlert('success', `Successfully clustered ${data.sample_size} business profiles into K=${data.k_clusters} patterns!`);

  } catch (err) {
    showAlert('error', `Connection error: ${err.message}. Ensure the Colab worker script is actively running.`);
  } finally {
    setLoadingState(false);
  }
}

function setLoadingState(isLoading) {
  btnSubmit.disabled = isLoading;
  if (isLoading) {
    btnSubmitText.textContent = 'Computing Clusters...';
    btnSpinner.classList.remove('hidden');
  } else {
    btnSubmitText.textContent = 'Execute ML Pipeline';
    btnSpinner.classList.add('hidden');
  }
}

function showAlert(type, message) {
  alertBox.classList.remove('hidden', 'bg-rose-950/60', 'border-rose-800', 'text-rose-300', 'bg-emerald-950/60', 'border-emerald-800', 'text-emerald-300', 'bg-amber-950/60', 'border-amber-800', 'text-amber-300');
  
  let icon = 'fa-circle-info';
  if (type === 'error') {
    alertBox.classList.add('bg-rose-950/60', 'border-rose-800', 'text-rose-300');
    icon = 'fa-triangle-exclamation';
  } else if (type === 'success') {
    alertBox.classList.add('bg-emerald-950/60', 'border-emerald-800', 'text-emerald-300');
    icon = 'fa-circle-check';
  } else {
    alertBox.classList.add('bg-amber-950/60', 'border-amber-800', 'text-amber-300');
    icon = 'fa-circle-exclamation';
  }

  alertBox.innerHTML = `
    <div class="flex items-start space-x-3">
      <i class="fa-solid ${icon} mt-0.5 text-base"></i>
      <div class="flex-1">${message}</div>
      <button onclick="hideAlert()" class="text-slate-400 hover:text-white">&times;</button>
    </div>
  `;
}

function hideAlert() {
  alertBox.classList.add('hidden');
}

// -------------------------------------------------------------
// Dashboard Rendering & Visualizations
// -------------------------------------------------------------
function renderDashboard(data) {
  // 1. Update Metrics Ribbon
  const metrics = data.metrics || {};
  
  // Silhouette
  const sil = metrics.silhouette_score ?? 0;
  metricSil.textContent = sil.toFixed(4);
  if (sil >= 0.5) {
    metricSilLabel.textContent = 'Strong Structure (>= 0.5)';
    metricSilLabel.className = 'text-[10px] text-emerald-400 font-semibold mt-1';
  } else if (sil >= 0.25) {
    metricSilLabel.textContent = 'Moderate Separation (0.25 - 0.5)';
    metricSilLabel.className = 'text-[10px] text-cyan-400 font-semibold mt-1';
  } else {
    metricSilLabel.textContent = 'Weak / Overlapping Separation (< 0.25)';
    metricSilLabel.className = 'text-[10px] text-amber-400 font-semibold mt-1';
  }

  // Davies-Bouldin
  const db = metrics.davies_bouldin_index ?? 0;
  metricDb.textContent = db.toFixed(4);

  // Inertia
  const inertia = metrics.inertia ?? 0;
  metricInertia.textContent = Number(inertia).toLocaleString(undefined, { maximumFractionDigits: 1 });

  // PCA Variance
  const pcaVar = metrics.pca_variance_ratio || [0, 0];
  const pc1 = (pcaVar[0] * 100).toFixed(1);
  const pc2 = (pcaVar[1] * 100).toFixed(1);
  const totalVar = ((pcaVar[0] + pcaVar[1]) * 100).toFixed(1);
  metricPcaVar.textContent = `${totalVar}%`;
  metricPcaVarSub.textContent = `PC1: ${pc1}% | PC2: ${pc2}%`;

  // Schema & Meta
  scatterMeta.textContent = `${data.sample_size} businesses (${data.k_clusters} clusters)`;
  detectedSchemaPill.textContent = data.schema_detected || 'Aggregated Dataset';
  chartPlaceholder.classList.add('hidden');

  // Cache results for dedicated Analytics & Market Recommendations page.
  // The server also keeps a copy (per client_id), so this is a best-effort cache:
  // large datasets can exceed the ~5MB localStorage quota.
  const activeFileName = (fileInput && fileInput.files && fileInput.files[0]) ? fileInput.files[0].name : (data.dataset_name || data.schema_detected || 'Processed Dataset');
  localStorage.removeItem('city_service_ml_results'); // never leave stale results behind
  localStorage.setItem('city_service_dataset_name', activeFileName);
  try {
    localStorage.setItem('city_service_ml_results', JSON.stringify(data));
  } catch (e) {
    try {
      // Retry with a compact payload (drop heavy PCA scatter points)
      const compact = Object.assign({}, data, { scatter_points: [] });
      localStorage.setItem('city_service_ml_results', JSON.stringify(compact));
    } catch (e2) {
      console.warn('Results too large for localStorage; Analytics page will load them from the server.', e2);
    }
  }

  // Reveal Analytics CTA Banner
  const ctaBanner = document.getElementById('analytics-cta-banner');
  if (ctaBanner) {
    ctaBanner.classList.remove('hidden');
    const ctaCount = document.getElementById('cta-cluster-count');
    if (ctaCount) {
      ctaCount.textContent = `${data.k_clusters} Archetype Segments Formed`;
    }
  }

  // 2. Render PCA Reviews & Profiles Charts
  renderPcaScatterPlot(data);
  renderProfilesChart(data);
  renderDistributionPieChart(data);

  // 3. Populate Archetype Profile Matrix Table
  populateProfilesTable(data);
}

// --- 2D PCA Scatter Plot ---
function renderPcaScatterPlot(data) {
  const ctx = document.getElementById('pcaScatterChart').getContext('2d');
  
  if (pcaScatterChart) {
    pcaScatterChart.destroy();
  }

  const k = data.k_clusters;
  const datasets = [];

  // Group scatter points by cluster
  for (let c = 0; c < k; c++) {
    const clusterPoints = data.scatter_points
      .filter(p => p.cluster === c)
      .map(p => ({
        x: p.pca_x,
        y: p.pca_y,
        id: p.id,
        cluster: c
      }));

    const color = CLUSTER_COLORS[c % CLUSTER_COLORS.length];

    datasets.push({
      label: `Cluster ${c}`,
      data: clusterPoints,
      backgroundColor: color.fill,
      borderColor: color.border,
      borderWidth: 1,
      pointRadius: 4.5,
      pointHoverRadius: 7,
      pointHoverBackgroundColor: '#ffffff',
      pointHoverBorderColor: color.border,
      pointHoverBorderWidth: 2,
    });
  }

  // Add Centroids dataset if present & enabled
  if (showCentroids && data.centroids && data.centroids.length > 0) {
    const centroidPoints = data.centroids.map(ct => ({
      x: ct.pca_x,
      y: ct.pca_y,
      cluster: ct.cluster
    }));

    datasets.push({
      label: 'Centroids (Centers)',
      data: centroidPoints,
      backgroundColor: '#ffffff',
      borderColor: '#fbbf24',
      borderWidth: 2,
      pointStyle: 'rectRot',
      pointRadius: 9,
      pointHoverRadius: 11,
      pointHoverBackgroundColor: '#fbbf24',
    });
  }

  pcaScatterChart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 750,
        easing: 'easeOutQuart'
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: '#cbd5e1',
            usePointStyle: true,
            boxWidth: 8,
            font: { family: 'Plus Jakarta Sans', size: 11 }
          }
        },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#f8fafc',
          bodyColor: '#cbd5e1',
          borderColor: '#334155',
          borderWidth: 1,
          padding: 10,
          boxPadding: 4,
          usePointStyle: true,
          callbacks: {
            label: function(context) {
              const raw = context.raw;
              if (raw.id) {
                return [
                  `Business ID: ${raw.id}`,
                  `Cluster: ${raw.cluster}`,
                  `PC1: ${raw.x.toFixed(3)}, PC2: ${raw.y.toFixed(3)}`
                ];
              } else {
                return `Centroid Cluster ${raw.cluster} (PC1: ${raw.x.toFixed(3)}, PC2: ${raw.y.toFixed(3)})`;
              }
            }
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: `Principal Component 1 (${((data.metrics.pca_variance_ratio[0] || 0) * 100).toFixed(1)}% Var)`,
            color: '#94a3b8',
            font: { family: 'Plus Jakarta Sans', size: 11, weight: 'bold' }
          },
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          title: {
            display: true,
            text: `Principal Component 2 (${((data.metrics.pca_variance_ratio[1] || 0) * 100).toFixed(1)}% Var)`,
            color: '#94a3b8',
            font: { family: 'Plus Jakarta Sans', size: 11, weight: 'bold' }
          },
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
        }
      }
    }
  });
}

// --- Cluster Feature Profiles Bar Chart ---
function renderProfilesChart(data) {
  const ctx = document.getElementById('clusterProfilesChart').getContext('2d');
  
  if (clusterProfilesChart) {
    clusterProfilesChart.destroy();
  }

  const features = data.feature_names || ['mean_rating', 'std_rating', 'log_review_count', 'avg_review_length'];
  const labels = features.map(f => f.replace(/_/g, ' ').toUpperCase());

  const datasets = data.profiles.map(p => {
    const color = CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length];
    return {
      label: `Cluster ${p.cluster}`,
      data: features.map(f => p[f] ?? 0),
      backgroundColor: color.fill,
      borderColor: color.border,
      borderWidth: 1.5,
      borderRadius: 4
    };
  });

  clusterProfilesChart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#cbd5e1', font: { size: 10 } }
        },
        tooltip: {
          backgroundColor: '#0f172a',
          borderColor: '#334155',
          borderWidth: 1,
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#94a3b8', font: { size: 9, weight: 'bold' } }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', font: { size: 9 } }
        }
      }
    }
  });
}

// --- Market Distribution Doughnut Chart ---
function renderDistributionPieChart(data) {
  const ctx = document.getElementById('clusterPieChart').getContext('2d');
  
  if (clusterPieChart) {
    clusterPieChart.destroy();
  }

  const labels = data.profiles.map(p => `Cluster ${p.cluster}`);
  const counts = data.profiles.map(p => p.size);
  const colors = data.profiles.map(p => CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length].border);

  clusterPieChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: counts,
        backgroundColor: colors,
        borderColor: '#0f172a',
        borderWidth: 2,
        hoverOffset: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#cbd5e1', boxWidth: 10, font: { size: 10 } }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const count = context.raw;
              const total = data.sample_size;
              const pct = ((count / total) * 100).toFixed(1);
              return `${context.label}: ${count} businesses (${pct}%)`;
            }
          }
        }
      },
      cutout: '65%'
    }
  });
}

// --- Archetype Classification & Table Population ---
function getArchetypeInsight(profile) {
  // Heuristic rule engine for business pattern archetypes
  const rating = profile.mean_rating ?? 0;
  const std = profile.std_rating ?? 0;
  const vol = profile.log_review_count ?? 0;

  if (rating >= 4.4 && vol >= 2.5) {
    return {
      title: 'High-Volume Champions',
      badge: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
      desc: 'Top customer satisfaction backed by strong community review volume.'
    };
  } else if (std >= 0.9) {
    return {
      title: 'Polarized / Volatile Service',
      badge: 'bg-rose-500/10 text-rose-300 border-rose-500/30',
      desc: 'Bimodal customer opinions with wide discrepancy between satisfied and upset users.'
    };
  } else if (rating >= 4.3 && vol < 2.5) {
    return {
      title: 'Emerging High-Quality Niches',
      badge: 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30',
      desc: 'Boutique or newly opened services with pristine ratings but lower volume.'
    };
  } else if (rating < 3.0) {
    return {
      title: 'At-Risk / Low Satisfaction',
      badge: 'bg-red-500/10 text-red-400 border-red-500/30',
      desc: 'Persistent complaints, urgent service restructuring or reputation recovery needed.'
    };
  } else {
    return {
      title: 'Consistent Mid-Tier Providers',
      badge: 'bg-indigo-500/10 text-indigo-300 border-indigo-500/30',
      desc: 'Dependable standard performance with moderate review traction.'
    };
  }
}

function populateProfilesTable(data) {
  profilesTableBody.innerHTML = '';

  data.profiles.forEach(p => {
    const archetype = getArchetypeInsight(p);
    const color = CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length];

    const tr = document.createElement('tr');
    tr.className = 'hover:bg-slate-800/40 transition';

    tr.innerHTML = `
      <td class="py-3 px-3">
        <span class="inline-flex items-center space-x-1.5 font-bold" style="color: ${color.border};">
          <span class="w-2.5 h-2.5 rounded-full" style="background-color: ${color.border};"></span>
          <span>Cluster ${p.cluster}</span>
        </span>
      </td>
      <td class="py-3 px-3">
        <div class="space-y-0.5">
          <span class="inline-block px-2 py-0.5 rounded text-[10px] font-semibold border ${archetype.badge}">
            ${archetype.title}
          </span>
          <p class="text-[10px] text-slate-500 font-sans">${archetype.desc}</p>
        </div>
      </td>
      <td class="py-3 px-3 text-right text-slate-200 font-semibold">${p.size}</td>
      <td class="py-3 px-3 text-right text-slate-300">${p.percentage}%</td>
      <td class="py-3 px-3 text-right text-slate-200 font-semibold">${p.mean_rating !== undefined ? p.mean_rating.toFixed(2) : '&mdash;'}</td>
      <td class="py-3 px-3 text-right text-slate-300">${p.std_rating !== undefined ? p.std_rating.toFixed(2) : '&mdash;'}</td>
      <td class="py-3 px-3 text-right text-slate-300">${p.log_review_count !== undefined ? p.log_review_count.toFixed(2) : '&mdash;'}</td>
      <td class="py-3 px-3 text-right text-slate-300">${p.avg_review_length !== undefined ? p.avg_review_length.toFixed(1) : '&mdash;'}</td>
    `;

    profilesTableBody.appendChild(tr);
  });
}

// -------------------------------------------------------------
// Market Trends & Business Venture Decision Analytics
// -------------------------------------------------------------

// --- Market Opportunity vs Risk Quadrant Map (Bubble Chart) ---
function renderMarketMatrixChart(data) {
  const canvas = document.getElementById('marketMatrixChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (marketMatrixChart) {
    marketMatrixChart.destroy();
  }

  const strategicDecisions = data.strategic_decisions || [];
  const datasets = strategicDecisions.map(sd => {
    const color = CLUSTER_COLORS[sd.cluster % CLUSTER_COLORS.length];
    return {
      label: `C${sd.cluster}: ${sd.archetype}`,
      data: [{
        x: sd.metrics_summary ? sd.metrics_summary.volume : 2.0,
        y: sd.metrics_summary ? sd.metrics_summary.rating : 3.5,
        r: Math.max(9, Math.min(32, Math.sqrt(sd.percentage || 10) * 5.2)),
        cluster: sd.cluster,
        archetype: sd.archetype,
        oppScore: sd.opportunity_score,
        pct: sd.percentage,
        venture: sd.venture_title,
        size: sd.size
      }],
      backgroundColor: color.fill,
      borderColor: color.border,
      borderWidth: 2,
      hoverBorderWidth: 3,
      hoverBorderColor: '#ffffff'
    };
  });

  marketMatrixChart = new Chart(ctx, {
    type: 'bubble',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: {
            color: '#cbd5e1',
            boxWidth: 8,
            usePointStyle: true,
            font: { family: 'Plus Jakarta Sans', size: 10 }
          }
        },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#f8fafc',
          bodyColor: '#cbd5e1',
          borderColor: '#334155',
          borderWidth: 1,
          padding: 10,
          callbacks: {
            title: (items) => {
              const d = items[0].raw;
              return `Cluster ${d.cluster} • ${d.archetype}`;
            },
            label: (context) => {
              const d = context.raw;
              return [
                `Opportunity Score: ${d.oppScore}/100`,
                `Market Share: ${d.pct}% (${d.size} businesses)`,
                `Quality (Rating): ${d.y.toFixed(2)} ★`,
                `Demand Volume: ${d.x.toFixed(2)}`,
                `Proposed Venture: ${d.venture}`
              ];
            }
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Market Demand Volume (Log Scale / Review Traction)',
            color: '#94a3b8',
            font: { family: 'Plus Jakarta Sans', size: 11, weight: 'bold' }
          },
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          title: {
            display: true,
            text: 'Customer Satisfaction / Quality (Mean Rating)',
            color: '#94a3b8',
            font: { family: 'Plus Jakarta Sans', size: 11, weight: 'bold' }
          },
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
        }
      }
    }
  });
}

// --- Multi-Vector Strategic Viability Radar Chart ---
function renderVentureRadarChart(data) {
  const canvas = document.getElementById('ventureRadarChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (ventureRadarChart) {
    ventureRadarChart.destroy();
  }

  const strategicDecisions = data.strategic_decisions || [];
  const labels = [
    'Satisfaction',
    'Demand Volume',
    'Consistency',
    'Feedback Depth',
    'Venture Opportunity'
  ];

  const datasets = strategicDecisions.map(sd => {
    const color = CLUSTER_COLORS[sd.cluster % CLUSTER_COLORS.length];
    const vec = sd.radar_vector || {};
    return {
      label: `Cluster ${sd.cluster}`,
      data: [
        vec.satisfaction ?? 50,
        vec.demand ?? 50,
        vec.consistency ?? 50,
        vec.engagement ?? 50,
        vec.opportunity ?? 50
      ],
      backgroundColor: color.fill.replace('0.75', '0.2'),
      borderColor: color.border,
      borderWidth: 2,
      pointBackgroundColor: color.border,
      pointBorderColor: '#ffffff',
      pointHoverRadius: 6
    };
  });

  ventureRadarChart = new Chart(ctx, {
    type: 'radar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          pointLabels: {
            color: '#cbd5e1',
            font: { family: 'Plus Jakarta Sans', size: 10, weight: '600' }
          },
          ticks: {
            display: false,
            backdropColor: 'transparent',
            min: 0,
            max: 100
          }
        }
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#cbd5e1', boxWidth: 8, font: { family: 'Plus Jakarta Sans', size: 10 } }
        },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#f8fafc',
          bodyColor: '#cbd5e1',
          borderColor: '#334155',
          borderWidth: 1
        }
      }
    }
  });
}

// --- Proposed Business Ventures & Real-World Decision Cards ---
function renderStrategicDecisionCards(data, activeFilter = 'all') {
  const container = document.getElementById('strategic-cards-container');
  const tabsContainer = document.getElementById('cluster-filter-tabs');
  if (!container) return;

  const decisions = data.strategic_decisions || [];
  if (decisions.length === 0) {
    container.innerHTML = `
      <div class="col-span-2 text-center py-8 text-slate-500 text-xs">
        No strategic decision profiles generated.
      </div>
    `;
    return;
  }

  // Render Tabs
  if (tabsContainer) {
    let tabsHtml = `
      <button onclick="filterDecisions('all')" class="px-2.5 py-1 rounded-lg text-[11px] font-semibold transition cursor-pointer ${activeFilter === 'all' ? 'bg-indigo-600 text-white shadow-sm' : 'bg-slate-800 text-slate-400 hover:text-white border border-slate-700'}">
        All Clusters (${decisions.length})
      </button>
    `;
    decisions.forEach(d => {
      const color = CLUSTER_COLORS[d.cluster % CLUSTER_COLORS.length];
      const isAct = activeFilter === String(d.cluster);
      tabsHtml += `
        <button onclick="filterDecisions('${d.cluster}')" class="px-2.5 py-1 rounded-lg text-[11px] font-semibold transition cursor-pointer flex items-center space-x-1 ${isAct ? 'bg-slate-700 text-white ring-1 ring-white/20' : 'bg-slate-800 text-slate-400 hover:text-white border border-slate-700'}">
          <span class="w-2 h-2 rounded-full" style="background-color: ${color.border};"></span>
          <span>C${d.cluster}</span>
        </button>
      `;
    });
    tabsContainer.innerHTML = tabsHtml;
  }

  // Filter Cards
  const filtered = activeFilter === 'all' ? decisions : decisions.filter(d => String(d.cluster) === activeFilter);

  let cardsHtml = '';
  filtered.forEach(d => {
    const color = CLUSTER_COLORS[d.cluster % CLUSTER_COLORS.length];
    
    let oppBadgeClass = 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30';
    if (d.opportunity_score >= 85) oppBadgeClass = 'bg-rose-500/20 text-rose-300 border-rose-500/30 animate-pulse';
    else if (d.opportunity_score >= 70) oppBadgeClass = 'bg-amber-500/20 text-amber-300 border-amber-500/30';

    cardsHtml += `
      <div class="rounded-2xl bg-slate-900/90 border border-slate-800 hover:border-slate-700 transition shadow-xl p-5 space-y-4 flex flex-col justify-between">
        
        <!-- Header -->
        <div class="space-y-2">
          <div class="flex items-center justify-between flex-wrap gap-2">
            <div class="flex items-center space-x-2">
              <span class="w-3 h-3 rounded-full shadow" style="background-color: ${color.border};"></span>
              <span class="text-xs font-bold uppercase tracking-wider text-white">Cluster ${d.cluster}</span>
              <span class="px-2 py-0.5 rounded text-[10px] font-semibold border ${d.archetype_badge}">
                ${d.archetype}
              </span>
            </div>
            <div class="flex items-center space-x-1.5">
              <span class="px-2.5 py-0.5 rounded-full text-[10px] font-bold border ${oppBadgeClass}">
                <i class="fa-solid fa-fire-flame-curved mr-1 text-amber-400"></i>Opp: ${d.opportunity_score}/100
              </span>
            </div>
          </div>

          <!-- Proposed Business Venture Box -->
          <div class="rounded-xl bg-gradient-to-r from-slate-950 via-slate-950 to-indigo-950/40 border border-indigo-900/40 p-3 space-y-1 shadow-inner">
            <div class="flex items-center justify-between text-[11px] flex-wrap gap-1">
              <span class="text-indigo-300 font-bold flex items-center space-x-1.5">
                <i class="fa-solid fa-rocket text-amber-400"></i>
                <span>Proposed Venture: ${d.venture_title}</span>
              </span>
              <span class="text-[10px] text-emerald-400 font-mono font-semibold bg-emerald-950/50 px-1.5 py-0.5 rounded border border-emerald-800/40">${d.roi_potential}</span>
            </div>
            <p class="text-[11px] text-slate-300 leading-relaxed font-sans mt-0.5">
              ${d.venture_concept}
            </p>
          </div>
        </div>

        <!-- Empirical Decisions List -->
        <div class="space-y-2">
          <h5 class="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
            <i class="fa-solid fa-list-check text-cyan-400"></i>
            <span>Real-World Statistical Decisions</span>
          </h5>
          <ul class="space-y-1.5 text-[11px] text-slate-300">
            ${d.decisions.map(dec => {
              let icon = 'fa-solid fa-arrow-right text-indigo-400';
              if (dec.startsWith('Capital Allocation')) icon = 'fa-solid fa-coins text-amber-400';
              else if (dec.startsWith('Pricing')) icon = 'fa-solid fa-tags text-emerald-400';
              else if (dec.startsWith('Operational')) icon = 'fa-solid fa-gear text-cyan-400';
              else if (dec.startsWith('Marketing')) icon = 'fa-solid fa-bullhorn text-pink-400';
              else if (dec.startsWith('Tech')) icon = 'fa-solid fa-laptop-code text-indigo-400';
              else if (dec.startsWith('Acquisition')) icon = 'fa-solid fa-handshake text-purple-400';
              return `
                <li class="flex items-start space-x-2 bg-slate-800/40 p-2 rounded-lg border border-slate-800/80">
                  <i class="${icon} text-xs mt-0.5 shrink-0"></i>
                  <span class="leading-relaxed">${dec}</span>
                </li>
              `;
            }).join('')}
          </ul>
        </div>

        <!-- Bottom Metrics Strip -->
        <div class="pt-3 border-t border-slate-800/80 flex items-center justify-between text-[10px] text-slate-400 font-mono flex-wrap gap-1">
          <span>Quality: <strong class="text-slate-200">${d.metrics_summary ? d.metrics_summary.rating : '3.5'} ★</strong></span>
          <span>Demand: <strong class="text-slate-200">${d.metrics_summary ? d.metrics_summary.volume : '2.0'}</strong></span>
          <span>Volatility: <strong class="text-slate-200">${d.metrics_summary ? d.metrics_summary.volatility : '0.5'}</strong></span>
          <span>Market Share: <strong class="text-indigo-300 font-bold">${d.percentage}%</strong></span>
        </div>

      </div>
    `;
  });

  container.innerHTML = cardsHtml;
}

window.filterDecisions = function(filterVal) {
  if (currentClusteringData) {
    renderStrategicDecisionCards(currentClusteringData, filterVal);
  }
};

window.copyStrategicBrief = function() {
  if (!currentClusteringData || !currentClusteringData.strategic_decisions) {
    alert('Please execute the clustering pipeline first.');
    return;
  }
  const mt = currentClusteringData.market_trends || {};
  let brief = `# CITY SERVICE CLUSTERING - STRATEGIC EXECUTIVE BRIEF\n`;
  brief += `Generated: ${new Date().toLocaleString()}\n`;
  brief += `Market Health Index: ${mt.market_health_score}/100 | Avg Rating: ${mt.avg_market_rating}★\n`;
  brief += `Top Identified Venture: ${mt.top_venture_target} (Score: ${mt.top_opportunity_score}/100)\n\n`;
  brief += `## REGIONAL MARKET GAP ANALYSIS\n${mt.market_gap_analysis}\n\n`;
  brief += `## CLUSTER VENTURE PROPOSALS & ACTIONABLE DECISIONS\n`;

  currentClusteringData.strategic_decisions.forEach(d => {
    brief += `\n### [Cluster ${d.cluster}] ${d.archetype} (${d.percentage}% Market Share)\n`;
    brief += `Venture: ${d.venture_title} (Opportunity Score: ${d.opportunity_score}/100 | ROI: ${d.roi_potential})\n`;
    brief += `Concept: ${d.venture_concept}\n`;
    brief += `Decisions:\n`;
    d.decisions.forEach(dec => {
      brief += `- ${dec}\n`;
    });
  });

  navigator.clipboard.writeText(brief).then(() => {
    const btnText = document.getElementById('btn-copy-text');
    if (btnText) {
      const orig = btnText.textContent;
      btnText.textContent = 'Copied to Clipboard!';
      setTimeout(() => { btnText.textContent = orig; }, 2500);
    }
  }).catch(() => {
    alert('Failed to copy. Please allow clipboard permissions.');
  });
};

window.exportStrategicJSON = function() {
  if (!currentClusteringData || !currentClusteringData.strategic_decisions) {
    alert('Please execute the clustering pipeline first.');
    return;
  }
  const exportPayload = {
    timestamp: new Date().toISOString(),
    dataset_schema: currentClusteringData.schema_detected,
    sample_size: currentClusteringData.sample_size,
    k_clusters: currentClusteringData.k_clusters,
    metrics: currentClusteringData.metrics,
    market_trends: currentClusteringData.market_trends,
    strategic_decisions: currentClusteringData.strategic_decisions
  };

  const blob = new Blob([JSON.stringify(exportPayload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `market_venture_decisions_k${currentClusteringData.k_clusters}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};
