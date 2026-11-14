/**
 * Analytics, Real-World Location Mapping & GPS Directions Dashboard
 * Manages Leaflet Geospatial Intelligence Map, OSRM Live GPS Routing API,
 * Turn-by-Turn Road Navigation, Business Sector & District Breakdown,
 * Real-World Entity Directory Table, and Strategic Decision Playbooks.
 */

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

// Global Instances & Data
let marketMatrixChart = null;
let ventureRadarChart = null;
let leafletMap = null;
let markersLayer = null;
let activeTileLayer = null;
let mapBaseLayers = {};
let activeBasemapKey = 'hybrid';
let mapMarkerRegistry = {};

// Real GPS Directions & Navigation State
let userGpsPosition = null; // { lat, lng, isDetected: boolean }
const DEFAULT_ORIGIN = { lat: 40.0175, lng: -105.2750, name: "Downtown Civic Center" };
let activeDestination = null; // { lat, lng, name, address, id, googleMapsUrl }
let activeTravelMode = 'driving';
let currentRoutePolyline = null;
let currentRouteCasing = null;
let userLocationMarker = null;
let destinationFlagMarker = null;

let currentAnalyticsData = null;
let currentActiveFilter = 'all';
let allEntities = [];
let filteredEntities = [];
let currentEntityPage = 1;
const ENTITIES_PER_PAGE = 12;

// DOM Elements
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const latencyTag = document.getElementById('latency-tag');

document.addEventListener('DOMContentLoaded', () => {
  checkWorkerHealth();
  setInterval(checkWorkerHealth, 30000);
  loadAnalyticsData();
});

// -------------------------------------------------------------
// Engine Ping
// -------------------------------------------------------------
async function checkWorkerHealth() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.status === 'online') {
      if (statusDot) statusDot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400';
      if (statusText) statusText.textContent = data.engine_type || 'Engine Online';
      if (latencyTag) {
        latencyTag.textContent = `${data.latency_ms || 1.0}ms`;
        latencyTag.classList.remove('hidden');
      }
    } else {
      if (statusDot) statusDot.className = 'w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse';
      if (statusText) statusText.textContent = 'Fallback Active';
    }
  } catch (e) {
    if (statusDot) statusDot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400';
    if (statusText) statusText.textContent = 'Integrated Engine';
  }
}

// -------------------------------------------------------------
// Data Loader (Persistent from Ingestion or Auto-Fetch)
// -------------------------------------------------------------
function getCityServiceClientId() {
  let id = localStorage.getItem('city_service_client_id');
  if (!id) {
    id = 'c_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 10);
    localStorage.setItem('city_service_client_id', id);
  }
  return id;
}

function isUsableResult(data) {
  return !!(data && data.status !== 'error' && Array.isArray(data.strategic_decisions) && data.strategic_decisions.length > 0);
}

function showAnalyticsStatus(message, isError) {
  const elGapText = document.getElementById('market-gap-text');
  if (elGapText) {
    elGapText.textContent = message;
    elGapText.classList.toggle('text-rose-300', !!isError);
  }
  const ctxDatasetName = document.getElementById('ctx-dataset-name');
  if (ctxDatasetName && isError) ctxDatasetName.textContent = 'No analysis loaded';
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 120000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(url, Object.assign({}, options, { signal: ctrl.signal }));
  } finally {
    clearTimeout(timer);
  }
}

async function loadAnalyticsData() {
  showAnalyticsStatus('Loading clustering results from the ML pipeline...');

  // 1. Server-side hand-off (works even when results exceed the localStorage quota)
  try {
    const res = await fetchWithTimeout(`/api/latest-results?client_id=${encodeURIComponent(getCityServiceClientId())}`, {}, 15000);
    if (res.ok) {
      const data = await res.json();
      if (isUsableResult(data)) {
        renderAnalyticsView(data, data.dataset_name || localStorage.getItem('city_service_dataset_name') || 'Uploaded Dataset');
        return;
      }
    }
  } catch (e) {
    console.warn('Server result hand-off unavailable:', e);
  }

  // 2. Browser cache from the Ingestion page
  const cached = localStorage.getItem('city_service_ml_results');
  if (cached) {
    try {
      const data = JSON.parse(cached);
      if (isUsableResult(data)) {
        renderAnalyticsView(data, localStorage.getItem('city_service_dataset_name') || 'Custom Uploaded Dataset');
        return;
      }
    } catch (e) {
      console.warn('Invalid cached ML results:', e);
    }
  }

  // 3. Nothing found: run the bundled GPS sample so the page is never empty
  await triggerSampleClustering();
}

async function triggerSampleClustering() {
  const ctxDatasetName = document.getElementById('ctx-dataset-name');
  if (ctxDatasetName) ctxDatasetName.textContent = 'Analyzing Sample City Reviews...';
  showAnalyticsStatus('No clustering run found for this session. Running ML clustering on the sample dataset...');

  try {
    const formData = new FormData();
    formData.append('k', '4');
    formData.append('use_sample', 'true');
    formData.append('sample_type', 'reviews');
    formData.append('client_id', getCityServiceClientId());
    const workerUrl = localStorage.getItem('colab_worker_url') || localStorage.getItem('hf_worker_url');
    if (workerUrl) formData.append('worker_url', workerUrl);

    const res = await fetchWithTimeout('/api/train', { method: 'POST', body: formData }, 120000);
    const data = await res.json();
    if (isUsableResult(data)) {
      localStorage.setItem('city_service_dataset_name', 'sample_reviews.csv (Sample, K=4)');
      try { localStorage.setItem('city_service_ml_results', JSON.stringify(data)); } catch (e) { /* quota */ }
      renderAnalyticsView(data, 'sample_reviews.csv (Sample Reviews)');
    } else {
      console.error('Clustering error:', data.message);
      showAnalyticsStatus(`Clustering failed: ${data.message || 'no strategic decisions returned'}. Run the pipeline on the Dataset & PCA page.`, true);
    }
  } catch (e) {
    console.error('Fetch error:', e);
    showAnalyticsStatus(`Could not reach the ML engine (${e.name === 'AbortError' ? 'timed out' : e.message}). Run the pipeline on the Dataset & PCA page and try again.`, true);
  }
}

// -------------------------------------------------------------
// Main View Renderer
// -------------------------------------------------------------
function renderAnalyticsView(data, datasetName) {
  currentAnalyticsData = data;
  allEntities = data.entity_locations || [];

  // 1. Context Bar
  const ctxDatasetName = document.getElementById('ctx-dataset-name');
  const ctxClusterMeta = document.getElementById('ctx-cluster-meta');
  if (ctxDatasetName) ctxDatasetName.textContent = datasetName || 'Analyzed Review Dataset';
  if (ctxClusterMeta) ctxClusterMeta.textContent = `${data.sample_size} businesses • ${data.k_clusters} clusters (K=${data.k_clusters})`;

  // 2. KPI Ribbon
  const mt = data.market_trends || {};
  const elHealth = document.getElementById('kpi-market-health');
  const elAvgR = document.getElementById('kpi-avg-rating');
  const elTopOpp = document.getElementById('kpi-top-opp-score');
  const elPrimeTarget = document.getElementById('kpi-prime-target-badge');
  const elVentureName = document.getElementById('kpi-top-venture-name');
  const elArchetype = document.getElementById('kpi-top-archetype');

  if (elHealth) elHealth.textContent = `${mt.market_health_score ?? 80}/100`;
  if (elAvgR) elAvgR.textContent = `${mt.avg_market_rating ?? 4.0} ★`;
  if (elTopOpp) elTopOpp.textContent = `${mt.top_opportunity_score ?? 90}/100`;
  if (elPrimeTarget) elPrimeTarget.textContent = `Cluster ${mt.prime_cluster_target ?? 0} Prime Target`;
  if (elVentureName) {
    elVentureName.textContent = mt.top_venture_target || 'Venture Expansion';
    elVentureName.title = mt.top_venture_target || '';
  }
  if (elArchetype) elArchetype.textContent = mt.prime_archetype || 'Strategic Recommendation';

  // 3. Regional Market Gap Banner
  const elGapText = document.getElementById('market-gap-text');
  if (elGapText) {
    elGapText.textContent = mt.market_gap_analysis || 'No significant satisfaction deficit detected. Primary opportunity lies in boutique digital scaling.';
  }

  // 4-6. Render each section independently so one failure never blanks the page
  const sections = [
    ['Market Matrix Chart', () => renderMarketMatrixChart(data)],
    ['Venture Radar Chart', () => renderVentureRadarChart(data)],
    ['GPS Map', () => initOrUpdateLeafletMap(data)],
    ['Sector Breakdown', () => renderSectorsBreakdown(data)],
    ['District Breakdown', () => renderDistrictsBreakdown(data)],
    ['Entity Directory', () => initEntityDirectory(data)],
    ['Strategic Decision Cards', () => renderStrategicDecisionCards(data, 'all')],
  ];
  sections.forEach(([name, fn]) => {
    try { fn(); } catch (e) { console.error(`[Analytics] ${name} failed to render:`, e); }
  });
}

// -------------------------------------------------------------
// Geospatial Leaflet Map Integration with Real-World GPS Pins
// (Google Satellite Hybrid / Aerial / Streets / Terrain + Esri World Imagery)
// -------------------------------------------------------------
function buildMapBaseLayers() {
  return {
    // Google Maps Satellite + Road Labels Hybrid (lyrs=y) — no API key required
    hybrid: L.tileLayer('https://{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', {
      maxZoom: 20,
      subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
      attribution: '&copy; Google Maps Satellite'
    }),
    // Pure Satellite / Aerial (Esri World Imagery + Google Satellite)
    satellite: L.tileLayer('https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
      maxZoom: 20,
      subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
      attribution: '&copy; Google Earth / Esri World Imagery'
    }),
    // Google Maps Standard Streets & Places View (lyrs=m)
    streets: L.tileLayer('https://{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', {
      maxZoom: 20,
      subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
      attribution: '&copy; Google Maps'
    }),
    // Geological / Topographical Terrain View (lyrs=p)
    terrain: L.tileLayer('https://{s}.google.com/vt/lyrs=p&x={x}&y={y}&z={z}', {
      maxZoom: 20,
      subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
      attribution: '&copy; Google Terrain / OpenStreetMap'
    }),
    // Esri World Imagery + OpenStreetMap fallback if Google tile servers are blocked on network
    esri_fallback: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 19,
      attribution: 'Tiles &copy; Esri'
    })
  };
}

window.switchMapBasemap = function(mode) {
  if (!leafletMap || !mapBaseLayers[mode]) return;
  if (activeTileLayer) {
    leafletMap.removeLayer(activeTileLayer);
  }
  activeTileLayer = mapBaseLayers[mode];
  activeTileLayer.addTo(leafletMap);
  activeBasemapKey = mode;

  ['hybrid', 'satellite', 'streets', 'terrain'].forEach(k => {
    const btn = document.getElementById(`basemap-btn-${k}`);
    if (!btn) return;
    if (k === mode) {
      btn.className = 'px-2.5 py-1 rounded-md bg-emerald-600/30 text-emerald-200 border border-emerald-500/40 flex items-center space-x-1 cursor-pointer transition';
    } else {
      btn.className = 'px-2.5 py-1 rounded-md text-slate-400 hover:text-white flex items-center space-x-1 cursor-pointer transition';
    }
  });
};

function initOrUpdateLeafletMap(data) {
  const mapContainer = document.getElementById('opportunity-map');
  if (!mapContainer || typeof L === 'undefined') return;

  const entities = data.entity_locations || [];
  const mapPinCount = document.getElementById('map-pin-count');
  if (mapPinCount) mapPinCount.textContent = entities.length;

  if (!leafletMap) {
    // Initial map position from dataset coordinates
    const initLat = entities.length > 0 && entities[0].lat ? entities[0].lat : 40.0175;
    const initLng = entities.length > 0 && entities[0].lng ? entities[0].lng : -105.2750;

    leafletMap = L.map('opportunity-map', {
      zoomControl: true,
      attributionControl: true
    }).setView([initLat, initLng], 14);

    mapBaseLayers = buildMapBaseLayers();
    activeTileLayer = mapBaseLayers.hybrid;
    activeTileLayer.on('tileerror', () => {
      // Automatic fallback to Esri World Satellite Imagery if network blocks mt0..3.google.com
      if (activeBasemapKey === 'hybrid' && activeTileLayer !== mapBaseLayers.esri_fallback) {
        leafletMap.removeLayer(activeTileLayer);
        activeTileLayer = mapBaseLayers.esri_fallback;
        activeTileLayer.addTo(leafletMap);
      }
    });
    activeTileLayer.addTo(leafletMap);

    markersLayer = L.layerGroup().addTo(leafletMap);
  }

  plotMapMarkers(entities);
}

function plotMapMarkers(entities, filterCluster = 'all', filterDistrict = 'all') {
  if (!markersLayer || !leafletMap) return;
  markersLayer.clearLayers();
  mapMarkerRegistry = {};

  const filtered = entities.filter(e => {
    const matchC = filterCluster === 'all' || String(e.cluster) === String(filterCluster);
    const matchD = filterDistrict === 'all' || e.district === filterDistrict;
    return matchC && matchD;
  });

  const bounds = [];

  filtered.forEach(e => {
    if (!e.lat || !e.lng) return;
    bounds.push([e.lat, e.lng]);

    const color = CLUSTER_COLORS[e.cluster % CLUSTER_COLORS.length];
    const isPrimeDisruption = e.opportunity_score >= 88;

    // Custom SVG Pin Icon
    const customIcon = L.divIcon({
      className: 'custom-map-pin',
      html: `
        <div style="position: relative; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;">
          ${isPrimeDisruption ? `<div class="pin-pulse" style="position: absolute; width: 24px; height: 24px; border-radius: 50%; background: ${color.border}; opacity: 0.5;"></div>` : ''}
          <div style="width: 20px; height: 20px; border-radius: 50%; background: #0f172a; border: 3px solid ${color.border}; box-shadow: 0 0 10px ${color.glow}; display: flex; align-items: center; justify-content: center;">
            <div style="width: 6px; height: 6px; border-radius: 50%; background: #ffffff;"></div>
          </div>
        </div>
      `,
      iconSize: [28, 28],
      iconAnchor: [14, 14]
    });

    const marker = L.marker([e.lat, e.lng], { icon: customIcon });

    const safeName = (e.name || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
    const safeAddr = (e.address || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
    const gmapsUrl = e.google_maps_url || `https://www.google.com/maps/dir/?api=1&destination=${e.lat},${e.lng}`;
    const gmapsSatUrl = `https://www.google.com/maps/@${e.lat},${e.lng},19z/data=!3m1!1e3`;

    // Glassmorphism Rich Popup with GPS Routing Trigger
    const popupHtml = `
      <div class="space-y-2 text-xs" style="min-width: 240px;">
        <div class="flex items-center justify-between border-b border-slate-700/80 pb-1.5">
          <span class="font-bold text-white tracking-tight">${e.name}</span>
          <span class="px-1.5 py-0.5 rounded text-[10px] font-bold" style="background: ${color.fill}; color: #ffffff;">
            C${e.cluster}
          </span>
        </div>
        
        <div class="space-y-1 text-slate-300 text-[11px]">
          <div class="flex items-center space-x-1.5 text-emerald-300 font-semibold">
            <i class="fa-solid fa-tag text-[10px]"></i>
            <span>${e.business_type}</span>
          </div>
          <div class="flex items-start space-x-1.5 text-slate-400">
            <i class="fa-solid fa-location-dot text-[10px] mt-0.5 shrink-0 text-cyan-400"></i>
            <span>${e.address}</span>
          </div>
          <div class="font-mono text-[9px] text-cyan-300 flex items-center justify-between">
            <span class="flex items-center space-x-1">
              <i class="fa-solid fa-satellite text-[8px] text-emerald-400"></i>
              <span>Lat/Lng: ${e.lat.toFixed(5)}, ${e.lng.toFixed(5)}</span>
            </span>
            ${e.has_real_gps ? '<span class="px-1 py-0.2 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-[8px]">GPS Verified</span>' : ''}
          </div>
        </div>

        <div class="grid grid-cols-2 gap-1.5 py-1.5 border-t border-b border-slate-800 text-[10px] font-mono">
          <div class="bg-slate-900/80 p-1 rounded text-center">
            <span class="text-slate-400">Rating:</span>
            <strong class="text-amber-400 ml-1">${e.rating}★</strong>
          </div>
          <div class="bg-slate-900/80 p-1 rounded text-center">
            <span class="text-slate-400">Opp Score:</span>
            <strong class="text-emerald-400 ml-1">${e.opportunity_score}</strong>
          </div>
        </div>

        <div class="bg-slate-900/90 p-2 rounded-lg border border-slate-800 text-[10px] text-slate-300 leading-snug">
          <strong class="text-cyan-300 block mb-0.5"><i class="fa-solid fa-bolt mr-1"></i>Investment Play:</strong>
          ${e.investment_action}
        </div>

        <!-- Real GPS Route & Google Maps Satellite Buttons -->
        <div class="grid grid-cols-2 gap-1.5 pt-1 border-t border-slate-800">
          <button onclick="requestGpsRoute(${e.lat}, ${e.lng}, '${safeName}', '${safeAddr}', '${e.id}', '${gmapsUrl}')" class="py-1.5 px-2 rounded-lg bg-emerald-600/30 hover:bg-emerald-600/50 text-emerald-200 border border-emerald-500/40 text-[10px] font-bold text-center flex items-center justify-center space-x-1 cursor-pointer">
            <i class="fa-solid fa-diamond-turn-right text-emerald-400"></i>
            <span>Get Directions</span>
          </button>
          <a href="${gmapsSatUrl}" target="_blank" class="py-1.5 px-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-[10px] font-semibold text-center flex items-center justify-center space-x-1 transition">
            <i class="fa-solid fa-satellite text-amber-400"></i>
            <span>Google Satellite</span>
          </a>
        </div>
      </div>
    `;

    marker.bindPopup(popupHtml);
    marker.addTo(markersLayer);
    mapMarkerRegistry[e.id] = marker;
  });

  const mapPinCount = document.getElementById('map-pin-count');
  if (mapPinCount) mapPinCount.textContent = filtered.length;

  if (bounds.length > 0) {
    leafletMap.fitBounds(bounds, { padding: [35, 35], maxZoom: 17 });
  }
}

window.onMapFilterChange = function() {
  const cSel = document.getElementById('map-filter-cluster');
  const dSel = document.getElementById('map-filter-district');
  const cVal = cSel ? cSel.value : 'all';
  const dVal = dSel ? dSel.value : 'all';
  plotMapMarkers(allEntities, cVal, dVal);
};

window.resetMapView = function() {
  const cSel = document.getElementById('map-filter-cluster');
  const dSel = document.getElementById('map-filter-district');
  if (cSel) cSel.value = 'all';
  if (dSel) dSel.value = 'all';
  plotMapMarkers(allEntities, 'all', 'all');
  if (allEntities.length > 0) {
    const pts = allEntities.filter(e => e.lat && e.lng).map(e => [e.lat, e.lng]);
    if (pts.length > 0) leafletMap.fitBounds(pts, { padding: [30, 30] });
  }
};

window.flyToEntity = function(lat, lng, entityId) {
  if (!leafletMap) return;
  leafletMap.flyTo([lat, lng], 16, { animate: true, duration: 1.2 });
  const mapSection = document.getElementById('opportunity-map');
  if (mapSection) {
    mapSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
  setTimeout(() => {
    if (mapMarkerRegistry[entityId]) {
      mapMarkerRegistry[entityId].openPopup();
    }
  }, 1300);
};

// -------------------------------------------------------------
// Real-World GPS Routing & Directions Integration (OSRM API)
// -------------------------------------------------------------
window.useBrowserGpsLocation = function() {
  const btn = document.getElementById('btn-use-my-gps');
  const label = document.getElementById('route-origin-label');

  if (!navigator.geolocation) {
    alert("Geolocation is not supported by your browser. Using municipal center as default origin.");
    return;
  }

  if (btn) btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin text-[10px]"></i><span>Locating...</span>`;

  navigator.geolocation.getCurrentPosition(
    (pos) => {
      userGpsPosition = {
        lat: pos.coords.latitude,
        lng: pos.coords.longitude,
        isDetected: true
      };

      if (label) {
        label.textContent = `My GPS Position (${userGpsPosition.lat.toFixed(4)}, ${userGpsPosition.lng.toFixed(4)})`;
      }
      if (btn) {
        btn.innerHTML = `<i class="fa-solid fa-satellite-dish text-emerald-400 text-[10px]"></i><span class="text-emerald-300">GPS Locked</span>`;
      }

      plotUserLocationOnMap(userGpsPosition.lat, userGpsPosition.lng);

      if (activeDestination) {
        requestGpsRoute(
          activeDestination.lat,
          activeDestination.lng,
          activeDestination.name,
          activeDestination.address,
          activeDestination.id,
          activeDestination.googleMapsUrl
        );
      }
    },
    (err) => {
      console.warn("GPS Geolocation denied or unavailable:", err.message);
      if (btn) btn.innerHTML = `<i class="fa-solid fa-crosshairs text-[10px]"></i><span>Detect My GPS</span>`;
      alert(`Could not acquire GPS position (${err.message}). Using Downtown Civic Center as starting origin.`);
      userGpsPosition = { ...DEFAULT_ORIGIN, isDetected: false };
      if (label) label.textContent = DEFAULT_ORIGIN.name;
    },
    { enableHighAccuracy: true, timeout: 8000, maximumAge: 60000 }
  );
};

function plotUserLocationOnMap(lat, lng) {
  if (!leafletMap) return;
  if (userLocationMarker) {
    leafletMap.removeLayer(userLocationMarker);
  }

  const userIcon = L.divIcon({
    className: 'user-gps-marker',
    html: `
      <div style="position: relative; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center;">
        <div class="user-gps-pulse"></div>
        <div class="user-gps-dot"></div>
      </div>
    `,
    iconSize: [32, 32],
    iconAnchor: [16, 16]
  });

  userLocationMarker = L.marker([lat, lng], { icon: userIcon, zIndexOffset: 1000 }).addTo(leafletMap);
  userLocationMarker.bindPopup(`<strong class="text-cyan-300 text-xs"><i class="fa-solid fa-crosshairs mr-1"></i>Your Current GPS Position</strong><div class="text-[10px] text-slate-400 font-mono mt-0.5">${lat.toFixed(5)}, ${lng.toFixed(5)}</div>`);
}

window.requestGpsRoute = async function(destLat, destLng, destName, destAddr, entityId, googleMapsUrl) {
  activeDestination = {
    lat: destLat,
    lng: destLng,
    name: destName,
    address: destAddr,
    id: entityId,
    googleMapsUrl: googleMapsUrl || `https://www.google.com/maps/dir/?api=1&destination=${destLat},${destLng}`
  };

  const panel = document.getElementById('gps-directions-panel');
  if (panel) panel.classList.remove('hidden');

  // Scroll to map smoothly
  const mapSection = document.getElementById('opportunity-map');
  if (mapSection) mapSection.scrollIntoView({ behavior: 'smooth', block: 'center' });

  const nameEl = document.getElementById('route-dest-name');
  const addrEl = document.getElementById('route-dest-addr');
  const gmapsBtn = document.getElementById('btn-open-google-maps');
  const stepsContainer = document.getElementById('route-steps-container');
  const distEl = document.getElementById('route-distance-val');
  const durEl = document.getElementById('route-duration-val');

  if (nameEl) nameEl.textContent = destName;
  if (addrEl) addrEl.textContent = `${destAddr} (${destLat.toFixed(4)}, ${destLng.toFixed(4)})`;
  if (gmapsBtn) gmapsBtn.href = activeDestination.googleMapsUrl;

  if (stepsContainer) {
    stepsContainer.innerHTML = `
      <div class="text-center py-4 text-slate-400 text-xs flex items-center justify-center space-x-2">
        <i class="fa-solid fa-satellite fa-spin text-emerald-400"></i>
        <span>Calculating GPS road navigation route...</span>
      </div>
    `;
  }

  // Determine origin
  const origin = userGpsPosition || DEFAULT_ORIGIN;
  plotUserLocationOnMap(origin.lat, origin.lng);

  try {
    const apiUrl = `/api/directions?start_lat=${origin.lat}&start_lng=${origin.lng}&end_lat=${destLat}&end_lng=${destLng}&mode=${activeTravelMode}`;
    const res = await fetch(apiUrl);
    const data = await res.json();

    if (data.status === 'success') {
      if (distEl) distEl.textContent = `${data.distance_miles} mi (${data.distance_km} km)`;
      if (durEl) durEl.textContent = `${data.duration_mins} mins`;

      if (gmapsBtn && data.google_maps_url) {
        gmapsBtn.href = data.google_maps_url;
      }

      // Draw route on Leaflet
      if (leafletMap && data.geometry && data.geometry.length > 0) {
        if (currentRoutePolyline) leafletMap.removeLayer(currentRoutePolyline);
        if (currentRouteCasing) leafletMap.removeLayer(currentRouteCasing);
        if (destinationFlagMarker) leafletMap.removeLayer(destinationFlagMarker);

        // 1. Route Casing for contrast
        currentRouteCasing = L.polyline(data.geometry, {
          color: '#0369a1',
          weight: 8,
          opacity: 0.6,
          className: 'gps-route-casing'
        }).addTo(leafletMap);

        // 2. Route Foreground (Neon Cyan)
        currentRoutePolyline = L.polyline(data.geometry, {
          color: '#38bdf8',
          weight: 5,
          opacity: 0.95,
          className: 'gps-route-line'
        }).addTo(leafletMap);

        // 3. Destination Flag Marker
        const flagIcon = L.divIcon({
          className: 'dest-flag-pin',
          html: `
            <div style="background: #e11d48; width: 26px; height: 26px; border-radius: 50%; border: 2px solid #ffffff; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 12px rgba(225, 29, 72, 0.8);">
              <i class="fa-solid fa-flag-checkered text-white text-[11px]"></i>
            </div>
          `,
          iconSize: [26, 26],
          iconAnchor: [13, 26]
        });

        destinationFlagMarker = L.marker([destLat, destLng], { icon: flagIcon, zIndexOffset: 900 }).addTo(leafletMap);
        destinationFlagMarker.bindPopup(`
          <div class="text-xs space-y-1">
            <strong class="text-white">${destName}</strong>
            <div class="text-[10px] text-slate-400">${destAddr}</div>
            <div class="text-[10px] text-emerald-400 font-bold mt-1">
              <i class="fa-solid fa-route mr-1"></i>${data.distance_miles} mi &bull; ${data.duration_mins} mins
            </div>
          </div>
        `);

        // Fit map bounds to show complete driving route
        leafletMap.fitBounds(currentRoutePolyline.getBounds(), { padding: [50, 50] });
      }

      // Render Steps in Panel
      if (stepsContainer) {
        if (!data.steps || data.steps.length === 0) {
          stepsContainer.innerHTML = `<div class="text-center py-2 text-slate-400 text-xs">Direct GPS route calculated.</div>`;
        } else {
          let sHtml = '';
          data.steps.forEach((s, sIdx) => {
            let mIcon = 'fa-solid fa-arrow-up';
            const mod = (s.modifier || '').toLowerCase();
            const typ = (s.type || '').toLowerCase();

            if (typ === 'arrive') mIcon = 'fa-solid fa-flag-checkered text-rose-400';
            else if (typ === 'depart') mIcon = 'fa-solid fa-circle-dot text-cyan-400';
            else if (mod.includes('left')) mIcon = 'fa-solid fa-arrow-left text-amber-400';
            else if (mod.includes('right')) mIcon = 'fa-solid fa-arrow-right text-emerald-400';
            else if (mod.includes('u-turn')) mIcon = 'fa-solid fa-rotate-left text-pink-400';

            sHtml += `
              <div class="flex items-start space-x-2 py-1.5 border-b border-slate-800/60 last:border-0">
                <div class="step-maneuver-icon mt-0.5">
                  <i class="${mIcon}"></i>
                </div>
                <div class="flex-1 min-w-0">
                  <div class="text-slate-200 leading-snug font-medium text-[11px]">${s.instruction}</div>
                  <div class="text-[9px] font-mono text-slate-400 mt-0.5">${s.distance_str}</div>
                </div>
              </div>
            `;
          });
          stepsContainer.innerHTML = sHtml;
        }
      }

    } else {
      if (stepsContainer) stepsContainer.innerHTML = `<div class="text-center py-2 text-rose-400 text-xs">Route calculation error: ${data.message}</div>`;
    }
  } catch (err) {
    if (stepsContainer) stepsContainer.innerHTML = `<div class="text-center py-2 text-rose-400 text-xs">Directions service error: ${err.message}</div>`;
  }
};

window.closeDirectionsPanel = function() {
  const panel = document.getElementById('gps-directions-panel');
  if (panel) panel.classList.add('hidden');
};

window.clearCurrentRoute = function() {
  if (leafletMap) {
    if (currentRoutePolyline) leafletMap.removeLayer(currentRoutePolyline);
    if (currentRouteCasing) leafletMap.removeLayer(currentRouteCasing);
    if (destinationFlagMarker) leafletMap.removeLayer(destinationFlagMarker);
  }
  currentRoutePolyline = null;
  currentRouteCasing = null;
  destinationFlagMarker = null;
  activeDestination = null;

  const distEl = document.getElementById('route-distance-val');
  const durEl = document.getElementById('route-duration-val');
  const nameEl = document.getElementById('route-dest-name');
  const addrEl = document.getElementById('route-dest-addr');
  const stepsContainer = document.getElementById('route-steps-container');

  if (distEl) distEl.textContent = '-- mi';
  if (durEl) durEl.textContent = '-- mins';
  if (nameEl) nameEl.textContent = 'Select a business';
  if (addrEl) addrEl.textContent = 'Coordinates: --';
  if (stepsContainer) stepsContainer.innerHTML = '<div class="text-center py-3 text-slate-500 text-xs">No active route.</div>';

  closeDirectionsPanel();
};

window.changeTravelMode = function(mode) {
  activeTravelMode = mode;
  ['driving', 'walking', 'cycling'].forEach(m => {
    const btn = document.getElementById(`btn-mode-${m}`);
    if (btn) {
      if (m === mode) {
        btn.className = 'px-2 py-1 rounded-lg bg-emerald-600/30 text-emerald-300 font-bold border border-emerald-500/40 text-[10px] flex items-center space-x-1 cursor-pointer';
      } else {
        btn.className = 'px-2 py-1 rounded-lg bg-slate-800 text-slate-400 hover:text-white text-[10px] flex items-center space-x-1 cursor-pointer';
      }
    }
  });

  if (activeDestination) {
    requestGpsRoute(
      activeDestination.lat,
      activeDestination.lng,
      activeDestination.name,
      activeDestination.address,
      activeDestination.id,
      activeDestination.googleMapsUrl
    );
  }
};

// -------------------------------------------------------------
// Market Opportunity by Business Sector (Type)
// -------------------------------------------------------------
function renderSectorsBreakdown(data) {
  const container = document.getElementById('sectors-breakdown-container');
  if (!container) return;

  const sectors = (data.location_insights && data.location_insights.sectors) || [];
  if (sectors.length === 0) {
    container.innerHTML = `<div class="text-center py-6 text-slate-500 text-xs">No business sector breakdown available.</div>`;
    return;
  }

  let html = '';
  sectors.forEach((sec, idx) => {
    const color = CLUSTER_COLORS[sec.dominant_cluster % CLUSTER_COLORS.length];
    const isHighOpp = sec.opportunity_score >= 88;

    html += `
      <div class="rounded-xl bg-slate-950/70 border border-slate-800/80 hover:border-slate-700 p-3.5 space-y-2 transition shadow-sm">
        <div class="flex items-center justify-between flex-wrap gap-2">
          <div class="flex items-center space-x-2">
            <span class="w-2.5 h-2.5 rounded-full" style="background-color: ${color.border};"></span>
            <span class="font-bold text-white text-xs">${sec.sector}</span>
            <span class="text-[10px] font-mono text-slate-400 bg-slate-800/80 px-1.5 py-0.5 rounded">
              ${sec.count} businesses
            </span>
          </div>
          <span class="px-2 py-0.5 rounded text-[10px] font-bold ${isHighOpp ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30 animate-pulse' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'}">
            Opp: ${sec.opportunity_score}/100
          </span>
        </div>

        <div class="grid grid-cols-2 sm:grid-cols-3 gap-2 text-[10px] text-slate-400 font-mono">
          <div>Avg Quality: <strong class="text-amber-300">${sec.avg_rating}★</strong></div>
          <div>Dominant: <strong class="text-slate-200">Cluster ${sec.dominant_cluster}</strong></div>
          <div class="col-span-2 sm:col-span-1 truncate" title="${sec.top_location}">Hotspot: <strong class="text-cyan-300">${sec.top_location}</strong></div>
        </div>

        <p class="text-[11px] text-slate-300 bg-slate-900/90 p-2 rounded-lg border border-slate-800/80 leading-relaxed font-sans">
          <i class="fa-solid fa-arrow-trend-up text-emerald-400 mr-1 text-[10px]"></i>${sec.investment_verdict}
        </p>
      </div>
    `;
  });

  container.innerHTML = html;
}

// -------------------------------------------------------------
// Municipal District Deficit & Hotspot Analysis
// -------------------------------------------------------------
function renderDistrictsBreakdown(data) {
  const container = document.getElementById('districts-breakdown-container');
  if (!container) return;

  const districts = (data.location_insights && data.location_insights.districts) || [];
  if (districts.length === 0) {
    container.innerHTML = `<div class="text-center py-6 text-slate-500 text-xs">No district location breakdown available.</div>`;
    return;
  }

  let html = '';
  districts.forEach(d => {
    const isDeficit = d.deficit_detected;
    const color = CLUSTER_COLORS[d.top_opportunity_cluster % CLUSTER_COLORS.length];

    html += `
      <div class="rounded-xl bg-slate-950/70 border ${isDeficit ? 'border-rose-900/40 bg-gradient-to-r from-rose-950/20 via-slate-950 to-slate-950' : 'border-slate-800/80'} p-3.5 space-y-2 transition shadow-sm">
        <div class="flex items-center justify-between flex-wrap gap-2">
          <div class="flex items-center space-x-2">
            <i class="fa-solid fa-location-dot ${isDeficit ? 'text-rose-400' : 'text-emerald-400'} text-xs"></i>
            <span class="font-bold text-white text-xs">${d.name}</span>
            <span class="text-[10px] font-mono text-slate-400 bg-slate-800/80 px-1.5 py-0.5 rounded">
              ${d.total_businesses} entities
            </span>
          </div>

          <div class="flex items-center space-x-1.5">
            ${isDeficit ? `
              <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/30">
                <i class="fa-solid fa-triangle-exclamation mr-1"></i>Deficit Alert
              </span>
            ` : `
              <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                Stable / Moat
              </span>
            `}
          </div>
        </div>

        <div class="grid grid-cols-2 sm:grid-cols-3 gap-2 text-[10px] text-slate-400 font-mono">
          <div>Mean Rating: <strong class="${isDeficit ? 'text-rose-300 font-bold' : 'text-amber-300'}">${d.avg_rating}★</strong></div>
          <div>Primary Sector: <strong class="text-slate-200 truncate block" title="${d.primary_sector}">${d.primary_sector}</strong></div>
          <div>Top Opp: <strong class="text-emerald-400">${d.opportunity_score}/100</strong></div>
        </div>

        <div class="flex items-center justify-between pt-1 text-[11px]">
          <span class="text-slate-400 text-[10px]">
            Target Cluster: <strong style="color: ${color.border};">Cluster ${d.top_opportunity_cluster}</strong>
          </span>
          <button onclick="flyToDistrict(${d.lat}, ${d.lng})" class="text-[10px] text-cyan-400 hover:text-cyan-300 underline font-semibold cursor-pointer">
            <i class="fa-solid fa-magnifying-glass-location mr-1"></i>Inspect District on Map
          </button>
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}

window.flyToDistrict = function(lat, lng) {
  if (!leafletMap) return;
  leafletMap.flyTo([lat, lng], 14, { animate: true, duration: 1.0 });
  const mapSection = document.getElementById('opportunity-map');
  if (mapSection) mapSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
};

// -------------------------------------------------------------
// Real-World Business Entity Directory & Investment Explorer
// -------------------------------------------------------------
function initEntityDirectory(data) {
  const sectorSelect = document.getElementById('entity-sector-select');
  const districtSelect = document.getElementById('entity-district-select');
  const mapClusterSelect = document.getElementById('map-filter-cluster');
  const mapDistrictSelect = document.getElementById('map-filter-district');

  const entities = data.entity_locations || [];
  const sectors = Array.from(new Set(entities.map(e => e.business_type).filter(Boolean))).sort();
  const districts = Array.from(new Set(entities.map(e => e.district).filter(Boolean))).sort();

  if (sectorSelect) {
    let sHtml = '<option value="all">All Business Types</option>';
    sectors.forEach(s => { sHtml += `<option value="${s}">${s}</option>`; });
    sectorSelect.innerHTML = sHtml;
  }

  if (districtSelect) {
    let dHtml = '<option value="all">All Districts</option>';
    districts.forEach(d => { dHtml += `<option value="${d}">${d}</option>`; });
    districtSelect.innerHTML = dHtml;
  }

  if (mapClusterSelect) {
    let mcHtml = '<option value="all">All Archetype Clusters</option>';
    (data.strategic_decisions || []).forEach(sd => {
      mcHtml += `<option value="${sd.cluster}">Cluster ${sd.cluster}: ${sd.archetype}</option>`;
    });
    mapClusterSelect.innerHTML = mcHtml;
  }

  if (mapDistrictSelect) {
    let mdHtml = '<option value="all">All City Districts</option>';
    districts.forEach(d => { mdHtml += `<option value="${d}">${d}</option>`; });
    mapDistrictSelect.innerHTML = mdHtml;
  }

  currentEntityPage = 1;
  filterEntityTable();
}

window.filterEntityTable = function() {
  const searchInput = document.getElementById('entity-search-input');
  const sectorSelect = document.getElementById('entity-sector-select');
  const districtSelect = document.getElementById('entity-district-select');

  const query = (searchInput ? searchInput.value : '').toLowerCase().trim();
  const selSector = sectorSelect ? sectorSelect.value : 'all';
  const selDistrict = districtSelect ? districtSelect.value : 'all';

  filteredEntities = allEntities.filter(e => {
    const matchQ = !query || 
      (e.name && e.name.toLowerCase().includes(query)) ||
      (e.id && e.id.toLowerCase().includes(query)) ||
      (e.business_type && e.business_type.toLowerCase().includes(query)) ||
      (e.address && e.address.toLowerCase().includes(query)) ||
      (e.district && e.district.toLowerCase().includes(query));

    const matchS = selSector === 'all' || e.business_type === selSector;
    const matchD = selDistrict === 'all' || e.district === selDistrict;
    const matchC = currentActiveFilter === 'all' || String(e.cluster) === String(currentActiveFilter);

    return matchQ && matchS && matchD && matchC;
  });

  currentEntityPage = 1;
  renderEntityTablePage();
};

function renderEntityTablePage() {
  const tbody = document.getElementById('entities-table-body');
  const showingEl = document.getElementById('entity-showing-count');
  const totalEl = document.getElementById('entity-total-count');
  const paginationEl = document.getElementById('entity-pagination-controls');

  if (totalEl) totalEl.textContent = filteredEntities.length;

  if (!tbody) return;

  if (filteredEntities.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" class="text-center py-8 text-slate-500">
          No businesses match your filter criteria. Try adjusting your search query.
        </td>
      </tr>
    `;
    if (showingEl) showingEl.textContent = '0';
    if (paginationEl) paginationEl.innerHTML = '';
    return;
  }

  const startIdx = (currentEntityPage - 1) * ENTITIES_PER_PAGE;
  const pageSlice = filteredEntities.slice(startIdx, startIdx + ENTITIES_PER_PAGE);

  if (showingEl) showingEl.textContent = `${startIdx + 1} - ${startIdx + pageSlice.length}`;

  let rowsHtml = '';
  pageSlice.forEach(e => {
    const color = CLUSTER_COLORS[e.cluster % CLUSTER_COLORS.length];
    const isDisrupt = e.opportunity_score >= 88;
    const safeName = (e.name || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
    const safeAddr = (e.address || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
    const gmapsUrl = e.google_maps_url || `https://www.google.com/maps/dir/?api=1&destination=${e.lat},${e.lng}`;

    rowsHtml += `
      <tr class="hover:bg-slate-800/40 transition">
        <td class="py-3 px-4">
          <div class="font-bold text-white flex items-center space-x-1.5">
            <span>${e.name}</span>
          </div>
          <div class="text-[10px] text-slate-400 font-mono">${e.id}</div>
        </td>

        <td class="py-3 px-4">
          <span class="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-800 text-slate-200 border border-slate-700">
            ${e.business_type}
          </span>
        </td>

        <td class="py-3 px-4">
          <div class="text-slate-300 font-medium">${e.district}</div>
          <div class="text-[10px] text-slate-400 truncate max-w-xs">${e.address}</div>
          <div class="font-mono text-[9px] text-cyan-300 flex items-center space-x-1.5 mt-0.5">
            <i class="fa-solid fa-satellite text-[8px] text-emerald-400"></i>
            <span>${e.lat.toFixed(4)}, ${e.lng.toFixed(4)}</span>
            ${e.has_real_gps ? '<span class="text-[8px] bg-emerald-950/60 text-emerald-300 px-1 rounded border border-emerald-800/40 font-sans font-bold">GPS Verified</span>' : ''}
          </div>
        </td>

        <td class="py-3 px-3 text-center">
          <div class="font-mono font-bold ${e.rating < 3.5 ? 'text-rose-400' : 'text-amber-400'}">${e.rating} ★</div>
          <div class="text-[10px] text-slate-400 font-mono">${e.volume} vol</div>
        </td>

        <td class="py-3 px-3 text-center">
          <span class="px-2 py-0.5 rounded text-[10px] font-bold font-mono ${isDisrupt ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'}">
            ${e.opportunity_score}
          </span>
        </td>

        <td class="py-3 px-4">
          <div class="text-[11px] text-slate-300 max-w-sm leading-snug">
            ${e.investment_action}
          </div>
        </td>

        <td class="py-3 px-3 text-center">
          <div class="flex items-center justify-center space-x-1.5">
            <button onclick="requestGpsRoute(${e.lat}, ${e.lng}, '${safeName}', '${safeAddr}', '${e.id}', '${gmapsUrl}')" class="px-2 py-1.5 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/40 text-emerald-300 border border-emerald-500/30 text-[10px] font-bold flex items-center space-x-1 cursor-pointer transition" title="Get Real GPS Driving Directions">
              <i class="fa-solid fa-diamond-turn-right text-xs"></i>
              <span>Route</span>
            </button>
            <button onclick="flyToEntity(${e.lat}, ${e.lng}, '${e.id}')" class="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 hover:text-white transition cursor-pointer" title="Locate on Map">
              <i class="fa-solid fa-location-crosshairs text-xs"></i>
            </button>
            <a href="${gmapsUrl}" target="_blank" class="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-amber-300 border border-slate-700 hover:text-white transition" title="Open in Google Maps App">
              <i class="fa-solid fa-arrow-up-right-from-square text-xs"></i>
            </a>
          </div>
        </td>
      </tr>
    `;
  });

  tbody.innerHTML = rowsHtml;

  // Pagination Controls
  if (paginationEl) {
    const totalPages = Math.ceil(filteredEntities.length / ENTITIES_PER_PAGE);
    if (totalPages <= 1) {
      paginationEl.innerHTML = '';
      return;
    }

    let pButtons = `
      <button onclick="setEntityPage(${Math.max(1, currentEntityPage - 1)})" ${currentEntityPage === 1 ? 'disabled class="opacity-40 cursor-not-allowed"' : 'class="hover:bg-slate-700"'} class="px-2 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700 text-xs">
        &laquo;
      </button>
    `;

    for (let p = 1; p <= totalPages; p++) {
      if (p === 1 || p === totalPages || (p >= currentEntityPage - 1 && p <= currentEntityPage + 1)) {
        pButtons += `
          <button onclick="setEntityPage(${p})" class="px-2.5 py-1 rounded text-xs font-mono font-semibold ${p === currentEntityPage ? 'bg-emerald-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700'}">
            ${p}
          </button>
        `;
      } else if (p === currentEntityPage - 2 || p === currentEntityPage + 2) {
        pButtons += `<span class="text-slate-600 px-1">...</span>`;
      }
    }

    pButtons += `
      <button onclick="setEntityPage(${Math.min(totalPages, currentEntityPage + 1)})" ${currentEntityPage === totalPages ? 'disabled class="opacity-40 cursor-not-allowed"' : 'class="hover:bg-slate-700"'} class="px-2 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700 text-xs">
        &raquo;
      </button>
    `;
    paginationEl.innerHTML = pButtons;
  }
}

window.setEntityPage = function(page) {
  currentEntityPage = page;
  renderEntityTablePage();
};

window.exportEntitiesCSV = function() {
  if (!allEntities || allEntities.length === 0) {
    alert('No entities mapped to export.');
    return;
  }

  const headers = ["Business ID", "Business Name", "Business Type", "District", "Address", "Latitude", "Longitude", "Has Real GPS", "Google Maps URL", "Cluster", "Rating", "Review Volume", "Volatility", "Opportunity Score", "Investment Action"];
  const rows = allEntities.map(e => [
    `"${e.id}"`,
    `"${(e.name || '').replace(/"/g, '""')}"`,
    `"${(e.business_type || '').replace(/"/g, '""')}"`,
    `"${(e.district || '').replace(/"/g, '""')}"`,
    `"${(e.address || '').replace(/"/g, '""')}"`,
    e.lat,
    e.lng,
    e.has_real_gps ? "Yes" : "Estimated",
    `"${e.google_maps_url || ''}"`,
    e.cluster,
    e.rating,
    e.volume,
    e.volatility,
    e.opportunity_score,
    `"${(e.investment_action || '').replace(/"/g, '""')}"`
  ]);

  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `city_service_entities_locations_k${currentAnalyticsData ? currentAnalyticsData.k_clusters : 4}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// -------------------------------------------------------------
// Market Opportunity vs Risk Quadrant Map (Bubble Chart)
// -------------------------------------------------------------
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

// -------------------------------------------------------------
// Multi-Vector Strategic Viability Radar Chart
// -------------------------------------------------------------
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

// -------------------------------------------------------------
// Decision Cards & Filter Tabs (Enriched with Sectors & Locations)
// -------------------------------------------------------------
function renderStrategicDecisionCards(data, activeFilter = 'all') {
  currentActiveFilter = activeFilter;
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

  // Tabs
  if (tabsContainer) {
    let tabsHtml = `
      <button onclick="filterDecisions('all')" class="px-3 py-1 rounded-lg text-[11px] font-semibold transition cursor-pointer ${activeFilter === 'all' ? 'bg-emerald-600 text-white shadow-sm' : 'bg-slate-800 text-slate-400 hover:text-white border border-slate-700'}">
        All Clusters (${decisions.length})
      </button>
    `;
    decisions.forEach(d => {
      const color = CLUSTER_COLORS[d.cluster % CLUSTER_COLORS.length];
      const isAct = activeFilter === String(d.cluster);
      tabsHtml += `
        <button onclick="filterDecisions('${d.cluster}')" class="px-3 py-1 rounded-lg text-[11px] font-semibold transition cursor-pointer flex items-center space-x-1.5 ${isAct ? 'bg-slate-700 text-white ring-1 ring-white/20' : 'bg-slate-800 text-slate-400 hover:text-white border border-slate-700'}">
          <span class="w-2 h-2 rounded-full" style="background-color: ${color.border};"></span>
          <span>Cluster ${d.cluster}</span>
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

    const topSectors = d.top_sectors || [];
    const topLocations = d.top_locations || [];
    const locStrategy = d.location_strategy || `Target key city districts with high service volume and unsatisfied demand.`;

    cardsHtml += `
      <div class="rounded-2xl bg-slate-900/90 border border-slate-800 hover:border-slate-700 transition shadow-xl p-5 space-y-4 flex flex-col justify-between">
        
        <!-- Header -->
        <div class="space-y-2.5">
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

          <!-- Real-World Business Types & Location Concentrations -->
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[10px] pt-1">
            <div class="bg-slate-950/80 p-2 rounded-lg border border-slate-800/80 space-y-1">
              <div class="text-slate-400 font-bold uppercase tracking-wider flex items-center space-x-1">
                <i class="fa-solid fa-industry text-emerald-400"></i>
                <span>Dominant Business Types</span>
              </div>
              <div class="flex flex-wrap gap-1">
                ${topSectors.length > 0 ? topSectors.map(s => `<span class="bg-slate-800 text-slate-200 px-1.5 py-0.5 rounded text-[9px] font-mono">${s}</span>`).join('') : '<span class="text-slate-500">Commercial Services</span>'}
              </div>
            </div>

            <div class="bg-slate-950/80 p-2 rounded-lg border border-slate-800/80 space-y-1">
              <div class="text-slate-400 font-bold uppercase tracking-wider flex items-center space-x-1">
                <i class="fa-solid fa-location-dot text-cyan-400"></i>
                <span>Geographic Hotspots</span>
              </div>
              <div class="flex flex-wrap gap-1">
                ${topLocations.length > 0 ? topLocations.map(l => `<span class="bg-slate-800 text-cyan-200 px-1.5 py-0.5 rounded text-[9px] font-mono">${l}</span>`).join('') : '<span class="text-slate-500">Downtown Corridor</span>'}
              </div>
            </div>
          </div>

          <!-- Location Deployment Strategy Callout -->
          <div class="bg-indigo-950/20 border border-indigo-900/30 p-2 rounded-lg text-[10px] text-indigo-200 leading-relaxed">
            <strong class="text-emerald-400 font-bold"><i class="fa-solid fa-compass mr-1"></i>Location Strategy:</strong> ${locStrategy}
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
  if (currentAnalyticsData) {
    renderStrategicDecisionCards(currentAnalyticsData, filterVal);
    filterEntityTable();
    const mapClusterSelect = document.getElementById('map-filter-cluster');
    if (mapClusterSelect) {
      mapClusterSelect.value = filterVal;
      onMapFilterChange();
    }
  }
};

// -------------------------------------------------------------
// Copy & Export Actions
// -------------------------------------------------------------
window.copyStrategicBrief = function() {
  if (!currentAnalyticsData || !currentAnalyticsData.strategic_decisions) {
    alert('Please wait for analytics data to finish loading.');
    return;
  }
  const mt = currentAnalyticsData.market_trends || {};
  let brief = `# CITY SERVICE CLUSTERING - STRATEGIC EXECUTIVE BRIEF\n`;
  brief += `Generated: ${new Date().toLocaleString()}\n`;
  brief += `Market Health Index: ${mt.market_health_score}/100 | Mean Rating: ${mt.avg_market_rating}★\n`;
  brief += `Top Identified Venture: ${mt.top_venture_target} (Opportunity Score: ${mt.top_opportunity_score}/100)\n\n`;
  brief += `## REGIONAL MARKET GAP ANALYSIS\n${mt.market_gap_analysis}\n\n`;
  brief += `## CLUSTER VENTURE PROPOSALS & ACTIONABLE DECISIONS\n`;

  currentAnalyticsData.strategic_decisions.forEach(d => {
    brief += `\n### [Cluster ${d.cluster}] ${d.archetype} (${d.percentage}% Market Share)\n`;
    brief += `Venture: ${d.venture_title} (Opportunity Score: ${d.opportunity_score}/100 | Projected ROI: ${d.roi_potential})\n`;
    brief += `Concept: ${d.venture_concept}\n`;
    if (d.top_sectors) brief += `Dominant Business Types: ${d.top_sectors.join(', ')}\n`;
    if (d.top_locations) brief += `Geographic Hotspots: ${d.top_locations.join(', ')}\n`;
    if (d.location_strategy) brief += `Location Strategy: ${d.location_strategy}\n`;
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
  if (!currentAnalyticsData || !currentAnalyticsData.strategic_decisions) {
    alert('Please wait for analytics data to finish loading.');
    return;
  }
  const exportPayload = {
    timestamp: new Date().toISOString(),
    dataset_schema: currentAnalyticsData.schema_detected,
    sample_size: currentAnalyticsData.sample_size,
    k_clusters: currentAnalyticsData.k_clusters,
    metrics: currentAnalyticsData.metrics,
    market_trends: currentAnalyticsData.market_trends,
    strategic_decisions: currentAnalyticsData.strategic_decisions,
    location_insights: currentAnalyticsData.location_insights,
    entity_locations: currentAnalyticsData.entity_locations
  };

  const blob = new Blob([JSON.stringify(exportPayload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `market_venture_recommendations_k${currentAnalyticsData.k_clusters}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};
