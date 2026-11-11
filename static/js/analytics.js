/**
 * Analytics & Market Recommendations Dashboard
 * Manages Bubble Opportunity Matrix, Radar Viability Chart,
 * and Targeted Business Venture Decision Cards.
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

let marketMatrixChart = null;
let ventureRadarChart = null;
let currentAnalyticsData = null;
let currentActiveFilter = 'all';

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
async function loadAnalyticsData() {
  const cached = localStorage.getItem('city_service_ml_results');
  if (cached) {
    try {
      const data = JSON.parse(cached);
      if (data && data.strategic_decisions && data.strategic_decisions.length > 0) {
        renderAnalyticsView(data, localStorage.getItem('city_service_dataset_name') || 'Custom Uploaded Dataset');
        return;
      }
    } catch (e) {
      console.warn('Invalid cached ML results:', e);
    }
  }

  // If no custom dataset cached yet, trigger baseline sample analysis
  await triggerSampleClustering();
}

async function triggerSampleClustering() {
  const ctxDatasetName = document.getElementById('ctx-dataset-name');
  if (ctxDatasetName) ctxDatasetName.textContent = 'Analyzing Sample City Reviews...';

  try {
    const formData = new FormData();
    formData.append('k', '4');
    formData.append('use_sample', 'true');
    formData.append('sample_type', 'reviews');

    const res = await fetch('/api/train', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    if (data.status === 'success') {
      localStorage.setItem('city_service_ml_results', JSON.stringify(data));
      localStorage.setItem('city_service_dataset_name', 'sample_city_reviews.csv (4 Clusters)');
      renderAnalyticsView(data, 'sample_city_reviews.csv (Sample Reviews)');
    } else {
      console.error('Clustering error:', data.message);
    }
  } catch (e) {
    console.error('Fetch error:', e);
  }
}

// -------------------------------------------------------------
// Main View Renderer
// -------------------------------------------------------------
function renderAnalyticsView(data, datasetName) {
  currentAnalyticsData = data;

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

  // 4. Render Charts
  renderMarketMatrixChart(data);
  renderVentureRadarChart(data);

  // 5. Render Decision Cards
  renderStrategicDecisionCards(data, 'all');
}

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
// Decision Cards & Filter Tabs
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
    strategic_decisions: currentAnalyticsData.strategic_decisions
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
