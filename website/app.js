
'use strict';

const WEIGHTS_URL  = './weights.json';
const HYPIXEL_URL  = 'https://api.hypixel.net/v2/skyblock/bazaar';
const ELECTION_URL = 'https://api.hypixel.net/v2/resources/skyblock/election';
const COFL_BASE    = 'https://sky.coflnet.com/api/bazaar';
const REFRESH_MS   = 60_000;        // refresh bazaar every 60s
const PRED_BATCH   = 50;            // compute predictions for top N items first
const COFL_DELAY   = 50;            // ms between CoflNet requests (rate limit)
const ANCHOR_HOURS = [168, 72, 24, 1, 0]; // must match trainer/train.py

const state = {
  weights:    null,
  bazaar:     {},       // pid → quick_status row
  mayor:      'Unknown',
  coflCache:  {},       // pid → { week: [...], hour: [...] }
  preds:      {},       // pid → { pred_buy: float|null }
  items:      [],       // all filtered items (computed from bazaar)
  filtered:   [],       // after search
  sortCol:    'score',
  sortDir:    'desc',
  tab:        'picks',
  searchQuery:'',
  lastRefresh: null,
  countdown:   REFRESH_MS / 1000,
  predsLoaded: false,
  cfg: {
    minPrice:  500,
    maxPrice:  500_000_000,
    maxSpread: 50,
  },
};

const $  = id => document.getElementById(id);
const fmt = n  => Number.isFinite(n) ? n.toLocaleString('en-US', { maximumFractionDigits: 1 }) : '—';
const fmtC = n => Number.isFinite(n) ? Math.round(n).toLocaleString('en-US') : '—';
const fmtPct = n => Number.isFinite(n) ? n.toFixed(1) + '%' : '—';


/**
 * Build the 5×11 anchor feature array for one item.
 * CoflNet /history/week returns newest-first, 2h intervals, 84 entries.
 * CoflNet /history/hour returns newest-first, 1m intervals, 60 entries.
 */
function buildAnchorFeatures(pid, weekData, hourData, currentSnap) {
  if (!weekData || weekData.length < 10) return null;

  const W  = state.weights;
  const sc_min   = W.scaler_min[pid];
  const sc_scale = W.scaler_scale[pid];
  if (!sc_min || !sc_scale) return null;

  // Helper: scale a value using item's MinMaxScaler
  // sklearn MinMaxScaler: X_scaled = (X - data_min_) * scale_
  const scale = (v, col) => Math.max(0, Math.min(1, (v - sc_min[col]) * sc_scale[col]));
  const clamp01 = v => Math.max(0, Math.min(1, v));

  // Prepare anchor sources
  // weekData[0] = most recent 2h block, weekData[83] = ~1 week ago
  // 2h intervals → 1 day = index 12, 3 days = index 36, 1 week = index 83
  const sources = [
    weekData[Math.min(83, weekData.length - 1)],  // ~1 week ago
    weekData[Math.min(36, weekData.length - 1)],  // ~3 days ago
    weekData[Math.min(12, weekData.length - 1)],  // ~1 day ago
    hourData && hourData.length > 0 ? hourData[hourData.length - 1] : weekData[0], // ~1h ago
    null,  // current (from Hypixel live)
  ];

  return ANCHOR_HOURS.map((hours, idx) => {
    let src = sources[idx];
    let buy, sell, sv, bv, smw, bmw;

    if (idx === 4 || !src) {
      // Current snapshot from Hypixel
      buy  = currentSnap.buyPrice  || 0;
      sell = currentSnap.sellPrice || 0;
      sv   = currentSnap.sellVolume || 0;
      bv   = currentSnap.buyVolume  || 0;
      smw  = currentSnap.sellMovingWeek || 0;
      bmw  = currentSnap.buyMovingWeek  || 0;
    } else {
      buy  = src.buy  || src.buyPrice  || 0;
      sell = src.sell || src.sellPrice || 0;
      sv   = src.sellVolume || 0;
      bv   = src.buyVolume  || 0;
      smw  = src.sellMovingWeek || 0;
      bmw  = src.buyMovingWeek  || 0;
    }

    const spread    = Math.max(0, buy - sell);
    const spreadPct = sell > 0 ? spread / sell * 100 : 0;

    return [
      scale(sell, 0),             // sellPrice
      scale(buy,  1),             // buyPrice
      clamp01(spread   / 50_000), // spread  (normalised, matches training)
      clamp01(spreadPct / 100),   // spreadPct
      scale(sv,  2),              // sellVolume
      scale(bv,  3),              // buyVolume
      scale(smw, 4),              // sellMovingWeek
      scale(bmw, 5),              // buyMovingWeek
      0.0,                        // placeholder (maxBuy — not in training)
      0.0,                        // placeholder (minBuy)
      hours / 168.0,              // hours_delta normalised (1w=1.0, now=0.0)
    ];
  });
}


async function loadWeights() {
  try {
    const r = await fetch(WEIGHTS_URL);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const json = await r.json();
    BazaarInference.load(json);
    state.weights = json;
    const m = BazaarInference.getMeta();
    $('s-model-date').textContent = m.trained_at.slice(0, 10);
    return true;
  } catch (e) {
    console.warn('[Weights] Failed to load weights.json:', e.message);
    $('s-model-date').textContent = 'Not loaded';
    return false;
  }
}

async function fetchMayor() {
  try {
    const r    = await fetch(ELECTION_URL, { cache: 'no-cache' });
    const data = await r.json();
    if (data.success && data.mayor) {
      const name     = data.mayor.name || 'Unknown';
      const minister = data.mayor.minister?.name || '';
      state.mayor = minister ? `${name}+${minister}` : name;
      // Extract perks and strip Hypixel colour codes (§e, §7, etc) for native tooltips
      const stripCol = s => (s || '').replace(/\u00A7[0-9a-fk-or]/gi, '');
      state.mayorPerks = (data.mayor.perks || []).map(p => `• ${p.name}: ${stripCol(p.description)}`).join('\n');
      state.ministerPerks = data.mayor.minister?.perk 
        ? `• ${data.mayor.minister.perk.name}: ${stripCol(data.mayor.minister.perk.description)}` 
        : '';
    }
  } catch (_) { /* keep previous mayor */ }
  const parts = state.mayor.split('+');
  const badge = $('mayor-badge');
  if (badge) {
    if (parts.length > 1) {
      badge.innerHTML = `<span class="muted">Mayor:</span> <span style="color:var(--yellow);font-weight:500;cursor:help;" title="${state.mayorPerks}">${parts[0]}</span> <span class="muted" style="margin-left:6px">M:</span> <span style="color:var(--yellow);font-weight:500;cursor:help;" title="${state.ministerPerks}">${parts[1]}</span>`;
    } else {
      badge.innerHTML = `<span class="muted">Mayor:</span> <span style="color:var(--yellow);font-weight:500;cursor:help;" title="${state.mayorPerks}">${parts[0]}</span>`;
    }
  }
}

async function fetchBazaar() {
  try {
    const r    = await fetch(HYPIXEL_URL, { cache: 'no-cache' });
    const data = await r.json();
    if (!data.success) return;

    const cfg = state.cfg;
    state.bazaar = {};

    for (const [pid, prod] of Object.entries(data.products)) {
      const qs     = prod.quick_status || {};
      const sell   = qs.sellPrice  || 0;
      const buy    = qs.buyPrice   || 0;
      if (sell <= 0 || buy <= 0) continue;
      if (sell < cfg.minPrice || sell > cfg.maxPrice) continue;

      const spread    = buy  - sell;
      const spreadPct = spread / sell * 100;
      if (spreadPct > cfg.maxSpread) continue;

      state.bazaar[pid] = {
        sellPrice:        sell,
        buyPrice:         buy,
        spread,
        spreadPct,
        sellVolume:       qs.sellVolume       || 0,
        buyVolume:        qs.buyVolume        || 0,
        sellMovingWeek:   qs.sellMovingWeek   || 0,
        buyMovingWeek:    qs.buyMovingWeek    || 0,
      };
    }

    state.lastRefresh = Date.now();
    $('s-items').textContent = Object.keys(state.bazaar).length.toLocaleString();
    buildItemList();
  } catch (e) {
    console.error('[Bazaar] Fetch error:', e);
  }
}

async function fetchCoflAnchors(pid) {
  if (state.coflCache[pid]) return state.coflCache[pid];

  try {
    const [weekRes, hourRes] = await Promise.all([
      fetch(`${COFL_BASE}/${encodeURIComponent(pid)}/history/week`),
      fetch(`${COFL_BASE}/${encodeURIComponent(pid)}/history/hour`),
    ]);
    const weekData = weekRes.ok ? await weekRes.json() : [];
    const hourData = hourRes.ok ? await hourRes.json() : [];
    state.coflCache[pid] = { weekData, hourData };
    return state.coflCache[pid];
  } catch (_) {
    return { weekData: [], hourData: [] };
  }
}


function computeScore(snap, p12, p24) {
  // Score uses 12h prediction priority if 24h is severely untrained/off, but usually takes 24h
  if (!snap) return 0;
  const pred = p24 != null ? p24 : p12;
  if (!pred) return 0;
  const change_pct    = (pred - snap.buyPrice) / snap.buyPrice * 100;
  const vol_score     = Math.sqrt((snap.sellMovingWeek || 0) * (snap.buyMovingWeek || 0));
  const spread_penalty = 1 / (1 + snap.spreadPct / 10);
  return change_pct * vol_score * spread_penalty / 1e6;
}

async function fetchAndPredictBatch(pids) {
  for (const pid of pids) {
    const snap = state.bazaar[pid];
    if (!snap) continue;

    // Small delay to be polite to CoflNet
    await new Promise(r => setTimeout(r, COFL_DELAY));

    const { weekData, hourData } = await fetchCoflAnchors(pid);
    const feats = buildAnchorFeatures(pid, weekData, hourData, snap);
    if (!feats) {
      state.preds[pid] = { pred_buy_12h: null, pred_buy_24h: null };
      continue;
    }

    const result = BazaarInference.predict(pid, feats, state.mayor);
    state.preds[pid] = result;
  }

  // Re-render once batch is done
  buildItemList();
  renderCurrentTab();
}

async function runPredictions() {
  if (!BazaarInference.isLoaded()) return;
  state.predsLoaded = false;

  // Sort items by raw volume to prioritise most-traded items first
  const ranked = Object.entries(state.bazaar)
    .sort((a, b) => (b[1].sellMovingWeek + b[1].buyMovingWeek) -
                    (a[1].sellMovingWeek + a[1].buyMovingWeek))
    .map(([pid]) => pid);

  // First batch: top PRED_BATCH items (visible immediately)
  await fetchAndPredictBatch(ranked.slice(0, PRED_BATCH));
  state.predsLoaded = true;

  // Remaining items in background
  fetchAndPredictBatch(ranked.slice(PRED_BATCH));
}


function buildItemList() {
  const q = state.searchQuery.toLowerCase();

  state.items = Object.entries(state.bazaar).map(([pid, snap]) => {
    const pred  = state.preds[pid] || {};
    const p12   = pred.pred_buy_12h ?? null;
    const p24   = pred.pred_buy_24h ?? null;
    
    const chg12 = p12 ? (p12 - snap.buyPrice) / snap.buyPrice * 100 : null;
    const chg24 = p24 ? (p24 - snap.buyPrice) / snap.buyPrice * 100 : null;

    return {
      pid,
      ...snap,
      pred_buy_12h:   p12,
      pred_buy_24h:   p24,
      change_pct_12h: chg12,
      change_pct_24h: chg24,
      score:          computeScore(snap, p12, p24),
    };
  });

  // Search filter
  state.filtered = q
    ? state.items.filter(r => r.pid.toLowerCase().includes(q) ||
        r.pid.replace(/_/g, ' ').toLowerCase().includes(q))
    : [...state.items];

  // Sort
  const { sortCol, sortDir } = state;
  const dir = sortDir === 'desc' ? -1 : 1;
  state.filtered.sort((a, b) => {
    const av = a[sortCol] ?? -Infinity;
    const bv = b[sortCol] ?? -Infinity;
    return (av < bv ? -1 : av > bv ? 1 : 0) * dir;
  });
}


function renderCurrentTab() {
  if (state.tab === 'picks')   renderPicksTable();
  if (state.tab === 'market')  renderMarket();
  if (state.tab === 'stats')   renderStats();
}

function renderPicksTable() {
  const thead = $('picks-thead');
  const tbody = $('picks-tbody');
  if (!thead || !tbody) return;

  const cols = [
    { id: 'pid',         label: 'Item' },
    { id: 'sellPrice',   label: 'Sell Price' },
    { id: 'buyPrice',    label: 'Buy Price' },
    { id: 'spreadPct',   label: 'Spread %' },
    { id: 'sellMovingWeek', label: 'Sell Vol/wk' },
    { id: 'buyMovingWeek',  label: 'Buy Vol/wk' },
    { id: 'pred_buy_12h',label: 'Pred 12h' },
    { id: 'change_pct_12h',label: 'Δ 12h' },
    { id: 'pred_buy_24h',label: 'Pred 24h' },
    { id: 'change_pct_24h',label: 'Δ 24h' },
    { id: 'score',       label: 'Score' },
  ];

  thead.innerHTML = cols.map(c =>
    `<th class="sortable${state.sortCol === c.id ? ' active-sort' : ''}"
         data-col="${c.id}"
         data-dir="${state.sortCol === c.id ? state.sortDir : ''}">${c.label}</th>`
  ).join('');

  const hasPreds = Object.keys(state.preds).length > 0;

  if (state.filtered.length === 0) {
    $('picks-table-wrap').classList.add('hidden');
    $('picks-empty').classList.remove('hidden');
    $('picks-loading').classList.add('hidden');
    return;
  }

  $('picks-loading').classList.add('hidden');
  $('picks-empty').classList.add('hidden');
  $('picks-table-wrap').classList.remove('hidden');
  $('picks-no-results').classList.toggle('hidden', state.filtered.length > 0);

  const spreadColour = pct =>
    pct > 20 ? 'var(--red)' : pct > 5 ? 'var(--yellow)' : 'var(--text-muted)';
  const changeColour = pct =>
    !pct ? '' : pct > 0 ? `color:var(--green)` : `color:var(--red)`;

  tbody.innerHTML = state.filtered.slice(0, 200).map(r => {
    const name = r.pid.replace(/_/g, ' ');
    return `<tr data-pid="${r.pid}">
      <td class="item-id" title="${r.pid}">${name}</td>
      <td class="mono">${fmtC(r.sellPrice)}</td>
      <td class="mono">${fmtC(r.buyPrice)}</td>
      <td class="mono" style="color:${spreadColour(r.spreadPct)}">${fmtPct(r.spreadPct)}</td>
      <td class="mono">${fmt(r.sellMovingWeek)}</td>
      <td class="mono">${fmt(r.buyMovingWeek)}</td>
      <td class="mono">${r.pred_buy_12h ? fmtC(r.pred_buy_12h) : (hasPreds ? '—' : '<span class="muted">…</span>')}</td>
      <td class="mono" style="${changeColour(r.change_pct_12h)}">${r.change_pct_12h != null ? (r.change_pct_12h > 0 ? '+' : '') + r.change_pct_12h.toFixed(2) + '%' : '—'}</td>
      <td class="mono">${r.pred_buy_24h ? fmtC(r.pred_buy_24h) : (hasPreds ? '—' : '<span class="muted">…</span>')}</td>
      <td class="mono" style="${changeColour(r.change_pct_24h)}">${r.change_pct_24h != null ? (r.change_pct_24h > 0 ? '+' : '') + r.change_pct_24h.toFixed(2) + '%' : '—'}</td>
      <td class="mono">${r.score ? r.score.toFixed(2) : '—'}</td>
    </tr>`;
  }).join('');

  // Row click → open modal
  tbody.querySelectorAll('tr[data-pid]').forEach(row =>
    row.addEventListener('click', () => openModal(row.dataset.pid))
  );

  // Header sort
  thead.querySelectorAll('th[data-col]').forEach(th =>
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (state.sortCol === col) {
        state.sortDir = state.sortDir === 'desc' ? 'asc' : 'desc';
      } else {
        state.sortCol = col;
        state.sortDir = 'desc';
      }
      buildItemList();
      renderPicksTable();
    })
  );
}

function renderMarket() {
  const grid = $('market-grid');
  if (!grid) return;
  $('market-loading').classList.add('hidden');

  const items = state.filtered.slice(0, 300);
  $('market-count').textContent = `${state.filtered.length} items`;

  grid.innerHTML = items.map(r => {
    const name    = r.pid.replace(/_/g, ' ');
    const mkChg = (pct) => pct != null
      ? `<span style="color:${pct >= 0 ? 'var(--green)' : 'var(--red)'}">
           ${pct >= 0 ? '▲' : '▼'} ${Math.abs(pct).toFixed(2)}%
         </span>`
      : '<span class="muted">No prediction</span>';
      
    return `
      <div class="market-card" data-pid="${r.pid}">
        <div class="mc-name" title="${r.pid}">${name}</div>
        <div class="mc-row"><span class="muted">Sell</span><span class="mono">${fmtC(r.sellPrice)}</span></div>
        <div class="mc-row"><span class="muted">Buy</span><span class="mono">${fmtC(r.buyPrice)}</span></div>
        <div class="mc-row"><span class="muted">Spread</span><span class="mono" style="color:${r.spreadPct > 20 ? 'var(--red)' : r.spreadPct > 5 ? 'var(--yellow)' : 'var(--text-muted)'}">${fmtPct(r.spreadPct)}</span></div>
        <div class="mc-row"><span class="muted">Pred 12h</span>${mkChg(r.change_pct_12h)}</div>
        <div class="mc-row"><span class="muted">Pred 24h</span>${mkChg(r.change_pct_24h)}</div>
      </div>`;
  }).join('');

  grid.querySelectorAll('.market-card').forEach(card =>
    card.addEventListener('click', () => openModal(card.dataset.pid))
  );
}

function renderStats() {
  const items = state.items;
  if (!items.length) return;

  // Top items by sell volume
  const topVol = [...items].sort((a, b) => b.sellMovingWeek - a.sellMovingWeek).slice(0, 10);
  renderBarChart('chart-vol', topVol.map(r => r.pid.replace(/_/g, ' ').slice(0, 18)),
    topVol.map(r => r.sellMovingWeek), 'Weekly Volume', '#818cf8');

  // Spread distribution
  const buckets = { '0-5%': 0, '5-20%': 0, '20-50%': 0, '>50%': 0 };
  items.forEach(r => {
    if      (r.spreadPct <= 5)  buckets['0-5%']++;
    else if (r.spreadPct <= 20) buckets['5-20%']++;
    else if (r.spreadPct <= 50) buckets['20-50%']++;
    else                        buckets['>50%']++;
  });
  renderBarChart('chart-spread', Object.keys(buckets), Object.values(buckets),
    'Spread Distribution', '#34d399');

  // Prediction coverage
  const withPred  = items.filter(r => r.pred_buy != null).length;
  const withoutPred = items.length - withPred;
  renderBarChart('chart-snaps', ['With Prediction', 'No Prediction'],
    [withPred, withoutPred], 'Prediction Coverage', '#fbbf24');
}

let _charts = {};
function renderBarChart(canvasId, labels, data, label, color) {
  if (_charts[canvasId]) { _charts[canvasId].destroy(); delete _charts[canvasId]; }
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return;
  _charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label, data, backgroundColor: color + '99', borderColor: color, borderWidth: 1 }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 800, easing: 'easeOutQuart' },
      plugins: { 
        legend: { display: false },
        tooltip: {
          callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.raw.toLocaleString('en-US')}` }
        }
      },
      scales: {
        x: { ticks: { color: '#a1a1aa', font: { size: 11 } }, grid: { color: '#222' } },
        y: { 
          ticks: { 
            color: '#a1a1aa',
            callback: value => typeof value === 'number' ? value.toLocaleString('en-US') : value
          }, 
          grid: { color: '#222' } 
        },
      },
    },
  });
}


let _histChart = null;

async function openModal(pid) {
  const snap = state.bazaar[pid];
  if (!snap) return;

  $('modal-title').textContent = pid.replace(/_/g, ' ');
  $('modal-sub').textContent   = pid;

  const pred = state.preds[pid] || {};
  const chg12  = pred.pred_buy_12h ? ((pred.pred_buy_12h - snap.buyPrice) / snap.buyPrice * 100) : null;
  const chg24  = pred.pred_buy_24h ? ((pred.pred_buy_24h - snap.buyPrice) / snap.buyPrice * 100) : null;

  $('modal-stats-row').innerHTML = [
    { label: 'Sell Price',    val: fmtC(snap.sellPrice) },
    { label: 'Buy Price',     val: fmtC(snap.buyPrice) },
    { label: 'Spread',        val: fmtPct(snap.spreadPct) },
    { label: 'Pred 12h',      val: pred.pred_buy_12h ? fmtC(pred.pred_buy_12h) : '—' },
    { label: 'Δ% in 12h',     val: chg12 != null ? `${chg12 > 0 ? '+' : ''}${chg12.toFixed(2)}%` : '—' },
    { label: 'Pred 24h',      val: pred.pred_buy_24h ? fmtC(pred.pred_buy_24h) : '—' },
    { label: 'Δ% in 24h',     val: chg24 != null ? `${chg24 > 0 ? '+' : ''}${chg24.toFixed(2)}%` : '—' },
    { label: 'Sell Vol/wk',   val: fmt(snap.sellMovingWeek) },
    { label: 'Buy Vol/wk',    val: fmt(snap.buyMovingWeek) },
  ].map(s => `<div class="m-stat"><div class="m-stat-label">${s.label}</div>
    <div class="m-stat-val mono">${s.val}</div></div>`).join('');

  $('modal-overlay').classList.remove('hidden');
  document.body.style.overflow = 'hidden';

  // Immediately destroy old chart to prevent ghosting
  if (_histChart) { _histChart.destroy(); _histChart = null; }

  // Render price history chart using CoflNet data
  renderModalChart(pid, snap);
}

async function renderModalChart(pid, snap) {
  $('modal-no-data').classList.add('hidden');
  $('chart-history').style.opacity = '1';

  const { weekData } = await fetchCoflAnchors(pid);
  if (!weekData || weekData.length === 0) {
    $('modal-no-data').classList.remove('hidden');
    return;
  }

  const labels = [...weekData].reverse().map(d => d.timestamp
    ? new Date(d.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit' })
    : '');
  const buyPrices  = [...weekData].reverse().map(d => d.buy  || d.buyPrice);
  const sellPrices = [...weekData].reverse().map(d => d.sell || d.sellPrice);

  const ctx = $('chart-history').getContext('2d');
  _histChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Buy',
          data: buyPrices,
          borderColor: '#818cf8',
          backgroundColor: 'rgba(129,140,248,0.08)',
          tension: 0.3,
          pointRadius: 0,
          fill: true,
          borderWidth: 1.5,
        },
        {
          label: 'Sell',
          data: sellPrices,
          borderColor: '#34d399',
          backgroundColor: 'rgba(52,211,153,0.05)',
          tension: 0.3,
          pointRadius: 0,
          fill: true,
          borderWidth: 1.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 800, easing: 'easeOutQuart' },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { color: '#a1a1aa', boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.raw.toLocaleString('en-US')}` }
        }
      },
      scales: {
        x: {
          ticks: { color: '#a1a1aa', maxTicksLimit: 8, font: { size: 10 }, maxRotation: 0 },
          grid: { color: '#1e1e1e' },
        },
        y: {
          ticks: { 
            color: '#a1a1aa', font: { size: 11 },
            callback: value => typeof value === 'number' ? value.toLocaleString('en-US') : value
          },
          grid: { color: '#1e1e1e' },
        },
      },
    },
  });
}

function closeModal() {
  $('modal-overlay').classList.add('hidden');
  document.body.style.overflow = '';
}


function startCountdown() {
  setInterval(() => {
    state.countdown--;
    if (state.countdown <= 0) {
      state.countdown = REFRESH_MS / 1000;
      fetchBazaar().then(() => { buildItemList(); renderCurrentTab(); runPredictions(); });
      fetchMayor();
    }
    const pct = (1 - state.countdown / (REFRESH_MS / 1000)) * 100;
    const bar = $('countdown-bar');
    if (bar) bar.style.width = `${pct}%`;
    const txt = $('countdown-text');
    if (txt) txt.textContent = state.countdown;
    const live = $('live-text');
    if (live && state.lastRefresh) {
      const sec = Math.floor((Date.now() - state.lastRefresh) / 1000);
      live.textContent = sec < 60 ? `${sec}s ago` : `${Math.floor(sec / 60)}m ago`;
    }
  }, 1000);
}


function switchTab(tab) {
  state.tab = tab;
  document.querySelectorAll('.tab').forEach(t =>
    t.classList.toggle('active', t.dataset.tab === tab)
  );
  document.querySelectorAll('.panel').forEach(p =>
    p.classList.toggle('active', p.id === `panel-${tab}`)
  );
  document.querySelectorAll('.panel:not(.active)').forEach(p =>
    p.classList.add('hidden')
  );
  document.querySelector(`#panel-${tab}`)?.classList.remove('hidden');
  renderCurrentTab();
}


function initSettings() {
  $('cfg-min-price').value  = state.cfg.minPrice;
  $('cfg-max-price').value  = state.cfg.maxPrice;
  $('cfg-max-spread').value = state.cfg.maxSpread;

  $('cfg-apply').addEventListener('click', () => {
    state.cfg.minPrice  = parseFloat($('cfg-min-price').value)  || 500;
    state.cfg.maxPrice  = parseFloat($('cfg-max-price').value)  || 500_000_000;
    state.cfg.maxSpread = parseFloat($('cfg-max-spread').value) || 50;
    fetchBazaar().then(() => { buildItemList(); renderCurrentTab(); });
    const st = $('cfg-status');
    st.classList.remove('hidden');
    setTimeout(() => st.classList.add('hidden'), 2000);
  });
}


function initSearch() {
  $('search-input')?.addEventListener('input', e => {
    state.searchQuery = e.target.value.trim();
    buildItemList();
    renderCurrentTab();
  });
}


function initSortDropdown() {
  $('sort-select')?.addEventListener('change', e => {
    state.sortCol = e.target.value;
    buildItemList();
    renderPicksTable();
  });
}


async function boot() {
  console.log('[Bazaar Intel v3] Booting...');

  // 1. Load weights
  await loadWeights();

  // 2. Fetch mayor + bazaar data at same time
  await Promise.all([fetchMayor(), fetchBazaar()]);

  // 3. Build item list and render immediately (without predictions)
  buildItemList();
  renderCurrentTab();

  // 4. Run predictions (async, updates table as batches complete)
  runPredictions();

  // 5. Start auto-refresh countdown
  startCountdown();

  console.log('[Bazaar Intel v3] Ready.');
}


document.addEventListener('DOMContentLoaded', () => {
  // Tab buttons
  document.querySelectorAll('.tab').forEach(btn =>
    btn.addEventListener('click', () => switchTab(btn.dataset.tab))
  );

  // Modal close
  $('modal-close')?.addEventListener('click', closeModal);
  $('modal-overlay')?.addEventListener('click', e => {
    if (e.target === $('modal-overlay')) closeModal();
  });

  // Refresh button
  $('refresh-btn')?.addEventListener('click', () => {
    state.countdown = REFRESH_MS / 1000;
    fetchBazaar().then(() => { buildItemList(); renderCurrentTab(); runPredictions(); });
    fetchMayor();
  });

  initSearch();
  initSortDropdown();
  initSettings();
  boot();
});
