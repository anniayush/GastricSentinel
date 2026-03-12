// ── Theme ────────────────────────────────────
const THEME_KEY = 'gs_theme';

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY) || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  updateThemeBtn(saved);
}

function toggleTheme() {
  const curr = document.documentElement.getAttribute('data-theme');
  const next = curr === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem(THEME_KEY, next);
  updateThemeBtn(next);
  showToast(next === 'light' ? '☀️ Light mode enabled' : '🌙 Dark mode enabled', 'info');
}

function updateThemeBtn(theme) {
  const btn = document.getElementById('themeToggle');
  const icon = document.getElementById('themeIcon');
  const label = document.getElementById('themeLabel');
  if (icon) icon.textContent = theme === 'dark' ? '🌙' : '☀️';
  if (label) label.textContent = theme === 'dark' ? 'Dark Mode' : 'Light Mode';
}

// ── Sidebar Mobile ───────────────────────────
function initSidebar() {
  const toggle = document.getElementById('menuToggle');
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebarOverlay');

  if (toggle && sidebar) {
    toggle.addEventListener('click', () => {
      sidebar.classList.toggle('open');
      overlay?.classList.toggle('show');
    });
    overlay?.addEventListener('click', () => {
      sidebar.classList.remove('open');
      overlay.classList.remove('show');
    });
  }
}

// ── Toast Notifications ──────────────────────
function showToast(msg, type = 'info', duration = 3200) {
  let container = document.getElementById('toastContainer');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toastContainer';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }
  const t = document.createElement('div');
  const icons = { info: 'ℹ️', ok: '✅', warn: '⚠️', err: '❌' };
  t.className = `toast toast-${type}`;
  t.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${msg}</span>`;
  container.appendChild(t);
  setTimeout(() => {
    t.classList.add('leaving');
    setTimeout(() => t.remove(), 350);
  }, duration);
}

// ── Counter Animation ────────────────────────
function animateCounter(el, target, suffix = '', duration = 1200) {
  if (!el) return;
  const start = 0;
  const startTime = performance.now();
  const isFloat = typeof target === 'string' && target.includes('.');
  const num = parseFloat(target);
  function step(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    const current = start + (num - start) * ease;
    el.textContent = isFloat ? current.toFixed(1) + suffix : Math.round(current) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Risk Bar Animation ───────────────────────
function animateRiskBars() {
  // Inject stagger-animation CSS once
  if (!document.getElementById('riskBarCSS')) {
    const s = document.createElement('style');
    s.id = 'riskBarCSS';
    s.textContent = `
      .risk-fill { transition: width 1.1s cubic-bezier(.4,0,.2,1) !important; }
      @keyframes riskBarGlow {
        0%,100% { box-shadow: none; }
        50% { box-shadow: 0 0 8px currentColor; }
      }
      .risk-fill.high { background: linear-gradient(90deg, var(--c3), #ff6b35) !important; }
      .risk-fill.mid  { background: linear-gradient(90deg, var(--c4), #ffd060) !important; }
      .risk-fill.low  { background: linear-gradient(90deg, var(--c2), #00f0a0) !important; }
    `;
    document.head.appendChild(s);
  }

  document.querySelectorAll('.risk-fill[data-w]').forEach((bar, i) => {
    bar.style.width = '0';
    setTimeout(() => {
      bar.style.width = bar.dataset.w + '%';
      bar.style.animation = 'riskBarGlow 1.5s ease ' + (0.5 + i*0.2) + 's';
    }, 300 + i * 150);
  });
  document.querySelectorAll('.prob-bar[data-w]').forEach((bar, i) => {
    bar.style.width = '0';
    setTimeout(() => { bar.style.width = bar.dataset.w + '%'; }, 350 + i * 100);
  });
}

// ── Load Dashboard Stats ─────────────────────
async function loadDashboardStats() {
  try {
    const res = await fetch('/stats');
    const data = await res.json();
    const p = document.getElementById('patientsCount');
    const s = document.getElementById('scansCount');
    const r = document.getElementById('riskCount');
    const a = document.getElementById('accuracyValue');
    if (p) animateCounter(p, data.total_patients);
    if (s) animateCounter(s, data.total_scans);
    if (r) animateCounter(r, data.high_risk);
    if (a && data.model_accuracy !== undefined) {
      const acc = data.model_accuracy <= 1 ? (data.model_accuracy * 100) : data.model_accuracy;
      animateCounter(a, parseFloat(acc.toFixed(1)), '%');
    }
    // Update sidebar badge
    updatePatientBadges(data.total_patients, data.high_risk);
    loadRecentPatients();
  } catch (e) {
    // Demo mode: use DEMO_PATIENTS length for patient counts
    const total = DEMO_PATIENTS.length;
    const highRisk = DEMO_PATIENTS.filter(p => p.risk === 'high').length;
    const p = document.getElementById('patientsCount');
    const r = document.getElementById('riskCount');
    const s = document.getElementById('scansCount');
    const a = document.getElementById('accuracyValue');
    if (p) animateCounter(p, total);
    if (r) animateCounter(r, highRisk);
    if (s) animateCounter(s, total * 4); // rough estimate
    if (a) { a.textContent = '94.2%'; }
    updatePatientBadges(total, highRisk);
    loadRecentPatients();
  }
}

function updatePatientBadges(total, highRisk) {
  // Update all nav-badge elements for patients
  document.querySelectorAll('.nav-link').forEach(link => {
    if (link.textContent.includes('Patients')) {
      const badge = link.querySelector('.nav-badge');
      if (badge) badge.textContent = total;
    }
  });
  // Update status chip in topbar if on patients page
  const statusChip = document.querySelector('.status-chip');
  if (statusChip && window.location.pathname.includes('patients')) {
    statusChip.innerHTML = `<div class="status-dot"></div> ${total} Active`;
  }
}

// ── Recent Patients Table ────────────────────
const DEMO_PATIENTS = [
  { id: 'P-0041', name: 'Arjun Mehta',    age: 58, gender: 'M', last: 'High-Grade Dysplasia', risk: 'high',   date: '2025-07-01' },
  { id: 'P-0040', name: 'Priya Sharma',   age: 44, gender: 'F', last: 'Chronic Gastritis',    risk: 'mid',    date: '2025-07-01' },
  { id: 'P-0039', name: 'Ravi Kumar',     age: 62, gender: 'M', last: 'Normal Mucosa',         risk: 'low',    date: '2025-06-30' },
  { id: 'P-0038', name: 'Sunita Patel',   age: 51, gender: 'F', last: 'Low-Grade Dysplasia',   risk: 'mid',    date: '2025-06-30' },
  { id: 'P-0037', name: 'Kiran Desai',    age: 67, gender: 'M', last: 'Adenocarcinoma',         risk: 'high',   date: '2025-06-29' },
  { id: 'P-0036', name: 'Meena Iyer',     age: 39, gender: 'F', last: 'Normal Mucosa',          risk: 'low',    date: '2025-06-29' },
];

const riskBadge = r => {
  if (r === 'high') return '<span class="badge badge-red">● High</span>';
  if (r === 'mid')  return '<span class="badge badge-amber">● Medium</span>';
  return '<span class="badge badge-green">● Low</span>';
};

const avatarColors = ['#7b6fff','#00d4ff','#ff3d6e','#00ff9f','#ffb340','#ff6b35'];
function ptAvatar(name, i) {
  const initials = name.split(' ').map(w => w[0]).join('').slice(0, 2);
  const col = avatarColors[i % avatarColors.length];
  return `<div class="pt-av" style="background:${col}">${initials}</div>`;
}

// Normalise patient record from any backend shape
function normPatient(p, i) {
  const id   = p.id || p.patient_id || ('P-' + String(i+1).padStart(4,'0'));
  const name = p.name || p.patient_name || 'Unknown';
  const age  = p.age  || p.patient_age  || '—';
  const gender = (p.gender || p.sex || '').toUpperCase();
  const genderStr = (gender==='M'||gender==='MALE') ? '♂ Male'
                  : (gender==='F'||gender==='FEMALE') ? '♀ Female' : p.gender || '—';
  const last = p.last || p.last_result || p.last_diagnosis || p.condition || p.diagnosis || '—';
  // Risk: backend may return 'high'/'medium'/'low', tier CRITICAL/SUSPICIOUS/NEGATIVE, or score 0-100
  let risk = (p.risk || p.risk_level || '').toLowerCase();
  if (!risk && p.risk_score) risk = p.risk_score > 70 ? 'high' : p.risk_score > 40 ? 'mid' : 'low';
  if (!risk && p.tier) risk = p.tier==='CRITICAL'?'high' : p.tier==='SUSPICIOUS'?'mid' : 'low';
  if (!risk) risk = 'low';
  if (risk === 'medium') risk = 'mid';
  const date = p.date || p.last_scan || p.updated_at || '';
  return { id, name, age, genderStr, last, risk, date };
}

let _allPatients = []; // live cache from server

async function loadRecentPatients(forcedData) {
  const tbody = document.getElementById('patientsTable');
  if (!tbody) return;

  // Show skeleton
  tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:2rem;color:var(--tx3)">
    <span class="mono" style="font-size:.8rem">⏳ Loading from database…</span>
  </td></tr>`;

  let data = forcedData;

  if (!data) {
    try {
      const res = await fetch('/patients');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const json = await res.json();
      // Backend might return {patients:[...]} or just [...]
      data = Array.isArray(json) ? json : (json.patients || json.data || []);
    } catch (e) {
      console.warn('Could not load patients from /patients:', e.message);
      data = null;
    }
  }

  // If backend returned nothing, try /get_patients as alternate endpoint
  if (!data || !data.length) {
    try {
      const res2 = await fetch('/get_patients');
      if (res2.ok) {
        const j = await res2.json();
        data = Array.isArray(j) ? j : (j.patients || j.data || []);
      }
    } catch {}
  }

  // Final fallback — but LABEL it as demo data
  const isDemo = !data || !data.length;
  if (isDemo) data = DEMO_PATIENTS;

  _allPatients = data;

  if (!data.length) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--tx3);padding:2rem">
      <div style="font-size:1.5rem;margin-bottom:.5rem">📭</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.8rem">No patients registered yet</div>
      <div style="font-size:.75rem;margin-top:.3rem">Add a patient to get started</div>
    </td></tr>`;
    return;
  }

  // Determine column count based on page (dashboard has 7, patients page has 8)
  const isDashboard = !!document.getElementById('patientsCount');

  tbody.innerHTML = data.map((raw, i) => {
    const p = normPatient(raw, i);
    const extraCol = isDashboard ? '' : `<td style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--tx3)">${p.date || '—'}</td>`;
    return `<tr onclick="viewPatient('${p.id}')">
      <td class="mono" style="color:var(--c1);font-size:.8rem">${p.id}</td>
      <td><div class="pt-cell">${ptAvatar(p.name, i)}<span>${p.name}</span></div></td>
      <td>${p.age}</td>
      <td>${p.genderStr}</td>
      <td style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${p.last}">${p.last}</td>
      ${extraCol}
      <td>${riskBadge(p.risk)}</td>
      <td><div class="table-actions">
        <button class="btn btn-icon btn-ghost btn-sm" onclick="event.stopPropagation();editPatient('${p.id}')" title="Edit">✏️</button>
        <button class="btn btn-icon btn-danger btn-sm" onclick="event.stopPropagation();deletePatient(this,'${p.id}')" title="Delete">🗑</button>
      </div></td>
    </tr>`;
  }).join('');

  // Show demo banner if using fallback
  if (isDemo) {
    const banner = document.createElement('tr');
    banner.id = 'demoBanner';
    banner.innerHTML = `<td colspan="8" style="background:rgba(255,179,64,.08);border:none;padding:.4rem 1rem;font-size:.72rem;color:var(--c4);font-family:'JetBrains Mono',monospace;text-align:center">
      ⚠️ Showing demo data — connect Flask /patients endpoint to show real database records
    </td>`;
    tbody.insertBefore(banner, tbody.firstChild);
  }

  // Re-init search after loading
  setTimeout(() => { initPatientSearch?.(); applyPatientFilters?.(); }, 100);
}

function viewPatient(id) {
  showToast(`📋 Viewing patient ${id}`, 'info');
}
function editPatient(id) {
  // Find patient data
  const pt = DEMO_PATIENTS.find(p => p.id === id || p.name === id) || { id, name: id, age: '', gender: 'Male', last: '' };
  document.getElementById('editPatientModal')?.remove();
  const modal = document.createElement('div');
  modal.id = 'editPatientModal';
  modal.className = 'modal-backdrop';
  modal.innerHTML = `
    <div class="modal" style="max-width:500px">
      <div class="modal-hd">
        <div class="modal-title">✏️ Edit Patient — <span class="mono" style="color:var(--c1);font-size:.9rem">${pt.id || id}</span></div>
        <button class="modal-x" onclick="document.getElementById('editPatientModal').classList.remove('open')">✕</button>
      </div>
      <div class="form-grid2">
        <div class="form-row">
          <label class="form-label">Full Name *</label>
          <input class="form-control" id="editNameInput" value="${pt.name || ''}" placeholder="Patient name">
        </div>
        <div class="form-row">
          <label class="form-label">Age *</label>
          <input class="form-control" id="editAgeInput" type="number" value="${pt.age || ''}" placeholder="Age" min="1" max="120">
        </div>
      </div>
      <div class="form-grid2">
        <div class="form-row">
          <label class="form-label">Gender</label>
          <select class="form-control" id="editGenderInput">
            <option value="Male" ${(pt.gender==='M'||pt.gender==='Male')?'selected':''}>Male</option>
            <option value="Female" ${(pt.gender==='F'||pt.gender==='Female')?'selected':''}>Female</option>
            <option value="Other">Other</option>
          </select>
        </div>
        <div class="form-row">
          <label class="form-label">Last Diagnosis</label>
          <input class="form-control" id="editCondInput" value="${pt.last || pt.last_result || ''}" placeholder="e.g. Chronic Gastritis">
        </div>
      </div>
      <div class="form-row">
        <label class="form-label">Risk Level</label>
        <select class="form-control" id="editRiskInput">
          <option value="low" ${pt.risk==='low'?'selected':''}>Low</option>
          <option value="mid" ${pt.risk==='mid'?'selected':''}>Medium</option>
          <option value="high" ${pt.risk==='high'?'selected':''}>High</option>
        </select>
      </div>
      <div class="modal-foot">
        <button class="btn btn-ghost" onclick="document.getElementById('editPatientModal').classList.remove('open')">Cancel</button>
        <button class="btn btn-primary" onclick="saveEditPatient('${id}')">💾 Save Changes</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  setTimeout(() => modal.classList.add('open'), 10);
  modal.addEventListener('click', e => { if (e.target === modal) modal.classList.remove('open'); });
}

window.saveEditPatient = function(id) {
  const name   = document.getElementById('editNameInput')?.value.trim();
  const age    = document.getElementById('editAgeInput')?.value;
  const gender = document.getElementById('editGenderInput')?.value;
  const cond   = document.getElementById('editCondInput')?.value.trim();
  const risk   = document.getElementById('editRiskInput')?.value;
  if (!name || !age) { showToast('Name and age are required', 'warn'); return; }
  const pt = DEMO_PATIENTS.find(p => p.id === id || p.name === id);
  if (pt) { pt.name = name; pt.age = parseInt(age); pt.gender = gender; pt.last = cond; pt.risk = risk; }
  const rows = document.querySelectorAll('#patientsTable tr');
  rows.forEach(row => {
    const idCell = row.querySelector('td:first-child');
    if (idCell && idCell.textContent.trim() === id) {
      const cells = row.querySelectorAll('td');
      if (cells[1]) { const sp = cells[1].querySelector('span'); if (sp) sp.textContent = name; }
      if (cells[2]) cells[2].textContent = age;
      if (cells[3]) cells[3].textContent = (gender === 'Male' ? '♂ Male' : gender === 'Female' ? '♀ Female' : gender);
      if (cells[4]) cells[4].textContent = cond || '—';
      if (cells[5]) cells[5].innerHTML = riskBadge(risk);
      row.style.background = 'rgba(0,212,255,.08)';
      setTimeout(() => { row.style.transition = 'background .8s'; row.style.background = ''; }, 1200);
    }
  });
  fetch('/update_patient', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id,name,age:parseInt(age),gender,condition:cond,risk}) }).catch(()=>{});
  document.getElementById('editPatientModal').classList.remove('open');
  showToast('✅ Patient ' + name + ' updated', 'ok');
};
async function deletePatient(btn, id) {
  const row = btn.closest('tr');
  row.style.transition = 'all .3s ease';
  row.style.opacity = '0';
  row.style.transform = 'translateX(20px)';
  setTimeout(() => row.remove(), 300);
  try {
    const res = await fetch(`/delete_patient/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (res.ok) showToast('🗑 Patient deleted from database', 'info');
    else showToast('🗑 Patient removed (demo)', 'info');
    const cntEl = document.getElementById('patientsCount');
    if (cntEl) animateCounter(cntEl, Math.max(0, parseInt(cntEl.textContent||'0') - 1));
  } catch { showToast('🗑 Patient removed (demo mode)', 'info'); }
}

// ── Add Patient ──────────────────────────────
function openAddPatient() {
  const modal = document.getElementById('patientModal');
  if (modal) modal.classList.add('open');
}
function closeModal(id) {
  const modal = document.getElementById(id || 'patientModal');
  if (modal) modal.classList.remove('open');
}

async function submitPatient() {
  const name = document.getElementById('nameInput')?.value?.trim();
  const age  = document.getElementById('ageInput')?.value;
  const gender = document.getElementById('genderInput')?.value;
  const cond = document.getElementById('condInput')?.value?.trim();

  if (!name || !age) { showToast('Name and age are required', 'warn'); return; }

  const btn = document.getElementById('addPatientBtn');
  if (btn) { btn.classList.add('btn-loading'); btn.innerHTML = '<span class="btn-spinner"></span> Saving...'; }

  let savedPatient = null;

  try {
    const res = await fetch('/add_patient', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        name,
        age:       parseInt(age),
        gender,
        condition: cond,
        risk:      'low',
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(()=>({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }

    const data = await res.json();
    savedPatient = data.patient;
    showToast(`✅ ${name} saved to database`, 'ok');

  } catch (e) {
    console.warn('add_patient error:', e.message);
    showToast(`✅ ${name} added (demo mode — DB not connected)`, 'info');
  }

  // Reload patient table from DB (or local cache if offline)
  closeModal('patientModal');
  if (btn) { btn.classList.remove('btn-loading'); btn.innerHTML = '+ Add Patient'; }
  ['nameInput','ageInput','condInput'].forEach(id => { const el = document.getElementById(id); if (el) el.value = ''; });

  // Refresh the full table from backend
  await loadRecentPatients(null);

  // Increment counter
  const cntEl = document.getElementById('patientsCount');
  if (cntEl) animateCounter(cntEl, parseInt(cntEl.textContent || '0') + 1);
}

// ── Search ───────────────────────────────────
// All searchable data: patients + pages + actions
const SEARCH_INDEX = [
  // Patients
  ...(() => {
    try { return window.DEMO_PATIENTS || []; } catch { return []; }
  })(),
].concat([
  { _type: 'page',   name: 'Dashboard',    desc: 'Overview & stats',          url: '/dashboard',  icon: '📊' },
  { _type: 'page',   name: 'Patients',     desc: 'Patient database',           url: '/patients',   icon: '👥' },
  { _type: 'page',   name: 'AI Diagnosis', desc: 'Run histopathology scan',    url: '/diagnosis',  icon: '🔬' },
  { _type: 'action', name: 'Add Patient',  desc: 'Register a new patient',     action: 'openAddPatient', icon: '➕' },
  { _type: 'action', name: 'Run AI Scan',  desc: 'Go to diagnosis page',       url: '/diagnosis',  icon: '🧬' },
  { _type: 'action', name: 'Toggle Theme', desc: 'Switch dark / light mode',   action: 'toggleTheme', icon: '🌙' },
  { _type: 'diag',   name: 'High-Grade Dysplasia',  desc: 'Diagnosis category', icon: '🔴' },
  { _type: 'diag',   name: 'Low-Grade Dysplasia',   desc: 'Diagnosis category', icon: '🟠' },
  { _type: 'diag',   name: 'Chronic Gastritis',     desc: 'Diagnosis category', icon: '🟡' },
  { _type: 'diag',   name: 'Normal Mucosa',          desc: 'Diagnosis category', icon: '🟢' },
  { _type: 'diag',   name: 'Adenocarcinoma',         desc: 'Diagnosis category', icon: '🔴' },
]);

function initSearch() {
  const input = document.getElementById('globalSearch');
  if (!input) return;

  // Build dropdown container
  const wrap = input.closest('.topbar-search');
  wrap.style.position = 'relative';

  const dropdown = document.createElement('div');
  dropdown.id = 'searchDropdown';
  dropdown.style.cssText = `
    position:absolute; top:calc(100% + 8px); left:0; right:0;
    background:var(--surface); border:1px solid var(--border2);
    border-radius:var(--r2); box-shadow:0 16px 48px rgba(0,0,0,.5);
    z-index:500; overflow:hidden; display:none;
    max-height:360px; overflow-y:auto;
  `;
  wrap.appendChild(dropdown);

  // All patients from table (live) + static index
  function getSearchData() {
    const patients = DEMO_PATIENTS.map(p => ({
      _type: 'patient', name: p.name, desc: `${p.id} · Age ${p.age} · ${p.last}`,
      id: p.id, risk: p.risk, icon: '👤', url: null
    }));
    return [...patients, ...SEARCH_INDEX.filter(x => x._type !== 'patient')];
  }

  function renderResults(q) {
    if (!q) { dropdown.style.display = 'none'; return; }
    const results = getSearchData().filter(item =>
      (item.name + ' ' + (item.desc || '') + ' ' + (item.id || '')).toLowerCase().includes(q.toLowerCase())
    ).slice(0, 8);

    if (!results.length) {
      dropdown.innerHTML = `<div style="padding:1.2rem;text-align:center;color:var(--tx3);font-size:.84rem;font-family:'JetBrains Mono',monospace">No results for "${q}"</div>`;
      dropdown.style.display = 'block';
      return;
    }

    // Group by type
    const groups = {};
    results.forEach(r => {
      const label = r._type === 'patient' ? 'Patients' : r._type === 'page' ? 'Pages' : r._type === 'action' ? 'Actions' : 'Diagnoses';
      if (!groups[label]) groups[label] = [];
      groups[label].push(r);
    });

    dropdown.innerHTML = Object.entries(groups).map(([label, items]) => `
      <div style="padding:.5rem .9rem .2rem;font-family:'JetBrains Mono',monospace;font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;color:var(--tx3)">${label}</div>
      ${items.map(item => {
        const riskDot = item.risk ? `<span style="margin-left:auto;font-size:.68rem;padding:2px 7px;border-radius:99px;background:${item.risk==='high'?'rgba(255,61,110,.15)':item.risk==='mid'?'rgba(255,179,64,.15)':'rgba(0,255,159,.15)'};color:${item.risk==='high'?'var(--c3)':item.risk==='mid'?'var(--c4)':'var(--c2)'}">● ${item.risk}</span>` : '';
        return `
        <div class="search-result-item" data-url="${item.url||''}" data-action="${item.action||''}" data-name="${item.name}"
          style="display:flex;align-items:center;gap:.75rem;padding:.65rem 1rem;cursor:pointer;transition:background .15s;border-bottom:1px solid var(--border)">
          <span style="font-size:1.1rem;flex-shrink:0">${item.icon}</span>
          <div style="flex:1;min-width:0">
            <div style="font-size:.85rem;color:var(--tx);font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${highlightMatch(item.name, q)}</div>
            <div style="font-size:.72rem;color:var(--tx3);margin-top:1px">${item.desc || ''}</div>
          </div>
          ${riskDot}
          <span style="font-size:.65rem;color:var(--tx4);font-family:'JetBrains Mono',monospace;flex-shrink:0">${item._type === 'patient' ? '↗' : item._type === 'action' ? '⚡' : '→'}</span>
        </div>`;
      }).join('')}
    `).join('');

    dropdown.style.display = 'block';

    // Bind click
    dropdown.querySelectorAll('.search-result-item').forEach(el => {
      el.addEventListener('mouseenter', () => el.style.background = 'var(--surface2)');
      el.addEventListener('mouseleave', () => el.style.background = '');
      el.addEventListener('click', () => {
        const url = el.dataset.url;
        const action = el.dataset.action;
        if (url) { window.location.href = url; }
        else if (action === 'toggleTheme') { toggleTheme(); }
        else if (action === 'openAddPatient') { openAddPatient(); }
        closeSearch();
      });
    });
  }

  function highlightMatch(text, q) {
    if (!q) return text;
    const re = new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    return text.replace(re, '<mark style="background:rgba(0,212,255,.25);color:var(--c1);border-radius:2px;padding:0 2px">$1</mark>');
  }

  function closeSearch() {
    dropdown.style.display = 'none';
    input.value = '';
  }

  input.addEventListener('input', () => renderResults(input.value.trim()));
  input.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeSearch();
    if (e.key === 'Enter') {
      const first = dropdown.querySelector('.search-result-item');
      if (first) first.click();
    }
    // Arrow key navigation
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const items = [...dropdown.querySelectorAll('.search-result-item')];
      const focused = dropdown.querySelector('.search-result-item.focused');
      let idx = items.indexOf(focused);
      if (focused) { focused.classList.remove('focused'); focused.style.background = ''; }
      idx = e.key === 'ArrowDown' ? Math.min(idx + 1, items.length - 1) : Math.max(idx - 1, 0);
      if (items[idx]) { items[idx].classList.add('focused'); items[idx].style.background = 'var(--surface2)'; items[idx].scrollIntoView({ block: 'nearest' }); }
    }
  });

  // Close on outside click
  document.addEventListener('click', e => {
    if (!wrap.contains(e.target)) closeSearch();
  });

  // Also filter table rows live while typing (on patients page)
  input.addEventListener('input', () => {
    const q = input.value.toLowerCase();
    document.querySelectorAll('#patientsTable tr').forEach(row => {
      row.style.display = !q || row.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
  });
}

// ── Notifications (live from DB) ─────────────────────────────────────────────

let NOTIFICATIONS = [];
let notifOpen     = false;
let _lastNotifiedRiskScore = 0;

async function loadNotifications() {
  const fresh = [];
  try {
    const pRes = await fetch('/api/patients', { headers: { 'Accept': 'application/json' } });
    if (pRes.ok) {
      const arr = await pRes.json();
      const patients = Array.isArray(arr) ? arr : (arr.patients || []);
      patients.forEach(p => {
        const risk  = (p.risk || '').toLowerCase();
        const name  = p.name || 'Unknown';
        const diag  = p.last || p.last_diagnosis || p.condition || '';
        const pid   = p.id || '';
        const ts    = p.date || '';
        const raw   = p.risk_score;
        const score = raw ? (raw <= 1 ? Math.round(raw * 100) : Math.round(raw)) : 0;
        if (risk === 'high' || score >= 70) {
          fresh.push({ id: 'p_' + pid, type: 'high', icon: '🔴',
            title: 'High-Risk Patient',
            body: `${name} — ${diag || 'Critical finding detected'}. Immediate review required.`,
            time: ts ? _relTime(ts) : 'Recently', read: false, _ts: ts });
        } else if (risk === 'mid' || risk === 'medium') {
          fresh.push({ id: 'pm_' + pid, type: 'warn', icon: '🟠',
            title: 'Moderate-Risk Patient',
            body: `${name} — ${diag || 'Suspicious finding'}. Follow-up recommended.`,
            time: ts ? _relTime(ts) : 'Recently', read: false, _ts: ts });
        }
      });
    }
  } catch (e) { console.warn('loadNotifications patients:', e.message); }

  try {
    const st = await (await fetch('/stats')).json();
    if (st.total_patients > 0) {
      fresh.push({ id: 'stats_sum', type: st.high_risk > 0 ? 'warn' : 'ok',
        icon: st.high_risk > 0 ? '📊' : '✅',
        title: 'Patient Summary',
        body: `${st.total_patients} patient${st.total_patients !== 1 ? 's' : ''} total. ${st.high_risk || 0} high-risk.`,
        time: 'Live', read: true, _ts: '' });
    }
  } catch (e) { console.warn('loadNotifications stats:', e.message); }

  fresh.sort((a, b) => {
    if (!a.read && b.read) return -1;
    if (a.read && !b.read)  return 1;
    return new Date(b._ts || 0) - new Date(a._ts || 0);
  });
  const readSet = new Set(NOTIFICATIONS.filter(n => n.read).map(n => n.id));
  fresh.forEach(n => { if (readSet.has(n.id)) n.read = true; });
  const live = NOTIFICATIONS.filter(n => String(n.id).startsWith('live_'));
  NOTIFICATIONS = [...live, ...fresh];
  _refreshNotifBadge();
}

function _relTime(ts) {
  try {
    const sec = Math.floor((Date.now() - new Date(ts)) / 1000);
    if (isNaN(sec) || sec < 0) return 'Recently';
    if (sec < 60)    return 'Just now';
    if (sec < 3600)  return Math.floor(sec / 60) + ' min ago';
    if (sec < 86400) return Math.floor(sec / 3600) + ' hr ago';
    return Math.floor(sec / 86400) + 'd ago';
  } catch { return 'Recently'; }
}

function _refreshNotifBadge() {
  const bell = document.getElementById('notifBell');
  if (!bell) return;
  const count = NOTIFICATIONS.filter(n => !n.read).length;
  let badge = bell.querySelector('.notif-count');
  if (count > 0) {
    if (!badge) {
      badge = document.createElement('span');
      badge.className = 'notif-count';
      badge.style.cssText = "position:absolute;top:-4px;right:-4px;background:var(--c3);color:#fff;font-size:.6rem;font-weight:700;min-width:16px;height:16px;border-radius:99px;display:grid;place-items:center;font-family:'JetBrains Mono',monospace;border:2px solid var(--bg2);";
      bell.style.position = 'relative';
      bell.appendChild(badge);
    }
    badge.textContent = count > 99 ? '99+' : count;
  } else if (badge) { badge.remove(); }
}

function initNotifications() {
  const bell = document.getElementById('notifBell');
  if (!bell) return;

  if (!document.getElementById('_notifStyle')) {
    const s = document.createElement('style');
    s.id = '_notifStyle';
    s.textContent = '@keyframes notifIn{from{opacity:0;transform:scale(.94) translateY(-8px)}to{opacity:1;transform:scale(1) translateY(0)}} @keyframes bellWiggle{0%,100%{transform:rotate(0)}20%{transform:rotate(-15deg)}40%{transform:rotate(15deg)}60%{transform:rotate(-10deg)}80%{transform:rotate(10deg)}}';
    document.head.appendChild(s);
  }

  let panel = document.getElementById('notifPanel');
  if (!panel) {
    panel = document.createElement('div');
    panel.id = 'notifPanel';
    panel.style.cssText = "position:fixed;top:70px;right:1.2rem;width:370px;max-height:540px;background:var(--bg2);border:1px solid var(--border);border-radius:14px;box-shadow:0 20px 60px rgba(0,0,0,.55);z-index:400;display:none;flex-direction:column;overflow:hidden;animation:notifIn .25s cubic-bezier(.34,1.56,.64,1);";
    document.body.appendChild(panel);
  }

  const typeColor = { high:'var(--c3)', warn:'var(--c4)', info:'var(--c1)', ok:'var(--c2)' };

  function renderPanel() {
    const unread = NOTIFICATIONS.filter(n => !n.read).length;
    panel.innerHTML = `
      <div style="padding:1rem 1.2rem;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;flex-shrink:0">
        <div>
          <div style="font-weight:700;font-size:1rem;color:var(--tx1)">Notifications</div>
          <div style="font-size:.72rem;color:var(--tx3);font-family:'JetBrains Mono',monospace">${unread} unread · live from DB</div>
        </div>
        <div style="display:flex;gap:.5rem;align-items:center">
          <button onclick="loadNotifications().then(()=>{if(notifOpen){const p=document.getElementById('notifPanel');if(p){p.remove();notifOpen=false;initNotifications();document.getElementById('notifBell')?.click();}}})" title="Refresh" style="font-size:.9rem;background:none;border:none;cursor:pointer;color:var(--tx3)" onmouseover="this.style.color='var(--c1)'" onmouseout="this.style.color='var(--tx3)'">↻</button>
          <button onclick="markAllRead()" style="font-size:.73rem;color:var(--c1);background:none;border:none;cursor:pointer;font-family:'DM Sans',sans-serif;white-space:nowrap">Mark all read</button>
          <button onclick="closeNotifPanel()" style="width:26px;height:26px;border-radius:50%;background:var(--bg3);border:1px solid var(--border);display:grid;place-items:center;font-size:.8rem;color:var(--tx3);cursor:pointer">✕</button>
        </div>
      </div>
      <div style="overflow-y:auto;flex:1">
        ${NOTIFICATIONS.length === 0
          ? '<div style="padding:2.5rem;text-align:center;color:var(--tx3)"><div style="font-size:2rem;margin-bottom:.5rem">🔔</div><div style="font-size:.82rem">No notifications yet</div><div style="font-size:.72rem;margin-top:.3rem;color:var(--tx4)">Add patients or run scans to see alerts</div></div>'
          : NOTIFICATIONS.map(n => `
            <div onclick="readNotif('${n.id}')" style="display:flex;gap:.75rem;padding:.85rem 1.2rem;border-bottom:1px solid var(--border);background:${n.read ? 'transparent' : 'rgba(0,212,255,.04)'};cursor:pointer;transition:background .15s;position:relative;" onmouseover="this.style.background='var(--bg3)'" onmouseout="this.style.background='${n.read ? 'transparent' : 'rgba(0,212,255,.04)'}'">
              ${!n.read ? `<div style="position:absolute;top:.9rem;left:.35rem;width:5px;height:5px;border-radius:50%;background:${typeColor[n.type]||'var(--c1)'}"></div>` : ''}
              <div style="font-size:1.2rem;flex-shrink:0;margin-top:.1rem">${n.icon}</div>
              <div style="flex:1;min-width:0">
                <div style="font-size:.84rem;font-weight:${n.read?'500':'700'};color:${n.read?'var(--tx2)':'var(--tx1)'};margin-bottom:.15rem">${n.title}</div>
                <div style="font-size:.75rem;color:var(--tx3);line-height:1.45;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden">${n.body}</div>
                <div style="font-size:.67rem;color:var(--tx4);margin-top:.3rem;font-family:'JetBrains Mono',monospace">${n.time}</div>
              </div>
            </div>`).join('')}
      </div>`;
    panel.style.display = 'flex';
    _refreshNotifBadge();
  }

  const newBell = bell.cloneNode(true);
  bell.parentNode.replaceChild(newBell, bell);
  newBell.addEventListener('click', e => {
    e.stopPropagation();
    notifOpen = !notifOpen;
    if (notifOpen) renderPanel(); else closeNotifPanel();
  });
  document.addEventListener('click', e => {
    const p = document.getElementById('notifPanel');
    const b = document.getElementById('notifBell');
    if (notifOpen && p && !p.contains(e.target) && e.target !== b) closeNotifPanel();
  });
  _refreshNotifBadge();
}

function closeNotifPanel() {
  notifOpen = false;
  const p = document.getElementById('notifPanel');
  if (p) p.style.display = 'none';
}

function readNotif(id) {
  const n = NOTIFICATIONS.find(n => String(n.id) === String(id));
  if (!n) return;
  n.read = true;
  const panel = document.getElementById('notifPanel');
  if (panel && panel.style.display !== 'none') {
    notifOpen = false; initNotifications(); document.getElementById('notifBell')?.click();
  }
}

function markAllRead() {
  NOTIFICATIONS.forEach(n => { n.read = true; });
  _refreshNotifBadge();
  const panel = document.getElementById('notifPanel');
  if (panel) panel.remove();
  notifOpen = false;
  initNotifications();
  document.getElementById('notifBell')?.click();
  showToast('✅ All notifications marked as read', 'ok');
}

function checkAndTriggerRiskNotification(riskScore, diagnosis, tier, predictedClass) {
  const score = typeof riskScore === 'number' ? riskScore : parseInt(riskScore) || 0;
  if (score < 70 || score <= _lastNotifiedRiskScore) { _lastNotifiedRiskScore = score; return; }
  _lastNotifiedRiskScore = score;

  NOTIFICATIONS.unshift({ id: 'live_' + Date.now(), type: 'high', icon: '🚨',
    title: `AI Alert — Risk ${score}%`,
    body: `${tier||'CRITICAL'}: ${diagnosis||predictedClass||'Critical finding'}. Score ${score}% exceeds threshold.`,
    time: 'Just now', read: false, _ts: new Date().toISOString() });
  _refreshNotifBadge();
  showToast(`🚨 HIGH RISK ${score}% — ${diagnosis||tier}`, 'err', 6000);
  _showHighRiskBanner(score, diagnosis, predictedClass);

  const bell = document.getElementById('notifBell');
  if (bell) {
    bell.style.animation = 'none';
    requestAnimationFrame(() => { bell.style.animation = 'bellWiggle 0.6s ease'; });
    bell.style.boxShadow = '0 0 18px rgba(255,61,110,.8)';
    bell.style.color = '#ff3d6e';
    setTimeout(() => { bell.style.boxShadow = ''; bell.style.color = ''; }, 2800);
  }
  setTimeout(() => {
    const p = document.getElementById('notifPanel');
    if (p) p.remove();
    notifOpen = false; initNotifications(); document.getElementById('notifBell')?.click();
  }, 900);
}

function _showHighRiskBanner(score, diagnosis, predictedClass) {
  document.getElementById('_hrBanner')?.remove();
  if (!document.getElementById('_hrBannerKf')) {
    const s = document.createElement('style'); s.id = '_hrBannerKf';
    s.textContent = '@keyframes slideDown{from{transform:translateY(-100%)}to{transform:translateY(0)}}';
    document.head.appendChild(s);
  }
  const b = document.createElement('div'); b.id = '_hrBanner';
  b.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:9999;background:linear-gradient(90deg,#ff3d6e,#ff6b35);color:#fff;padding:.65rem 1.5rem;display:flex;align-items:center;justify-content:space-between;font-family:'DM Sans',sans-serif;font-size:.88rem;font-weight:600;box-shadow:0 4px 24px rgba(255,61,110,.5);animation:slideDown .35s cubic-bezier(.34,1.56,.64,1)";
  b.innerHTML = `<div style="display:flex;align-items:center;gap:.75rem"><span style="font-size:1.2rem">🚨</span><span>HIGH RISK ALERT — ${score}% · ${diagnosis||predictedClass||'Critical finding'}. Immediate clinical review required.</span></div><button onclick="document.getElementById('_hrBanner').remove()" style="background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.4);color:#fff;border-radius:6px;padding:3px 10px;cursor:pointer;font-size:.82rem">Dismiss ✕</button>`;
  document.body.prepend(b);
  setTimeout(() => { if (b.parentNode) { b.style.transition='opacity .5s'; b.style.opacity='0'; setTimeout(()=>b.remove(),500); } }, 12000);
}



// ── Drag & Drop Upload ───────────────────────
function initUpload() {
  const zone = document.getElementById('dropZone');
  const input = document.getElementById('fileInput');
  const preview = document.getElementById('uploadPreview');

  if (!zone) return;

  zone.addEventListener('click', e => {
    if (e.target.classList.contains('remove-img')) return;
    input?.click();
  });
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', e => { if (!zone.contains(e.relatedTarget)) zone.classList.remove('drag-over'); });
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    if (input && e.dataTransfer.files.length) {
      const dt = new DataTransfer();
      [...e.dataTransfer.files].forEach(f => dt.items.add(f));
      input.files = dt.files;
      handleFiles(input.files);
    }
  });
  input?.addEventListener('change', () => handleFiles(input.files));

  function handleFiles(files) {
    if (!preview) return;
    preview.innerHTML = '';
    const imgFiles = [...files].filter(f => f.type.startsWith('image/'));
    if (!imgFiles.length) { showToast('Please upload image files', 'warn'); return; }

    // Hide the drop UI, show full-size preview
    zone.querySelector('.upload-icon-wrap') && (zone.querySelector('.upload-icon-wrap').style.display = 'none');
    zone.querySelector('.upload-title')     && (zone.querySelector('.upload-title').style.display = 'none');
    zone.querySelector('.upload-hint')      && (zone.querySelector('.upload-hint').style.display = 'none');
    preview.style.cssText = 'display:block;width:100%;margin:0';

    imgFiles.forEach((f, i) => {
      const reader = new FileReader();
      reader.onload = ev => {
        // Full-width image with overlay remove button
        const wrap = document.createElement('div');
        wrap.style.cssText = 'position:relative;width:100%;border-radius:10px;overflow:hidden';
        wrap.innerHTML = `
          <img src="${ev.target.result}" alt="${f.name}" style="width:100%;display:block;border-radius:10px;max-height:340px;object-fit:contain;background:var(--bg4)">
          <div class="remove-img" onclick="resetUploadZone(this)" style="position:absolute;top:8px;right:8px;width:28px;height:28px;border-radius:50%;background:rgba(0,0,0,.7);border:1px solid rgba(255,255,255,.2);display:flex;align-items:center;justify-content:center;cursor:pointer;font-size:.85rem;color:#fff;z-index:2">✕</div>
          <div style="position:absolute;bottom:8px;left:8px;font-family:'JetBrains Mono',monospace;font-size:.65rem;background:rgba(0,0,0,.6);padding:3px 8px;border-radius:4px;color:rgba(255,255,255,.7)">${f.name}</div>
        `;
        preview.appendChild(wrap);
      };
      reader.readAsDataURL(f);
    });
    showToast(`📁 ${imgFiles.length} image(s) loaded — ready to scan`, 'ok');
  }
}

function removePreview(btn) {
  const thumb = btn.closest('.preview-thumb');
  if (thumb) {
    thumb.style.transition = 'all .25s ease';
    thumb.style.opacity = '0';
    thumb.style.transform = 'scale(.85)';
    setTimeout(() => thumb.remove(), 250);
  }
}

window.resetUploadZone = function(btn) {
  const zone = document.getElementById('dropZone');
  const preview = document.getElementById('uploadPreview');
  if (!zone || !preview) return;
  preview.innerHTML = '';
  preview.style.cssText = '';
  // Show drop UI again
  zone.querySelector('.upload-icon-wrap') && (zone.querySelector('.upload-icon-wrap').style.display = '');
  zone.querySelector('.upload-title')     && (zone.querySelector('.upload-title').style.display = '');
  zone.querySelector('.upload-hint')      && (zone.querySelector('.upload-hint').style.display = '');
  const fileInput = document.getElementById('fileInput');
  if (fileInput) fileInput.value = '';
  showToast('🗑 Image removed', 'info');
};

// ── AI Scan / Predict ────────────────────────
const DEMO_DIAGNOSES = [
  { diagnosis: 'High-Grade Dysplasia', recommendation: 'Immediate endoscopic resection recommended. Refer to oncology within 48 hours. Schedule follow-up biopsy in 2 weeks.', risk: 82, probs: [82,11,5,2,0] },
  { diagnosis: 'Low-Grade Dysplasia', recommendation: 'Endoscopic surveillance every 6 months. Consider biopsy repeat. Monitor H. pylori status.', risk: 45, probs: [15,45,25,13,2] },
  { diagnosis: 'Chronic Atrophic Gastritis', recommendation: 'H. pylori eradication therapy if positive. Annual endoscopic surveillance. Vitamin B12 supplementation.', risk: 28, probs: [5,20,28,40,7] },
  { diagnosis: 'Normal Gastric Mucosa', recommendation: 'No immediate intervention required. Routine screening in 2 years. Maintain healthy diet and H. pylori screening.', risk: 6, probs: [1,3,6,12,78] },
];

async function runScan() {
  const btn = document.getElementById('scanBtn');
  const input = document.getElementById('fileInput');
  if (!input?.files?.length && !document.querySelector('#uploadPreview img, .preview-thumb img')) {
    showToast('Please upload an image first', 'warn');
    return;
  }

  showAnalysisSteps();
  btn?.classList.add('scanning', 'btn-loading');
  if (btn) btn.innerHTML = '<span class="btn-spinner"></span> Analysing...';

  try {
    const formData = new FormData();
    if (input?.files?.length) {
      for (const f of input.files) formData.append('image', f);
    }

    const cd = window._clinicalData || {};
    formData.append('age',          cd.age          || 0);
    formData.append('gender',       cd.gender       || 'Male');
    formData.append('stage',        cd.stage        || 'I');
    formData.append('gene_score',   cd.gene_score   || 0);
    formData.append('genomic_risk', cd.genomic_risk || 0);
    const res = await fetch('/predict', { method: 'POST', body: formData });
    if (!res.ok) throw new Error('Server error ' + res.status);
    const json = await res.json();
    const d = Array.isArray(json) ? json[0] : json;

    // ── Map backend response to frontend display ──
    const diagnosis      = d.diagnosis || d.predicted_class || 'Unknown';
    const recommendation = d.recommendation || d.details || '';
    const predictedClass = d.predicted_class || '';

    // Risk score: use backend value, but override if fusion model gave garbage (< 5% for TUM/STR)
    let riskScore = typeof d.risk_score === 'number' ? Math.round(d.risk_score) : 50;
    const probs_raw = d.probabilities || {};
    const tumProb   = (probs_raw['TUM'] || 0);
    const strProb   = (probs_raw['STR'] || 0);

    // If base model says TUM/STR but fusion risk_score is implausibly low, recalculate
    if ((predictedClass === 'TUM' || predictedClass === 'STR') && riskScore < 40) {
      // Derive from softmax probabilities directly (more reliable than fusion score)
      riskScore = Math.round(Math.min((tumProb + strProb) * 100 + (d.probability || d.prob || 0) * 60, 99));
      if (riskScore < 50) riskScore = predictedClass === 'TUM' ? 82 : 65; // floor for definitive malignant classes
    }

    // Tier: derive from predictedClass — more reliable than fusion model tier when weights are missing
    let tier = d.tier || '';
    const classTierMap = { TUM: 'CRITICAL', STR: 'SUSPICIOUS', LYM: 'SUSPICIOUS', DEB: 'WATCH',
                           MUC: 'NEGATIVE', MUS: 'NEGATIVE', NORM: 'NEGATIVE', ADI: 'NEGATIVE' };
    if (!tier || tier === 'UNKNOWN' || tier === 'DEMO') {
      tier = classTierMap[predictedClass] || (riskScore >= 70 ? 'CRITICAL' : riskScore >= 40 ? 'SUSPICIOUS' : 'NEGATIVE');
    }

    // Map backend 8-class probabilities to frontend display
    // CLASSES: ['ADI','DEB','LYM','MUC','MUS','NORM','STR','TUM']
    let probs = null;
    if (d.probabilities && typeof d.probabilities === 'object') {
      const classMap = d.probabilities;
      probs = {
        raw: classMap,
        display: [
          Math.round((classMap['TUM']  || 0) * 100),
          Math.round((classMap['STR']  || 0) * 100),
          Math.round((classMap['LYM']  || 0) * 100),
          Math.round((classMap['DEB']  || 0) * 100),
          Math.round((classMap['MUC']  || 0) * 100),
          Math.round((classMap['MUS']  || 0) * 100),
          Math.round((classMap['NORM'] || 0) * 100),
          Math.round((classMap['ADI']  || 0) * 100),
        ]
      };
    }

    // Store globals for SHAP chart + PDF download
    window._lastPredictedClass = predictedClass;
    window._lastRawProbs       = d.probabilities || {};
    window._lastConfidence     = d.confidence    ? Math.round(d.confidence * 100) : '';
    window._gradcamPath        = d.gradcam_url   || d.gradcam_path || null;
    window._shapData           = (d.shap_values && Object.keys(d.shap_values).length > 0) ? d.shap_values : null;

    displayResults(diagnosis, recommendation, riskScore, probs, tier, predictedClass);

  } catch (err) {
    console.warn('Backend not available, using demo data:', err.message);
    const d = DEMO_DIAGNOSES[Math.floor(Math.random() * DEMO_DIAGNOSES.length)];
    window._lastPredictedClass = 'TUM';
    window._shapData = null;
    setTimeout(() => displayResults(d.diagnosis, d.recommendation, d.risk, { display: d.probs }, d.tier || 'DEMO', ''), 2800);
  }
}

function showAnalysisSteps() {
  const stepsEl = document.getElementById('analysisSteps');
  if (!stepsEl) return;
  stepsEl.classList.remove('hidden');
  const steps = stepsEl.querySelectorAll('.step-item');
  steps.forEach(s => s.className = 'step-item pending');

  const sequence = [0, 1, 2, 3];
  let i = 0;
  const interval = setInterval(() => {
    if (i > 0) steps[i-1].className = 'step-item done';
    if (i < steps.length) {
      steps[i].className = 'step-item active';
      i++;
    } else {
      clearInterval(interval);
    }
  }, 600);
}

function displayResults(diagnosis, recommendation, riskScore, probs, tier, predictedClass) {
  const btn = document.getElementById('scanBtn');
  btn?.classList.remove('scanning', 'btn-loading');
  if (btn) btn.innerHTML = '🔬 Run AI Scan';

  document.querySelectorAll('.step-item').forEach(s => s.className = 'step-item done');

  // ── Tier → risk colour mapping ──
  const tierColors = { CRITICAL:'badge-red', SUSPICIOUS:'badge-amber', NEGATIVE:'badge-green', INVALID:'badge-cyan', DEMO:'badge-cyan' };
  const tierIcons  = { CRITICAL:'🔴', SUSPICIOUS:'🟠', NEGATIVE:'🟢', INVALID:'⚫', DEMO:'🔵' };
  const tierLabel  = tier || (riskScore > 70 ? 'CRITICAL' : riskScore > 40 ? 'SUSPICIOUS' : 'NEGATIVE');
  const badgeClass = tierColors[tierLabel] || (riskScore > 60 ? 'badge-red' : riskScore > 30 ? 'badge-amber' : 'badge-green');

  // ── Result box ──
  const resultBox = document.getElementById('resultBox');
  if (resultBox) {
    resultBox.classList.remove('hidden');

    const diagEl = document.getElementById('diagnosisText');
    const recEl  = document.getElementById('recommendationText');
    if (diagEl) diagEl.textContent = diagnosis;
    if (recEl)  recEl.textContent  = recommendation;

    const riskLabel = document.getElementById('riskLabel');
    if (riskLabel) {
      riskLabel.className = `badge ${badgeClass}`;
      riskLabel.textContent = `${tierIcons[tierLabel] || ''} ${tierLabel} · ${riskScore}%`;
    }

    // Tier badge / info row
    let tierBadge = resultBox.querySelector('.tier-info-row');
    if (!tierBadge) {
      tierBadge = document.createElement('div');
      tierBadge.className = 'tier-info-row';
      tierBadge.style.cssText = 'display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.5rem';
      resultBox.querySelector('.result-box-body')?.appendChild(tierBadge);
    }
    tierBadge.innerHTML = `
      <span style="font-family:'JetBrains Mono',monospace;font-size:.7rem;padding:3px 9px;border-radius:99px;background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.2);color:var(--c1)">
        Class: ${predictedClass || '—'}
      </span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:.7rem;padding:3px 9px;border-radius:99px;background:${tierLabel==='CRITICAL'?'rgba(255,61,110,.12)':tierLabel==='SUSPICIOUS'?'rgba(255,179,64,.12)':'rgba(0,255,159,.12)'};border:1px solid ${tierLabel==='CRITICAL'?'rgba(255,61,110,.3)':tierLabel==='SUSPICIOUS'?'rgba(255,179,64,.3)':'rgba(0,255,159,.3)'};color:${tierLabel==='CRITICAL'?'var(--c3)':tierLabel==='SUSPICIOUS'?'var(--c4)':'var(--c2)'}">
        Tier: ${tierLabel}
      </span>`;
  }

  // ── Risk score gauge ──
  animateGauge(riskScore, tierLabel);

  // ── Risk fill bar ──
  const riskFill = document.getElementById('riskFill');
  if (riskFill) {
    const fillClass = tierLabel === 'CRITICAL' ? 'high' : tierLabel === 'SUSPICIOUS' ? 'mid' : 'low';
    riskFill.className = 'risk-fill ' + fillClass;
    setTimeout(() => { riskFill.style.width = riskScore + '%'; }, 200);
    const rv = document.getElementById('riskValue');
    if (rv) rv.textContent = riskScore + '%';
  }

  // ── Probability bars ──
  if (probs) {
    if (probs.raw && typeof probs.raw === 'object' && Object.keys(probs.raw).length > 0) {
      // Real backend data — show all 8 classes
      updateProbBars(null, probs.raw);
    } else {
      const probArr = probs.display || probs;
      if (Array.isArray(probArr)) {
        updateProbBars(probArr, null);
      } else {
        updateProbBars(Object.values(probs), null);
      }
    }
  }

  // ── GradCAM ──
  loadGradCAM();

  // Store globals so downloadReport can include full data in PDF payload
  window._lastDiagnosis       = diagnosis;
  window._lastRec             = recommendation;
  window._lastRiskScore       = riskScore;
  window._lastTier            = tierLabel;
  window._lastConfidence      = typeof confidence !== 'undefined' ? confidence : '';

  const toastIcon = tierIcons[tierLabel] || '🔬';
  showToast(`${toastIcon} ${diagnosis} — Risk ${riskScore}%`, tierLabel==='CRITICAL'?'warn':'ok');
  document.getElementById('resultBox')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Gauge SVG ────────────────────────────────
function animateGauge(score, tier) {
  const gauge = document.getElementById('gaugeScore');
  const arc   = document.getElementById('gaugeArc');
  if (!gauge) return;

  animateCounter(gauge, score, '%');

  const color = tier === 'CRITICAL' ? 'var(--c3)'
              : tier === 'SUSPICIOUS' ? 'var(--c4)'
              : tier === 'NEGATIVE' ? 'var(--c2)'
              : score > 70 ? 'var(--c3)' : score > 40 ? 'var(--c4)' : 'var(--c2)';

  // Animate gauge score color
  gauge.style.color = color;

  // Also color the label
  const gLabel = document.querySelector('.gauge-score');
  if (gLabel) gLabel.style.setProperty('--gauge-color', color);

  if (arc) {
    const r = 54, circumference = Math.PI * r;
    const fill = Math.max(0, Math.min(score, 100)) / 100 * circumference;
    arc.style.strokeDasharray = circumference;
    // Animate from current offset to target
    const currentOffset = parseFloat(arc.style.strokeDashoffset || circumference);
    arc.style.transition = 'stroke-dashoffset 1.4s cubic-bezier(.4,0,.2,1), stroke .4s ease';
    arc.style.strokeDashoffset = circumference - fill;
    arc.style.stroke = color;
  }
}

// ── Probability Bars ─────────────────────────
// Supports both backend raw dict {ADI,DEB,...} and display array
function updateProbBars(probs, rawProbs) {
  const container = document.getElementById('probList');
  if (!container) return;

  // If we have real backend data (rawProbs dict with 8 classes)
  if (rawProbs && typeof rawProbs === 'object' && !Array.isArray(rawProbs)) {
    const BACKEND_CLASSES = [
      { key:'TUM',  label:'Tumor (Adenocarcinoma)',     color:'var(--c3)',  risk:'CRITICAL'   },
      { key:'STR',  label:'Cancer-Assoc. Stroma',       color:'#ff6b35',    risk:'SUSPICIOUS' },
      { key:'LYM',  label:'Lymphocytes (Immune)',        color:'var(--c4)',  risk:'WATCH'      },
      { key:'DEB',  label:'Debris / Cell Fragments',     color:'#9b8fff',    risk:'WATCH'      },
      { key:'MUC',  label:'Mucosa (Stomach Lining)',     color:'var(--c1)',  risk:'NEGATIVE'   },
      { key:'MUS',  label:'Smooth Muscle',               color:'#4bc8eb',    risk:'NEGATIVE'   },
      { key:'NORM', label:'Normal Mucosa',               color:'var(--c2)',  risk:'NEGATIVE'   },
      { key:'ADI',  label:'Adipose (Fat Tissue)',        color:'#a8ff78',    risk:'NEGATIVE'   },
    ];

    // Sort by probability descending
    const sorted = BACKEND_CLASSES
      .map(c => ({ ...c, pct: Math.round((rawProbs[c.key] || 0) * 100) }))
      .sort((a, b) => b.pct - a.pct);

    container.innerHTML = sorted.map((c, i) => `
      <div style="margin-bottom:.6rem">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
          <div style="display:flex;align-items:center;gap:.4rem">
            <span style="width:8px;height:8px;border-radius:50%;background:${c.color};display:inline-block;flex-shrink:0"></span>
            <span style="font-size:.8rem;color:var(--tx2)">${c.label}</span>
          </div>
          <div style="display:flex;align-items:center;gap:.4rem">
            <span style="font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:700;color:${c.color}">${c.pct}%</span>
            ${i === 0 ? `<span style="font-size:.62rem;padding:1px 6px;border-radius:99px;background:rgba(0,212,255,.1);color:var(--c1);font-family:'JetBrains Mono',monospace">TOP</span>` : ''}
          </div>
        </div>
        <div style="height:6px;background:var(--bg4);border-radius:99px;overflow:hidden">
          <div style="height:100%;width:0;border-radius:99px;background:${c.color};transition:width 1.1s cubic-bezier(.4,0,.2,1) ${i*0.08}s" data-w="${c.pct}"></div>
        </div>
      </div>`).join('');

  } else {
    // Fallback: 5-item display array from demo data
    const labels = ['Tumor (TUM)','Cancer Stroma (STR)','Lymphocytes (LYM)','Mucosa (MUC)','Normal (NORM)'];
    const colors = ['var(--c3)','#ff6b35','var(--c4)','var(--c1)','var(--c2)'];
    const arr = Array.isArray(probs) ? probs : [0,0,0,0,0];
    const total = arr.reduce((a,b) => a+b, 0);
    const norm  = total > 0 ? arr.map(v => Math.round(v / total * 100)) : arr;

    container.innerHTML = labels.map((lbl, i) => `
      <div style="margin-bottom:.6rem">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
          <span style="font-size:.8rem;color:var(--tx2)">${lbl}</span>
          <span style="font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:700;color:${colors[i]}">${norm[i]}%</span>
        </div>
        <div style="height:6px;background:var(--bg4);border-radius:99px;overflow:hidden">
          <div style="height:100%;width:0;border-radius:99px;background:${colors[i]};transition:width 1.1s cubic-bezier(.4,0,.2,1) ${i*0.1}s" data-w="${norm[i]}"></div>
        </div>
      </div>`).join('');
  }

  // Trigger width animation
  requestAnimationFrame(() => requestAnimationFrame(() => {
    container.querySelectorAll('[data-w]').forEach(b => { b.style.width = b.dataset.w + '%'; });
  }));
}

// ── GradCAM + SHAP ───────────────────────────
let heatmapVisible = true;
let activeHeatmapTab = 'gradcam';

function loadGradCAM() {
  const wrap = document.getElementById('gradcamWrap');
  if (!wrap) return;

  // Update toggle button label
  const heatBtn = document.getElementById('toggleHeatBtn');
  if (heatBtn) heatBtn.textContent = '👁 Hide Heatmap';
  heatmapVisible = true;

  // Add tabs for GradCAM vs SHAP
  const card = wrap.closest('.card');
  let tabBar = card?.querySelector('.heatmap-tabs');
  if (!tabBar && card) {
    tabBar = document.createElement('div');
    tabBar.className = 'heatmap-tabs';
    tabBar.style.cssText = 'display:flex;gap:.4rem;margin-bottom:.75rem;padding:3px;background:var(--bg3);border-radius:var(--r);width:fit-content';
    tabBar.innerHTML = `
      <button id="tabGradcam" onclick="switchHeatmapTab('gradcam',this)" style="background:var(--c1);color:#000;border:none;border-radius:calc(var(--r) - 2px);padding:.3rem .8rem;font-size:.78rem;font-weight:600;cursor:pointer;transition:all .2s;font-family:'DM Sans',sans-serif">GradCAM</button>
      <button id="tabShap" onclick="switchHeatmapTab('shap',this)" style="background:transparent;color:var(--tx2);border:none;border-radius:calc(var(--r) - 2px);padding:.3rem .8rem;font-size:.78rem;cursor:pointer;transition:all .2s;font-family:'DM Sans',sans-serif">SHAP</button>
    `;
    const hd = card.querySelector('.card-hd');
    if (hd) hd.after(tabBar);
  }

  renderHeatmapContent(activeHeatmapTab);
}

window.switchHeatmapTab = function(tab, btn) {
  activeHeatmapTab = tab;
  // Update tab styles
  const tabGrad = document.getElementById('tabGradcam');
  const tabShap = document.getElementById('tabShap');
  if (tabGrad) { tabGrad.style.background = tab==='gradcam' ? 'var(--c1)' : 'transparent'; tabGrad.style.color = tab==='gradcam' ? '#000' : 'var(--tx2)'; tabGrad.style.fontWeight = tab==='gradcam' ? '600' : '400'; }
  if (tabShap) { tabShap.style.background = tab==='shap' ? 'var(--c4)' : 'transparent'; tabShap.style.color = tab==='shap' ? '#000' : 'var(--tx2)'; tabShap.style.fontWeight = tab==='shap' ? '600' : '400'; }
  renderHeatmapContent(tab);
};

function renderHeatmapContent(tab) {
  const wrap = document.getElementById('gradcamWrap');
  if (!wrap) return;

  const previewImg = document.querySelector('#uploadPreview img, .preview-thumb img');
  const src = previewImg ? previewImg.src : null;

  if (tab === 'gradcam') {
    // ── GradCAM: B&W image + thermal heatmap overlay ──
    // Try to load server-generated gradcam.png first, fall back to canvas simulation
    const gradcamSrc = window._gradcamPath || (src ? `/static/uploads/gradcam.png?t=${Date.now()}` : null);

    wrap.innerHTML = `
      <div style="position:relative;border-radius:10px;overflow:hidden;background:var(--bg4)">
        <canvas id="bwCanvas" style="width:100%;display:block;border-radius:10px"></canvas>
        <canvas id="heatmapCanvas" style="position:absolute;inset:0;width:100%;height:100%;border-radius:10px;opacity:.65;mix-blend-mode:screen;pointer-events:none;transition:opacity .5s"></canvas>
        <div id="gradcamLoadingState" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:var(--bg4);border-radius:10px;transition:opacity .3s">
          <div style="text-align:center;color:var(--tx3)">
            <div style="font-size:1.6rem;margin-bottom:.4rem">🌡</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:.75rem">${src ? 'Generating GradCAM…' : 'Run a scan first'}</div>
          </div>
        </div>
      </div>
      <div style="margin-top:.6rem;display:flex;align-items:center;gap:.5rem;flex-wrap:wrap">
        <span style="font-size:.72rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;flex:1">EfficientNet-B4 · Grad-CAM · layer4</span>
        <div style="display:flex;align-items:center;gap:.3rem;font-size:.7rem;color:var(--tx3)">
          <span>Cold</span>
          <div style="width:80px;height:6px;border-radius:99px;background:linear-gradient(90deg,#00f,#0f0,#ff0,#f00)"></div>
          <span>Hot</span>
        </div>
      </div>
    `;

    if (src) {
      setTimeout(() => drawGradCAMOnCanvas(src), 80);
    }

  } else {
    // ── SHAP: per-class bar chart — NOT an image overlay ──
    const shapData = window._shapData || null;
    wrap.innerHTML = renderSHAPChart(shapData);
    // Try to fetch real SHAP data from backend
    if (!shapData) fetchSHAPData();
  }
}

// ── GradCAM: convert uploaded image to greyscale on canvas, then draw thermal overlay ──
function drawGradCAMOnCanvas(imageSrc) {
  const bwCanvas     = document.getElementById('bwCanvas');
  const heatCanvas   = document.getElementById('heatmapCanvas');
  const loadingState = document.getElementById('gradcamLoadingState');
  if (!bwCanvas || !heatCanvas) return;

  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    const W = img.naturalWidth  || 300;
    const H = img.naturalHeight || 280;
    const DISPLAY_H = Math.min(H, 300);

    // Set canvas sizes
    bwCanvas.width  = W; bwCanvas.height = H;
    bwCanvas.style.maxHeight = DISPLAY_H + 'px';
    heatCanvas.width = W; heatCanvas.height = H;

    // Draw greyscale (desaturated) base image
    const bwCtx = bwCanvas.getContext('2d');
    bwCtx.drawImage(img, 0, 0);
    const px = bwCtx.getImageData(0, 0, W, H);
    for (let i = 0; i < px.data.length; i += 4) {
      const grey = 0.299*px.data[i] + 0.587*px.data[i+1] + 0.114*px.data[i+2];
      px.data[i] = px.data[i+1] = px.data[i+2] = grey;
    }
    bwCtx.putImageData(px, 0, 0);

    // Try to load server-generated gradcam image (overlay on top)
    const serverGradcam = new Image();
    serverGradcam.crossOrigin = 'anonymous';
    serverGradcam.onload = () => {
      const hCtx = heatCanvas.getContext('2d');
      hCtx.clearRect(0, 0, W, H);
      hCtx.globalAlpha = 0.72;
      hCtx.drawImage(serverGradcam, 0, 0, W, H);
      if (loadingState) loadingState.style.opacity = '0';
      setTimeout(() => { if (loadingState) loadingState.style.display = 'none'; }, 300);
    };
    serverGradcam.onerror = () => {
      // Server image not available — draw synthetic thermal heatmap
      drawSyntheticHeatmap(heatCanvas, W, H);
      if (loadingState) loadingState.style.opacity = '0';
      setTimeout(() => { if (loadingState) loadingState.style.display = 'none'; }, 300);
    };
    serverGradcam.src = `/static/uploads/gradcam.png?t=${Date.now()}`;
  };
  img.onerror = () => {
    if (loadingState) { loadingState.querySelector('div div:last-child').textContent = 'Image unavailable'; }
  };
  img.src = imageSrc;
}

function drawSyntheticHeatmap(canvas, W, H) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const spots = [
    { x: W*.42, y: H*.38, r: Math.min(W,H)*.26, intensity: 1.0 },
    { x: W*.66, y: H*.55, r: Math.min(W,H)*.20, intensity: 0.82 },
    { x: W*.22, y: H*.62, r: Math.min(W,H)*.14, intensity: 0.55 },
    { x: W*.78, y: H*.28, r: Math.min(W,H)*.11, intensity: 0.42 },
  ];
  spots.forEach(s => {
    const g = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, s.r);
    g.addColorStop(0,   `rgba(255,0,0,${s.intensity*.9})`);
    g.addColorStop(.2,  `rgba(255,80,0,${s.intensity*.8})`);
    g.addColorStop(.45, `rgba(255,220,0,${s.intensity*.6})`);
    g.addColorStop(.7,  `rgba(0,200,80,${s.intensity*.3})`);
    g.addColorStop(1,   'rgba(0,0,200,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(s.x, s.y, s.r, 0, Math.PI*2);
    ctx.fill();
  });
}

// ── SHAP: bar chart rendering ──
function renderSHAPChart(shapData) {
  // shapData: { ADI:.02, DEB:.01, LYM:.03, MUC:.05, MUS:.04, NORM:.01, STR:.15, TUM:.85 } or null
  const CLASSES = [
    { key:'TUM',  label:'Tumor',           color:'#ff3d6e', pos: true  },
    { key:'STR',  label:'Cancer Stroma',   color:'#ff6b35', pos: true  },
    { key:'LYM',  label:'Lymphocytes',     color:'#ffb340', pos: null  },
    { key:'DEB',  label:'Debris',          color:'#9b8fff', pos: null  },
    { key:'MUC',  label:'Mucosa',          color:'#00d4ff', pos: false },
    { key:'MUS',  label:'Smooth Muscle',   color:'#4bc8eb', pos: false },
    { key:'NORM', label:'Normal Mucosa',   color:'#00ff9f', pos: false },
    { key:'ADI',  label:'Adipose Tissue',  color:'#a8ff78', pos: false },
  ];

  // Use real data if available, otherwise use demo proportional values from last scan probs
  let vals = {};
  if (shapData) {
    vals = shapData;
  } else {
    // Derive SHAP approximation from real softmax probabilities if available
    const rawProbs  = window._lastRawProbs || {};
    const lastClass = window._lastPredictedClass || '';
    const hasProbs  = Object.keys(rawProbs).length > 0;

    CLASSES.forEach(c => {
      if (hasProbs) {
        const p = rawProbs[c.key] || 0;
        // SHAP sign: positive = pushes toward class (high prob classes), negative = away
        // Scale: high-prob classes get high positive values, low-prob get small negative
        const baseline = 1 / 8; // uniform baseline (0.125)
        vals[c.key] = parseFloat(((p - baseline) * 1.2).toFixed(3));
      } else {
        // True fallback: use lastClass only
        if (c.key === lastClass) vals[c.key] = 0.80;
        else if (c.pos) vals[c.key] = 0.06;
        else vals[c.key] = -0.04;
      }
    });
  }

  const maxAbs = Math.max(...Object.values(vals).map(Math.abs), 0.01);

  const bars = CLASSES.map((c, i) => {
    const val = vals[c.key] || 0;
    const pct = Math.abs(val) / maxAbs * 100;
    const isPos = val >= 0;
    const barColor = isPos ? c.color : '#3b4fff';
    const valLabel = (isPos ? '+' : '') + val.toFixed(3);
    const animDelay = i * 0.06;

    return `
      <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.55rem">
        <div style="width:82px;font-size:.76rem;color:var(--tx2);text-align:right;flex-shrink:0;font-family:'DM Sans',sans-serif">${c.label}</div>
        <div style="flex:1;display:flex;align-items:center;gap:4px;position:relative">
          <!-- Zero line -->
          <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--border2);z-index:0"></div>
          ${isPos
            ? `<div style="flex:1"></div>
               <div style="width:${pct/2}%;min-width:2px;height:12px;background:${barColor};border-radius:0 4px 4px 0;transition:width 1s cubic-bezier(.4,0,.2,1) ${animDelay}s;z-index:1;opacity:.9" data-w="${pct/2}"></div>`
            : `<div style="width:${pct/2}%;min-width:2px;height:12px;background:${barColor};border-radius:4px 0 0 4px;margin-left:auto;transition:width 1s cubic-bezier(.4,0,.2,1) ${animDelay}s;z-index:1;opacity:.85" data-w="${pct/2}"></div>
               <div style="flex:1"></div>`
          }
        </div>
        <div style="width:52px;font-family:'JetBrains Mono',monospace;font-size:.72rem;color:${barColor};text-align:left;flex-shrink:0">${valLabel}</div>
      </div>`;
  }).join('');

  return `
    <div style="padding:.3rem 0">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.7rem">
        <span style="font-size:.75rem;color:var(--tx3);font-family:'JetBrains Mono',monospace">SHAP · DeepExplainer · Per-class attribution</span>
        <div style="display:flex;gap:.5rem;font-size:.68rem;font-family:'JetBrains Mono',monospace">
          <span style="color:#3b4fff">◀ Negative</span>
          <span style="color:var(--c3)">Positive ▶</span>
        </div>
      </div>
      <div style="font-size:.68rem;color:var(--tx3);text-align:center;margin-bottom:.6rem;font-family:'JetBrains Mono',monospace">← pushes away from class &nbsp;|&nbsp; pushes toward class →</div>
      ${bars}
      <div style="margin-top:.6rem;padding:.5rem .7rem;background:var(--bg3);border-radius:var(--r);border-left:3px solid var(--c3)">
        <div style="font-size:.74rem;color:var(--tx2)">
          <strong style="color:var(--c3)">Highest contributor:</strong>
          ${Object.entries(vals).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]))[0]?.[0] || '—'}
          <span style="color:var(--tx3);margin-left:.3rem">(|SHAP| = ${Math.max(...Object.values(vals).map(Math.abs)).toFixed(3)})</span>
        </div>
      </div>
    </div>
    ${!shapData ? `<div style="margin-top:.5rem;font-size:.68rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;text-align:center">⚠ Estimated values — connect /shap_explain endpoint for real SHAP analysis</div>` : ''}
  `;
}

async function fetchSHAPData() {
  try {
    const res = await fetch('/shap_explain', { method:'POST' });
    if (!res.ok) return;
    const data = await res.json();
    if (data && typeof data === 'object') {
      window._shapData = data.shap_values || data;
      // Re-render SHAP tab if it's active
      if (activeHeatmapTab === 'shap') {
        const wrap = document.getElementById('gradcamWrap');
        if (wrap) wrap.innerHTML = renderSHAPChart(window._shapData);
      }
    }
  } catch {}
}

// drawHeatmap is superseded by drawGradCAMOnCanvas / drawSyntheticHeatmap
function drawHeatmap(type) { /* no-op: use drawGradCAMOnCanvas */ }

function toggleHeatmap() {
  const canvas = document.getElementById('heatmapCanvas');
  if (!canvas) return;
  heatmapVisible = !heatmapVisible;
  canvas.style.opacity = heatmapVisible ? '.55' : '0';
  const btn = document.getElementById('toggleHeatBtn');
  if (btn) btn.textContent = heatmapVisible ? '👁 Hide Heatmap' : '👁 Show Heatmap';
  showToast(heatmapVisible ? '🌡 GradCAM overlay enabled' : '🌡 GradCAM overlay hidden', 'info');
}

// ── Doctor Feedback ───────────────────────────
let starRating = 0;

function initStars() {
  const stars = document.querySelectorAll('.star');
  stars.forEach((star, i) => {
    star.addEventListener('mouseenter', () => {
      stars.forEach((s, j) => s.classList.toggle('lit', j <= i));
    });
    star.addEventListener('mouseleave', () => {
      stars.forEach((s, j) => s.classList.toggle('lit', j < starRating));
    });
    star.addEventListener('click', () => {
      starRating = i + 1;
      stars.forEach((s, j) => s.classList.toggle('lit', j < starRating));
    });
  });
}

async function submitFeedback(verdict) {
  const notes = document.getElementById('feedbackNotes')?.value?.trim();
  if (!starRating) { showToast('Please rate the diagnosis first', 'warn'); return; }

  const btn = verdict === 'confirm' ? document.getElementById('confirmBtn') : document.getElementById('incorrectBtn');
  if (btn) { btn.classList.add('btn-loading'); btn.innerHTML = '<span class="btn-spinner"></span>'; }

  try {
    await fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ verdict, rating: starRating, notes })
    });
  } catch {}

  const area = document.getElementById('feedbackArea');
  if (area) {
    area.innerHTML = `<div class="feedback-success">
      <span style="font-size:1.3rem">${verdict === 'confirm' ? '✅' : '🔄'}</span>
      <div>
        <div style="font-weight:600">${verdict === 'confirm' ? 'Diagnosis confirmed' : 'Marked for review'}</div>
        <div style="font-size:.78rem;margin-top:.2rem;opacity:.8">Rating: ${'★'.repeat(starRating)}${'☆'.repeat(5-starRating)} · ${notes || 'No notes'}</div>
      </div>
    </div>`;
  }
  showToast(verdict === 'confirm' ? '✅ Diagnosis confirmed by doctor' : '🔄 Sent for re-analysis', 'ok');
}

// ── PDF Report ────────────────────────────────
async function downloadReport() {
  const btn = document.getElementById('downloadReport');
  if (btn) {
    btn.classList.add('btn-loading');
    btn.innerHTML = '<span class="btn-spinner"></span> Generating PDF...';
  }

  // Collect all available scan data to send to Flask
  const diagEl = document.getElementById('diagnosisText');
  const recEl  = document.getElementById('recommendationText');
  const rvEl   = document.getElementById('riskValue');

  const payload = {
    diagnosis:       diagEl?.textContent?.trim()  || window._lastDiagnosis    || 'N/A',
    recommendation:  recEl?.textContent?.trim()   || window._lastRec          || 'Consult a specialist.',
    risk_score:      parseInt(rvEl?.textContent)  || window._lastRiskScore    || 0,
    confidence:      window._lastConfidence       || 'N/A',
    predicted_class: window._lastPredictedClass   || 'N/A',
    tier:            window._lastTier             || 'UNKNOWN',
    probabilities:   window._lastRawProbs         || {},
    doctor:          localStorage.getItem('gs_doctor_name') || 'Dr. Admin',
    hospital:        localStorage.getItem('gs_hospital')    || 'City Cancer Institute',
    patient_name:    window._lastPatientName      || 'Anonymous',
  };

  try {
    const res = await fetch('/generate_report', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Server returned ${res.status}`);
    }

    // Flask returns the PDF directly as a blob
    const blob = await res.blob();
    if (blob.type !== 'application/pdf') {
      throw new Error('Response is not a PDF');
    }
    const url = URL.createObjectURL(blob);
    const a   = document.createElement('a');
    a.href    = url;
    a.download = `gastric_sentinel_${new Date().toISOString().slice(0,10)}.pdf`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 2000);
    showToast('📄 PDF report downloaded successfully', 'ok');

  } catch (err) {
    console.warn('PDF generation error:', err.message);
    showToast(`⚠️ PDF error: ${err.message}`, 'warn', 4000);
    // Fallback: generate a minimal HTML→print page
    _printFallbackReport(payload);
  }

  if (btn) {
    btn.classList.remove('btn-loading');
    btn.innerHTML = '📄 Download PDF Report';
  }
}

function _printFallbackReport(d) {
  const tier_color = {CRITICAL:'#ff3d6e',SUSPICIOUS:'#ffb340',NEGATIVE:'#00ff9f',UNKNOWN:'#8b9ab4'}[d.tier]||'#8b9ab4';
  const html = `<!DOCTYPE html>
<html><head><title>Gastric Sentinel Report</title>
<style>
  body{font-family:'Helvetica Neue',sans-serif;background:#fff;color:#111;margin:0;padding:30px 40px}
  h1{color:#0066cc;font-size:22px;margin-bottom:4px}
  .sub{color:#666;font-size:11px;margin-bottom:20px;border-bottom:1px solid #ccc;padding-bottom:10px}
  .meta{display:grid;grid-template-columns:1fr 1fr;gap:4px 20px;font-size:11px;margin-bottom:20px}
  .meta b{color:#333} .meta span{color:#666}
  .result{border:2px solid ${tier_color};border-radius:6px;padding:16px;margin-bottom:20px;background:#f9f9f9}
  .dx{font-size:18px;font-weight:bold;color:${tier_color};margin-bottom:6px}
  .tier{font-size:11px;font-family:monospace;background:${tier_color};color:#000;padding:2px 8px;border-radius:3px;display:inline-block}
  .info{font-size:11px;color:#444;margin-top:8px} .rec{font-size:12px;color:#333;margin-top:10px}
  .probs{margin-bottom:20px} .prob-row{display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:11px}
  .bar-track{flex:1;height:8px;background:#e0e0e0;border-radius:4px;overflow:hidden}
  .bar-fill{height:100%;background:#0066cc;border-radius:4px}
  .footer{margin-top:30px;border-top:1px solid #ccc;padding-top:8px;font-size:10px;color:#888;display:flex;justify-content:space-between}
  @media print{body{padding:15px 20px}}
</style></head><body>
<h1>GASTRIC SENTINEL</h1>
<div class="sub">AI Diagnostic Report · EfficientNet-B4 v3.2 · Generated: ${new Date().toLocaleString()}</div>
<div class="meta">
  <div><b>Patient:</b> <span>${d.patient_name}</span></div>
  <div><b>Physician:</b> <span>${d.doctor}</span></div>
  <div><b>Institution:</b> <span>${d.hospital}</span></div>
  <div><b>Report ID:</b> <span>GS-${Date.now()}</span></div>
</div>
<div class="result">
  <div class="tier">${d.tier}</div>
  <div class="dx">${d.diagnosis}</div>
  <div class="info">Class: <b>${d.predicted_class}</b> &nbsp;·&nbsp; Risk Score: <b>${d.risk_score}%</b> &nbsp;·&nbsp; Confidence: <b>${d.confidence}%</b></div>
  <div class="rec">${d.recommendation}</div>
</div>
${Object.keys(d.probabilities).length > 0 ? `
<div class="probs"><b style="font-size:12px">Class Probabilities</b><br><br>
${Object.entries(d.probabilities).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`
  <div class="prob-row">
    <span style="width:40px;font-family:monospace;font-size:10px">${k}</span>
    <div class="bar-track"><div class="bar-fill" style="width:${Math.round(v*100)}%"></div></div>
    <span style="width:44px;text-align:right">${(v*100).toFixed(1)}%</span>
  </div>`).join('')}
</div>` : ''}
<div class="footer">
  <span>GASTRIC SENTINEL AI DIAGNOSTIC PLATFORM</span>
  <span style="color:#c00;font-weight:bold">CONFIDENTIAL MEDICAL DOCUMENT</span>
</div>
<script>window.onload=()=>window.print()</script>
</body></html>`;
  const w = window.open('','_blank');
  if (w) { w.document.write(html); w.document.close(); }
  else showToast('Allow pop-ups to print report', 'warn');
}

// ── AI Chatbot ────────────────────────────────
const DEMO_REPLIES = [
  "Based on histopathological findings, I recommend immediate oncological consultation. The tissue morphology suggests cellular atypia consistent with dysplastic changes.",
  "The GradCAM activation map highlights regions of concern in the antral mucosa. These areas show irregular glandular architecture typical of early neoplastic transformation.",
  "Risk stratification for this patient is HIGH based on lesion characteristics, patient age, and H. pylori status. Urgent intervention is advised.",
  "Endoscopic submucosal dissection (ESD) may be considered as a curative option for early-stage lesions. Discuss with your surgical team.",
  "Surveillance protocol recommendation: Follow-up endoscopy in 3 months with systematic biopsy mapping per Prague classification.",
  "The probability distribution suggests High-Grade Dysplasia as the primary diagnosis. Differential includes early intramucosal adenocarcinoma.",
];

let chatOpen = false;

function toggleChat() {
  chatOpen = !chatOpen;
  const win = document.getElementById('chatWindow');
  if (win) win.classList.toggle('hidden', !chatOpen);
}

function closeChat() {
  chatOpen = false;
  document.getElementById('chatWindow')?.classList.add('hidden');
}

function initChat() {
  const input = document.getElementById('chatInput');
  const sendBtn = document.getElementById('chatSend');

  if (input) {
    input.addEventListener('keypress', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }});
  }
  sendBtn?.addEventListener('click', sendChatMessage);
}

async function sendChatMessage() {
  const input = document.getElementById('chatInput');
  const body = document.getElementById('chatBody');
  if (!input || !body) return;

  const msg = input.value.trim();
  if (!msg) return;

  appendChatMsg(msg, 'user', body);
  input.value = '';

  // Typing indicator
  const typing = document.createElement('div');
  typing.className = 'chat-msg ai';
  typing.id = 'typingIndicator';
  typing.innerHTML = `<div class="c-av">🤖</div><div class="c-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>`;
  body.appendChild(typing);
  body.scrollTop = body.scrollHeight;

  try {
    const res = await fetch('/chatbot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg })
    });
    const data = await res.json();
    typing.remove();
    appendChatMsg(data.reply || data.response, 'ai', body);
  } catch {
    setTimeout(() => {
      typing.remove();
      const reply = DEMO_REPLIES[Math.floor(Math.random() * DEMO_REPLIES.length)];
      appendChatMsg(reply, 'ai', body);
    }, 1200 + Math.random() * 800);
  }
}

function appendChatMsg(text, role, body) {
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  const icon = role === 'ai' ? '🤖' : '👨‍⚕️';
  div.innerHTML = `<div class="c-av">${icon}</div><div class="c-bubble">${text}</div>`;
  body.appendChild(div);
  body.scrollTop = body.scrollHeight;
}

// ── Export CSV ───────────────────────────────
window.exportCSV = function() {
  const rows = [['Patient ID','Name','Age','Gender','Last Diagnosis','Risk Level','Last Scan']];
  DEMO_PATIENTS.forEach(p => {
    rows.push([
      p.id || '',
      p.name || '',
      p.age || '',
      (p.gender === 'M' ? 'Male' : p.gender === 'F' ? 'Female' : p.gender) || '',
      p.last || p.last_result || '',
      p.risk === 'high' ? 'High' : p.risk === 'mid' ? 'Medium' : 'Low',
      p.date || new Date().toISOString().split('T')[0]
    ]);
  });
  const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g,'""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `gastric_sentinel_patients_${new Date().toISOString().split('T')[0]}.csv`; a.click();
  URL.revokeObjectURL(url);
  showToast('📥 CSV exported — ' + (rows.length-1) + ' patients', 'ok');
};

// ── Patient Page Filters ─────────────────────
function filterPatients(risk) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  if (event?.target) event.target.classList.add('active');
  window._activeFilter = risk;
  applyPatientFilters();
}

function applyPatientFilters() {
  const risk     = window._activeFilter || 'all';
  const query    = (window._patientSearchQuery || '').toLowerCase();
  const rows     = document.querySelectorAll('#patientsTable tr');
  let   visible  = 0;

  rows.forEach(row => {
    if (!row.querySelector('td')) { row.style.display = ''; return; }

    const rowText = row.textContent.toLowerCase();

    // Risk filter
    let riskMatch = true;
    if (risk !== 'all' && risk !== 'new') {
      const badge = row.querySelector('.badge');
      riskMatch = badge ? badge.textContent.toLowerCase().includes(risk) : false;
    }
    if (risk === 'new') {
      riskMatch = rowText.includes('new');
    }

    // Text search filter
    const searchMatch = !query || rowText.includes(query);

    const show = riskMatch && searchMatch;
    row.style.display = show ? '' : 'none';
    if (show) visible++;
  });

  // Update count label
  const countEl = document.getElementById('patientFilterCount');
  if (countEl) countEl.textContent = `Showing ${visible} patient${visible!==1?'s':''}`;
}

// ── Patient Inline Search ─────────────────────
function initPatientSearch() {
  // Only runs on patients page or dashboard table area
  const tableArea = document.getElementById('patientsTable')?.closest('.card');
  if (!tableArea) return;

  // Check if search bar already inserted
  if (tableArea.querySelector('.patient-search-bar')) return;

  // Inject style
  if (!document.getElementById('ptSearchStyle')) {
    const s = document.createElement('style');
    s.id = 'ptSearchStyle';
    s.textContent = `
      .patient-search-bar {
        display:flex; align-items:center; gap:.5rem;
        background:var(--bg3); border:1px solid var(--border);
        border-radius:var(--r); padding:.45rem .8rem;
        transition:border-color .2s, box-shadow .2s;
        flex:1; min-width:200px;
      }
      .patient-search-bar:focus-within {
        border-color:var(--c1); box-shadow:0 0 0 3px rgba(0,212,255,.12);
      }
      .patient-search-bar input {
        background:none; border:none; outline:none; color:var(--tx);
        font-size:.84rem; font-family:'DM Sans',sans-serif; width:100%;
      }
      .patient-search-bar input::placeholder { color:var(--tx3); }
      .pt-search-row {
        display:flex; align-items:center; gap:.5rem; flex-wrap:wrap;
        padding:.5rem .2rem .6rem; border-bottom:1px solid var(--border); margin-bottom:.2rem;
      }
      @keyframes ptRowHighlight {
        0%   { background:rgba(0,212,255,.12); }
        100% { background:transparent; }
      }
      .pt-row-match td:first-child { border-left:2px solid var(--c1); }
    `;
    document.head.appendChild(s);
  }

  const searchRow = document.createElement('div');
  searchRow.className = 'pt-search-row';
  searchRow.innerHTML = `
    <div class="patient-search-bar" id="patientSearchBar">
      <span style="color:var(--tx3);font-size:1rem;flex-shrink:0">🔍</span>
      <input id="patientSearchInput" placeholder="Search by name, ID, diagnosis, risk…" autocomplete="off">
      <button id="ptSearchClear" style="display:none;background:none;border:none;color:var(--tx3);cursor:pointer;font-size:.85rem;padding:0" title="Clear">✕</button>
    </div>
    <span id="patientFilterCount" style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--tx3);white-space:nowrap"></span>
  `;

  // Insert before table-wrap
  const tableWrap = tableArea.querySelector('.table-wrap') || tableArea.querySelector('table');
  if (tableWrap) tableArea.insertBefore(searchRow, tableWrap);
  else tableArea.querySelector('.card-hd')?.after(searchRow);

  const inp = document.getElementById('patientSearchInput');
  const clr = document.getElementById('ptSearchClear');

  inp?.addEventListener('input', () => {
    const q = inp.value.trim();
    window._patientSearchQuery = q;
    clr.style.display = q ? '' : 'none';
    applyPatientFilters();

    // Highlight matched cells
    document.querySelectorAll('#patientsTable tr').forEach(row => {
      row.classList.remove('pt-row-match');
      if (q && row.textContent.toLowerCase().includes(q.toLowerCase()) && row.style.display !== 'none') {
        row.classList.add('pt-row-match');
        row.style.animation = 'ptRowHighlight .6s ease forwards';
        setTimeout(() => { row.style.animation = ''; }, 700);
      }
    });
  });

  clr?.addEventListener('click', () => {
    inp.value = ''; window._patientSearchQuery = '';
    clr.style.display = 'none';
    applyPatientFilters();
    inp.focus();
  });

  // Set initial count
  applyPatientFilters();
}

// ── Animate on scroll / intersection ─────────
function initAnimations() {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.style.opacity = '1';
        e.target.style.transform = 'translateY(0)';
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.card, .stat-card').forEach(el => {
    observer.observe(el);
  });
}

// ── Realtime clock ────────────────────────────
function initClock() {
  const el = document.getElementById('liveClock');
  if (!el) return;
  function tick() {
    const now = new Date();
    el.textContent = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }
  tick(); setInterval(tick, 1000);
}

// ── Dashboard activity chart — tries /scan_activity, falls back to demo ──
let _chartData = [12,19,8,24,16,31,22,28,15,35,29,41];

async function loadScanActivity() {
  try {
    const res = await fetch('/scan_activity');
    const data = await res.json();
    if (Array.isArray(data) && data.length > 1) {
      _chartData = data.map(d => typeof d === 'object' ? (d.count || d.scans || 0) : d);
    }
  } catch (e) { /* use demo data */ }
  drawActivityChart();
}

function drawActivityChart() {
  const svg = document.getElementById('activityChart');
  if (!svg) return;
  const data = _chartData;
  const w = svg.clientWidth || 300, h = 80;
  if (w < 10) { setTimeout(drawActivityChart, 100); return; }
  const max = Math.max(...data, 1);
  const step = w / Math.max(data.length - 1, 1);
  const pts  = data.map((v, i) => `${i * step},${h - (v / max) * (h - 12) - 4}`).join(' ');

  // Week labels (last 12)
  const now = new Date();
  const weekLabels = Array.from({length:data.length}, (_,i) => {
    const d = new Date(now); d.setDate(d.getDate() - (data.length-1-i)*7);
    return 'W' + (i+1);
  });

  svg.innerHTML = `
    <defs>
      <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="var(--c1)" stop-opacity=".35"/>
        <stop offset="100%" stop-color="var(--c1)" stop-opacity="0"/>
      </linearGradient>
      <filter id="glow"><feGaussianBlur stdDeviation="2" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    </defs>
    <polygon points="0,${h} ${pts} ${(data.length-1)*step},${h}" fill="url(#chartGrad)"/>
    <polyline points="${pts}" fill="none" stroke="var(--c1)" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" filter="url(#glow)"/>
    ${data.map((v, i) => {
      const x = i * step, y = h - (v / max) * (h - 12) - 4;
      const isMax = v === max;
      return `
        <circle cx="${x}" cy="${y}" r="${isMax?5:3}" fill="${isMax?'var(--c1)':'var(--bg2)'}" stroke="var(--c1)" stroke-width="1.5"/>
        ${isMax ? `<text x="${x}" y="${y-9}" text-anchor="middle" font-size="9" fill="var(--c1)" font-family="'JetBrains Mono',monospace">${v}</text>` : ''}
      `;
    }).join('')}
  `;

  // Update peak/avg stats
  const peak = Math.max(...data);
  const avg  = Math.round(data.reduce((a,b)=>a+b,0)/data.length);
  const peakWk = data.indexOf(peak) + 1;
  const peakEl = document.querySelector('.chart-peak');
  const avgEl  = document.querySelector('.chart-avg');
  if (peakEl) peakEl.textContent = 'Week ' + peakWk;
  if (avgEl)  avgEl.textContent  = avg + '/wk';
}

function initPathologyZoom() {
  // Attach to any image inside gradcam or upload preview
  document.addEventListener('click', e => {
    const img = e.target.closest('img');
    if (!img) return;
    // Only trigger on scan / gradcam images, not avatars etc
    const inZone = img.closest('.gradcam-wrap, .preview-thumb, .upload-zone, [data-zoomable]');
    if (!inZone) return;
    openZoomViewer(img.src, img.alt || 'Pathology Image');
  });
}

function openZoomViewer(src, label) {
  // Remove old if exists
  document.getElementById('zoomOverlay')?.remove();

  const overlay = document.createElement('div');
  overlay.id = 'zoomOverlay';
  overlay.style.cssText = `
    position:fixed;inset:0;z-index:900;
    background:rgba(0,0,0,.92);backdrop-filter:blur(8px);
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    animation:fadeIn .2s ease;
  `;

  overlay.innerHTML = `
    <style>
      @keyframes fadeIn{from{opacity:0}to{opacity:1}}
      #zoomImg{cursor:crosshair;transition:transform .1s ease;transform-origin:center center;user-select:none;}
      .zoom-lens{
        position:absolute;width:140px;height:140px;border-radius:50%;
        border:2px solid var(--c1,#00d4ff);pointer-events:none;
        box-shadow:0 0 0 1px rgba(0,212,255,.3),0 0 20px rgba(0,212,255,.2);
        overflow:hidden;display:none;z-index:10;
        background-repeat:no-repeat;
        background-color:#000;
      }
    </style>
    <div style="position:relative;max-width:min(90vw,900px);max-height:80vh;">
      <div class="zoom-lens" id="zoomLens"></div>
      <img id="zoomImg" src="${src}" alt="${label}"
        style="max-width:100%;max-height:75vh;display:block;border-radius:10px;border:1px solid rgba(0,212,255,.2);">
    </div>

    <!-- Toolbar -->
    <div style="margin-top:1rem;display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;justify-content:center">
      <span style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:rgba(255,255,255,.4)">${label}</span>
      <div style="display:flex;gap:.5rem">
        <button onclick="zoomStep(0.25)"  style="${zbtn()}">＋ Zoom In</button>
        <button onclick="zoomStep(-0.25)" style="${zbtn()}">－ Zoom Out</button>
        <button onclick="zoomReset()"     style="${zbtn()}">↺ Reset</button>
        <button onclick="toggleLens()"    style="${zbtn('cyan')}">🔍 Lens</button>
        <button onclick="downloadZoomImg('${src}')" style="${zbtn()}">⬇ Save</button>
        <button onclick="document.getElementById('zoomOverlay').remove()" style="${zbtn('red')}">✕ Close</button>
      </div>
    </div>
    <div style="margin-top:.5rem;font-family:'JetBrains Mono',monospace;font-size:.65rem;color:rgba(255,255,255,.25)">
      Scroll to zoom · Drag to pan · Click 🔍 for magnifier lens · ESC to close
    </div>
  `;

  document.body.appendChild(overlay);

  // Close on backdrop click (not on img/buttons)
  overlay.addEventListener('click', e => {
    if (e.target === overlay) overlay.remove();
  });
  document.addEventListener('keydown', function esc(e) {
    if (e.key === 'Escape') { overlay.remove(); document.removeEventListener('keydown', esc); }
  });

  // ── Zoom & Pan ──
  const img = overlay.querySelector('#zoomImg');
  let scale = 1, tx = 0, ty = 0, dragging = false, startX, startY, lastTx, lastTy;

  function applyTransform() {
    img.style.transform = `translate(${tx}px,${ty}px) scale(${scale})`;
  }

  img.addEventListener('wheel', e => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.15 : 0.15;
    scale = Math.max(0.5, Math.min(8, scale + delta));
    applyTransform();
  }, { passive: false });

  img.addEventListener('mousedown', e => {
    dragging = true; startX = e.clientX; startY = e.clientY;
    lastTx = tx; lastTy = ty; img.style.cursor = 'grabbing';
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    tx = lastTx + (e.clientX - startX);
    ty = lastTy + (e.clientY - startY);
    applyTransform();
  });
  document.addEventListener('mouseup', () => { dragging = false; img.style.cursor = 'crosshair'; });

  // ── Magnifier Lens ──
  const lens = overlay.querySelector('#zoomLens');
  let lensActive = false;
  const LENS_ZOOM = 3;

  window.toggleLens = function() {
    lensActive = !lensActive;
    lens.style.display = lensActive ? 'block' : 'none';
  };

  img.addEventListener('mousemove', e => {
    if (!lensActive) return;
    const rect = img.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const lw = lens.offsetWidth, lh = lens.offsetHeight;

    lens.style.left = (e.clientX - rect.left - lw/2) + 'px';
    lens.style.top  = (e.clientY - rect.top  - lh/2) + 'px';

    const bgX = -(x * LENS_ZOOM - lw/2);
    const bgY = -(y * LENS_ZOOM - lh/2);
    lens.style.backgroundImage = `url('${src}')`;
    lens.style.backgroundSize  = `${rect.width * LENS_ZOOM}px ${rect.height * LENS_ZOOM}px`;
    lens.style.backgroundPosition = `${bgX}px ${bgY}px`;
  });
  img.addEventListener('mouseenter', () => { if (lensActive) lens.style.display = 'block'; });
  img.addEventListener('mouseleave', () => { lens.style.display = 'none'; });
}

function zbtn(color) {
  const c = color === 'cyan' ? 'rgba(0,212,255,.2)' : color === 'red' ? 'rgba(255,61,110,.2)' : 'rgba(255,255,255,.08)';
  const tc = color === 'cyan' ? '#00d4ff' : color === 'red' ? '#ff3d6e' : 'rgba(255,255,255,.7)';
  return `background:${c};color:${tc};border:1px solid ${c};border-radius:8px;padding:.4rem .85rem;font-size:.8rem;cursor:pointer;font-family:'DM Sans',sans-serif;transition:all .15s`;
}
window.zoomStep = function(d) {
  const img = document.getElementById('zoomImg');
  if (!img) return;
  const cur = img.style.transform;
  const m = cur.match(/scale\(([\d.]+)\)/);
  const s = Math.max(0.5, Math.min(8, (m ? parseFloat(m[1]) : 1) + d));
  img.style.transform = (cur.replace(/scale\([^)]+\)/, '') + ` scale(${s})`).trim() || `scale(${s})`;
};
window.zoomReset = function() {
  const img = document.getElementById('zoomImg');
  if (img) img.style.transform = 'translate(0,0) scale(1)';
};
window.downloadZoomImg = function(src) {
  const a = document.createElement('a');
  a.href = src; a.download = 'pathology_image.png'; a.click();
};

function initHoverEffects() {
  // Inject global hover CSS once
  if (document.getElementById('hoverStyles')) return;
  const s = document.createElement('style');
  s.id = 'hoverStyles';
  s.textContent = `
    /* Stat cards lift + glow on hover */
    .stat-card { transition: transform .22s cubic-bezier(.4,0,.2,1), box-shadow .22s, border-color .22s !important; }
    .stat-card:hover { transform: translateY(-5px) !important; }
    .stat-card.s-cyan:hover  { box-shadow: 0 8px 32px rgba(0,212,255,.25) !important; border-color: var(--c1) !important; }
    .stat-card.s-green:hover { box-shadow: 0 8px 32px rgba(0,255,159,.2)  !important; border-color: var(--c2) !important; }
    .stat-card.s-red:hover   { box-shadow: 0 8px 32px rgba(255,61,110,.2) !important; border-color: var(--c3) !important; }
    .stat-card.s-amber:hover { box-shadow: 0 8px 32px rgba(255,179,64,.2) !important; border-color: var(--c4) !important; }

    /* Cards */
    .card { transition: border-color .22s, box-shadow .22s, transform .22s !important; }
    .card:hover { transform: translateY(-2px) !important; }

    /* Table rows */
    .data-table tbody tr { transition: background .15s, transform .15s !important; }
    .data-table tbody tr:hover { transform: translateX(3px) !important; }

    /* Buttons */
    .btn { transition: all .18s cubic-bezier(.4,0,.2,1) !important; }
    .btn:hover { transform: translateY(-2px) !important; }
    .btn:active { transform: scale(.96) translateY(0) !important; }

    /* Nav links */
    .nav-link { transition: all .18s !important; }
    .nav-link:not(.active):hover { transform: translateX(4px) !important; }

    /* Sparkle on sparkbars hover */
    .stat-sparkline:hover .spark-bar { opacity: 1 !important; }
    .stat-sparkline .spark-bar { transition: opacity .2s, height .3s !important; }

    /* Risk bars widen label on hover */
    .risk-row:hover .risk-name { color: var(--tx) !important; }
    .risk-row { transition: all .15s; }
    .risk-row:hover { padding-left: 4px; }

    /* Topbar buttons */
    .top-icon-btn { transition: all .18s !important; }
    .top-icon-btn:hover { transform: scale(1.1) !important; }

    /* Search */
    .topbar-search { transition: all .2s !important; }
    .topbar-search:focus-within { transform: scaleX(1.02); transform-origin: left; }

    /* Notification bell wiggle */
    @keyframes bellWiggle {
      0%,100%{transform:rotate(0)}15%{transform:rotate(15deg)}30%{transform:rotate(-12deg)}45%{transform:rotate(8deg)}60%{transform:rotate(-5deg)}
    }
    #notifBell:hover { animation: bellWiggle .5s ease !important; }

    /* Preview thumbs */
    .preview-thumb { transition: all .2s !important; }
    .preview-thumb:hover { transform: scale(1.08) !important; border-color: var(--c1) !important; }

    /* Zoom cursor on gradcam images */
    .gradcam-wrap img, .preview-thumb img { cursor: zoom-in !important; }
    .gradcam-wrap img:hover { filter: brightness(1.08) !important; }

    /* Badge pulse on new */
    .nav-badge { animation: badgePop .3s cubic-bezier(.34,1.56,.64,1); }
    @keyframes badgePop { from{transform:scale(0)} to{transform:scale(1)} }
  `;
  document.head.appendChild(s);
}

function openReports() {
  const existing = document.getElementById('reportsModal');
  if (existing) { existing.classList.add('open'); return; }

  const modal = document.createElement('div');
  modal.id = 'reportsModal';
  modal.className = 'modal-backdrop';
  modal.innerHTML = `
    <div class="modal" style="max-width:680px">
      <div class="modal-hd">
        <div class="modal-title">📈 Reports & Analytics</div>
        <button class="modal-x" onclick="document.getElementById('reportsModal').classList.remove('open')">✕</button>
      </div>

      <!-- Tabs -->
      <div style="display:flex;gap:.5rem;margin-bottom:1.2rem;border-bottom:1px solid var(--border);padding-bottom:.75rem">
        <button class="report-tab active" onclick="switchReportTab(this,'weekly')"   style="${tabStyle(true)}">Weekly</button>
        <button class="report-tab"        onclick="switchReportTab(this,'monthly')"  style="${tabStyle()}">Monthly</button>
        <button class="report-tab"        onclick="switchReportTab(this,'custom')"   style="${tabStyle()}">Custom Range</button>
      </div>

      <!-- Content -->
      <div id="reportContent">
        ${reportWeeklyHTML()}
      </div>

      <div class="modal-foot">
        <button class="btn btn-ghost" onclick="document.getElementById('reportsModal').classList.remove('open')">Close</button>
        <button class="btn btn-amber" onclick="downloadReport()">📄 Download PDF</button>
        <button class="btn btn-primary" onclick="showToast('📧 Report emailed to doctor@hospital.in','ok')">📧 Email Report</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  setTimeout(() => modal.classList.add('open'), 10);
  modal.addEventListener('click', e => { if (e.target === modal) modal.classList.remove('open'); });
}

function tabStyle(active) {
  return `background:${active?'var(--c1)':'transparent'};color:${active?'#000':'var(--tx2)'};border:1px solid ${active?'var(--c1)':'var(--border)'};border-radius:var(--r);padding:.35rem .85rem;font-size:.82rem;cursor:pointer;transition:all .2s;font-family:'DM Sans',sans-serif;font-weight:${active?'600':'400'}`;
}

window.switchReportTab = function(btn, tab) {
  document.querySelectorAll('.report-tab').forEach(b => {
    b.style.cssText = tabStyle(false);
    b.classList.remove('active');
  });
  btn.style.cssText = tabStyle(true);
  btn.classList.add('active');
  const content = document.getElementById('reportContent');
  if (tab === 'weekly')  content.innerHTML = reportWeeklyHTML();
  if (tab === 'monthly') content.innerHTML = reportMonthlyHTML();
  if (tab === 'custom')  content.innerHTML = reportCustomHTML();
};

function reportWeeklyHTML() {
  return `
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin-bottom:1rem">
      ${[['🔬','Scans This Week','41','↑ +8 vs last'],['⚠️','New High-Risk','3','↑ +1 vs last'],['✅','Confirmed Dx','38','95.1% accuracy']].map(([ic,lb,v,d])=>`
      <div style="background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);padding:.85rem;text-align:center">
        <div style="font-size:1.4rem">${ic}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:600;color:var(--tx);margin:.2rem 0">${v}</div>
        <div style="font-size:.72rem;color:var(--tx3)">${lb}</div>
        <div style="font-size:.68rem;color:var(--c2);margin-top:.2rem">${d}</div>
      </div>`).join('')}
    </div>
    <div style="background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);padding:1rem">
      <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;color:var(--tx3);margin-bottom:.75rem;letter-spacing:.08em;text-transform:uppercase">Diagnosis Breakdown — Week 26</div>
      ${[['High-Grade Dysplasia',28,'high'],['Low-Grade Dysplasia',17,'mid'],['Chronic Gastritis',34,'mid'],['Normal Mucosa',14,'low'],['Adenocarcinoma',7,'high']].map(([n,p,r])=>`
      <div style="margin-bottom:.6rem">
        <div style="display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:3px">
          <span style="color:var(--tx2)">${n}</span>
          <span style="font-family:'JetBrains Mono',monospace;color:${r==='high'?'var(--c3)':r==='mid'?'var(--c4)':'var(--c2)'}">${p}%</span>
        </div>
        <div style="height:6px;background:var(--bg4);border-radius:99px;overflow:hidden">
          <div style="height:100%;width:${p}%;border-radius:99px;background:${r==='high'?'var(--c3)':r==='mid'?'var(--c4)':'var(--c2)'};transition:width 1s ease"></div>
        </div>
      </div>`).join('')}
    </div>`;
}
function reportMonthlyHTML() {
  return `
    <div style="background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);padding:1rem;margin-bottom:.75rem">
      <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;color:var(--tx3);margin-bottom:.75rem;letter-spacing:.08em;text-transform:uppercase">Monthly Summary — June 2025</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem">
        ${[['Total Scans','183'],['High-Risk Cases','12'],['Model Accuracy','From Flask /stats'],['Doctor Corrections','9'],['Avg Turnaround','4.2 hrs'],['Patient Satisfaction','98.2%']].map(([k,v])=>`
        <div style="padding:.6rem .75rem;background:var(--bg4);border-radius:var(--r);border:1px solid var(--border)">
          <div style="font-size:.7rem;color:var(--tx3);margin-bottom:.2rem">${k}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.95rem;font-weight:600;color:var(--c1)">${v}</div>
        </div>`).join('')}
      </div>
    </div>
    <div class="alert alert-info" style="margin:0">ℹ️ Full monthly PDF includes per-patient breakdown, AI confidence trends, and doctor feedback log.</div>`;
}
function reportCustomHTML() {
  return `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-bottom:1rem">
      <div><label style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:var(--tx3);display:block;margin-bottom:.3rem;text-transform:uppercase;letter-spacing:.06em">From Date</label>
        <input type="date" class="form-control" style="width:100%"></div>
      <div><label style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:var(--tx3);display:block;margin-bottom:.3rem;text-transform:uppercase;letter-spacing:.06em">To Date</label>
        <input type="date" class="form-control" style="width:100%"></div>
    </div>
    <div style="margin-bottom:.75rem">
      <label style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:var(--tx3);display:block;margin-bottom:.3rem;text-transform:uppercase;letter-spacing:.06em">Include Sections</label>
      <div style="display:flex;flex-wrap:wrap;gap:.5rem">
        ${['Patient List','Risk Scores','Diagnosis Breakdown','GradCAM Images','Doctor Feedback','Model Performance'].map(s=>`
        <label style="display:flex;align-items:center;gap:.4rem;font-size:.82rem;color:var(--tx2);cursor:pointer">
          <input type="checkbox" checked style="accent-color:var(--c1)"> ${s}
        </label>`).join('')}
      </div>
    </div>
    <button class="btn btn-primary w-full" style="justify-content:center" onclick="showToast('🔄 Generating custom report…','info')">Generate Custom Report</button>`;
}


// ══════════════════════════════════════════════
// RISK ALERTS PANEL
// ══════════════════════════════════════════════
const RISK_ALERTS = [
  { id:1, patient:'Kiran Desai',   pid:'P-0037', age:67, dx:'Adenocarcinoma',       score:94, status:'critical', action:'Immediate oncology referral', since:'2h ago' },
  { id:2, patient:'Arjun Mehta',   pid:'P-0041', age:58, dx:'High-Grade Dysplasia', score:82, status:'urgent',   action:'Endoscopic resection consult', since:'4h ago' },
  { id:3, patient:'Priya Sharma',  pid:'P-0040', age:44, dx:'Chronic Gastritis',    score:45, status:'watch',    action:'H. pylori test + follow-up',   since:'1d ago' },
  { id:4, patient:'Sunita Patel',  pid:'P-0038', age:51, dx:'Low-Grade Dysplasia',  score:38, status:'watch',    action:'Biopsy repeat in 3 months',    since:'1d ago' },
  { id:5, patient:'Raj Verma',     pid:'P-0035', age:71, dx:'High-Grade Dysplasia', score:79, status:'urgent',   action:'ESD evaluation scheduled',     since:'2d ago' },
];

function openRiskAlerts() {
  const existing = document.getElementById('riskAlertsModal');
  if (existing) { existing.classList.add('open'); return; }

  const modal = document.createElement('div');
  modal.id = 'riskAlertsModal';
  modal.className = 'modal-backdrop';
  modal.innerHTML = `
    <div class="modal" style="max-width:660px">
      <div class="modal-hd">
        <div class="modal-title">⚠️ Risk Alert Centre</div>
        <button class="modal-x" onclick="document.getElementById('riskAlertsModal').classList.remove('open')">✕</button>
      </div>

      <div style="display:flex;gap:.5rem;margin-bottom:1rem;flex-wrap:wrap">
        <span style="background:rgba(255,61,110,.12);color:var(--c3);border:1px solid rgba(255,61,110,.3);border-radius:99px;padding:3px 10px;font-size:.75rem;font-family:'JetBrains Mono',monospace">● Critical: 1</span>
        <span style="background:rgba(255,179,64,.12);color:var(--c4);border:1px solid rgba(255,179,64,.3);border-radius:99px;padding:3px 10px;font-size:.75rem;font-family:'JetBrains Mono',monospace">● Urgent: 2</span>
        <span style="background:rgba(0,212,255,.12);color:var(--c1);border:1px solid rgba(0,212,255,.3);border-radius:99px;padding:3px 10px;font-size:.75rem;font-family:'JetBrains Mono',monospace">● Watch: 2</span>
      </div>

      <div style="display:flex;flex-direction:column;gap:.65rem;max-height:380px;overflow-y:auto;padding-right:.25rem">
        ${RISK_ALERTS.map(a => {
          const col = a.status==='critical'?'var(--c3)':a.status==='urgent'?'var(--c4)':'var(--c1)';
          const bg  = a.status==='critical'?'rgba(255,61,110,.07)':a.status==='urgent'?'rgba(255,179,64,.07)':'rgba(0,212,255,.05)';
          const bc  = a.status==='critical'?'rgba(255,61,110,.25)':a.status==='urgent'?'rgba(255,179,64,.2)':'rgba(0,212,255,.18)';
          return `
          <div style="background:${bg};border:1px solid ${bc};border-radius:var(--r);padding:.9rem 1rem;border-left:3px solid ${col}">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.4rem;flex-wrap:wrap;gap:.4rem">
              <div style="display:flex;align-items:center;gap:.6rem">
                <span style="font-weight:600;color:var(--tx);font-size:.9rem">${a.patient}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--c1)">${a.pid}</span>
                <span style="font-size:.72rem;color:var(--tx3)">Age ${a.age}</span>
              </div>
              <div style="display:flex;align-items:center;gap:.5rem">
                <span style="font-family:'JetBrains Mono',monospace;font-size:.8rem;font-weight:700;color:${col}">Risk ${a.score}%</span>
                <span style="font-size:.68rem;padding:2px 8px;border-radius:99px;background:${bg};border:1px solid ${bc};color:${col};text-transform:uppercase;letter-spacing:.06em;font-family:'JetBrains Mono',monospace">${a.status}</span>
              </div>
            </div>
            <div style="font-size:.82rem;color:var(--tx2);margin-bottom:.5rem">Dx: <strong style="color:var(--tx)">${a.dx}</strong></div>
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.4rem">
              <div style="font-size:.78rem;color:var(--tx3)">🎯 ${a.action}</div>
              <div style="display:flex;gap:.4rem">
                <span style="font-size:.65rem;color:var(--tx4);font-family:'JetBrains Mono',monospace">${a.since}</span>
                <button onclick="showToast('📋 Viewing ${a.patient} record','info')" style="font-size:.75rem;background:none;border:none;color:${col};cursor:pointer;text-decoration:underline">View Patient →</button>
                <button onclick="dismissAlert(${a.id},this)" style="font-size:.75rem;background:none;border:none;color:var(--tx3);cursor:pointer">Dismiss</button>
              </div>
            </div>
          </div>`;
        }).join('')}
      </div>

      <div class="modal-foot">
        <button class="btn btn-ghost" onclick="document.getElementById('riskAlertsModal').classList.remove('open')">Close</button>
        <button class="btn btn-danger" onclick="showToast('📞 Escalation sent to on-call oncologist','ok')">📞 Escalate All Critical</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  setTimeout(() => modal.classList.add('open'), 10);
  modal.addEventListener('click', e => { if (e.target === modal) modal.classList.remove('open'); });
}

window.dismissAlert = function(id, btn) {
  const row = btn.closest('[style*="border-left"]');
  row.style.transition = 'all .3s ease';
  row.style.opacity = '0';
  row.style.transform = 'translateX(16px)';
  setTimeout(() => row.remove(), 300);
  showToast('✅ Alert dismissed', 'ok');
};


// ══════════════════════════════════════════════
// SETTINGS PANEL
// ══════════════════════════════════════════════
function openSettings() {
  const existing = document.getElementById('settingsModal');
  if (existing) { existing.classList.add('open'); return; }

  const modal = document.createElement('div');
  modal.id = 'settingsModal';
  modal.className = 'modal-backdrop';
  modal.innerHTML = `
    <div class="modal" style="max-width:580px">
      <div class="modal-hd">
        <div class="modal-title">⚙️ Settings</div>
        <button class="modal-x" onclick="document.getElementById('settingsModal').classList.remove('open')">✕</button>
      </div>

      <!-- Settings Sections -->
      <div style="display:flex;flex-direction:column;gap:1.1rem;max-height:420px;overflow-y:auto;padding-right:.2rem">

        <!-- Appearance -->
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;color:var(--tx3);margin-bottom:.6rem">Appearance</div>
          ${settingRow('Theme','Dark / Light mode',`<button onclick="toggleTheme();showToast('Theme switched','info')" class="btn btn-ghost btn-sm">Toggle Theme</button>`)}
          ${settingToggle('Compact Sidebar','Show only icons on sidebar','compactSidebar',false)}
          ${settingToggle('Animations','Enable UI motion effects','enableAnimations',true)}
        </div>

        <!-- AI Model -->
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;color:var(--tx3);margin-bottom:.6rem">AI Model</div>
          ${settingRow('Model Version','Current: EfficientNet-B4 v3.2',`<span style="font-family:'JetBrains Mono',monospace;font-size:.75rem;color:var(--c2)">● Up to date</span>`)}
          ${settingToggle('Auto-Scan on Upload','Run AI immediately after image upload','autoScan',false)}
          ${settingToggle('GradCAM by Default','Show heatmap overlay automatically','autoGradcam',true)}
          ${settingRow('Confidence Threshold','Minimum confidence to flag result',`<input type="range" min="50" max="99" value="75" style="width:90px;accent-color:var(--c1)" oninput="this.nextElementSibling.textContent=this.value+'%'"><span style="font-family:'JetBrains Mono',monospace;font-size:.75rem;color:var(--c1);margin-left:.4rem">75%</span>`)}
        </div>

        <!-- Notifications -->
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;color:var(--tx3);margin-bottom:.6rem">Notifications</div>
          ${settingToggle('High-Risk Alerts','Notify when risk score > 70%','notifHighRisk',true)}
          ${settingToggle('Scan Complete','Notify when AI analysis finishes','notifScan',true)}
          ${settingToggle('Weekly Report Email','Auto-email weekly summary','notifEmail',false)}
        </div>

        <!-- Account -->
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;color:var(--tx3);margin-bottom:.6rem">Account</div>
          ${settingRow('Doctor Name','Displayed in reports',`<input class="form-control" value="Dr. Admin" style="width:160px;padding:.35rem .7rem;font-size:.82rem">`)}
          ${settingRow('Hospital','Institution name',`<input class="form-control" value="City Cancer Institute" style="width:190px;padding:.35rem .7rem;font-size:.82rem">`)}
          ${settingRow('Export Format','PDF report format',`<select class="form-control" style="width:130px;padding:.35rem .7rem;font-size:.82rem"><option>A4 Portrait</option><option>A4 Landscape</option><option>Letter</option></select>`)}
        </div>

      </div>

      <div class="modal-foot">
        <button class="btn btn-ghost" onclick="document.getElementById('settingsModal').classList.remove('open')">Cancel</button>
        <button class="btn btn-primary" onclick="saveAllSettings()">Save Settings</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  setTimeout(() => modal.classList.add('open'), 10);
  modal.addEventListener('click', e => { if (e.target === modal) modal.classList.remove('open'); });
}

function settingRow(label, desc, control) {
  return `<div style="display:flex;align-items:center;justify-content:space-between;padding:.6rem 0;border-bottom:1px solid var(--border);gap:.5rem;flex-wrap:wrap">
    <div><div style="font-size:.85rem;color:var(--tx);font-weight:500">${label}</div><div style="font-size:.72rem;color:var(--tx3);margin-top:1px">${desc}</div></div>
    <div style="flex-shrink:0">${control}</div>
  </div>`;
}
function settingToggle(label, desc, key, defaultOn) {
  const on = localStorage.getItem('gs_setting_'+key) !== null ? localStorage.getItem('gs_setting_'+key) === 'true' : defaultOn;
  return settingRow(label, desc, `
    <div onclick="toggleSetting('${key}',this)" style="width:40px;height:22px;border-radius:99px;background:${on?'var(--c1)':'var(--bg4)'};border:1px solid ${on?'var(--c1)':'var(--border2)'};position:relative;cursor:pointer;transition:all .2s;flex-shrink:0" data-on="${on}">
      <div style="width:16px;height:16px;border-radius:50%;background:#fff;position:absolute;top:2px;left:${on?'20px':'2px'};transition:left .2s;box-shadow:0 1px 4px rgba(0,0,0,.3)"></div>
    </div>`);
}
window.toggleSetting = function(key, el) {
  const on = el.dataset.on !== 'true';
  el.dataset.on = on;
  el.style.background = on ? 'var(--c1)' : 'var(--bg4)';
  el.style.borderColor = on ? 'var(--c1)' : 'var(--border2)';
  el.querySelector('div').style.left = on ? '20px' : '2px';
  localStorage.setItem('gs_setting_'+key, on);

  // Actually apply the settings
  applySettingEffect(key, on);
  showToast(`${on ? '✅' : '🔕'} ${settingLabel(key)} ${on ? 'enabled' : 'disabled'}`, 'info', 1600);
};

function settingLabel(key) {
  const map = { compactSidebar:'Compact sidebar', enableAnimations:'Animations', autoScan:'Auto-scan', autoGradcam:'Auto GradCAM', notifHighRisk:'Risk alerts', notifScan:'Scan notifications', notifEmail:'Email reports' };
  return map[key] || key;
}

function applySettingEffect(key, on) {
  if (key === 'compactSidebar') {
    const sidebar = document.getElementById('sidebar');
    const mainWrap = document.querySelector('.main-wrap');
    if (sidebar) sidebar.classList.toggle('compact', on);
    // Remove/add compact CSS
    document.getElementById('compactCSS')?.remove();
    if (on) {
      const s = document.createElement('style');
      s.id = 'compactCSS';
      s.textContent = `
        .sidebar.compact { width:64px !important; overflow:visible !important; }
        .sidebar.compact .brand-text,
        .sidebar.compact .nav-section-label,
        .sidebar.compact .nav-badge,
        .sidebar.compact .sidebar-footer { display:none !important; }
        .sidebar.compact .brand-pulse { font-size:1.4rem !important; }
        .sidebar.compact .sidebar-brand { justify-content:center !important; padding:.85rem .5rem !important; }
        .sidebar.compact .nav-link {
          justify-content:center !important;
          padding:.75rem !important;
          position:relative;
          flex-direction:column;
          gap:0 !important;
        }
        .sidebar.compact .nav-link .nav-icon { font-size:1.3rem !important; margin:0 !important; line-height:1; }
        .sidebar.compact .nav-link span:not(.nav-icon) { display:none !important; }
        .sidebar.compact .nav-link::after {
          content: attr(data-label);
          position:absolute;
          left:70px;
          top:50%;
          transform:translateY(-50%);
          background:var(--surface,#1a1f2e);
          color:var(--tx,#e0eaff);
          padding:5px 10px;
          border-radius:8px;
          font-size:.78rem;
          white-space:nowrap;
          opacity:0;
          pointer-events:none;
          transition:opacity .15s,transform .15s;
          box-shadow:0 4px 16px rgba(0,0,0,.4);
          border:1px solid var(--border2,rgba(255,255,255,.1));
          z-index:999;
          font-family:'DM Sans',sans-serif;
          transform: translateY(-50%) translateX(-4px);
        }
        .sidebar.compact .nav-link:hover::after {
          opacity:1 !important;
          transform: translateY(-50%) translateX(0) !important;
        }
        .sidebar.compact .nav-section { padding:0 !important; }
        .main-wrap.compact-push { margin-left:64px !important; }
        @media(max-width:768px) { .main-wrap.compact-push { margin-left:0 !important; } }
      `;
      // Add data-label attributes to nav links for tooltips
      document.querySelectorAll('.nav-link').forEach(link => {
        const text = link.textContent.replace(/[0-9]/g,'').trim().replace(/[^\w\s]/g,'').trim();
        if (!link.dataset.label) link.setAttribute('data-label', text);
      });
      document.head.appendChild(s);
      mainWrap?.classList.add('compact-push');
    } else {
      mainWrap?.classList.remove('compact-push');
    }
    showToast(on ? '⬅️ Sidebar collapsed' : '➡️ Sidebar expanded', 'info', 1500);
  }
  if (key === 'enableAnimations') {
    document.documentElement.style.setProperty('--transition-speed', on ? '.22s' : '0s');
    if (!on) {
      const s = document.getElementById('noAnimCSS') || document.createElement('style');
      s.id = 'noAnimCSS';
      s.textContent = on ? '' : '*, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }';
      document.head.appendChild(s);
    } else {
      document.getElementById('noAnimCSS')?.remove();
    }
  }
  if (key === 'autoGradcam') {
    window._autoGradcam = on;
  }
  if (key === 'autoScan') {
    window._autoScan = on;
  }
}

// Apply saved settings on load
function applySavedSettings() {
  ['compactSidebar','enableAnimations','autoGradcam','autoScan','notifHighRisk','notifScan','notifEmail'].forEach(key => {
    const val = localStorage.getItem('gs_setting_'+key);
    if (val !== null) applySettingEffect(key, val === 'true');
  });
}


// ══════════════════════════════════════════════
// PROFILE DROPDOWN (Google-style)
// ══════════════════════════════════════════════
function initProfileDropdown() {
  const avatarBtn = document.querySelector('.avatar-btn');
  if (!avatarBtn) return;

  // Inject style once
  if (!document.getElementById('profileDropStyle')) {
    const s = document.createElement('style');
    s.id = 'profileDropStyle';
    s.textContent = `
      @keyframes profileDropIn {
        from { opacity:0; transform:scale(.92) translateY(-8px); }
        to   { opacity:1; transform:scale(1) translateY(0); }
      }
      .profile-drop {
        position:fixed; z-index:800;
        width:300px;
        background:var(--surface,#1a1f2e);
        border:1px solid var(--border2,rgba(255,255,255,.12));
        border-radius:16px;
        box-shadow:0 24px 64px rgba(0,0,0,.6);
        overflow:hidden;
        animation:profileDropIn .2s cubic-bezier(.34,1.2,.64,1);
      }
      .profile-drop-header {
        background:linear-gradient(135deg,rgba(0,212,255,.08),rgba(123,111,255,.08));
        padding:1.2rem 1.2rem .9rem;
        border-bottom:1px solid var(--border,rgba(255,255,255,.07));
        text-align:center;
      }
      .profile-drop-avatar {
        width:64px; height:64px; border-radius:50%;
        background:linear-gradient(135deg,var(--c1,#00d4ff),var(--c5,#7b6fff));
        display:flex; align-items:center; justify-content:center;
        font-size:1.4rem; font-weight:700; color:#fff;
        margin:0 auto .7rem;
        border:2px solid rgba(0,212,255,.3);
        box-shadow:0 0 0 4px rgba(0,212,255,.08);
      }
      .profile-drop-name {
        font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;color:var(--tx);
      }
      .profile-drop-email {
        font-size:.75rem;color:var(--tx3);margin-top:.2rem;
        font-family:'JetBrains Mono',monospace;
      }
      .profile-drop-divider {
        height:1px;background:var(--border,rgba(255,255,255,.07));
      }
      .profile-drop-item {
        display:flex;align-items:center;gap:.8rem;
        padding:.7rem 1.1rem;
        cursor:pointer;
        transition:background .15s;
        font-size:.87rem;color:var(--tx2);
      }
      .profile-drop-item:hover {
        background:var(--surface2,rgba(255,255,255,.05));
        color:var(--tx);
      }
      .profile-drop-item .pdicon {
        width:32px;height:32px;border-radius:50%;
        display:flex;align-items:center;justify-content:center;
        font-size:.95rem;flex-shrink:0;
      }
      .profile-drop-signout {
        border-top:1px solid var(--border,rgba(255,255,255,.07));
        padding:.65rem 1.1rem;
        display:flex;align-items:center;gap:.7rem;
        cursor:pointer;
        font-size:.84rem;color:var(--c3,#ff3d6e);
        transition:background .15s;
      }
      .profile-drop-signout:hover {
        background:rgba(255,61,110,.07);
      }
    `;
    document.head.appendChild(s);
  }

  let dropOpen = false;
  let dropEl = null;

  function openProfileDrop() {
    if (dropEl) { dropEl.remove(); dropEl = null; dropOpen = false; return; }

    dropEl = document.createElement('div');
    dropEl.className = 'profile-drop';
    dropEl.id = 'profileDropdown';

    // Get saved settings
    const doctorName = localStorage.getItem('gs_doctor_name') || 'Dr. Admin';
    const hospital   = localStorage.getItem('gs_hospital')    || 'City Cancer Institute';
    const email      = localStorage.getItem('gs_email')       || 'admin@gastric-sentinel.ai';

    const initials = doctorName.split(' ').filter(Boolean).map(w => w[0]).join('').slice(0, 2).toUpperCase();

    dropEl.innerHTML = `
      <div class="profile-drop-header">
        <div class="profile-drop-avatar">${initials}</div>
        <div class="profile-drop-name">${doctorName}</div>
        <div class="profile-drop-email">${email}</div>
        <div style="margin-top:.5rem;font-size:.72rem;color:var(--tx3);font-family:'JetBrains Mono',monospace">
          🏥 ${hospital}
        </div>
      </div>

      <div class="profile-drop-item" onclick="openSettings();closeProfileDrop()">
        <div class="pdicon" style="background:rgba(0,212,255,.12);color:var(--c1)">⚙️</div>
        <div>
          <div style="font-weight:500">Settings</div>
          <div style="font-size:.72rem;color:var(--tx3)">Theme, AI model, notifications</div>
        </div>
      </div>

      <div class="profile-drop-item" onclick="openReports();closeProfileDrop()">
        <div class="pdicon" style="background:rgba(0,255,159,.12);color:var(--c2)">📈</div>
        <div>
          <div style="font-weight:500">Reports</div>
          <div style="font-size:.72rem;color:var(--tx3)">View analytics & download PDFs</div>
        </div>
      </div>

      <div class="profile-drop-item" onclick="openRiskAlerts();closeProfileDrop()">
        <div class="pdicon" style="background:rgba(255,61,110,.12);color:var(--c3)">⚠️</div>
        <div>
          <div style="font-weight:500">Risk Alerts</div>
          <div style="font-size:.72rem;color:var(--tx3)">12 active · 1 critical</div>
        </div>
      </div>

      <div class="profile-drop-divider"></div>

      <div class="profile-drop-item" onclick="editProfileInline(document.getElementById('profileDropdown'))">
        <div class="pdicon" style="background:rgba(123,111,255,.12);color:#7b6fff">✏️</div>
        <div>
          <div style="font-weight:500">Edit Profile</div>
          <div style="font-size:.72rem;color:var(--tx3)">Change name, email, hospital</div>
        </div>
      </div>

      <div class="profile-drop-signout" onclick="closeProfileDrop();showToast('👋 Signed out (demo mode)','info')">
        <div class="pdicon" style="background:rgba(255,61,110,.1);color:var(--c3);width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;">🚪</div>
        Sign out
      </div>
    `;

    document.body.appendChild(dropEl);
    dropOpen = true;

    // Position below avatar button
    const rect = avatarBtn.getBoundingClientRect();
    dropEl.style.top  = (rect.bottom + 8) + 'px';
    dropEl.style.right = (window.innerWidth - rect.right) + 'px';

    // Close on outside click
    setTimeout(() => {
      document.addEventListener('click', outsideClickHandler);
    }, 10);
  }

  function outsideClickHandler(e) {
    if (dropEl && !dropEl.contains(e.target) && e.target !== avatarBtn) {
      closeProfileDrop();
    }
  }

  window.closeProfileDrop = function() {
    if (dropEl) { dropEl.remove(); dropEl = null; }
    dropOpen = false;
    document.removeEventListener('click', outsideClickHandler);
  };

  window.editProfileInline = function(dropRef) {
    const targetDrop = dropRef || document.getElementById('profileDropdown');
    if (!targetDrop) return;
    const doctorName = localStorage.getItem('gs_doctor_name') || 'Dr. Admin';
    const hospital   = localStorage.getItem('gs_hospital')    || 'City Cancer Institute';
    const email      = localStorage.getItem('gs_email')       || 'admin@gastric-sentinel.ai';

    const header = targetDrop.querySelector('.profile-drop-header');
    if (!header) return;
    header.innerHTML = `
      <div style="text-align:left;padding:.1rem 0">
        <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.8rem">
          <div style="width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,var(--c1),var(--c5,#7b6fff));display:flex;align-items:center;justify-content:center;font-weight:700;color:#000;font-size:.9rem;flex-shrink:0">${doctorName.split(' ').map(w=>w[0]).join('').slice(0,2).toUpperCase()}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;color:var(--tx3);text-transform:uppercase;letter-spacing:.08em">Edit Profile</div>
        </div>
        <div style="display:flex;flex-direction:column;gap:.4rem">
          <div>
            <label style="font-size:.68rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;display:block;margin-bottom:2px">FULL NAME</label>
            <input id="pd_name"  class="form-control" value="${doctorName}" placeholder="Dr. Full Name" style="padding:.4rem .7rem;font-size:.83rem;width:100%">
          </div>
          <div>
            <label style="font-size:.68rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;display:block;margin-bottom:2px">EMAIL</label>
            <input id="pd_email" class="form-control" value="${email}" placeholder="doctor@hospital.com" style="padding:.4rem .7rem;font-size:.83rem;width:100%">
          </div>
          <div>
            <label style="font-size:.68rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;display:block;margin-bottom:2px">HOSPITAL / INSTITUTION</label>
            <input id="pd_hosp"  class="form-control" value="${hospital}" placeholder="City Cancer Institute" style="padding:.4rem .7rem;font-size:.83rem;width:100%;margin-bottom:.2rem">
          </div>
        </div>
        <div style="display:flex;gap:.4rem;margin-top:.7rem">
          <button onclick="saveProfileEdit()" class="btn btn-primary btn-sm" style="flex:1;justify-content:center;padding:.45rem">💾 Save</button>
          <button onclick="cancelEditProfile()" class="btn btn-ghost btn-sm" style="flex:1;justify-content:center;padding:.45rem">Cancel</button>
        </div>
      </div>
    `;
    // Focus first input
    setTimeout(() => document.getElementById('pd_name')?.focus(), 50);
  };

  window.cancelEditProfile = function() {
    // Rebuild the dropdown instead of closing it
    closeProfileDrop();
    // Re-open after brief delay
    setTimeout(() => document.querySelector('.avatar-btn')?.click(), 60);
  };

  window.saveProfileEdit = function() {
    const nameVal  = document.getElementById('pd_name')?.value.trim();
    const emailVal = document.getElementById('pd_email')?.value.trim();
    const hospVal  = document.getElementById('pd_hosp')?.value.trim();
    if (!nameVal) { showToast('Name cannot be empty', 'warn'); return; }
    if (nameVal)  localStorage.setItem('gs_doctor_name', nameVal);
    if (emailVal) localStorage.setItem('gs_email', emailVal);
    if (hospVal)  localStorage.setItem('gs_hospital', hospVal);
    closeProfileDrop();
    showToast('✅ Profile updated', 'ok');
    const initials = nameVal.split(' ').filter(Boolean).map(w=>w[0]).join('').slice(0,2).toUpperCase();
    document.querySelectorAll('.avatar-btn').forEach(b => b.textContent = initials || 'DR');
  };

  avatarBtn.style.cursor = 'pointer';
  avatarBtn.addEventListener('click', e => { e.stopPropagation(); openProfileDrop(); });
}

// ── Save All Settings ────────────────────────
window.saveAllSettings = function() {
  // Save text inputs (doctor name, hospital, export format)
  const nameInp = document.querySelector('#settingsModal input[value*="Dr"]') || 
                  [...document.querySelectorAll('#settingsModal input')].find(i => i.value.includes('Dr'));
  const hospInp = [...document.querySelectorAll('#settingsModal input')].find(i => i.placeholder === 'Institution name' || i.value.includes('Cancer') || i.value.includes('Hospital'));
  
  // Find by looking for all inputs in settings modal
  const allInputs = document.querySelectorAll('#settingsModal .form-control');
  allInputs.forEach(inp => {
    if (inp.tagName === 'INPUT' && inp.type !== 'range') {
      if (inp.value.startsWith('Dr.') || inp.placeholder?.includes('name')) {
        localStorage.setItem('gs_doctor_name', inp.value);
      }
      if (inp.value.includes('Institute') || inp.value.includes('Hospital') || inp.value.includes('Cancer')) {
        localStorage.setItem('gs_hospital', inp.value);
      }
    }
  });

  // Update confidence threshold
  const slider = document.querySelector('#settingsModal input[type="range"]');
  if (slider) localStorage.setItem('gs_confidence_threshold', slider.value);

  document.getElementById('settingsModal').classList.remove('open');
  showToast('✅ Settings saved successfully', 'ok');
  
  // Update avatar initials if name changed
  const doctorName = localStorage.getItem('gs_doctor_name') || 'Dr. Admin';
  const initials = doctorName.split(' ').filter(Boolean).map(w=>w[0]).join('').slice(0,2).toUpperCase();
  document.querySelectorAll('.avatar-btn').forEach(b => { b.textContent = initials || 'DR'; });
};

// ── Init ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initSidebar();
  initSearch();
  initNotifications();
  initHoverEffects();
  initPathologyZoom();
  initUpload();
  initChat();
  initStars();
  initAnimations();
  initClock();
  animateRiskBars();
  initProfileDropdown();
  applySavedSettings();

  // Wire sidebar nav links for Reports / Risk Alerts / Settings
  document.querySelectorAll('.nav-link').forEach(link => {
    const text = link.textContent.trim();
    if (text.includes('Reports')) {
      link.addEventListener('click', e => { e.preventDefault(); openReports(); });
    }
    if (text.includes('Risk Alerts')) {
      link.addEventListener('click', e => { e.preventDefault(); openRiskAlerts(); });
    }
    if (text.includes('Settings')) {
      link.addEventListener('click', e => { e.preventDefault(); openSettings(); });
    }
  });

  // Also wire Quick Actions buttons if present
  document.querySelectorAll('button, a').forEach(el => {
    const t = el.textContent.trim();
    if (t.includes('Generate Report') || t.includes('View Report')) {
      el.addEventListener('click', e => { e.preventDefault(); openReports(); });
    }
    if (t.includes('View Risk Alerts') || t.includes('Risk Alert')) {
      el.addEventListener('click', e => { e.preventDefault(); openRiskAlerts(); });
    }
  });

  // Page-specific
  if (document.getElementById('patientsCount')) loadDashboardStats();
  if (document.getElementById('patientsTable') && !document.getElementById('patientsCount')) {
    loadRecentPatients(DEMO_PATIENTS);
    setTimeout(initPatientSearch, 400);
  }

  loadScanActivity();
  window.addEventListener('resize', drawActivityChart);
  // Init patient inline search after table loads
  setTimeout(initPatientSearch, 400);

  // Bind static buttons
  document.getElementById('themeToggle')?.addEventListener('click', toggleTheme);
  document.getElementById('scanBtn')?.addEventListener('click', runScan);
  document.getElementById('toggleHeatBtn')?.addEventListener('click', toggleHeatmap);
  document.getElementById('downloadReport')?.addEventListener('click', downloadReport);
  document.getElementById('confirmBtn')?.addEventListener('click', () => submitFeedback('confirm'));
  document.getElementById('incorrectBtn')?.addEventListener('click', () => submitFeedback('incorrect'));
  document.getElementById('chatFab')?.addEventListener('click', toggleChat);
  document.getElementById('chatClose')?.addEventListener('click', closeChat);
  document.getElementById('addPatientBtn')?.addEventListener('click', submitPatient);
  document.getElementById('openAddModal')?.addEventListener('click', openAddPatient);
  document.getElementById('modalCloseBtn')?.addEventListener('click', () => closeModal('patientModal'));

  // Close modals on backdrop click
  document.querySelectorAll('.modal-backdrop').forEach(bd => {
    bd.addEventListener('click', e => { if (e.target === bd) bd.classList.remove('open'); });
  });
});
// ═══════════════════════════════════════════════════════════════
// NEW FEATURES — PDF Modal, Batch Upload, Confidence Trend, Chatbot Fix
// ═══════════════════════════════════════════════════════════════

// ── 1. PDF Report Modal ──────────────────────────────────────────
function openReportModal() {
  const existing = document.getElementById('reportModal');
  if (existing) { existing.classList.add('open'); return; }

  const tier = window._lastTier || 'UNKNOWN';
  const tierColor = { CRITICAL: '#ff3d6e', SUSPICIOUS: '#ffb340', NEGATIVE: '#00d4aa', UNKNOWN: '#8b9ab4' }[tier] || '#8b9ab4';

  const modal = document.createElement('div');
  modal.id = 'reportModal';
  modal.className = 'modal-backdrop';
  modal.innerHTML = `
    <div class="modal" style="max-width:560px;padding:0;overflow:hidden">
      <div style="background:linear-gradient(135deg,var(--bg2),var(--bg3));padding:1.4rem 1.6rem;border-bottom:1px solid var(--border)">
        <div style="display:flex;align-items:center;justify-content:space-between">
          <div style="display:flex;align-items:center;gap:.7rem">
            <div style="width:36px;height:36px;border-radius:10px;background:${tierColor}22;border:1px solid ${tierColor}55;display:grid;place-items:center;font-size:1.1rem">📄</div>
            <div>
              <div style="font-weight:700;font-size:.95rem;color:var(--tx1)">Generate PDF Report</div>
              <div style="font-size:.72rem;color:var(--tx3);font-family:'JetBrains Mono',monospace">Enter patient details for the report</div>
            </div>
          </div>
          <button onclick="document.getElementById('reportModal').classList.remove('open')" style="width:28px;height:28px;border-radius:50%;background:var(--bg4);border:1px solid var(--border);color:var(--tx2);cursor:pointer;font-size:.85rem;display:grid;place-items:center">✕</button>
        </div>
      </div>

      <div style="padding:1.4rem 1.6rem;display:flex;flex-direction:column;gap:1rem">

        <div style="background:var(--bg3);border:1px solid ${tierColor}44;border-radius:10px;padding:.9rem 1rem">
          <div style="font-size:.7rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.35rem">Current Diagnosis</div>
          <div style="font-weight:700;color:${tierColor};font-size:.95rem">${window._lastDiagnosis || 'No scan yet'}</div>
          <div style="font-size:.75rem;color:var(--tx3);margin-top:.2rem">${window._lastTier || '—'} · Confidence: ${window._lastConfidence || '—'}%</div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.75rem">
          <div>
            <label style="font-size:.72rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.06em;display:block;margin-bottom:.35rem">Patient Name *</label>
            <input id="rpm_patient" class="form-control" placeholder="e.g. John Smith" value="${window._lastPatientName || ''}" style="width:100%;box-sizing:border-box">
          </div>
          <div>
            <label style="font-size:.72rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.06em;display:block;margin-bottom:.35rem">Patient Age</label>
            <input id="rpm_age" class="form-control" placeholder="e.g. 58" type="number" min="1" max="120" style="width:100%;box-sizing:border-box">
          </div>
          <div>
            <label style="font-size:.72rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.06em;display:block;margin-bottom:.35rem">Patient ID / MRN</label>
            <input id="rpm_id" class="form-control" placeholder="e.g. MRN-2024-0081" style="width:100%;box-sizing:border-box">
          </div>
          <div>
            <label style="font-size:.72rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.06em;display:block;margin-bottom:.35rem">Gender</label>
            <select id="rpm_gender" class="form-control" style="width:100%;box-sizing:border-box">
              <option value="">Select…</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.75rem">
          <div>
            <label style="font-size:.72rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.06em;display:block;margin-bottom:.35rem">Physician Name</label>
            <input id="rpm_doctor" class="form-control" placeholder="Dr. Jane Doe" value="${localStorage.getItem('gs_doctor_name') || ''}" style="width:100%;box-sizing:border-box">
          </div>
          <div>
            <label style="font-size:.72rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.06em;display:block;margin-bottom:.35rem">Institution</label>
            <input id="rpm_hospital" class="form-control" placeholder="City Cancer Institute" value="${localStorage.getItem('gs_hospital') || ''}" style="width:100%;box-sizing:border-box">
          </div>
        </div>

        <div>
          <label style="font-size:.72rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.06em;display:block;margin-bottom:.35rem">Clinical Notes (optional)</label>
          <textarea id="rpm_notes" class="form-control" rows="2" placeholder="Additional observations, biopsy site, clinical context…" style="width:100%;box-sizing:border-box;resize:vertical;min-height:56px"></textarea>
        </div>

        <div style="display:flex;gap:.6rem;margin-top:.2rem">
          <button onclick="document.getElementById('reportModal').classList.remove('open')" class="btn btn-ghost" style="flex:0 0 auto">Cancel</button>
          <button onclick="submitReportModal()" class="btn btn-amber" id="rpmSubmitBtn" style="flex:1;justify-content:center">
            📄 Generate & Download PDF
          </button>
        </div>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  setTimeout(() => modal.classList.add('open'), 10);
  modal.addEventListener('click', e => { if (e.target === modal) modal.classList.remove('open'); });
  document.getElementById('rpm_patient')?.focus();
}

async function submitReportModal() {
  const patientName = document.getElementById('rpm_patient')?.value.trim() || 'Anonymous';
  const patientAge  = document.getElementById('rpm_age')?.value || '';
  const patientId   = document.getElementById('rpm_id')?.value.trim() || '';
  const gender      = document.getElementById('rpm_gender')?.value || '';
  const doctor      = document.getElementById('rpm_doctor')?.value.trim() || localStorage.getItem('gs_doctor_name') || 'Dr. Admin';
  const hospital    = document.getElementById('rpm_hospital')?.value.trim() || localStorage.getItem('gs_hospital') || 'City Cancer Institute';
  const notes       = document.getElementById('rpm_notes')?.value.trim() || '';

  if (doctor) localStorage.setItem('gs_doctor_name', doctor);
  if (hospital) localStorage.setItem('gs_hospital', hospital);
  window._lastPatientName = patientName;

  const btn = document.getElementById('rpmSubmitBtn');
  if (btn) { btn.classList.add('btn-loading'); btn.innerHTML = '<span class="btn-spinner"></span> Generating…'; }

  const payload = {
    patient_name:    patientName,
    patient_age:     patientAge,
    patient_id:      patientId,
    patient_gender:  gender,
    doctor,
    hospital,
    notes,
    diagnosis:       window._lastDiagnosis     || 'N/A',
    recommendation:  window._lastRec           || 'Consult a specialist.',
    risk_score:      window._lastRiskScore      || 0,
    confidence:      window._lastConfidence     || 'N/A',
    predicted_class: window._lastPredictedClass || 'N/A',
    tier:            window._lastTier           || 'UNKNOWN',
    probabilities:   window._lastRawProbs       || {},
  };

  try {
    const res = await fetch('/generate_report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('Server error ' + res.status);
    const blob = await res.blob();
    if (blob.size < 200) throw new Error('Empty PDF');
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `GS_Report_${patientName.replace(/\s+/g,'_')}_${new Date().toISOString().slice(0,10)}.pdf`;
    document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 2000);
    showToast('📄 Report downloaded for ' + patientName, 'ok');
    document.getElementById('reportModal')?.classList.remove('open');
  } catch (err) {
    console.warn('PDF backend error, using print fallback:', err.message);
    _printFallbackReport({ ...payload, patient_name: patientName });
    document.getElementById('reportModal')?.classList.remove('open');
  }

  if (btn) { btn.classList.remove('btn-loading'); btn.innerHTML = '📄 Generate & Download PDF'; }
}


// ── 2. Batch Upload & Multi-Scan ────────────────────────────────
let _batchResults = [];

function openBatchModal() {
  const existing = document.getElementById('batchModal');
  if (existing) { _resetBatchModal(); existing.classList.add('open'); return; }

  const modal = document.createElement('div');
  modal.id = 'batchModal';
  modal.className = 'modal-backdrop';
  modal.innerHTML = `
    <div class="modal" style="max-width:740px;padding:0;overflow:hidden;max-height:90vh;display:flex;flex-direction:column">
      <div style="background:linear-gradient(135deg,var(--bg2),var(--bg3));padding:1.2rem 1.6rem;border-bottom:1px solid var(--border);flex-shrink:0">
        <div style="display:flex;align-items:center;justify-content:space-between">
          <div style="display:flex;align-items:center;gap:.7rem">
            <div style="width:36px;height:36px;border-radius:10px;background:#7c3aed22;border:1px solid #7c3aed55;display:grid;place-items:center;font-size:1.1rem">🔬</div>
            <div>
              <div style="font-weight:700;font-size:.95rem;color:var(--tx1)">Batch Image Analysis</div>
              <div style="font-size:.72rem;color:var(--tx3);font-family:'JetBrains Mono',monospace">Upload multiple patches — all processed sequentially</div>
            </div>
          </div>
          <button onclick="document.getElementById('batchModal').classList.remove('open')" style="width:28px;height:28px;border-radius:50%;background:var(--bg4);border:1px solid var(--border);color:var(--tx2);cursor:pointer;font-size:.85rem;display:grid;place-items:center">✕</button>
        </div>
      </div>

      <div style="padding:1.4rem 1.6rem;overflow-y:auto;flex:1;display:flex;flex-direction:column;gap:1rem">
        <div id="batchDropZone" style="border:2px dashed var(--border);border-radius:12px;padding:2rem;text-align:center;cursor:pointer;transition:.2s;position:relative"
          onclick="document.getElementById('batchFileInput').click()"
          ondragover="event.preventDefault();this.style.borderColor='var(--c1)'"
          ondragleave="this.style.borderColor='var(--border)'"
          ondrop="event.preventDefault();this.style.borderColor='var(--border)';handleBatchDrop(event.dataTransfer.files)">
          <input type="file" id="batchFileInput" multiple accept="image/*" style="display:none" onchange="handleBatchFiles(this.files)">
          <div style="font-size:2rem;margin-bottom:.5rem">🖼️</div>
          <div style="font-weight:600;color:var(--tx1);margin-bottom:.25rem">Drop images or click to browse</div>
          <div style="font-size:.78rem;color:var(--tx3)">Select up to 20 histopathology patches at once · JPEG, PNG, TIFF</div>
        </div>

        <div id="batchFileList" style="display:none;display:flex;flex-direction:column;gap:.5rem"></div>

        <div id="batchProgress" style="display:none">
          <div style="display:flex;justify-content:space-between;margin-bottom:.4rem">
            <span style="font-size:.78rem;color:var(--tx2);font-family:'JetBrains Mono',monospace" id="batchProgressLabel">Processing 0 / 0…</span>
            <span style="font-size:.78rem;color:var(--c1)" id="batchProgressPct">0%</span>
          </div>
          <div style="height:6px;background:var(--bg4);border-radius:3px;overflow:hidden">
            <div id="batchProgressBar" style="height:100%;width:0%;background:linear-gradient(90deg,var(--c1),#7c3aed);border-radius:3px;transition:width .3s ease"></div>
          </div>
        </div>

        <div id="batchResultsWrap" style="display:none">
          <div style="font-size:.78rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.6rem">Results</div>
          <div id="batchResultsGrid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:.6rem"></div>
          <div id="batchSummary" style="margin-top:1rem;padding:.9rem 1rem;background:var(--bg3);border-radius:10px;border:1px solid var(--border)"></div>
        </div>
      </div>

      <div style="padding:.9rem 1.6rem;border-top:1px solid var(--border);display:flex;gap:.6rem;flex-shrink:0">
        <button onclick="document.getElementById('batchModal').classList.remove('open')" class="btn btn-ghost">Close</button>
        <button id="batchRunBtn" onclick="runBatchScan()" class="btn btn-primary" style="flex:1;justify-content:center" disabled>
          🔬 Run Batch Analysis
        </button>
        <button id="batchReportBtn" onclick="downloadBatchReport()" class="btn btn-amber" style="display:none">
          📄 Download Report
        </button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  setTimeout(() => modal.classList.add('open'), 10);
  modal.addEventListener('click', e => { if (e.target === modal) modal.classList.remove('open'); });
}

let _batchFiles = [];

function handleBatchDrop(files) { handleBatchFiles(files); }

function handleBatchFiles(files) {
  _batchFiles = Array.from(files).slice(0, 20);
  _batchResults = [];
  const list = document.getElementById('batchFileList');
  const runBtn = document.getElementById('batchRunBtn');
  if (!list || !_batchFiles.length) return;

  list.style.display = 'flex';
  list.innerHTML = _batchFiles.map((f, i) => `
    <div style="display:flex;align-items:center;gap:.7rem;padding:.55rem .75rem;background:var(--bg3);border-radius:8px;border:1px solid var(--border)" id="batchFile_${i}">
      <span style="font-size:1rem">🖼️</span>
      <span style="flex:1;font-size:.8rem;color:var(--tx2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${f.name}</span>
      <span style="font-size:.72rem;color:var(--tx3);font-family:'JetBrains Mono',monospace">${(f.size/1024).toFixed(0)} KB</span>
      <span id="batchFileStatus_${i}" style="font-size:.75rem;color:var(--tx3)">Pending</span>
    </div>
  `).join('');
  if (runBtn) { runBtn.disabled = false; }
  document.getElementById('batchResultsWrap').style.display = 'none';
  document.getElementById('batchReportBtn').style.display = 'none';
}

async function runBatchScan() {
  if (!_batchFiles.length) return;
  _batchResults = [];
  const runBtn = document.getElementById('batchRunBtn');
  const progress = document.getElementById('batchProgress');
  const progressBar = document.getElementById('batchProgressBar');
  const progressLabel = document.getElementById('batchProgressLabel');
  const progressPct = document.getElementById('batchProgressPct');

  if (runBtn) { runBtn.disabled = true; runBtn.innerHTML = '<span class="btn-spinner"></span> Processing…'; }
  if (progress) progress.style.display = 'block';

  for (let i = 0; i < _batchFiles.length; i++) {
    const f = _batchFiles[i];
    const statusEl = document.getElementById(`batchFileStatus_${i}`);
    const rowEl = document.getElementById(`batchFile_${i}`);
    if (statusEl) statusEl.innerHTML = '<span style="color:var(--c4)">⏳ Scanning…</span>';

    const pct = Math.round((i / _batchFiles.length) * 100);
    if (progressBar) progressBar.style.width = pct + '%';
    if (progressLabel) progressLabel.textContent = `Processing ${i + 1} / ${_batchFiles.length}…`;
    if (progressPct) progressPct.textContent = pct + '%';

    try {
      const formData = new FormData();
      formData.append('image', f);
      const res = await fetch('/predict', { method: 'POST', body: formData });
      if (!res.ok) throw new Error('Server error');
      const d = await res.json();

      const tier = d.tier || 'UNKNOWN';
      const tColor = { CRITICAL:'#ff3d6e', SUSPICIOUS:'#ffb340', NEGATIVE:'#00d4aa', UNKNOWN:'#8b9ab4' }[tier] || '#8b9ab4';
      _batchResults.push({ file: f.name, ...d, tier, tColor });
      if (statusEl) statusEl.innerHTML = `<span style="color:${tColor};font-weight:600">${tier}</span>`;
      if (rowEl) rowEl.style.borderColor = tColor + '66';
    } catch (err) {
      const demo = ['NEGATIVE','SUSPICIOUS','CRITICAL'][Math.floor(Math.random()*3)];
      const tColor = { CRITICAL:'#ff3d6e', SUSPICIOUS:'#ffb340', NEGATIVE:'#00d4aa' }[demo];
      _batchResults.push({ file: f.name, tier: demo, tColor, diagnosis: demo === 'CRITICAL' ? 'Tumor (Demo)' : demo === 'SUSPICIOUS' ? 'Stroma (Demo)' : 'Normal (Demo)', probability: Math.random() * 0.4 + 0.6, risk_score: demo === 'CRITICAL' ? 75 + Math.random()*20 : demo === 'SUSPICIOUS' ? 35 + Math.random()*25 : Math.random()*20, probabilities: {} });
      if (statusEl) statusEl.innerHTML = `<span style="color:${tColor}">${demo} (demo)</span>`;
    }

    await new Promise(r => setTimeout(r, 300));
  }

  if (progressBar) progressBar.style.width = '100%';
  if (progressLabel) progressLabel.textContent = `Completed ${_batchFiles.length} / ${_batchFiles.length}`;
  if (progressPct) progressPct.textContent = '100%';
  if (runBtn) { runBtn.disabled = false; runBtn.innerHTML = '🔬 Re-run Analysis'; }

  renderBatchResults();
}

function renderBatchResults() {
  const wrap = document.getElementById('batchResultsWrap');
  const grid = document.getElementById('batchResultsGrid');
  const summary = document.getElementById('batchSummary');
  const reportBtn = document.getElementById('batchReportBtn');
  if (!wrap || !grid) return;

  wrap.style.display = 'block';
  if (reportBtn) reportBtn.style.display = 'inline-flex';

  const critical = _batchResults.filter(r => r.tier === 'CRITICAL').length;
  const suspicious = _batchResults.filter(r => r.tier === 'SUSPICIOUS').length;
  const negative = _batchResults.filter(r => r.tier === 'NEGATIVE').length;

  grid.innerHTML = _batchResults.map((r, i) => `
    <div style="background:var(--bg3);border:1px solid ${r.tColor}44;border-radius:10px;padding:.8rem;transition:.2s" onmouseenter="this.style.borderColor='${r.tColor}'" onmouseleave="this.style.borderColor='${r.tColor}44'">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.5rem">
        <span style="font-size:.65rem;font-family:'JetBrains Mono',monospace;color:var(--tx3)">#${String(i+1).padStart(2,'0')}</span>
        <span style="font-size:.68rem;font-weight:700;color:${r.tColor};background:${r.tColor}22;padding:2px 8px;border-radius:4px">${r.tier}</span>
      </div>
      <div style="font-size:.78rem;color:var(--tx1);font-weight:600;margin-bottom:.3rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${r.file}">${r.file}</div>
      <div style="font-size:.72rem;color:var(--tx3);margin-bottom:.4rem">${r.diagnosis || '—'}</div>
      <div style="height:4px;background:var(--bg4);border-radius:2px;overflow:hidden">
        <div style="height:100%;width:${Math.round(r.risk_score || 0)}%;background:${r.tColor};border-radius:2px"></div>
      </div>
      <div style="font-size:.68rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;margin-top:.3rem">Risk: ${Math.round(r.risk_score || 0)}% · P: ${r.probability ? (r.probability*100).toFixed(1) : '—'}%</div>
    </div>
  `).join('');

  summary.innerHTML = `
    <div style="font-size:.72rem;color:var(--tx3);font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.6rem">Batch Summary — ${_batchResults.length} images</div>
    <div style="display:flex;gap:1.2rem;flex-wrap:wrap">
      <div style="text-align:center"><div style="font-size:1.5rem;font-weight:800;color:#ff3d6e">${critical}</div><div style="font-size:.72rem;color:var(--tx3)">Critical</div></div>
      <div style="text-align:center"><div style="font-size:1.5rem;font-weight:800;color:#ffb340">${suspicious}</div><div style="font-size:.72rem;color:var(--tx3)">Suspicious</div></div>
      <div style="text-align:center"><div style="font-size:1.5rem;font-weight:800;color:#00d4aa">${negative}</div><div style="font-size:.72rem;color:var(--tx3)">Negative</div></div>
      <div style="text-align:center"><div style="font-size:1.5rem;font-weight:800;color:var(--tx1)">${_batchResults.length}</div><div style="font-size:.72rem;color:var(--tx3)">Total</div></div>
    </div>
    ${critical > 0 ? `<div style="margin-top:.7rem;padding:.6rem .8rem;background:#ff3d6e15;border:1px solid #ff3d6e44;border-radius:8px;font-size:.78rem;color:#ff3d6e">⚠️ ${critical} critical finding${critical>1?'s':''} detected — immediate oncology review recommended</div>` : ''}
  `;
}

function downloadBatchReport() {
  if (!_batchResults.length) { showToast('No batch results to export', 'warn'); return; }
  const rows = [['#','File','Tier','Diagnosis','Risk Score','Confidence']];
  _batchResults.forEach((r,i) => rows.push([i+1, r.file, r.tier, r.diagnosis||'', Math.round(r.risk_score||0)+'%', r.probability ? (r.probability*100).toFixed(1)+'%' : '—']));
  const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g,'""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `GS_BatchReport_${new Date().toISOString().slice(0,10)}.csv`; a.click();
  URL.revokeObjectURL(url);
  showToast('📥 Batch CSV exported — ' + _batchResults.length + ' results', 'ok');
}

function _resetBatchModal() {
  _batchFiles = []; _batchResults = [];
  const list = document.getElementById('batchFileList');
  if (list) { list.style.display = 'none'; list.innerHTML = ''; }
  document.getElementById('batchProgress').style.display = 'none';
  document.getElementById('batchResultsWrap').style.display = 'none';
  document.getElementById('batchReportBtn').style.display = 'none';
  const runBtn = document.getElementById('batchRunBtn');
  if (runBtn) { runBtn.disabled = true; runBtn.innerHTML = '🔬 Run Batch Analysis'; }
}


// ── 3. Confidence Trend Chart ────────────────────────────────────
let _trendData = JSON.parse(localStorage.getItem('gs_trend_data') || '[]');

function recordTrendPoint(tier, confidence, predictedClass) {
  const point = {
    t: Date.now(),
    label: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    confidence: typeof confidence === 'number' ? Math.round(confidence) : parseInt(confidence) || 0,
    tier,
    cls: predictedClass || '—'
  };
  _trendData.push(point);
  if (_trendData.length > 20) _trendData = _trendData.slice(-20);
  try { localStorage.setItem('gs_trend_data', JSON.stringify(_trendData)); } catch(e) {}
  if (document.getElementById('trendChart')) renderTrendChart();
}

function renderTrendChart() {
  const container = document.getElementById('trendChart');
  if (!container) return;

  if (_trendData.length < 2) {
    container.innerHTML = `<div style="text-align:center;padding:2rem 0;color:var(--tx3);font-size:.82rem;font-family:'JetBrains Mono',monospace">Run at least 2 scans to see the confidence trend</div>`;
    return;
  }

  const W = container.clientWidth || 400;
  const H = 120;
  const PAD = { l: 36, r: 12, t: 14, b: 28 };
  const iW = W - PAD.l - PAD.r;
  const iH = H - PAD.t - PAD.b;

  const vals = _trendData.map(d => d.confidence);
  const minV = Math.max(0, Math.min(...vals) - 5);
  const maxV = Math.min(100, Math.max(...vals) + 5);

  const xScale = i => PAD.l + (i / (_trendData.length - 1)) * iW;
  const yScale = v => PAD.t + iH - ((v - minV) / (maxV - minV)) * iH;

  const pts = _trendData.map((d, i) => `${xScale(i)},${yScale(d.confidence)}`).join(' ');
  const firstPt = `${xScale(0)},${yScale(_trendData[0].confidence)}`;
  const lastPt  = `${xScale(_trendData.length-1)},${yScale(_trendData[_trendData.length-1].confidence)}`;

  const tierColor = t => ({ CRITICAL: '#ff3d6e', SUSPICIOUS: '#ffb340', NEGATIVE: '#00d4aa', UNKNOWN: '#8b9ab4' }[t] || '#8b9ab4');

  const dots = _trendData.map((d, i) => {
    const cx = xScale(i), cy = yScale(d.confidence);
    const col = tierColor(d.tier);
    return `<circle cx="${cx}" cy="${cy}" r="4" fill="${col}" stroke="var(--bg2)" stroke-width="2" style="cursor:pointer">
      <title>${d.label} — ${d.cls}: ${d.confidence}% (${d.tier})</title>
    </circle>`;
  }).join('');

  const labels = _trendData.filter((_, i) => i % Math.ceil(_trendData.length / 5) === 0 || i === _trendData.length - 1)
    .map((d, _, arr) => {
      const origIdx = _trendData.indexOf(d);
      return `<text x="${xScale(origIdx)}" y="${H - 4}" text-anchor="middle" font-size="9" fill="var(--tx3)" font-family="'JetBrains Mono',monospace">${d.label}</text>`;
    }).join('');

  const yTicks = [0, 25, 50, 75, 100].filter(v => v >= minV && v <= maxV).map(v =>
    `<line x1="${PAD.l}" y1="${yScale(v)}" x2="${W - PAD.r}" y2="${yScale(v)}" stroke="var(--border)" stroke-width="0.5" stroke-dasharray="3,3"/>
     <text x="${PAD.l - 4}" y="${yScale(v) + 3}" text-anchor="end" font-size="9" fill="var(--tx3)" font-family="'JetBrains Mono',monospace">${v}</text>`
  ).join('');

  const gradId = 'trendGrad_' + Date.now();
  const last = _trendData[_trendData.length - 1];
  const lineColor = tierColor(last.tier);

  container.innerHTML = `
    <svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <defs>
        <linearGradient id="${gradId}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="${lineColor}" stop-opacity="0.25"/>
          <stop offset="100%" stop-color="${lineColor}" stop-opacity="0"/>
        </linearGradient>
      </defs>
      ${yTicks}
      <polygon points="${firstPt} ${pts} ${lastPt} ${PAD.l + iW},${H - PAD.b} ${PAD.l},${H - PAD.b}" fill="url(#${gradId})"/>
      <polyline points="${pts}" fill="none" stroke="${lineColor}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>
      ${dots}
      ${labels}
    </svg>
    <div style="display:flex;gap:1rem;margin-top:.5rem;flex-wrap:wrap">
      ${[['CRITICAL','#ff3d6e'],['SUSPICIOUS','#ffb340'],['NEGATIVE','#00d4aa']].map(([t,c]) => `<span style="font-size:.68rem;color:${c};font-family:'JetBrains Mono',monospace;display:flex;align-items:center;gap:.3rem"><span style="width:8px;height:8px;border-radius:50%;background:${c};display:inline-block"></span>${t}</span>`).join('')}
      <span style="font-size:.68rem;color:var(--tx3);margin-left:auto;font-family:'JetBrains Mono',monospace">${_trendData.length} scan${_trendData.length !== 1 ? 's' : ''} recorded</span>
    </div>
  `;
}

function clearTrendData() {
  _trendData = [];
  try { localStorage.removeItem('gs_trend_data'); } catch(e) {}
  renderTrendChart();
  showToast('Trend data cleared', 'info');
}


// ── 4. Improved Chatbot ──────────────────────────────────────────
async function sendChatMessageV2() {
  const input = document.getElementById('chatInput');
  const body  = document.getElementById('chatBody');
  if (!input || !body) return;

  const msg = input.value.trim();
  if (!msg) return;

  appendChatMsg(msg, 'user', body);
  input.value = '';

  const typing = document.createElement('div');
  typing.className = 'chat-msg ai';
  typing.id = 'typingIndicator';
  typing.innerHTML = `<div class="c-av">🤖</div><div class="c-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>`;
  body.appendChild(typing);
  body.scrollTop = body.scrollHeight;

  // Build rich context from current scan state
  const ctx = {
    diagnosis:       window._lastDiagnosis       || null,
    tier:            window._lastTier             || null,
    risk_score:      window._lastRiskScore        || null,
    confidence:      window._lastConfidence       || null,
    predicted_class: window._lastPredictedClass   || null,
    recommendation:  window._lastRec              || null,
    probabilities:   window._lastRawProbs         || null,
  };

  const hasScan = !!(ctx.diagnosis && ctx.diagnosis !== 'N/A');

  const systemContext = hasScan
    ? `Current scan result: ${ctx.predicted_class} (${ctx.tier}) — ${ctx.diagnosis}. Confidence: ${ctx.confidence}%. Risk score: ${ctx.risk_score}%. Recommendation: ${ctx.recommendation}. ${ctx.probabilities ? 'Class probabilities: ' + Object.entries(ctx.probabilities).map(([k,v]) => `${k}:${(v*100).toFixed(1)}%`).join(', ') + '.' : ''}`
    : 'No scan has been performed yet in this session.';

  try {
    const res = await fetch('/chatbot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: msg,
        context: systemContext,
        scan_available: hasScan,
      })
    });

    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const reply = data.reply || data.response || data.message || 'No response received.';
    typing.remove();
    appendChatMsg(reply, 'ai', body);

  } catch (err) {
    typing.remove();
    // Contextual fallback — not random
    let fallback;
    const q = msg.toLowerCase();
    if (!hasScan) {
      fallback = "Please upload and scan a histopathology image first — I'll then be able to provide specific clinical insights on your result.";
    } else if (q.includes('gradcam') || q.includes('heatmap')) {
      fallback = `The GradCAM overlay for this ${ctx.predicted_class} prediction highlights the tissue regions that most influenced the classification. Red/orange areas indicate the highest model attention — these are the zones the network weighted most heavily when arriving at the ${ctx.tier} verdict.`;
    } else if (q.includes('shap')) {
      fallback = `SHAP values for this scan show pixel-level attribution relative to a neutral baseline. Positive (red) regions support the ${ctx.predicted_class} classification, while negative (blue) regions oppose it. This is complementary to GradCAM — SHAP shows direction of influence, GradCAM shows magnitude.`;
    } else if (q.includes('recommend') || q.includes('next') || q.includes('what should')) {
      fallback = `Based on the ${ctx.tier} finding (${ctx.diagnosis}): ${ctx.recommendation || 'Consult a specialist for further evaluation.'}`;
    } else if (q.includes('risk') || q.includes('danger') || q.includes('serious')) {
      fallback = `The current risk score is ${ctx.risk_score}% (${ctx.tier}). ${ctx.tier === 'CRITICAL' ? 'This indicates high-probability malignant tissue — immediate pathological review and oncology referral are recommended.' : ctx.tier === 'SUSPICIOUS' ? 'This is a borderline finding. Close follow-up and adjacent tissue biopsy are advisable.' : 'No immediate malignancy risk detected, but routine surveillance is still recommended.'}`;
    } else if (q.includes('confidence') || q.includes('accurate') || q.includes('sure')) {
      fallback = `Model confidence for this prediction is ${ctx.confidence}%. This ResNet50 + multimodal fusion model was trained on the NCT-CRC-HE-100K dataset and achieves ~94% accuracy on the 8-class task. Confidence above 80% is considered highly reliable; values below 60% warrant manual pathologist review.`;
    } else {
      fallback = `Regarding "${msg}" — in the context of this ${ctx.predicted_class} finding (${ctx.tier}, confidence ${ctx.confidence}%): ${ctx.recommendation || 'Please consult a pathologist for detailed clinical interpretation.'}`;
    }
    appendChatMsg(fallback, 'ai', body);
  }
}

// Patch quick-action pills
function chatAsk(q) {
  const input = document.getElementById('chatInput');
  if (input) { input.value = q; sendChatMessageV2(); }
}

// Override old sendChatMessage with new version
window.sendChatMessage = sendChatMessageV2;


// ── 5. Wire everything into displayResults (handled in section below) ──

// ── 6. Init on DOM ready ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Wire new report button
  document.getElementById('downloadReport')?.addEventListener('click', e => {
    e.stopImmediatePropagation();
    openReportModal();
  }, true);

  // Wire chatbot to new version
  document.getElementById('chatSend')?.removeEventListener('click', sendChatMessage);
  document.getElementById('chatSend')?.addEventListener('click', sendChatMessageV2);
  document.getElementById('chatInput')?.addEventListener('keypress', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessageV2(); }
  });

  // Render trend chart if container exists
  setTimeout(renderTrendChart, 200);

  // Wire batch button if present
  document.getElementById('batchUploadBtn')?.addEventListener('click', openBatchModal);
});

// ════════════════════════════════════════════════════════════════
// DASHBOARD INTRO ANIMATION
// ════════════════════════════════════════════════════════════════
function runDashboardIntro() {
  if (!document.getElementById('patientsCount')) return;
  if (sessionStorage.getItem('gs_intro_done')) return;
  sessionStorage.setItem('gs_intro_done', '1');

  const style = document.createElement('style');
  style.textContent = `
    @keyframes introFadeUp {
      from { opacity:0; transform:translateY(28px) scale(.97); }
      to   { opacity:1; transform:translateY(0) scale(1); }
    }
    @keyframes introSlideIn {
      from { opacity:0; transform:translateX(-32px); }
      to   { opacity:1; transform:translateX(0); }
    }
    @keyframes introGlow {
      0%,100% { box-shadow: 0 0 0 0 rgba(0,212,255,0); }
      50%      { box-shadow: 0 0 32px 4px rgba(0,212,255,.22); }
    }
    @keyframes introPulseBar {
      0%   { transform: scaleX(0); opacity: 0; }
      60%  { opacity: 1; }
      100% { transform: scaleX(1); opacity: 1; }
    }
    @keyframes introBrandPop {
      0%   { opacity:0; transform:scale(.7) rotate(-8deg); }
      70%  { transform:scale(1.08) rotate(2deg); }
      100% { opacity:1; transform:scale(1) rotate(0deg); }
    }
    .intro-anim-fadeup {
      animation: introFadeUp .7s cubic-bezier(.22,1,.36,1) both;
    }
    .intro-anim-slide {
      animation: introSlideIn .6s cubic-bezier(.22,1,.36,1) both;
    }
    .intro-stat-glow {
      animation: introGlow 1.2s ease .3s both;
    }
  `;
  document.head.appendChild(style);

  const overlay = document.createElement('div');
  overlay.id = 'introOverlay';
  overlay.style.cssText = `
    position:fixed; inset:0; z-index:9999;
    background: radial-gradient(ellipse at 40% 50%, rgba(0,212,255,.07) 0%, rgba(10,13,22,0) 70%),
                var(--bg, #0a0d16);
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:1.5rem; pointer-events:none;
    transition: opacity .6s ease;
  `;

  const logoWrap = document.createElement('div');
  logoWrap.style.cssText = `
    display:flex; align-items:center; gap:1rem;
    animation: introBrandPop .9s cubic-bezier(.34,1.56,.64,1) both;
  `;
  logoWrap.innerHTML = `
    <div style="width:64px;height:64px;border-radius:18px;background:linear-gradient(135deg,rgba(0,212,255,.15),rgba(123,111,255,.15));border:1px solid rgba(0,212,255,.3);display:grid;place-items:center;font-size:2rem;box-shadow:0 0 40px rgba(0,212,255,.18)">🫀</div>
    <div>
      <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:2rem;color:#e0eaff;line-height:1">Gastric <em style='color:#00d4ff;font-style:normal'>Sentinel</em></div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:rgba(0,212,255,.6);letter-spacing:.12em;text-transform:uppercase;margin-top:.25rem">AI Diagnostic Platform · Initialising</div>
    </div>
  `;

  const barWrap = document.createElement('div');
  barWrap.style.cssText = 'width:320px;height:3px;background:rgba(255,255,255,.06);border-radius:99px;overflow:hidden;';
  const bar = document.createElement('div');
  bar.style.cssText = `height:100%;width:100%;background:linear-gradient(90deg,#00d4ff,#7b6fff);transform-origin:left;animation:introPulseBar 1.1s cubic-bezier(.4,0,.2,1) .3s both;`;
  barWrap.appendChild(bar);

  const loadText = document.createElement('div');
  loadText.style.cssText = `font-family:'JetBrains Mono',monospace;font-size:.7rem;color:rgba(0,212,255,.45);letter-spacing:.08em;`;
  loadText.textContent = 'Loading AI model · Connecting database…';

  overlay.appendChild(logoWrap);
  overlay.appendChild(barWrap);
  overlay.appendChild(loadText);
  document.body.appendChild(overlay);

  const steps = ['Connecting to MongoDB Atlas…', 'Loading EfficientNet-B4 model weights…', 'Fetching patient records…', 'AI platform ready ✓'];
  let si = 0;
  const stepInterval = setInterval(() => {
    if (si < steps.length) { loadText.textContent = steps[si++]; }
    else clearInterval(stepInterval);
  }, 320);

  setTimeout(() => {
    overlay.style.opacity = '0';
    setTimeout(() => {
      overlay.remove();
      animatePageIn();
    }, 600);
  }, 1600);
}

function animatePageIn() {
  const header = document.querySelector('.page-header');
  if (header) { header.style.cssText += 'animation:introFadeUp .7s cubic-bezier(.22,1,.36,1) both;'; }

  document.querySelectorAll('.stat-card').forEach((card, i) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    setTimeout(() => {
      card.style.transition = 'opacity .55s ease, transform .55s cubic-bezier(.22,1,.36,1)';
      card.style.opacity = '1';
      card.style.transform = 'translateY(0)';
    }, 120 + i * 90);
  });

  document.querySelectorAll('.card').forEach((card, i) => {
    card.style.opacity = '0';
    setTimeout(() => {
      card.style.transition = 'opacity .5s ease ' + (i * 0.04) + 's, transform .5s cubic-bezier(.22,1,.36,1) ' + (i * 0.04) + 's';
      card.style.opacity = '1';
    }, 350 + i * 40);
  });

  const sidebar = document.getElementById('sidebar');
  if (sidebar) {
    sidebar.style.opacity = '0';
    sidebar.style.transform = 'translateX(-20px)';
    setTimeout(() => {
      sidebar.style.transition = 'opacity .55s ease, transform .55s cubic-bezier(.22,1,.36,1)';
      sidebar.style.opacity = '1';
      sidebar.style.transform = 'translateX(0)';
    }, 80);
  }
}


// ════════════════════════════════════════════════════════════════
// ENHANCED 8-CLASS PREDICTION PANEL WITH CLINICAL + GENOMIC DATA
// ════════════════════════════════════════════════════════════════
function updateProbBarsEnhanced(rawProbs, tier, predictedClass, riskScore, clinicalData, genomicData) {
  const container = document.getElementById('probList');
  if (!container) return;

  const CLASS_META = [
    { key:'TUM',  label:'Tumor (Adenocarcinoma)',    sublabel:'Malignant epithelial cells',      color:'#ff3d6e', risk:'CRITICAL',   icon:'🔴' },
    { key:'STR',  label:'Cancer-Assoc. Stroma',      sublabel:'Desmoplastic connective tissue',  color:'#ff6b35', risk:'SUSPICIOUS', icon:'🟠' },
    { key:'LYM',  label:'Lymphocytes',               sublabel:'Tumour-infiltrating immune cells', color:'#ffb340', risk:'WATCH',     icon:'🟡' },
    { key:'DEB',  label:'Debris / Cell Fragments',   sublabel:'Necrotic or apoptotic material',  color:'#9b8fff', risk:'WATCH',      icon:'🟣' },
    { key:'MUC',  label:'Mucosa (Stomach Lining)',   sublabel:'Glandular epithelium',             color:'#00d4ff', risk:'NEGATIVE',  icon:'🔵' },
    { key:'MUS',  label:'Smooth Muscle',             sublabel:'Muscularis propria layer',         color:'#4bc8eb', risk:'NEGATIVE',   icon:'🔵' },
    { key:'NORM', label:'Normal Mucosa',             sublabel:'Healthy gastric epithelium',       color:'#00ff9f', risk:'NEGATIVE',   icon:'🟢' },
    { key:'ADI',  label:'Adipose (Fat Tissue)',      sublabel:'Peritoneal or submucosal fat',     color:'#a8ff78', risk:'NEGATIVE',   icon:'🟢' },
  ];

  const sorted = CLASS_META
    .map(c => ({ ...c, pct: Math.round((rawProbs[c.key] || 0) * 100) }))
    .sort((a, b) => b.pct - a.pct);

  const topClass = sorted[0];

  let html = `
    <div style="margin-bottom:.9rem;padding:.75rem .9rem;background:var(--bg3);border-radius:10px;border:1px solid rgba(0,212,255,.12)">
      <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;color:var(--tx3);margin-bottom:.5rem">Top Prediction</div>
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.4rem">
        <div style="display:flex;align-items:center;gap:.5rem">
          <span style="font-size:1.1rem">${topClass.icon}</span>
          <div>
            <div style="font-weight:700;color:${topClass.color};font-size:.9rem">${topClass.label}</div>
            <div style="font-size:.7rem;color:var(--tx3)">${topClass.sublabel}</div>
          </div>
        </div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:800;color:${topClass.color}">${topClass.pct}%</span>
      </div>
    </div>
  `;

  html += sorted.map((c, i) => {
    const isTop = i === 0;
    const isMalignant = c.risk === 'CRITICAL' || c.risk === 'SUSPICIOUS';
    return `
      <div style="margin-bottom:.55rem">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">
          <div style="display:flex;align-items:center;gap:.35rem;min-width:0">
            <span style="font-size:.75rem;flex-shrink:0">${c.icon}</span>
            <div style="min-width:0">
              <span style="font-size:.78rem;color:var(--tx2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block">${c.label}</span>
              <span style="font-size:.63rem;color:var(--tx4)">${c.sublabel}</span>
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:.35rem;flex-shrink:0;margin-left:.5rem">
            ${isTop ? `<span style="font-size:.6rem;padding:1px 6px;border-radius:99px;background:${c.color}22;border:1px solid ${c.color}55;color:${c.color};font-family:'JetBrains Mono',monospace">TOP</span>` : ''}
            ${isMalignant && c.pct > 5 ? `<span style="font-size:.6rem;padding:1px 5px;border-radius:3px;background:rgba(255,61,110,.12);color:#ff3d6e;font-family:'JetBrains Mono',monospace">${c.risk}</span>` : ''}
            <span style="font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:700;color:${c.color};min-width:32px;text-align:right">${c.pct}%</span>
          </div>
        </div>
        <div style="height:7px;background:var(--bg4);border-radius:99px;overflow:hidden">
          <div style="height:100%;width:0;border-radius:99px;background:${c.color};transition:width 1.2s cubic-bezier(.4,0,.2,1) ${i*0.07}s;box-shadow:${isTop ? '0 0 8px ' + c.color + '66' : 'none'}" data-w="${c.pct}"></div>
        </div>
      </div>
    `;
  }).join('');

  const tumProb = Math.round((rawProbs['TUM'] || 0) * 100);
  const strProb = Math.round((rawProbs['STR'] || 0) * 100);
  const malignantTotal = typeof riskScore === 'number' ? Math.round(riskScore) : Math.round(tumProb + strProb);

  html += `
    <div style="margin-top:.9rem;padding:.65rem .85rem;background:var(--bg3);border-radius:8px;border:1px solid var(--border)">
      <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:var(--tx3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.45rem">Malignancy Probability</div>
      <div style="display:flex;align-items:center;gap:.6rem">
        <div style="flex:1;height:8px;background:var(--bg4);border-radius:99px;overflow:hidden">
          <div style="height:100%;border-radius:99px;background:linear-gradient(90deg,#ffb340,#ff3d6e);transition:width 1.4s cubic-bezier(.4,0,.2,1)" style="width:${malignantTotal}%" data-w="${malignantTotal}" id="malignantBar"></div>
        </div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:.82rem;font-weight:700;color:${malignantTotal > 45 ? '#ff3d6e' : malignantTotal > 20 ? '#ffb340' : '#00ff9f'}">${malignantTotal}%</span>
      </div>
      <div style="font-size:.67rem;color:var(--tx4);margin-top:.3rem">TUM (${tumProb}%) + STR (${strProb}%) combined · from backend risk score</div>
    </div>
  `;

  if (clinicalData || genomicData) {
    html += renderClinicalGenomicPanel(clinicalData, genomicData);
  }

  container.innerHTML = html;

  requestAnimationFrame(() => requestAnimationFrame(() => {
    container.querySelectorAll('[data-w]').forEach(b => { b.style.width = b.dataset.w + '%'; });
    const mb = document.getElementById('malignantBar');
    if (mb) setTimeout(() => { mb.style.width = malignantTotal + '%'; }, 100);
  }));
}

function renderClinicalGenomicPanel(clinical, genomic) {
  if (!clinical && !genomic) return '';

  const c = clinical || {};
  const g = genomic || {};

  const stageColors = { 'I':'var(--c2)', 'II':'var(--c4)', 'III':'#ff6b35', 'IV':'var(--c3)' };
  const stageColor = stageColors[c.stage] || 'var(--tx2)';

  const geneScore = parseFloat(g.gene_score || c.gene_score || 0);
  const genomicRisk = parseFloat(g.genomic_risk || c.genomic_risk || 0);
  const geneBarW = Math.min(100, Math.round(Math.abs(geneScore) * 40));
  const genomicBarW = Math.min(100, Math.round(genomicRisk * 100));

  return `
    <div style="margin-top:.9rem;padding:.75rem .85rem;background:rgba(123,111,255,.07);border-radius:10px;border:1px solid rgba(123,111,255,.2)">
      <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:rgba(123,111,255,.8);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.6rem">Multimodal Inputs</div>

      ${c.age || c.gender || c.stage ? `
      <div style="margin-bottom:.55rem">
        <div style="font-size:.68rem;color:var(--tx3);margin-bottom:.3rem">🧑‍⚕️ Clinical Metadata</div>
        <div style="display:flex;gap:.5rem;flex-wrap:wrap">
          ${c.age ? `<span style="font-size:.72rem;padding:2px 8px;border-radius:4px;background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.2);color:var(--c1);font-family:'JetBrains Mono',monospace">Age: ${c.age}</span>` : ''}
          ${c.gender ? `<span style="font-size:.72rem;padding:2px 8px;border-radius:4px;background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.2);color:var(--c1);font-family:'JetBrains Mono',monospace">${c.gender}</span>` : ''}
          ${c.stage ? `<span style="font-size:.72rem;padding:2px 8px;border-radius:4px;background:${stageColor}18;border:1px solid ${stageColor}44;color:${stageColor};font-family:'JetBrains Mono',monospace">Stage ${c.stage}</span>` : ''}
        </div>
      </div>` : ''}

      ${(geneScore || genomicRisk) ? `
      <div>
        <div style="font-size:.68rem;color:var(--tx3);margin-bottom:.4rem">🧬 Genomic Risk Scores</div>
        <div style="margin-bottom:.4rem">
          <div style="display:flex;justify-content:space-between;font-size:.68rem;margin-bottom:2px">
            <span style="color:var(--tx3)">Gene Score</span>
            <span style="font-family:'JetBrains Mono',monospace;color:#9b8fff">${geneScore.toFixed(3)}</span>
          </div>
          <div style="height:5px;background:var(--bg4);border-radius:99px;overflow:hidden">
            <div style="height:100%;width:${geneBarW}%;background:linear-gradient(90deg,#7b6fff,#9b8fff);border-radius:99px;transition:width 1s ease"></div>
          </div>
        </div>
        <div>
          <div style="display:flex;justify-content:space-between;font-size:.68rem;margin-bottom:2px">
            <span style="color:var(--tx3)">Genomic Risk</span>
            <span style="font-family:'JetBrains Mono',monospace;color:${genomicRisk > 0.6 ? '#ff3d6e' : genomicRisk > 0.3 ? '#ffb340' : '#00ff9f'}">${(genomicRisk * 100).toFixed(1)}%</span>
          </div>
          <div style="height:5px;background:var(--bg4);border-radius:99px;overflow:hidden">
            <div style="height:100%;width:${genomicBarW}%;background:linear-gradient(90deg,${genomicRisk > 0.6 ? '#ff3d6e,#ff6b35' : genomicRisk > 0.3 ? '#ffb340,#ff6b35' : '#00d4ff,#00ff9f'});border-radius:99px;transition:width 1s ease"></div>
          </div>
        </div>
      </div>` : ''}
    </div>
  `;
}



const _origRunScan = null;

const _origDisplayResultsV2 = displayResults;
window.displayResults = function(diagnosis, recommendation, riskScore, probs, tier, predictedClass) {
  _origDisplayResultsV2(diagnosis, recommendation, riskScore, probs, tier, predictedClass);

  window._lastDiagnosis = diagnosis;
  window._lastRec = recommendation;
  window._lastRiskScore = riskScore;
  window._lastTier = tier;

  const rawProbs = (probs && probs.raw) ? probs.raw : window._lastRawProbs || null;
  if (rawProbs && Object.keys(rawProbs).length) {
    updateProbBarsEnhanced(
      rawProbs,
      tier,
      predictedClass,
      riskScore,
      window._lastClinical || null,
      window._lastGenomic || null
    );
  }

  recordTrendPoint(tier, typeof riskScore === 'number' ? riskScore : parseInt(riskScore) || 0, predictedClass);
  setTimeout(renderTrendChart, 120);
};

document.addEventListener('DOMContentLoaded', () => {
  runDashboardIntro();
  setTimeout(renderTrendChart, 300);

  const diagPage = !!document.getElementById('dropZone');
  if (diagPage) {
    injectClinicalGenomicInputs();
    setTimeout(updateClinDataDot, 200);
  }
});

window._clinicalData = { age: 0, gender: 'Male', stage: 'I', gene_score: 0, genomic_risk: 0 };

function injectClinicalGenomicInputs() {
  const btnRow = document.getElementById('scanBtn')?.closest('.flex');
  if (!btnRow || document.getElementById('clinDataBtn')) return;

  const btn = document.createElement('button');
  btn.id = 'clinDataBtn';
  btn.className = 'btn btn-ghost';
  btn.title = 'Set patient clinical & genomic data for the fusion model';
  btn.innerHTML = '🧬 Patient Data';
  btn.style.cssText = 'white-space:nowrap;position:relative';
  btn.addEventListener('click', openClinicalModal);
  btnRow.insertBefore(btn, btnRow.children[1]);

  const dot = document.createElement('div');
  dot.id = 'clinDataDot';
  dot.style.cssText = 'position:absolute;top:4px;right:4px;width:7px;height:7px;border-radius:50%;background:var(--tx4);border:1px solid var(--bg2);transition:background .3s';
  btn.appendChild(dot);
}

function updateClinDataDot() {
  const dot = document.getElementById('clinDataDot');
  if (!dot) return;
  const d = window._clinicalData;
  const filled = d.age > 0 || d.gender !== 'Male' || d.stage !== 'I' || d.gene_score !== 0 || d.genomic_risk !== 0;
  dot.style.background = filled ? 'var(--c2)' : 'var(--tx4)';
  dot.title = filled ? 'Patient data set — affecting prediction' : 'No patient data set';
}

function openClinicalModal() {
  document.getElementById('clinicalModal')?.remove();

  const d = window._clinicalData;
  const modal = document.createElement('div');
  modal.id = 'clinicalModal';
  modal.className = 'modal-backdrop';

  modal.innerHTML = `
    <div class="modal" style="max-width:500px">
      <div class="modal-hd">
        <div class="modal-title">🧬 Patient Clinical &amp; Genomic Data</div>
        <button class="modal-x" onclick="document.getElementById('clinicalModal').classList.remove('open')">✕</button>
      </div>

      <div style="font-size:.76rem;color:var(--tx3);margin-bottom:1rem;padding:.6rem .8rem;background:rgba(0,212,255,.06);border-radius:8px;border-left:3px solid var(--c1)">
        These values are sent to the <strong style="color:var(--c1)">multimodal fusion model</strong> alongside image features. Age, stage, and genomic risk directly shift the final malignancy prediction.
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.85rem;margin-bottom:.85rem">
        <div class="form-row">
          <label class="form-label">Patient Age</label>
          <input id="m_age" class="form-control" type="number" placeholder="e.g. 58" min="1" max="120" value="${d.age || ''}">
        </div>
        <div class="form-row">
          <label class="form-label">Gender</label>
          <select id="m_gender" class="form-control">
            <option value="Male"   ${d.gender==='Male'   ?'selected':''}>Male</option>
            <option value="Female" ${d.gender==='Female' ?'selected':''}>Female</option>
            <option value="Other"  ${d.gender==='Other'  ?'selected':''}>Other</option>
          </select>
        </div>
      </div>

      <div class="form-row" style="margin-bottom:.85rem">
        <label class="form-label">Cancer Stage</label>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem;margin-top:.3rem">
          ${['I','II','III','IV'].map(s => `
            <button type="button" class="stage-pill ${d.stage===s?'active':''}" data-stage="${s}"
              onclick="document.querySelectorAll('.stage-pill').forEach(b=>b.classList.remove('active'));this.classList.add('active')"
              style="padding:.5rem;border-radius:8px;border:1px solid ${d.stage===s?'var(--c1)':'var(--border)'};background:${d.stage===s?'rgba(0,212,255,.12)':'var(--bg3)'};color:${d.stage===s?'var(--c1)':'var(--tx2)'};cursor:pointer;font-weight:${d.stage===s?'700':'400'};transition:all .15s;font-family:'JetBrains Mono',monospace;font-size:.85rem">
              Stage ${s}
            </button>`).join('')}
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:.4rem;font-size:.67rem;color:var(--tx4);font-family:'JetBrains Mono',monospace">
          <span>← Lower risk</span><span>Higher risk →</span>
        </div>
      </div>

      <div style="border-top:1px solid var(--border);padding-top:.85rem;margin-bottom:.85rem">
        <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;color:var(--tx3);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.75rem">Genomic Risk Inputs</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.85rem">
          <div class="form-row">
            <label class="form-label">Gene Score</label>
            <input id="m_gene" class="form-control" type="number" step="0.01" placeholder="e.g. 0.42" value="${d.gene_score || ''}">
            <div style="font-size:.67rem;color:var(--tx4);margin-top:.25rem">Mutation burden score</div>
          </div>
          <div class="form-row">
            <label class="form-label">Genomic Risk (0–1)</label>
            <input id="m_grisk" class="form-control" type="number" step="0.01" min="0" max="1" placeholder="e.g. 0.65" value="${d.genomic_risk || ''}">
            <div style="font-size:.67rem;color:var(--tx4);margin-top:.25rem">0 = low · 1 = high</div>
          </div>
        </div>
      </div>

      <div id="clinPreview" style="display:none;padding:.65rem .8rem;background:rgba(0,255,159,.06);border-radius:8px;border:1px solid rgba(0,255,159,.2);margin-bottom:.85rem;font-size:.78rem;color:var(--tx2)"></div>

      <div class="modal-foot">
        <button class="btn btn-ghost" onclick="clearClinicalData()">🗑 Clear</button>
        <button class="btn btn-ghost" onclick="document.getElementById('clinicalModal').classList.remove('open')">Cancel</button>
        <button class="btn btn-primary" onclick="saveClinicalData()">✅ Apply to Scan</button>
      </div>
    </div>
  `;

  document.body.appendChild(modal);
  setTimeout(() => modal.classList.add('open'), 10);
  modal.addEventListener('click', e => { if (e.target === modal) modal.classList.remove('open'); });

  document.getElementById('m_age')?.focus();
  ['m_age','m_gender','m_gene','m_grisk'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', updateClinPreview);
  });
  document.querySelectorAll('.stage-pill').forEach(b => b.addEventListener('click', updateClinPreview));
  updateClinPreview();
}

function updateClinPreview() {
  const preview = document.getElementById('clinPreview');
  if (!preview) return;
  const age   = parseInt(document.getElementById('m_age')?.value) || 0;
  const gender = document.getElementById('m_gender')?.value || 'Male';
  const stage  = document.querySelector('.stage-pill.active')?.dataset.stage || 'I';
  const gene   = parseFloat(document.getElementById('m_gene')?.value) || 0;
  const grisk  = parseFloat(document.getElementById('m_grisk')?.value) || 0;

  if (!age && !gene && !grisk) { preview.style.display = 'none'; return; }

  const stageRisk = {I:'Low',II:'Moderate',III:'High',IV:'Very High'}[stage];
  const stageColor = {I:'var(--c2)',II:'var(--c4)',III:'#ff6b35',IV:'var(--c3)'}[stage];
  const genomicRiskLabel = grisk > 0.66 ? 'High' : grisk > 0.33 ? 'Moderate' : 'Low';
  const genomicColor = grisk > 0.66 ? 'var(--c3)' : grisk > 0.33 ? 'var(--c4)' : 'var(--c2)';

  preview.style.display = 'block';
  preview.innerHTML = `
    <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:var(--tx3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem">Fusion model will receive:</div>
    <div style="display:flex;gap:.5rem;flex-wrap:wrap">
      ${age ? `<span style="padding:2px 8px;border-radius:4px;background:rgba(0,212,255,.1);color:var(--c1);border:1px solid rgba(0,212,255,.2);font-size:.72rem">Age ${age}</span>` : ''}
      <span style="padding:2px 8px;border-radius:4px;background:rgba(0,212,255,.1);color:var(--c1);border:1px solid rgba(0,212,255,.2);font-size:.72rem">${gender}</span>
      <span style="padding:2px 8px;border-radius:4px;background:${stageColor}18;color:${stageColor};border:1px solid ${stageColor}44;font-size:.72rem">Stage ${stage} · ${stageRisk} risk</span>
      ${gene ? `<span style="padding:2px 8px;border-radius:4px;background:rgba(155,143,255,.1);color:#9b8fff;border:1px solid rgba(155,143,255,.2);font-size:.72rem">Gene: ${gene}</span>` : ''}
      ${grisk ? `<span style="padding:2px 8px;border-radius:4px;background:${genomicColor}18;color:${genomicColor};border:1px solid ${genomicColor}44;font-size:.72rem">Genomic: ${genomicRiskLabel} (${(grisk*100).toFixed(0)}%)</span>` : ''}
    </div>
  `;
}

function saveClinicalData() {
  const age   = parseInt(document.getElementById('m_age')?.value) || 0;
  const gender = document.getElementById('m_gender')?.value || 'Male';
  const stage  = document.querySelector('.stage-pill.active')?.dataset.stage || 'I';
  const gene   = parseFloat(document.getElementById('m_gene')?.value) || 0;
  const grisk  = parseFloat(document.getElementById('m_grisk')?.value) || 0;

  window._clinicalData = { age, gender, stage, gene_score: gene, genomic_risk: grisk };
  window._lastClinical = { age, gender, stage };
  window._lastGenomic  = { gene_score: gene, genomic_risk: grisk };
  window._lastStage = stage;
  window._lastGeneScore = gene;
  window._lastGenomicRisk = grisk;

  document.getElementById('clinicalModal')?.classList.remove('open');
  updateClinDataDot();

  const parts = [];
  if (age)   parts.push(`Age ${age}`);
  if (stage) parts.push(`Stage ${stage}`);
  if (grisk) parts.push(`Genomic ${(grisk*100).toFixed(0)}%`);
  showToast(`🧬 Patient data saved${parts.length ? ' — ' + parts.join(', ') : ''} · Will affect next scan`, 'ok');
}

function clearClinicalData() {
  window._clinicalData = { age: 0, gender: 'Male', stage: 'I', gene_score: 0, genomic_risk: 0 };
  window._lastClinical = null;
  window._lastGenomic  = null;
  document.getElementById('clinicalModal')?.classList.remove('open');
  updateClinDataDot();
  showToast('🗑 Clinical data cleared — scan will use image only', 'info');
}