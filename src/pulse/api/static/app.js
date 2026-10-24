// Pulse dashboard: polls the API and renders fleet health, alerts and rollout state.
// Plain JS, no build step. The API key input is kept in localStorage for convenience only.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const keyInput = $("api-key");
  try { keyInput.value = localStorage.getItem("pulse.apiKey") || ""; } catch (e) { /* storage unavailable */ }
  keyInput.addEventListener("change", () => {
    try { localStorage.setItem("pulse.apiKey", keyInput.value); } catch (e) { /* ignore */ }
    refresh();
  });

  function headers() {
    const h = { "Content-Type": "application/json" };
    if (keyInput.value) h["X-API-Key"] = keyInput.value;
    return h;
  }
  async function api(path, options) {
    const res = await fetch(path, Object.assign({ headers: headers() }, options || {}));
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (e) { /* not json */ }
      throw new Error(res.status + ": " + detail);
    }
    return res.json();
  }

  const fmt = (v, d) => (v === null || v === undefined) ? "-" : Number(v).toFixed(d === undefined ? 0 : d);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const BAND_GLYPH = { healthy: "●", watch: "▲", critical: "■" };

  function bandCell(health) {
    if (!health) return '<span class="empty">no score</span>';
    const b = health.band;
    return '<span class="band band-' + b + '"><span class="dot" aria-hidden="true"></span>' + BAND_GLYPH[b] + " " + b + "</span>";
  }

  const tooltip = $("tooltip");
  function sparkline(canvas, values) {
    const dpr = window.devicePixelRatio || 1;
    const w = 160, h = 32;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    const css = getComputedStyle(document.documentElement);
    ctx.strokeStyle = css.getPropertyValue("--line").trim();
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, h - 0.5); ctx.lineTo(w, h - 0.5); ctx.stroke();
    if (values.length < 2) return;
    const step = w / (values.length - 1);
    const y = (v) => h - 2 - (Math.max(0, Math.min(100, v)) / 100) * (h - 4);
    ctx.strokeStyle = css.getPropertyValue("--series-1").trim();
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    values.forEach((v, i) => { i === 0 ? ctx.moveTo(0, y(v)) : ctx.lineTo(i * step, y(v)); });
    ctx.stroke();
    canvas.onmousemove = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const i = Math.max(0, Math.min(values.length - 1, Math.round((ev.clientX - rect.left) / step)));
      tooltip.textContent = "score " + fmt(values[i], 1) + " (" + (i + 1 - values.length) + ")";
      tooltip.style.left = (ev.clientX + 10) + "px";
      tooltip.style.top = (ev.clientY - 28) + "px";
      tooltip.hidden = false;
    };
    canvas.onmouseleave = () => { tooltip.hidden = true; };
  }

  function renderDevices(devices) {
    const body = $("devices-body");
    if (!devices.length) return;
    body.innerHTML = devices.map((d) => {
      const score = d.health ? d.health.score : null;
      return "<tr>" +
        "<td><code>" + esc(d.device_id) + "</code></td>" +
        "<td>" + bandCell(d.health) + "</td>" +
        '<td class="num"><span class="score"><span class="bar"><span style="width:' + fmt(score || 0, 0) + '%"></span></span>' + fmt(score, 1) + "</span></td>" +
        '<td><canvas class="spark" data-id="' + esc(d.device_id) + '" aria-label="health trend for ' + esc(d.device_id) + '"></canvas></td>' +
        "<td>" + (d.health ? esc(d.health.dominant_class) : "-") + "</td>" +
        '<td class="num">' + fmt(d.last_rpm, 0) + "</td>" +
        '<td class="num">' + fmt(d.last_rms_g, 3) + "</td>" +
        "<td><code>" + esc(d.model_version) + "</code></td>" +
        '<td class="num">' + d.n_reports + "</td></tr>";
    }).join("");
    const byId = Object.fromEntries(devices.map((d) => [d.device_id, d]));
    body.querySelectorAll("canvas.spark").forEach((c) => sparkline(c, byId[c.dataset.id].recent_scores || []));
  }

  function renderAlerts(alerts) {
    const list = $("alerts-list");
    if (!alerts.length) { list.innerHTML = '<li class="empty">None open.</li>'; return; }
    list.innerHTML = alerts.map((a) =>
      "<li>" +
        '<span class="sev sev-' + esc(a.severity) + '">' + (a.severity === "critical" ? "■ " : "▲ ") + esc(a.severity) + "</span>" +
        "<span><code>" + esc(a.device_id) + "</code> " + esc(a.message) +
          '<div class="meta">' + esc(a.rule) + " · seen " + a.occurrences + "x · since " + new Date(a.created_at * 1000).toLocaleTimeString() +
          (a.acknowledged ? " · acknowledged" : "") + "</div></span>" +
        (a.acknowledged ? "<span></span>" : '<button class="quiet" data-ack="' + a.id + '">Acknowledge</button>') +
      "</li>").join("");
    list.querySelectorAll("button[data-ack]").forEach((b) => b.addEventListener("click", async () => {
      try { await api("/v1/alerts/" + b.dataset.ack + "/ack", { method: "POST" }); refresh(); } catch (e) { setMsg(e.message); }
    }));
  }

  function setMsg(text) { $("release-msg").textContent = text; }

  function renderReleases(releases) {
    const body = $("releases-body");
    if (!releases.length) return;
    body.innerHTML = releases.map((r) => {
      const m = r.manifest, ro = r.rollout, c = r.cohort || {};
      const acc = m.metrics && m.metrics.accuracy !== undefined ? fmt(m.metrics.accuracy, 4) : "-";
      const canPromote = ["draft", "canary", "pilot"].includes(ro.stage);
      const canRollback = ro.stage !== "rolled_back";
      return "<tr><td><code>" + esc(m.version) + "</code></td>" +
        '<td class="stage">' + esc(ro.stage) + "</td>" +
        '<td class="num">' + ro.percent + "</td>" +
        '<td class="num">' + (c.n_devices === undefined ? "-" : c.n_devices) + "</td>" +
        '<td class="num">' + (c.n_reports === undefined ? "-" : c.n_reports) + "</td>" +
        '<td class="num">' + (c.fault_rate === undefined ? "-" : fmt(c.fault_rate, 3)) + "</td>" +
        '<td class="num">' + acc + "</td>" +
        "<td>" + (canPromote ? '<button data-promote="' + esc(m.version) + '">Promote</button> ' : "") +
                 (canRollback ? '<button class="quiet" data-rollback="' + esc(m.version) + '">Roll back</button>' : "") + "</td></tr>";
    }).join("");
    body.querySelectorAll("button[data-promote]").forEach((b) => b.addEventListener("click", async () => {
      try { const r = await api("/v1/registry/releases/" + b.dataset.promote + "/promote", { method: "POST" }); setMsg(r.manifest.version + " is now at " + r.rollout.stage); refresh(); }
      catch (e) { setMsg(e.message); }
    }));
    body.querySelectorAll("button[data-rollback]").forEach((b) => b.addEventListener("click", async () => {
      try { await api("/v1/registry/releases/" + b.dataset.rollback + "/rollback", { method: "POST", body: JSON.stringify({ reason: "rolled back from dashboard" }) }); setMsg(b.dataset.rollback + " rolled back"); refresh(); }
      catch (e) { setMsg(e.message); }
    }));
  }

  function renderSummary(s) {
    $("kpi-devices").textContent = s.devices;
    $("kpi-health").textContent = fmt(s.mean_score, 1);
    $("kpi-bands").textContent = s.bands.healthy + " / " + s.bands.watch + " / " + s.bands.critical;
    $("kpi-alerts").textContent = s.open_alerts;
    $("kpi-reports").textContent = s.reports_total;
  }

  async function refresh() {
    try {
      const [summary, devices, alerts, releases] = await Promise.all([
        api("/v1/fleet/summary"), api("/v1/devices"), api("/v1/alerts?open=true"), api("/v1/registry/releases"),
      ]);
      renderSummary(summary); renderDevices(devices); renderAlerts(alerts); renderReleases(releases);
    } catch (e) {
      setMsg("API error: " + e.message);
    }
  }
  refresh();
  setInterval(refresh, 3000);
})();
