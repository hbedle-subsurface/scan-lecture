/* The SCAN029 case: seismic section, well control, attributes and SOM facies. */
(() => {
  "use strict";

  const CLASS_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948", "#76b7b2", "#ff9da7"];
  const ZOOMS = { full: [15.0, 40.0], well: [30.0, 39.0] };
  const MARGIN = { l: 58, r: 66, t: 14, b: 42 };
  const KEY_TOPS = new Set(["Rupel Clay Member", "Houthem Formation", "Zechstein Upper Claystone Formation",
    "Epen Formation", "Zeeland Formation", "Bosscheveld Formation"]);

  const GLOSSARY = {
    impedance: ["Acoustic impedance", "Density multiplied by P-wave velocity. A reflection forms where impedance changes across a boundary; the size and sign of the change set the reflection amplitude and polarity."],
    reflection: ["Reflection", "Energy returned toward the surface where a seismic wave meets a change in acoustic impedance."],
    twt: ["Two-way time", "Time for a seismic wave to travel from the surface down to a reflector and back. Depth and two-way time are related through velocity, which increases with depth, so the two scales are not proportional."],
    som: ["Self-organizing map (SOM)", "An unsupervised neural network that arranges prototype vectors on a 2D grid so that similar attribute combinations sit near each other (Kohonen, 1982). Each sample is assigned to its closest prototype."],
    zscore: ["Standard deviations", "Each attribute is rescaled by subtracting its mean and dividing by its standard deviation over the whole window, so attributes with different units can be compared."],
  };

  const state = {
    stage: 1, zoom: "full", showWell: true, showHorizons: true, showUnits: true, showInterp: true,
    hideControl: false, attr: "coherence", attrOpacity: 0.75, preset: "combined", somOpacity: 0.7,
    verdictOpacity: 0.55, traceKm: 34.19, hover: null,
  };

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  let meta, section, attrCache = {}, somCache = {}, baseImg, overlayImg = null, overlayKey = "";

  async function loadBin(name, Type) {
    const r = await fetch(`data/${name}`);
    if (!r.ok) throw new Error(`Could not load data/${name} (${r.status}).`);
    return new Type(await r.arrayBuffer());
  }

  /* ---------- rasters ---------- */
  function makeCanvas(w, h) { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; }

  function buildBase() {
    const { nx } = meta.section, nt = meta.nt, c = makeCanvas(nx, nt), ctx = c.getContext("2d");
    const img = ctx.createImageData(nx, nt);
    for (let i = 0; i < nx; i++) for (let j = 0; j < nt; j++) {
      const v = section[i * nt + j];                     // -127..127, gained amplitude
      const g = Math.max(0, Math.min(255, 128 - v * 1.6)); // negative (impedance increase) dark
      const p = (j * nx + i) * 4; img.data[p] = img.data[p + 1] = img.data[p + 2] = g; img.data[p + 3] = 255;
    }
    ctx.putImageData(img, 0, 0); return c;
  }

  function buildAttrOverlay(key, data) {
    const { nx } = meta.grid, nt = meta.nt, lut = meta.attributes[key].lut, c = makeCanvas(nx, nt), ctx = c.getContext("2d");
    const img = ctx.createImageData(nx, nt);
    for (let i = 0; i < nx; i++) for (let j = 0; j < nt; j++) {
      const col = lut[data[i * nt + j]], p = (j * nx + i) * 4;
      img.data[p] = col[0]; img.data[p + 1] = col[1]; img.data[p + 2] = col[2]; img.data[p + 3] = 255;
    }
    ctx.putImageData(img, 0, 0); return c;
  }

  function buildClassOverlay(data) {
    const { nx } = meta.grid, nt = meta.nt, c = makeCanvas(nx, nt), ctx = c.getContext("2d");
    const img = ctx.createImageData(nx, nt), rgb = CLASS_COLORS.map(hexToRgb);
    for (let i = 0; i < nx; i++) for (let j = 0; j < nt; j++) {
      const col = rgb[data[i * nt + j]], p = (j * nx + i) * 4;
      img.data[p] = col[0]; img.data[p + 1] = col[1]; img.data[p + 2] = col[2]; img.data[p + 3] = 255;
    }
    ctx.putImageData(img, 0, 0); return c;
  }

  function hexToRgb(h) { return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)); }

  async function getAttr(key) { return attrCache[key] ??= await loadBin(`attr_${key}.bin`, Uint8Array); }
  async function getSom(key) { return somCache[key] ??= await loadBin(`som_${key}.bin`, Uint8Array); }

  /* ---------- geometry ---------- */
  const cv = $("#section"), cx = cv.getContext("2d");
  let W = 0, H = 0, dpr = 1;
  function view() { const [a, b] = ZOOMS[state.zoom]; return { kmA: a, kmB: b, tA: meta.t_min, tB: meta.t_max }; }
  function X(km) { const v = view(); return MARGIN.l + (km - v.kmA) / (v.kmB - v.kmA) * (W - MARGIN.l - MARGIN.r); }
  function Y(t) { const v = view(); return MARGIN.t + (t - v.tA) / (v.tB - v.tA) * (H - MARGIN.t - MARGIN.b); }
  function invX(px) { const v = view(); return v.kmA + (px - MARGIN.l) / (W - MARGIN.l - MARGIN.r) * (v.kmB - v.kmA); }
  function invY(py) { const v = view(); return v.tA + (py - MARGIN.t) / (H - MARGIN.t - MARGIN.b) * (v.tB - v.tA); }

  function resize() {
    const r = cv.getBoundingClientRect(); dpr = window.devicePixelRatio || 1;
    W = r.width; H = r.height; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    cx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
  }

  function drawRaster(img, kmMin, kmMax, alpha) {
    const v = view(), nx = img.width, nt = img.height;
    const sx = (v.kmA - kmMin) / (kmMax - kmMin) * nx, sw = (v.kmB - v.kmA) / (kmMax - kmMin) * nx;
    cx.save(); cx.globalAlpha = alpha; cx.imageSmoothingEnabled = true;
    cx.beginPath(); cx.rect(MARGIN.l, MARGIN.t, W - MARGIN.l - MARGIN.r, H - MARGIN.t - MARGIN.b); cx.clip();
    cx.drawImage(img, sx, 0, sw, nt, MARGIN.l, MARGIN.t, W - MARGIN.l - MARGIN.r, H - MARGIN.t - MARGIN.b);
    cx.restore();
  }

  /* ---------- drawing ---------- */
  function draw() {
    if (!meta || !W) return;
    cx.clearRect(0, 0, W, H);
    cx.fillStyle = "#111"; cx.fillRect(0, 0, W, H);
    drawRaster(baseImg, meta.km_min, meta.km_max, 1);

    const s = state.stage;
    const wellControl = s === 5 || (!state.hideControl && (s === 1 || s === 2 || s === 3));
    if (s === 1 && state.showUnits && !state.hideControl) drawUnits(state.zoom === "well");
    if ((s === 2 || s === 3 || s === 5) && overlayImg) {
      const a = s === 2 ? state.attrOpacity : s === 3 ? state.somOpacity : state.verdictOpacity;
      drawRaster(overlayImg, meta.grid.km_min, meta.grid.km_max, a);
    }
    if (wellControl && state.showHorizons) drawHorizons(s !== 1);
    if (wellControl && (s === 1 || s === 5) && state.showWell) drawWell();
    drawTraceMarker();
    drawAxes(wellControl);
  }

  function clipPlot() { cx.beginPath(); cx.rect(MARGIN.l, MARGIN.t, W - MARGIN.l - MARGIN.r, H - MARGIN.t - MARGIN.b); cx.clip(); }

  function horizon(unit) { return meta.horizons.items.find((h) => h.unit === unit); }

  function drawUnits(withNames) {
    const km = meta.horizons.km; cx.save(); clipPlot();
    for (const u of meta.units) {
      const base = horizon(u.base), top = u.top ? horizon(u.top) : null;
      cx.beginPath();
      km.forEach((k, i) => { const y = top ? Y(top.twt[i]) : Y(meta.t_min); i ? cx.lineTo(X(k), y) : cx.moveTo(X(k), y); });
      for (let i = km.length - 1; i >= 0; i--) cx.lineTo(X(km[i]), Y(base.twt[i]));
      cx.closePath(); cx.globalAlpha = u.reservoir ? 0.42 : 0.24; cx.fillStyle = u.color; cx.fill();
    }
    cx.restore();
    if (!withNames) return;
    // unit names at the east end of the tracked window
    cx.save(); clipPlot(); cx.font = "600 13px Barlow, Arial, sans-serif";
    const i = km.length - 40, xk = X(km[i]);
    for (const u of meta.units) {
      const base = horizon(u.base), top = u.top ? horizon(u.top) : null;
      const yTop = top ? Y(top.twt[i]) : Y(meta.t_min + 0.08), y = (yTop + Y(base.twt[i])) / 2;
      if (xk < MARGIN.l + 40 || xk > W - MARGIN.r) continue;
      const w = cx.measureText(u.name).width + 12;
      cx.fillStyle = "rgba(239,229,200,.92)"; cx.fillRect(xk - w, y - 10, w, 20);
      cx.fillStyle = u.reservoir ? "#9d0208" : "#1f1d18"; cx.textAlign = "right"; cx.textBaseline = "middle"; cx.fillText(u.name, xk - 6, y);
    }
    cx.restore();
  }

  function drawHorizons(thin) {
    const km = meta.horizons.km; cx.save(); clipPlot();
    for (const h of meta.horizons.items) {
      cx.strokeStyle = h.color; cx.lineWidth = thin ? 1.6 : 2.4;
      // tracked segments solid
      cx.setLineDash([]); cx.beginPath(); let pen = false;
      km.forEach((k, i) => { if (h.tracked[i]) { pen ? cx.lineTo(X(k), Y(h.twt[i])) : cx.moveTo(X(k), Y(h.twt[i])); pen = true; } else pen = false; });
      cx.stroke();
      if (state.showInterp) {
        cx.setLineDash([6, 5]); cx.lineWidth = thin ? 1.2 : 1.6; cx.beginPath(); pen = false;
        km.forEach((k, i) => { const on = !h.tracked[i] || (i > 0 && !h.tracked[i - 1]); if (on) { pen ? cx.lineTo(X(k), Y(h.twt[i])) : cx.moveTo(X(k), Y(h.twt[i])); pen = true; } else pen = false; });
        cx.stroke();
      }
    }
    cx.setLineDash([]);
    // labels at west end
    cx.font = "600 13px Barlow, Arial, sans-serif"; cx.textBaseline = "bottom";
    const i = 12, xk = Math.max(X(km[i]), MARGIN.l + 4);
    for (const h of meta.horizons.items) {
      const below = h.unit === "Epen Formation";
      const y = below ? Y(h.twt[i]) + 20 : Y(h.twt[i]) - 3, w = cx.measureText(h.label).width + 10;
      cx.fillStyle = "rgba(20,24,26,.82)"; cx.fillRect(xk, y - 17, w, 17);
      cx.fillStyle = h.color; cx.textAlign = "left"; cx.fillText(h.label, xk + 5, y - 1);
    }
    cx.restore();
  }

  function drawWell() {
    const p = meta.well.path; cx.save(); clipPlot();
    cx.lineCap = "round";
    for (const [col, w] of [["#000", 5], ["#ffd166", 2.4]]) {
      cx.strokeStyle = col; cx.lineWidth = w; cx.beginPath();
      p.forEach((q, i) => (i ? cx.lineTo(X(q.km), Y(q.twt)) : cx.moveTo(X(q.km), Y(q.twt)))); cx.stroke();
    }
    cx.font = "500 12px Barlow, Arial, sans-serif"; cx.textBaseline = "middle";
    for (const t of meta.well.tops) {
      if (t.twt < meta.t_min) continue;
      const x = X(t.km), y = Y(t.twt), key = KEY_TOPS.has(t.unit);
      cx.fillStyle = key ? "#ffd166" : "rgba(255,209,102,.6)";
      cx.beginPath(); cx.arc(x, y, key ? 4 : 2.5, 0, Math.PI * 2); cx.fill();
    }
    const top = p.find((q) => q.twt >= meta.t_min + 0.02) || p[0];
    const label = `${meta.well.name}, projected 0.5–1.4 km from the line`;
    cx.font = "600 13px Barlow, Arial, sans-serif"; const w = cx.measureText(label).width + 12;
    let lx = X(top.km) + 8; if (lx + w > W - MARGIN.r - 4) lx = X(top.km) - 8 - w;
    cx.fillStyle = "#ffd166"; cx.fillRect(lx, Y(top.twt) - 2, w, 20);
    cx.fillStyle = "#1f1d18"; cx.textAlign = "left"; cx.fillText(label, lx + 6, Y(top.twt) + 8);
    cx.restore();
  }

  function drawTraceMarker() {
    if (state.stage !== 1) return;
    const x = X(state.traceKm); if (x < MARGIN.l || x > W - MARGIN.r) return;
    cx.save(); cx.strokeStyle = "rgba(200,54,45,.9)"; cx.lineWidth = 1.5; cx.setLineDash([3, 3]);
    cx.beginPath(); cx.moveTo(x, MARGIN.t); cx.lineTo(x, H - MARGIN.b); cx.stroke(); cx.restore();
  }

  function niceStep(range, target) {
    const raw = range / target, mag = 10 ** Math.floor(Math.log10(raw));
    return [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  }

  function drawAxes(depthAxis) {
    const v = view(); cx.save();
    cx.strokeStyle = "#dfe6e9"; cx.lineWidth = 1; cx.strokeRect(MARGIN.l, MARGIN.t, W - MARGIN.l - MARGIN.r, H - MARGIN.t - MARGIN.b);
    cx.fillStyle = "#dfe6e9"; cx.font = "13px Barlow, Arial, sans-serif";
    // km
    const ks = niceStep(v.kmB - v.kmA, 10); cx.textAlign = "center"; cx.textBaseline = "top";
    for (let k = Math.ceil(v.kmA / ks) * ks; k <= v.kmB + 1e-9; k += ks) {
      const x = X(k); cx.fillRect(x, H - MARGIN.b, 1, 5); cx.fillText(k.toFixed(ks < 1 ? 1 : 0), x, H - MARGIN.b + 7);
    }
    cx.fillText("Distance along SCAN029 (km)", (MARGIN.l + W - MARGIN.r) / 2, H - 18);
    // twt
    cx.textAlign = "right"; cx.textBaseline = "middle";
    for (let t = 0.2; t <= v.tB + 1e-9; t += 0.2) { const y = Y(t); cx.fillRect(MARGIN.l - 5, y, 5, 1); cx.fillText(t.toFixed(1), MARGIN.l - 8, y); }
    cx.save(); cx.translate(16, (MARGIN.t + H - MARGIN.b) / 2); cx.rotate(-Math.PI / 2); cx.textAlign = "center"; cx.fillText("Two-way time (s)", 0, 0); cx.restore();
    // depth at well
    if (depthAxis) {
      cx.textAlign = "left";
      for (const d of meta.well.depth_axis) {
        if (d.depth % 500 || d.twt < v.tA || d.twt > v.tB) continue;
        const y = Y(d.twt); cx.fillRect(W - MARGIN.r, y, 5, 1); cx.fillText(String(d.depth), W - MARGIN.r + 8, y);
      }
      cx.save(); cx.translate(W - 12, (MARGIN.t + H - MARGIN.b) / 2); cx.rotate(Math.PI / 2); cx.textAlign = "center";
      cx.fillText(`Depth below NAP at ${meta.well.name} (m)`, 0, 0); cx.restore();
    }
    cx.restore();
  }

  /* ---------- wiggle trace ---------- */
  function drawWiggle() {
    const c = $("#wiggle"), g = c.getContext("2d"), w = c.width, h = c.height, pad = { l: 34, r: 8, t: 8, b: 8 };
    g.clearRect(0, 0, w, h); g.fillStyle = "#fffaf0"; g.fillRect(0, 0, w, h);
    const { nx } = meta.section, nt = meta.nt;
    const i = Math.round((state.traceKm - meta.km_min) / (meta.km_max - meta.km_min) * (nx - 1));
    const x0 = pad.l + (w - pad.l - pad.r) / 2, scale = (w - pad.l - pad.r) / 2 / 127 * 0.55;   // fixed amplitude scale
    const y = (j) => pad.t + j / (nt - 1) * (h - pad.t - pad.b);
    g.strokeStyle = "#9a8f73"; g.beginPath(); g.moveTo(x0, pad.t); g.lineTo(x0, h - pad.b); g.stroke();
    g.beginPath(); g.moveTo(x0, y(0));
    for (let j = 0; j < nt; j++) g.lineTo(x0 + section[i * nt + j] * scale, y(j));
    for (let j = nt - 1; j >= 0; j--) g.lineTo(x0, y(j));
    g.save(); g.clip(); g.fillStyle = "#1f1d18"; g.fillRect(pad.l, 0, x0 - pad.l, h); g.restore();   // fill negative side
    g.strokeStyle = "#1f1d18"; g.lineWidth = 1; g.beginPath();
    for (let j = 0; j < nt; j++) (j ? g.lineTo : g.moveTo).call(g, x0 + section[i * nt + j] * scale, y(j));
    g.stroke();
    g.fillStyle = "#5a5446"; g.font = "11px Barlow, Arial, sans-serif"; g.textAlign = "right"; g.textBaseline = "middle";
    for (let t = 0.2; t <= meta.t_max + 1e-9; t += 0.2) g.fillText(t.toFixed(1), pad.l - 6, y((t - meta.t_min) / meta.dt));
    $("#traceKm").textContent = state.traceKm.toFixed(2);
  }

  /* ---------- controls ---------- */
  async function refreshOverlay() {
    let key = "";
    if (state.stage === 2) key = "a:" + state.attr;
    if (state.stage === 3 || state.stage === 5) key = "s:" + state.preset;
    if (!key) { overlayImg = null; overlayKey = ""; draw(); return; }
    if (key !== overlayKey) {
      overlayImg = key.startsWith("a:") ? buildAttrOverlay(state.attr, await getAttr(state.attr)) : buildClassOverlay(await getSom(state.preset));
      overlayKey = key;
    }
    draw();
  }

  function setStage(n) {
    state.stage = n;
    $$(".tag").forEach((b) => b.setAttribute("aria-current", String(+b.dataset.stage === n)));
    $$(".card").forEach((c) => (c.hidden = +c.dataset.for !== n));
    refreshOverlay();
  }

  function fillAttrUI() {
    const sel = $("#attrSelect"), groups = {};
    for (const [k, a] of Object.entries(meta.attributes)) (groups[a.family] ??= []).push([k, a]);
    for (const [fam, list] of Object.entries(groups)) {
      const og = document.createElement("optgroup"); og.label = fam;
      for (const [k, a] of list) { const o = document.createElement("option"); o.value = k; o.textContent = a.label; og.append(o); }
      sel.append(og);
    }
    sel.value = state.attr;
    const show = () => {
      const a = meta.attributes[state.attr];
      $("#attrMeasures").textContent = a.measures; $("#attrGeology").textContent = a.geology; $("#attrSource").textContent = a.source;
      const c = $("#colorbar"), g = c.getContext("2d"); g.clearRect(0, 0, c.width, c.height);
      a.lut.forEach((col, i) => { g.fillStyle = `rgb(${col})`; g.fillRect(i / 256 * c.width, 0, c.width / 256 + 1, 20); });
      g.fillStyle = "#1f1d18"; g.font = "12px Barlow, Arial, sans-serif"; g.textBaseline = "top";
      g.textAlign = "left"; g.fillText(fmt(a.min), 0, 24); g.textAlign = "right"; g.fillText(fmt(a.max), c.width, 24);
      g.textAlign = "center"; g.fillText(a.unit, c.width / 2, 24);
    };
    sel.addEventListener("change", () => { state.attr = sel.value; show(); refreshOverlay(); });
    show();
  }
  const fmt = (x) => (Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 1 ? x.toFixed(1) : x.toFixed(2));

  function fillPresetUI(listSel, legendSel) {
    const list = $(listSel);
    for (const [k, p] of Object.entries(meta.presets)) {
      const b = document.createElement("button"); b.setAttribute("role", "radio"); b.dataset.preset = k;
      b.innerHTML = `${p.label}<small>${p.features.map((f) => meta.attributes[f].label).join(", ")}</small>`;
      b.addEventListener("click", () => { state.preset = k; syncPresets(); refreshOverlay(); });
      list.append(b);
    }
  }

  function syncPresets() {
    $$(".presets button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.preset === state.preset)));
    for (const sel of ["#classLegend", "#classLegendVerdict"]) drawLegend($(sel));
  }

  function drawLegend(el) {
    const p = meta.presets[state.preset], feats = p.features, K = p.class_means_z.length;
    el.innerHTML = "";
    const cellW = 30, cellH = 22, left = 64, top = 64, w = left + feats.length * cellW + 34, h = top + K * cellH + 2;
    const c = makeCanvas(w * 2, h * 2), g = c.getContext("2d"); g.scale(2, 2);
    c.style.width = "100%"; c.setAttribute("aria-label", "Mean attribute value for each class");
    const div = (v) => { // fixed diverging scale, -2 blue to +2 red
      const u = Math.max(-2, Math.min(2, v)) / 2, a = [67, 110, 170], b = [247, 244, 236], r = [196, 64, 52];
      const e = u < 0 ? a : r, t = Math.abs(u); return `rgb(${b.map((x, i) => Math.round(x + (e[i] - x) * t))})`;
    };
    g.font = "11px Barlow, Arial, sans-serif"; g.fillStyle = "#1f1d18";
    feats.forEach((f, i) => { g.save(); g.translate(left + i * cellW + cellW / 2 + 4, top - 6); g.rotate(-Math.PI / 3); g.textAlign = "left"; g.fillText(shortName(f), 0, 0); g.restore(); });
    p.class_means_z.forEach((z, k) => {
      const y = top + k * cellH;
      g.fillStyle = CLASS_COLORS[k]; g.fillRect(4, y + 3, 16, cellH - 6);
      g.fillStyle = "#1f1d18"; g.textAlign = "left"; g.textBaseline = "middle";
      g.fillText(`${k + 1}  ${(p.class_fraction[k] * 100).toFixed(0)}%`, 24, y + cellH / 2);
      z.forEach((v, i) => {
        g.fillStyle = div(v); g.fillRect(left + i * cellW, y + 1, cellW - 2, cellH - 2);
        g.fillStyle = Math.abs(v) > 1.2 ? "#fff" : "#1f1d18"; g.textAlign = "center"; g.fillText(v.toFixed(1), left + i * cellW + cellW / 2 - 1, y + cellH / 2);
      });
    });
    el.append(c);
  }
  const shortName = (f) => ({ rms_amplitude: "RMS amp.", instantaneous_frequency: "Inst. freq.", spectral_ratio: "Spec. ratio",
    apparent_dip: "Dip", dip_variability: "Dip var.", coherence: "Coherence", far_minus_near: "Far − near" }[f] || f);

  function wireControls() {
    $$(".tag").forEach((b) => b.addEventListener("click", () => !b.disabled && setStage(+b.dataset.stage)));
    $$("[data-zoom]").forEach((b) => b.addEventListener("click", () => {
      state.zoom = b.dataset.zoom; $$("[data-zoom]").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); draw();
    }));
    const bindCheck = (id, key) => $(id).addEventListener("change", (e) => { state[key] = e.target.checked; draw(); });
    bindCheck("#showWell", "showWell"); bindCheck("#showUnits", "showUnits"); bindCheck("#showInterp", "showInterp");
    const horizonBoxes = ["#showHorizons", ...$$(".syncHorizons")].map((x) => (typeof x === "string" ? $(x) : x));
    horizonBoxes.forEach((box) => box.addEventListener("change", (e) => {
      state.showHorizons = e.target.checked; horizonBoxes.forEach((b) => (b.checked = state.showHorizons)); draw();
    }));
    $("#hideWellControl").addEventListener("click", (e) => {
      state.hideControl = !state.hideControl; e.target.setAttribute("aria-pressed", String(state.hideControl));
      e.target.textContent = state.hideControl ? "Show well control" : "Hide all well control"; draw();
    });
    $("#attrOpacity").addEventListener("input", (e) => { state.attrOpacity = +e.target.value; draw(); });
    $("#somOpacity").addEventListener("input", (e) => { state.somOpacity = +e.target.value; draw(); });
    $("#verdictOpacity").addEventListener("input", (e) => { state.verdictOpacity = +e.target.value; draw(); });

    cv.addEventListener("mousemove", async (e) => {
      const r = cv.getBoundingClientRect(), km = invX(e.clientX - r.left), t = invY(e.clientY - r.top);
      if (km < view().kmA || km > view().kmB || t < meta.t_min || t > meta.t_max) { $("#readout").textContent = "Move over the section to read position and values. Click to show a trace."; return; }
      let txt = `${km.toFixed(2)} km, ${t.toFixed(3)} s`;
      const gi = Math.round((km - meta.grid.km_min) / (meta.grid.km_max - meta.grid.km_min) * (meta.grid.nx - 1));
      const j = Math.round((t - meta.t_min) / meta.dt);
      if (state.stage === 2 && attrCache[state.attr]) {
        const a = meta.attributes[state.attr], q = attrCache[state.attr][gi * meta.nt + j];
        txt += `, ${a.label} ${fmt(a.min + q / 255 * (a.max - a.min))} ${a.unit}`;
      } else if ((state.stage === 3 || state.stage === 5) && somCache[state.preset]) {
        txt += `, class ${somCache[state.preset][gi * meta.nt + j] + 1}`;
      }
      $("#readout").textContent = txt;
    });
    cv.addEventListener("click", (e) => {
      const r = cv.getBoundingClientRect(), km = invX(e.clientX - r.left);
      if (km < view().kmA || km > view().kmB) return;
      state.traceKm = km; drawWiggle(); draw();
    });

    const gl = $("#glossary");
    document.addEventListener("click", (e) => {
      const t = e.target.closest(".term");
      if (t) {
        const [title, text] = GLOSSARY[t.dataset.term]; $("#glossTitle").textContent = title; $("#glossText").textContent = text;
        const r = t.getBoundingClientRect(); gl.hidden = false;
        gl.style.left = Math.min(window.innerWidth - 340, r.left) + "px"; gl.style.top = Math.min(window.innerHeight - gl.offsetHeight - 10, r.bottom + 6) + "px";
      } else if (e.target.closest(".close") || !e.target.closest(".glossary")) gl.hidden = true;
    });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") gl.hidden = true; });
    new ResizeObserver(resize).observe($(".canvaswrap"));
  }

  async function init() {
    try {
      meta = await (await fetch("data/meta.json")).json();
      section = await loadBin("section.bin", Int8Array);
    } catch (err) {
      $("#readout").textContent = `${err.message} The page needs to be served over http (for example GitHub Pages or "python -m http.server"), not opened as a file.`;
      return;
    }
    baseImg = buildBase();
    fillAttrUI(); fillPresetUI("#presetList"); fillPresetUI("#presetListVerdict"); syncPresets();
    wireControls(); drawWiggle(); resize();
  }
  init();
})();
