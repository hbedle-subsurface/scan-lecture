/* The SCAN029 case: seismic section, well control, attributes, SOM facies and SHAP. */
(() => {
  "use strict";

  const CLASS_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948", "#76b7b2", "#ff9da7"];
  const ZOOMS = { full: [15.0, 40.0], well: [30.0, 39.0] };
  const MARGIN = { l: 58, r: 66, t: 14, b: 42 };
  const SHAP_RANGE = 0.5;       // fixed color and bar range for SHAP values (membership units)
  const KEY_TOPS = new Set(["Rupel Clay Member", "Houthem Formation", "Zechstein Upper Claystone Formation",
    "Epen Formation", "Zeeland Formation", "Bosscheveld Formation"]);
  const PICK_MODE = new URLSearchParams(location.search).has("pick");

  const GLOSSARY = {
    impedance: ["Acoustic impedance", "Density multiplied by P-wave velocity. A reflection forms where impedance changes across a boundary; the size and sign of the change set the reflection amplitude and polarity."],
    som: ["Self-organizing map (SOM)", "An unsupervised neural network that arranges prototype vectors on a 2D grid so that similar attribute combinations sit near each other (Kohonen, 1982). Each sample is assigned to its closest prototype, and here the prototypes are grouped into 8 classes."],
    zscore: ["Standard deviations", "Each attribute is rescaled by subtracting its mean and dividing by its standard deviation over the whole window, so attributes with different units can be compared."],
    shap: ["SHAP values", "Shapley additive explanations (Lundberg and Lee, 2017). For one sample, each attribute receives the change it makes to the model output, averaged over every order in which attributes can be added. The base value plus all SHAP values equals the output. Here the output is the sample's soft membership in its SOM class, computed from distances to the SOM prototypes."],
  };

  const state = {
    stage: 1, zoom: "full", showWell: true, showHorizons: true, showUnits: true, showInterp: true, hideControl: false,
    attr: "coherence", attrOpacity: 0.75, preset: "combined", somOpacity: 0.7, verdictOpacity: 0.55,
    shapView: "classes", shapOpacity: 0.7, sample: null, traceKm: 34.19,
  };

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const cache = {};
  let meta, section, baseImg, overlay = null, overlayKey = "", autoHorizons, picks = {};

  const loadBin = async (name, Type) => {
    if (cache[name]) return cache[name];
    const r = await fetch(`data/${name}`);
    if (!r.ok) throw new Error(`Could not load data/${name} (${r.status}).`);
    return (cache[name] = new Type(await r.arrayBuffer()));
  };
  const makeCanvas = (w, h) => Object.assign(document.createElement("canvas"), { width: w, height: h });
  const hexToRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const fmt = (x) => (Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 1 ? x.toFixed(1) : x.toFixed(2));
  const shortName = (f) => ({ relative_acoustic_impedance: "Rel. AI", amplitude_volume_transform: "AVT", envelope: "Envelope", sweetness: "Sweetness", rms_amplitude: "RMS amp.", instantaneous_frequency: "Inst. freq.", spectral_ratio: "Spec. ratio",
    apparent_dip: "Dip", dip_variability: "Dip var.", coherence: "Coherence", far_minus_near: "Far − near" }[f] || f);

  /* ---------- rasters ---------- */
  function raster(nx, nt, colorAt) {
    const c = makeCanvas(nx, nt), ctx = c.getContext("2d"), img = ctx.createImageData(nx, nt);
    for (let i = 0; i < nx; i++) for (let j = 0; j < nt; j++) {
      const col = colorAt(i, j), p = (j * nx + i) * 4;
      img.data[p] = col[0]; img.data[p + 1] = col[1]; img.data[p + 2] = col[2]; img.data[p + 3] = 255;
    }
    ctx.putImageData(img, 0, 0); return c;
  }
  const grayAt = (i, j) => { const g = Math.max(0, Math.min(255, 128 - section[i * meta.nt + j] * 1.6)); return [g, g, g]; };
  const DIVERGING = (() => { // blue - off-white - red, fixed ±SHAP_RANGE
    const a = [49, 99, 173], b = [247, 244, 236], r = [190, 45, 40];
    return Array.from({ length: 256 }, (_, k) => { const u = k / 127.5 - 1, e = u < 0 ? a : r, t = Math.abs(u); return b.map((x, i) => Math.round(x + (e[i] - x) * t)); });
  })();

  async function buildOverlay() {
    const s = state.stage, g = meta.grid, nt = meta.nt;
    if (s === 2) {
      const d = await loadBin(`attr_${state.attr}.bin`, Uint8Array), lut = meta.attributes[state.attr].lut;
      return { img: raster(g.nx, nt, (i, j) => lut[d[i * nt + j]]), km: [g.km_min, g.km_max], t: [meta.t_min, meta.t_max] };
    }
    if (s === 3 || s === 5 || (s === 4 && state.shapView === "classes")) {
      const d = await loadBin(`som_${state.preset}.bin`, Uint8Array), rgb = CLASS_COLORS.map(hexToRgb);
      return { img: raster(g.nx, nt, (i, j) => rgb[d[i * nt + j]]), km: [g.km_min, g.km_max], t: [meta.t_min, meta.t_max] };
    }
    if (s === 4) {
      const sh = meta.shap[state.preset], feats = meta.presets[state.preset].features, fi = feats.indexOf(state.shapView);
      const d = await loadBin(`shap_${state.preset}.bin`, Int8Array), M = feats.length;
      const img = raster(sh.nx, sh.nt, (i, j) => {
        const v = d[(i * sh.nt + j) * M + fi] / 127 * sh.scale;
        return DIVERGING[Math.max(0, Math.min(255, Math.round((v / SHAP_RANGE + 1) * 127.5)))];
      });
      const dk = (g.km_max - g.km_min) / (g.nx - 1);
      return { img, km: [g.km_min - dk * sh.trace_step / 2, g.km_min + dk * (sh.nx * sh.trace_step - sh.trace_step / 2)],
        t: [meta.t_min - meta.dt * sh.sample_step / 2, meta.t_min + meta.dt * (sh.nt * sh.sample_step - sh.sample_step / 2)] };
    }
    return null;
  }

  /* ---------- geometry ---------- */
  const cv = $("#section"), cx = cv.getContext("2d");
  let W = 0, H = 0;
  const view = () => { const [a, b] = ZOOMS[state.zoom]; return { kmA: a, kmB: b, tA: meta.t_min, tB: meta.t_max }; };
  const X = (km) => { const v = view(); return MARGIN.l + (km - v.kmA) / (v.kmB - v.kmA) * (W - MARGIN.l - MARGIN.r); };
  const Y = (t) => { const v = view(); return MARGIN.t + (t - v.tA) / (v.tB - v.tA) * (H - MARGIN.t - MARGIN.b); };
  const invX = (px) => { const v = view(); return v.kmA + (px - MARGIN.l) / (W - MARGIN.l - MARGIN.r) * (v.kmB - v.kmA); };
  const invY = (py) => { const v = view(); return v.tA + (py - MARGIN.t) / (H - MARGIN.t - MARGIN.b) * (v.tB - v.tA); };

  function resize() {
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    W = r.width; H = r.height; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    cx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
  }
  function clipPlot() { cx.beginPath(); cx.rect(MARGIN.l, MARGIN.t, W - MARGIN.l - MARGIN.r, H - MARGIN.t - MARGIN.b); cx.clip(); }

  function drawRaster(o, alpha) {
    const nx = o.img.width, nt = o.img.height;
    // destination rectangle of the whole raster in canvas coordinates; the clip trims it to the view
    cx.save(); clipPlot(); cx.globalAlpha = alpha; cx.imageSmoothingEnabled = true;
    cx.drawImage(o.img, 0, 0, nx, nt, X(o.km[0]), Y(o.t[0]), X(o.km[1]) - X(o.km[0]), Y(o.t[1]) - Y(o.t[0]));
    cx.restore();
  }

  /* ---------- horizons (automatic near the well, replaced by picks where they exist) ---------- */
  function mergedHorizons() {
    const km = meta.horizons.km;
    return autoHorizons.map((h) => {
      const p = (picks[h.unit] || []).slice().sort((a, b) => a[0] - b[0]);
      if (p.length < 2) return h;
      const twt = h.twt.slice(), tracked = h.tracked.slice();
      km.forEach((k, i) => {
        if (k < p[0][0] || k > p[p.length - 1][0]) return;
        let j = 1; while (p[j][0] < k) j++;
        const [k0, t0] = p[j - 1], [k1, t1] = p[j];
        twt[i] = t0 + (t1 - t0) * (k - k0) / (k1 - k0 || 1); tracked[i] = 1;
      });
      return { ...h, twt, tracked, picked: true };
    });
  }
  const horizonsNow = () => (horizonsNow.v ??= mergedHorizons());
  const invalidateHorizons = () => { horizonsNow.v = null; };
  const hz = (unit) => horizonsNow().find((h) => h.unit === unit);

  /* ---------- drawing ---------- */
  function draw() {
    if (!meta || !W) return;
    cx.clearRect(0, 0, W, H); cx.fillStyle = "#111"; cx.fillRect(0, 0, W, H);
    drawRaster({ img: baseImg, km: [meta.km_min, meta.km_max], t: [meta.t_min, meta.t_max] }, 1);
    const s = state.stage, control = s === 5 || (!state.hideControl && s !== 5);
    if (s === 1 && state.showUnits && control) drawUnits(state.zoom === "well");
    if (overlay && s > 1) drawRaster(overlay, { 2: state.attrOpacity, 3: state.somOpacity, 4: state.shapOpacity, 5: state.verdictOpacity }[s]);
    if (control && state.showHorizons) drawHorizons(s !== 1);
    if (control && (s === 1 || s === 5) && state.showWell) drawWell();
    if (s === 1) drawTraceMarker();
    if (s === 1 && PICK_MODE) drawPicks();
    if (s === 4 && state.sample) drawSampleMarker();
    drawAxes(control);
  }

  function runs(n, ok) { const out = []; let a = -1; for (let i = 0; i <= n; i++) { if (i < n && ok(i)) { if (a < 0) a = i; } else if (a >= 0) { out.push([a, i - 1]); a = -1; } } return out; }

  function drawUnits(withNames) {
    const km = meta.horizons.km; cx.save(); clipPlot();
    for (const u of meta.units) {
      const base = hz(u.base), top = u.top ? hz(u.top) : null;
      for (const [a, b] of runs(km.length, (i) => base.twt[i] != null && (!top || top.twt[i] != null))) {
        cx.beginPath();
        for (let i = a; i <= b; i++) { const y = top ? Y(top.twt[i]) : Y(meta.t_min); i === a ? cx.moveTo(X(km[i]), y) : cx.lineTo(X(km[i]), y); }
        for (let i = b; i >= a; i--) cx.lineTo(X(km[i]), Y(base.twt[i]));
        cx.closePath(); cx.globalAlpha = u.reservoir ? 0.42 : 0.24; cx.fillStyle = u.color; cx.fill();
      }
    }
    cx.restore();
    if (!withNames) return;
    cx.save(); clipPlot(); cx.font = "600 13px Barlow, Arial, sans-serif"; cx.textAlign = "right"; cx.textBaseline = "middle";
    for (const u of meta.units) {
      const base = hz(u.base), top = u.top ? hz(u.top) : null;
      const ok = (k) => base.twt[k] != null && (!top || top.twt[k] != null);
      let i = -1; for (let k = km.length - 1; k >= 0; k--) if (km[k] < view().kmB - 0.3 && ok(k)) { i = k; break; }
      if (i < 0) continue;
      let a = i; while (a > 0 && ok(a - 1)) a--;
      if (km[i] - Math.max(km[a], view().kmA) < 3) continue;   // name only units shaded over at least 3 km
      const y = ((top ? Y(top.twt[i]) : Y(meta.t_min + 0.08)) + Y(base.twt[i])) / 2, xk = X(km[i]), w = cx.measureText(u.name).width + 12;
      cx.fillStyle = "rgba(239,229,200,.92)"; cx.fillRect(xk - w, y - 10, w, 20);
      cx.fillStyle = u.reservoir ? "#9d0208" : "#1f1d18"; cx.fillText(u.name, xk - 6, y);
    }
    cx.restore();
  }

  function drawHorizons(thin) {
    const km = meta.horizons.km; cx.save(); clipPlot();
    for (const h of horizonsNow()) {
      const valid = (i) => h.twt[i] != null;
      cx.strokeStyle = h.color;
      for (const [solid, test] of [[true, (i) => valid(i) && h.tracked[i]], [false, (i) => valid(i) && !h.tracked[i]]]) {
        if (!solid && !state.showInterp) continue;
        cx.setLineDash(solid ? [] : [6, 5]); cx.lineWidth = solid ? (thin ? 1.6 : 2.4) : (thin ? 1.2 : 1.6);
        for (const [a, b] of runs(km.length, test)) {
          cx.beginPath(); const a0 = Math.max(0, a - 1), b0 = Math.min(km.length - 1, b + 1);
          for (let i = a0; i <= b0; i++) if (valid(i)) (i === a0 ? cx.moveTo : cx.lineTo).call(cx, X(km[i]), Y(h.twt[i]));
          cx.stroke();
        }
      }
    }
    cx.setLineDash([]); cx.font = "600 13px Barlow, Arial, sans-serif"; cx.textBaseline = "bottom"; cx.textAlign = "left";
    for (const h of horizonsNow()) {
      let i = km.findIndex((k, j) => k >= view().kmA + 0.1 && h.twt[j] != null);
      if (i < 0 || km[i] > view().kmB - 1) continue;
      const below = h.unit === "Epen Formation", y = below ? Y(h.twt[i]) + 20 : Y(h.twt[i]) - 3, xk = Math.max(X(km[i]), MARGIN.l + 4), w = cx.measureText(h.label).width + 10;
      cx.fillStyle = "rgba(20,24,26,.82)"; cx.fillRect(xk, y - 17, w, 17); cx.fillStyle = h.color; cx.fillText(h.label, xk + 5, y - 1);
    }
    cx.restore();
  }

  function drawWell() {
    const p = meta.well.path; cx.save(); clipPlot(); cx.lineCap = "round";
    for (const [col, w] of [["#000", 5], ["#ffd166", 2.4]]) { cx.strokeStyle = col; cx.lineWidth = w; cx.beginPath(); p.forEach((q, i) => (i ? cx.lineTo : cx.moveTo).call(cx, X(q.km), Y(q.twt))); cx.stroke(); }
    for (const t of meta.well.tops) {
      if (t.twt < meta.t_min) continue;
      const key = KEY_TOPS.has(t.unit); cx.fillStyle = key ? "#ffd166" : "rgba(255,209,102,.6)";
      cx.beginPath(); cx.arc(X(t.km), Y(t.twt), key ? 4 : 2.5, 0, Math.PI * 2); cx.fill();
    }
    const top = p.find((q) => q.twt >= meta.t_min + 0.02) || p[0], label = `${meta.well.name}, projected 0.5–1.4 km from the line`;
    cx.font = "600 13px Barlow, Arial, sans-serif"; cx.textBaseline = "middle"; const w = cx.measureText(label).width + 12;
    let lx = X(top.km) + 8; if (lx + w > W - MARGIN.r - 4) lx = X(top.km) - 8 - w;
    cx.fillStyle = "#ffd166"; cx.fillRect(lx, Y(top.twt) - 2, w, 20); cx.fillStyle = "#1f1d18"; cx.textAlign = "left"; cx.fillText(label, lx + 6, Y(top.twt) + 8);
    cx.restore();
  }

  function drawTraceMarker() {
    const x = X(state.traceKm); if (x < MARGIN.l || x > W - MARGIN.r) return;
    cx.save(); cx.strokeStyle = "rgba(200,54,45,.9)"; cx.lineWidth = 1.5; cx.setLineDash([3, 3]);
    cx.beginPath(); cx.moveTo(x, MARGIN.t); cx.lineTo(x, H - MARGIN.b); cx.stroke(); cx.restore();
  }

  function drawSampleMarker() {
    const x = X(state.sample.km), y = Y(state.sample.t); cx.save(); clipPlot();
    cx.lineWidth = 3; cx.strokeStyle = "#000"; cx.beginPath(); cx.arc(x, y, 9, 0, Math.PI * 2); cx.stroke();
    cx.lineWidth = 1.8; cx.strokeStyle = "#ffd166"; cx.beginPath(); cx.arc(x, y, 9, 0, Math.PI * 2); cx.stroke(); cx.restore();
  }

  function drawPicks() {
    const unit = $("#pickHorizon").value, h = meta.horizons.items.find((x) => x.unit === unit), p = picks[unit] || [];
    cx.save(); clipPlot();
    for (const [k, t] of p) { cx.fillStyle = "#000"; cx.beginPath(); cx.arc(X(k), Y(t), 5, 0, Math.PI * 2); cx.fill(); cx.fillStyle = h.color; cx.beginPath(); cx.arc(X(k), Y(t), 3.5, 0, Math.PI * 2); cx.fill(); }
    cx.restore();
  }

  function niceStep(range, target) { const raw = range / target, mag = 10 ** Math.floor(Math.log10(raw)); return [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw); }

  function drawAxes(depthAxis) {
    const v = view(); cx.save();
    cx.strokeStyle = "#dfe6e9"; cx.lineWidth = 1; cx.strokeRect(MARGIN.l, MARGIN.t, W - MARGIN.l - MARGIN.r, H - MARGIN.t - MARGIN.b);
    cx.fillStyle = "#dfe6e9"; cx.font = "13px Barlow, Arial, sans-serif";
    const ks = niceStep(v.kmB - v.kmA, 10); cx.textAlign = "center"; cx.textBaseline = "top";
    for (let k = Math.ceil(v.kmA / ks) * ks; k <= v.kmB + 1e-9; k += ks) { const x = X(k); cx.fillRect(x, H - MARGIN.b, 1, 5); cx.fillText(k.toFixed(ks < 1 ? 1 : 0), x, H - MARGIN.b + 7); }
    cx.fillText("Distance along SCAN029 (km)", (MARGIN.l + W - MARGIN.r) / 2, H - 18);
    cx.textAlign = "right"; cx.textBaseline = "middle";
    for (let t = 0.2; t <= v.tB + 1e-9; t += 0.2) { const y = Y(t); cx.fillRect(MARGIN.l - 5, y, 5, 1); cx.fillText(t.toFixed(1), MARGIN.l - 8, y); }
    cx.save(); cx.translate(16, (MARGIN.t + H - MARGIN.b) / 2); cx.rotate(-Math.PI / 2); cx.textAlign = "center"; cx.fillText("Two-way time (s)", 0, 0); cx.restore();
    if (depthAxis) {
      cx.textAlign = "left";
      for (const d of meta.well.depth_axis) { if (d.depth % 500 || d.twt < v.tA || d.twt > v.tB) continue; const y = Y(d.twt); cx.fillRect(W - MARGIN.r, y, 5, 1); cx.fillText(String(d.depth), W - MARGIN.r + 8, y); }
      cx.save(); cx.translate(W - 12, (MARGIN.t + H - MARGIN.b) / 2); cx.rotate(Math.PI / 2); cx.textAlign = "center"; cx.fillText(`Depth below NAP at ${meta.well.name} (m)`, 0, 0); cx.restore();
    }
    cx.restore();
  }

  /* ---------- side panels ---------- */
  function drawWiggle() {
    const c = $("#wiggle"), g = c.getContext("2d"), w = c.width, h = c.height, pad = { l: 34, r: 8, t: 8, b: 8 }, nx = meta.section.nx, nt = meta.nt;
    g.fillStyle = "#fffaf0"; g.fillRect(0, 0, w, h);
    const i = Math.round((state.traceKm - meta.km_min) / (meta.km_max - meta.km_min) * (nx - 1));
    const x0 = pad.l + (w - pad.l - pad.r) / 2, scale = (w - pad.l - pad.r) / 2 / 127 * 0.55, y = (j) => pad.t + j / (nt - 1) * (h - pad.t - pad.b);
    g.strokeStyle = "#9a8f73"; g.beginPath(); g.moveTo(x0, pad.t); g.lineTo(x0, h - pad.b); g.stroke();
    g.beginPath(); g.moveTo(x0, y(0));
    for (let j = 0; j < nt; j++) g.lineTo(x0 + section[i * nt + j] * scale, y(j));
    for (let j = nt - 1; j >= 0; j--) g.lineTo(x0, y(j));
    g.save(); g.clip(); g.fillStyle = "#1f1d18"; g.fillRect(pad.l, 0, x0 - pad.l, h); g.restore();
    g.strokeStyle = "#1f1d18"; g.lineWidth = 1; g.beginPath();
    for (let j = 0; j < nt; j++) (j ? g.lineTo : g.moveTo).call(g, x0 + section[i * nt + j] * scale, y(j));
    g.stroke();
    g.fillStyle = "#5a5446"; g.font = "11px Barlow, Arial, sans-serif"; g.textAlign = "right"; g.textBaseline = "middle";
    for (let t = 0.2; t <= meta.t_max + 1e-9; t += 0.2) g.fillText(t.toFixed(1), pad.l - 6, y((t - meta.t_min) / meta.dt));
    $("#traceKm").textContent = state.traceKm.toFixed(2);
  }

  function drawColorbar() {
    const a = meta.attributes[state.attr], c = $("#colorbar"), g = c.getContext("2d"); g.clearRect(0, 0, c.width, c.height);
    a.lut.forEach((col, i) => { g.fillStyle = `rgb(${col})`; g.fillRect(i / 256 * c.width, 0, c.width / 256 + 1, 20); });
    g.fillStyle = "#1f1d18"; g.font = "12px Barlow, Arial, sans-serif"; g.textBaseline = "top";
    g.textAlign = "left"; g.fillText(fmt(a.min), 0, 24); g.textAlign = "right"; g.fillText(fmt(a.max), c.width, 24); g.textAlign = "center"; g.fillText(a.unit, c.width / 2, 24);
    $("#attrMeasures").textContent = a.measures; $("#attrGeology").textContent = a.geology; $("#attrSource").textContent = a.source;
  }

  function drawLegend(el) {
    const p = meta.presets[state.preset], feats = p.features, K = p.class_means_z.length; el.innerHTML = "";
    const cellW = 30, cellH = 22, left = 64, top = 64, w = left + feats.length * cellW + 34, h = top + K * cellH + 2;
    const c = makeCanvas(w * 2, h * 2), g = c.getContext("2d"); g.scale(2, 2); c.style.width = "100%";
    const div = (v) => `rgb(${DIVERGING[Math.round((Math.max(-2, Math.min(2, v)) / 2 + 1) * 127.5)]})`;
    g.font = "11px Barlow, Arial, sans-serif"; g.fillStyle = "#1f1d18";
    feats.forEach((f, i) => { g.save(); g.translate(left + i * cellW + cellW / 2 + 4, top - 6); g.rotate(-Math.PI / 3); g.fillText(shortName(f), 0, 0); g.restore(); });
    p.class_means_z.forEach((z, k) => {
      const y = top + k * cellH; g.fillStyle = CLASS_COLORS[k]; g.fillRect(4, y + 3, 16, cellH - 6);
      g.fillStyle = "#1f1d18"; g.textAlign = "left"; g.textBaseline = "middle"; g.fillText(`${k + 1}  ${(p.class_fraction[k] * 100).toFixed(0)}%`, 24, y + cellH / 2);
      z.forEach((v, i) => { g.fillStyle = div(v); g.fillRect(left + i * cellW, y + 1, cellW - 2, cellH - 2); g.fillStyle = Math.abs(v) > 1.2 ? "#fff" : "#1f1d18"; g.textAlign = "center"; g.fillText(v.toFixed(1), left + i * cellW + cellW / 2 - 1, y + cellH / 2); });
    });
    el.append(c);
  }

  function hbars(canvas, labels, values, { min, max, colors, title, zeroLine = true, valueFmt = (v) => v.toFixed(2) }) {
    const g = canvas.getContext("2d"), w = canvas.width, h = canvas.height, left = 74, right = 30, top = title ? 20 : 6, bottom = 18;
    g.fillStyle = "#fffaf0"; g.fillRect(0, 0, w, h);
    const X0 = (v) => left + (v - min) / (max - min) * (w - left - right), rowH = (h - top - bottom) / labels.length;
    g.font = "12px Barlow, Arial, sans-serif"; g.textBaseline = "middle";
    if (title) { g.fillStyle = "#1f1d18"; g.textAlign = "left"; g.fillText(title, 6, 10); }
    g.strokeStyle = "#cfc4a6"; g.lineWidth = 1;
    const ticks = niceStep(max - min, 4);
    g.fillStyle = "#5a5446"; g.textAlign = "center"; g.font = "11px Barlow, Arial, sans-serif";
    for (let t = Math.ceil(min / ticks) * ticks; t <= max + 1e-9; t += ticks) { const x = X0(t); g.beginPath(); g.moveTo(x, top); g.lineTo(x, h - bottom); g.stroke(); g.fillText(fmt(t), x, h - bottom + 10); }
    if (zeroLine) { g.strokeStyle = "#1f1d18"; g.beginPath(); g.moveTo(X0(0), top); g.lineTo(X0(0), h - bottom); g.stroke(); }
    labels.forEach((lab, i) => {
      const y = top + i * rowH + rowH / 2, v = values[i], vc = Math.max(min, Math.min(max, v));
      g.fillStyle = "#1f1d18"; g.textAlign = "right"; g.font = "12px Barlow, Arial, sans-serif"; g.fillText(lab, left - 6, y);
      g.fillStyle = colors[i]; g.fillRect(Math.min(X0(0), X0(vc)), y - rowH * 0.32, Math.abs(X0(vc) - X0(0)), rowH * 0.64);
      g.fillStyle = "#1f1d18"; g.font = "11px Barlow, Arial, sans-serif"; g.textAlign = v >= 0 ? "left" : "right";
      g.fillText(valueFmt(v), X0(vc) + (v >= 0 ? 4 : -4), y);
    });
  }

  async function drawShapPanels() {
    const p = meta.presets[state.preset], sh = meta.shap[state.preset], feats = p.features, names = feats.map(shortName);
    const local = $("#shapLocal"), global = $("#shapGlobal");
    // local explanation
    if (!state.sample) {
      const g = local.getContext("2d"); g.fillStyle = "#fffaf0"; g.fillRect(0, 0, local.width, local.height);
      $("#sampleTitle").textContent = "Click the section to explain a sample";
    } else {
      const d = await loadBin(`shap_${state.preset}.bin`, Int8Array), mem = await loadBin(`membership_${state.preset}.bin`, Uint8Array);
      const som = await loadBin(`som_${state.preset}.bin`, Uint8Array), g = meta.grid, M = feats.length;
      const gi = Math.round((state.sample.km - g.km_min) / (g.km_max - g.km_min) * (g.nx - 1));
      const si = Math.max(0, Math.min(sh.nx - 1, Math.round(gi / sh.trace_step))), sj = Math.max(0, Math.min(sh.nt - 1, Math.round((state.sample.t - meta.t_min) / meta.dt / sh.sample_step)));
      const k = som[(si * sh.trace_step) * meta.nt + sj * sh.sample_step];
      const vals = feats.map((_, f) => d[(si * sh.nt + sj) * M + f] / 127 * sh.scale), base = sh.base[k], out = mem[si * sh.nt + sj] / 255;
      $("#sampleTitle").textContent = `${state.sample.km.toFixed(2)} km, ${state.sample.t.toFixed(2)} s: class ${k + 1}`;
      hbars(local, ["Base value", ...names, "Membership"], [base, ...vals, out], {
        min: -SHAP_RANGE, max: 1, title: `Class ${k + 1}: base value + SHAP = membership`,
        colors: ["#9a8f73", ...vals.map((v) => (v >= 0 ? "#be2d28" : "#3163ad")), CLASS_COLORS[k]] });
      state.sample.cls = k;
    }
    // global importance: the selected sample's class, or all samples
    const k = state.sample?.cls;
    const vals = k == null ? sh.importance_overall : sh.importance_by_class[k];
    hbars(global, names, vals, { min: 0, max: 0.4, zeroLine: false, colors: feats.map(() => (k == null ? "#5a5446" : CLASS_COLORS[k])),
      title: k == null ? "All samples" : `Samples in class ${k + 1}`, valueFmt: (v) => v.toFixed(3) });
  }

  function drawShapBar() {
    const c = $("#shapBar"), g = c.getContext("2d"); c.hidden = state.shapView === "classes"; if (c.hidden) return;
    g.fillStyle = "#fffaf0"; g.fillRect(0, 0, c.width, c.height);
    DIVERGING.forEach((col, i) => { g.fillStyle = `rgb(${col})`; g.fillRect(i / 256 * c.width, 0, c.width / 256 + 1, 18); });
    g.fillStyle = "#1f1d18"; g.font = "12px Barlow, Arial, sans-serif"; g.textBaseline = "top";
    g.textAlign = "left"; g.fillText(`−${SHAP_RANGE}`, 0, 22); g.textAlign = "right"; g.fillText(`+${SHAP_RANGE}`, c.width, 22);
    g.textAlign = "center"; g.fillText("SHAP value (membership)", c.width / 2, 22);
  }

  /* ---------- state changes ---------- */
  async function refresh() {
    const key = [state.stage, state.stage === 2 ? state.attr : state.preset, state.stage === 4 ? state.shapView : ""].join(":");
    if (state.stage === 1) { overlay = null; overlayKey = ""; }
    else if (key !== overlayKey) { overlay = await buildOverlay(); overlayKey = key; }
    if (state.stage === 4) { drawShapBar(); await drawShapPanels(); }
    draw();
  }

  function setStage(n) {
    state.stage = n;
    $$(".tag").forEach((b) => b.setAttribute("aria-current", String(+b.dataset.stage === n)));
    $$(".card").forEach((c) => (c.hidden = +c.dataset.for !== n));
    refresh();
  }

  function fillShapViews() {
    const sel = $("#shapView"); sel.innerHTML = "";
    const add = (v, t) => { const o = document.createElement("option"); o.value = v; o.textContent = t; sel.append(o); };
    add("classes", "SOM classes");
    for (const f of meta.presets[state.preset].features) add(f, `SHAP value of ${meta.attributes[f].label}`);
    if (![...sel.options].some((o) => o.value === state.shapView)) state.shapView = "classes";
    sel.value = state.shapView;
  }

  function syncPresets() {
    $$(".presets button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.preset === state.preset)));
    drawLegend($("#classLegend")); drawLegend($("#classLegendVerdict")); fillShapViews();
  }

  function wire() {
    for (const sel of ["#presetList", "#presetListShap", "#presetListVerdict"]) {
      for (const [k, p] of Object.entries(meta.presets)) {
        const b = document.createElement("button"); b.setAttribute("role", "radio"); b.dataset.preset = k;
        b.innerHTML = `${p.label}<small>${p.features.map((f) => meta.attributes[f].label).join(", ")}</small>`;
        b.addEventListener("click", () => { state.preset = k; syncPresets(); refresh(); });
        $(sel).append(b);
      }
    }
    const aSel = $("#attrSelect"), groups = {};
    for (const [k, a] of Object.entries(meta.attributes)) (groups[a.family] ??= []).push([k, a]);
    for (const [fam, list] of Object.entries(groups)) {
      const og = document.createElement("optgroup"); og.label = fam;
      for (const [k, a] of list) og.append(Object.assign(document.createElement("option"), { value: k, textContent: a.label }));
      aSel.append(og);
    }
    aSel.value = state.attr; aSel.addEventListener("change", () => { state.attr = aSel.value; drawColorbar(); refresh(); });
    $("#shapView").addEventListener("change", (e) => { state.shapView = e.target.value; refresh(); });

    $$(".tag").forEach((b) => b.addEventListener("click", () => setStage(+b.dataset.stage)));
    $$("[data-zoom]").forEach((b) => b.addEventListener("click", () => { state.zoom = b.dataset.zoom; $$("[data-zoom]").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); draw(); }));
    for (const [id, key] of [["#showWell", "showWell"], ["#showUnits", "showUnits"], ["#showInterp", "showInterp"]]) $(id).addEventListener("change", (e) => { state[key] = e.target.checked; draw(); });
    const hBoxes = [$("#showHorizons"), ...$$(".syncHorizons")];
    hBoxes.forEach((box) => box.addEventListener("change", (e) => { state.showHorizons = e.target.checked; hBoxes.forEach((b) => (b.checked = state.showHorizons)); draw(); }));
    $("#hideWellControl").addEventListener("click", (e) => {
      state.hideControl = !state.hideControl; e.target.setAttribute("aria-pressed", String(state.hideControl));
      e.target.textContent = state.hideControl ? "Show well control" : "Hide all well control"; draw();
    });
    for (const [id, key] of [["#attrOpacity", "attrOpacity"], ["#somOpacity", "somOpacity"], ["#shapOpacity", "shapOpacity"], ["#verdictOpacity", "verdictOpacity"]])
      $(id).addEventListener("input", (e) => { state[key] = +e.target.value; draw(); });

    const idle = "Move over the section to read position and values. Click to show a trace.";
    cv.addEventListener("mousemove", (e) => {
      const r = cv.getBoundingClientRect(), km = invX(e.clientX - r.left), t = invY(e.clientY - r.top), v = view();
      if (km < v.kmA || km > v.kmB || t < meta.t_min || t > meta.t_max) { $("#readout").textContent = idle; return; }
      const g = meta.grid, gi = Math.round((km - g.km_min) / (g.km_max - g.km_min) * (g.nx - 1)), j = Math.round((t - meta.t_min) / meta.dt);
      let txt = `${km.toFixed(2)} km, ${t.toFixed(3)} s`;
      const attr = cache[`attr_${state.attr}.bin`], som = cache[`som_${state.preset}.bin`];
      if (state.stage === 2 && attr && gi >= 0 && gi < g.nx) { const a = meta.attributes[state.attr]; txt += `, ${a.label} ${fmt(a.min + attr[gi * meta.nt + j] / 255 * (a.max - a.min))} ${a.unit}`; }
      if (state.stage >= 3 && som && gi >= 0 && gi < g.nx) txt += `, class ${som[gi * meta.nt + j] + 1}`;
      $("#readout").textContent = txt;
    });
    cv.addEventListener("click", (e) => {
      const r = cv.getBoundingClientRect(), km = invX(e.clientX - r.left), t = invY(e.clientY - r.top), v = view();
      if (km < v.kmA || km > v.kmB || t < meta.t_min || t > meta.t_max) return;
      if (state.stage === 1 && PICK_MODE) return pickAt(km, t, e.shiftKey);
      if (state.stage === 1) { state.traceKm = km; drawWiggle(); draw(); }
      if (state.stage === 4) { state.sample = { km, t }; drawShapPanels().then(draw); }
    });

    const gl = $("#glossary");
    document.addEventListener("click", (e) => {
      const t = e.target.closest(".term");
      if (t) {
        const [title, text] = GLOSSARY[t.dataset.term]; $("#glossTitle").textContent = title; $("#glossText").textContent = text; gl.hidden = false;
        const r = t.getBoundingClientRect(); gl.style.left = Math.max(8, Math.min(window.innerWidth - 340, r.left)) + "px"; gl.style.top = Math.max(8, Math.min(window.innerHeight - gl.offsetHeight - 10, r.bottom + 6)) + "px";
      } else if (e.target.closest(".close") || !e.target.closest(".glossary")) gl.hidden = true;
    });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") gl.hidden = true; });
    new ResizeObserver(resize).observe($(".canvaswrap"));
  }

  /* ---------- pick mode (open the page with ?pick) ---------- */
  function pickAt(km, t, remove) {
    const unit = $("#pickHorizon").value, list = (picks[unit] ??= []);
    if (remove) {
      if (!list.length) return;
      let best = 0, bd = Infinity; list.forEach(([k, tt], i) => { const d = Math.hypot(X(k) - X(km), Y(tt) - Y(t)); if (d < bd) { bd = d; best = i; } });
      list.splice(best, 1);
    } else {
      if ($("#pickSnap").checked) {
        const h = meta.horizons.items.find((x) => x.unit === unit), nx = meta.section.nx, nt = meta.nt;
        const i = Math.round((km - meta.km_min) / (meta.km_max - meta.km_min) * (nx - 1)), j = Math.round((t - meta.t_min) / meta.dt);
        let bj = j, bv = -Infinity;
        for (let q = Math.max(0, j - 4); q <= Math.min(nt - 1, j + 4); q++) { const v = h.polarity * section[i * nt + q]; if (v > bv) { bv = v; bj = q; } }
        t = meta.t_min + bj * meta.dt;
      }
      list.push([+km.toFixed(4), +t.toFixed(4)]); list.sort((a, b) => a[0] - b[0]);
    }
    invalidateHorizons(); draw();
  }

  function setupPickMode() {
    if (!PICK_MODE) return;
    $("#pickPanel").hidden = false;
    const sel = $("#pickHorizon");
    for (const h of meta.horizons.items) sel.append(Object.assign(document.createElement("option"), { value: h.unit, textContent: h.label }));
    sel.addEventListener("change", draw);
    $("#pickClear").addEventListener("click", () => { delete picks[sel.value]; invalidateHorizons(); draw(); });
    $("#pickSave").addEventListener("click", () => {
      const blob = new Blob([JSON.stringify(picks, null, 1)], { type: "application/json" });
      Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "horizon_picks.json" }).click();
    });
  }

  async function init() {
    try {
      meta = await (await fetch("data/meta.json")).json();
      section = await loadBin("section.bin", Int8Array);
    } catch (err) {
      $("#readout").textContent = `${err.message} The page needs to be served over http (for example GitHub Pages or "python -m http.server"), not opened as a file.`;
      return;
    }
    try { const r = await fetch("data/horizon_picks.json"); if (r.ok) picks = await r.json(); } catch (_) { /* no picks yet */ }
    autoHorizons = meta.horizons.items;
    baseImg = raster(meta.section.nx, meta.nt, grayAt);
    wire(); setupPickMode(); drawColorbar(); syncPresets(); drawWiggle(); resize();
  }
  init();
})();
