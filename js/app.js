/* The SCAN029 case: seismic section, well control, attributes, SOM facies and SHAP. */
(() => {
  "use strict";

  const ZOOMS = { full: [0.0, 48.5], someren: [0.0, 21.0], well: [29.0, 40.0], custom: [0.0, 48.5] };
  const MARGIN = { l: 58, r: 66, t: 70, b: 42 };   // top margin holds the labels, above the seismic
  const KEY_TOPS = new Set(["Rupel Clay Member", "Houthem Formation", "Zechstein Upper Claystone Formation",
    "Epen Formation", "Zeeland Formation", "Bosscheveld Formation"]);
  const PICK_MODE = new URLSearchParams(location.search).has("pick");

  const GLOSSARY = {
    impedance: ["Acoustic impedance", "Density multiplied by P-wave velocity. A reflection forms where impedance changes across a boundary; the size and sign of the change set the reflection amplitude and polarity."],
    som: ["Self-organizing map (SOM)", "An unsupervised neural network that arranges prototype vectors on a 2D grid so that similar attribute combinations sit near each other (Kohonen, 1982). Each sample is assigned to its closest prototype, and here the prototypes are grouped into 8 classes."],
    neuron: ["Neuron", "One prototype on the SOM grid: a vector with one value per attribute. Each sample is assigned to the neuron whose prototype is closest to its attribute values, after each attribute is converted to standard deviations from its mean."],
    polarity: ["Polarity", "The SCAN029 data are zero phase, and the processing header states that an increase in acoustic impedance is recorded as a negative number. A boundary where impedance increases downward, such as shale over limestone, is therefore a trough."],
    zscore: ["Standard deviations", "Each attribute is rescaled by subtracting its mean and dividing by its standard deviation over the whole window, so attributes with different units can be compared."],
    shap: ["SHAP values", "Shapley additive explanations (Lundberg and Lee, 2017). For one sample, each attribute receives the change it makes to the model output, averaged over the orders in which attributes can be added. Here the output is the sample's position on the SOM grid, which sets its color. The average position of all samples plus every attribute's SHAP value gives the sample's position. Values are estimated from random attribute orderings (Strumbelj and Kononenko, 2014)."],
  };

  const state = {
    stage: 1, zoom: "full", showWell: true, showHorizons: true, showUnits: false, seisMap: "gray_black", attrLut: "default", showInterp: true, hideControl: false,
    attr: "coherence", attrOpacity: 0.75, somOpacity: 0.75, verdictOpacity: 0.55, sample: null, explained: null, traceKm: 34.19,
    runs: [], current: -1, busy: false, showGeoColumns: true, geoFocus: null, showSomeren: true, showNames: true, showKarst: true, showFault: true, hiddenWells: new Set(),
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
  const shortName = (f) => ({ instantaneous_phase: "Inst. phase", cos_instantaneous_phase: "Cos phase", quadrature_trace: "Quadrature", full_stack_amplitude: "Full amp.", near_stack_amplitude: "Near amp.", mid_stack_amplitude: "Mid amp.", far_stack_amplitude: "Far amp.", relative_acoustic_impedance: "Rel. AI", amplitude_volume_transform: "AVT", envelope: "Envelope", sweetness: "Sweetness", rms_amplitude: "RMS amp.", instantaneous_frequency: "Inst. freq.", spectral_ratio: "Spec. ratio",
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
  // In this dataset an increase in acoustic impedance is a negative number (zero-phase data, processing header).
  const SEIS_MAPS = {
    gray_black: { label: "Grayscale, impedance increase black", f: (v) => { const g = Math.max(0, Math.min(255, 128 + v * 1.6)); return [g, g, g]; } },
    gray_white: { label: "Grayscale, impedance increase white", f: (v) => { const g = Math.max(0, Math.min(255, 128 - v * 1.6)); return [g, g, g]; } },
    red_blue: { label: "Red–white–blue, impedance increase red", f: (v) => { const u = Math.max(-1, Math.min(1, v / 80)), e = u < 0 ? [178, 24, 43] : [33, 102, 172], t = Math.abs(u); return [247, 247, 247].map((x, k) => Math.round(x + (e[k] - x) * t)); } },
  };
  const grayAt = (i, j) => SEIS_MAPS[state.seisMap].f(section[i * meta.nt + j]);
  const DIVERGING = (() => { // blue - off-white - red
    const a = [49, 99, 173], b = [247, 244, 236], r = [190, 45, 40];
    return Array.from({ length: 256 }, (_, k) => { const u = k / 127.5 - 1, e = u < 0 ? a : r, t = Math.abs(u); return b.map((x, i) => Math.round(x + (e[i] - x) * t)); });
  })();

  async function buildOverlay() {
    const s = state.stage, g = meta.grid, nt = g.nt, tg = [meta.t_min - g.dt / 2, meta.t_min + (nt - 0.5) * g.dt];
    if (s === 3) {
      const d = await loadBin(`attr_${state.attr}.bin`, Uint8Array), lut = state.attrLut === "default" ? meta.attributes[state.attr].lut : meta.luts[state.attrLut].lut;
      return { img: raster(g.nx, nt, (i, j) => lut[d[i * nt + j]]), km: [g.km_min, g.km_max], t: tg };
    }
    if (s >= 4 && run()) {
      const r = run(), cols = neuronColors(r.side);
      const img = raster(g.nx, nt, (i, j) => { const k = r.bmu[i * nt + j]; return k === 255 ? [0, 0, 0] : cols[k]; });
      const cctx = img.getContext("2d"), id = cctx.getImageData(0, 0, g.nx, nt);
      for (let i = 0; i < g.nx; i++) for (let j = 0; j < nt; j++) if (r.bmu[i * nt + j] === 255) id.data[(j * g.nx + i) * 4 + 3] = 0;
      cctx.putImageData(id, 0, 0);
      return { img, km: [g.km_min, g.km_max], t: tg };
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
  let topLabels = [];
  const topLabel = (x, text, bg, fg = "#1f1d18") => { if (x >= MARGIN.l - 2 && x <= W - MARGIN.r + 2) topLabels.push({ x, text, bg, fg }); };

  function renderTopLabels() {
    // labels above the plot, placed in up to three rows so they do not overlap, with a tick down to the plot edge
    cx.save(); cx.font = "600 12px Barlow, Arial, sans-serif"; cx.textBaseline = "middle"; cx.textAlign = "left";
    const rows = [[], [], []], rowH = 19;
    for (const l of topLabels.sort((a, b) => a.x - b.x)) {
      const w = cx.measureText(l.text).width + 12;
      let x0 = Math.min(Math.max(l.x - w / 2, MARGIN.l), W - MARGIN.r - w), r = 0;
      for (; r < rows.length; r++) if (!rows[r].some(([a, b]) => x0 < b + 6 && x0 + w > a - 6)) break;
      if (r === rows.length) r = rows.length - 1;
      rows[r].push([x0, x0 + w]);
      const y = 4 + r * rowH;
      cx.strokeStyle = l.bg; cx.lineWidth = 1; cx.beginPath(); cx.moveTo(l.x, y + 16); cx.lineTo(l.x, MARGIN.t); cx.stroke();
      cx.fillStyle = l.bg; cx.fillRect(x0, y, w, 16); cx.fillStyle = l.fg; cx.fillText(l.text, x0 + 6, y + 8);
    }
    cx.restore();
  }

  function draw() {
    if (!meta || !W) return;
    topLabels = [];
    cx.clearRect(0, 0, W, H); cx.fillStyle = "#111"; cx.fillRect(0, 0, W, H);
    drawRaster({ img: baseImg, km: [meta.km_min, meta.km_max], t: [meta.t_min, meta.t_max] }, 1);
    const s = state.stage, control = !(s === 1 && state.hideControl);
    if (((s === 1 && state.showUnits) || (s === 2 && state.showGeoColumns)) && control) drawWellColumns();
    if (overlay && s > 2) drawRaster(overlay, { 3: state.attrOpacity, 4: state.somOpacity, 5: state.somOpacity, 6: state.verdictOpacity }[s]);
    if (control && state.showHorizons) drawHorizons(s !== 1);
    if (control && state.showWell) drawWell();
    drawSomerenLimits();
    if (s === 1) drawTraceMarker();
    if (s === 1 && PICK_MODE) drawPicks();
    if (s === 5 && state.sample) drawSampleMarker();
    if (s === 2) drawGeologyOverlay();
    if (s >= 4 && run()) { const w = run().window; cx.save(); clipPlot(); cx.setLineDash([10, 5]); cx.strokeStyle = "#ffffff"; cx.lineWidth = 1.5;
      cx.strokeRect(X(w.km[0]), Y(w.t[0]), X(w.km[1]) - X(w.km[0]), Y(w.t[1]) - Y(w.t[0])); cx.restore(); }
    drawAxes(control);
    renderTopLabels();
  }

  const PICK_HALF_KM = 0.6;   // horizons are drawn as short picks this far either side of the well top
  const topKm = (unit) => { const t = meta.well.tops.find((x) => x.unit === unit); return t ? t.km : null; };
  const nearPick = (h, i) => h.picked || (topKm(h.unit) != null && Math.abs(meta.horizons.km[i] - topKm(h.unit)) <= PICK_HALF_KM);

  /* ---------- formation groups beside each well, after the scheme of Doornenbal et al. (2019, fig. 4) ---------- */
  const GROUPS = {
    N: { name: "North Sea Supergroup (Cenozoic)", color: "#f2d46b" },
    NU: { name: "Upper North Sea Group (Miocene–Quaternary)", color: "#f7e39a" },
    NM: { name: "Middle North Sea Group (Oligocene)", color: "#f2c94c" },
    NL: { name: "Lower North Sea Group (Paleocene–Eocene)", color: "#e0a93b" },
    CK: { name: "Chalk Group (Late Cretaceous–Danian)", color: "#9bd18b" },
    "ZE+RB": { name: "Zechstein and Lower Germanic Trias groups (Permian–Triassic)", color: "#a88bd1" },
    DC: { name: "Limburg Group (Namurian, Upper Carboniferous)", color: "#8ec9ea" },
    CL: { name: "Carboniferous Limestone Group (Dinantian)", color: "#7d8fa3" },
    OB: { name: "Banjaard Group (Devonian–?Dinantian)", color: "#b8906f" },
  };
  const WELL_GROUPS = {
    "CAL-GT-04": [["N", "Upper North Sea Group"], ["CK", "Houthem Formation"], ["ZE+RB", "Nederweert Sandstone Member"], ["DC", "Epen Formation"], ["CL", "Zeeland Formation"], ["OB", "Bosscheveld Formation"]],
    "CAL-GT-01": [["N", "Kieseloolite Formation"], ["CK", "Houthem Formation"], ["ZE+RB", "Nederweert Sandstone Member"], ["DC", "Epen Formation"], ["CL", "Zeeland Formation"]],
    "ASTEN-GT-02": [["NU", "Veghel Formation"], ["NM", "Veldhoven Clay Member"], ["NL", "Basal Dongen Sand Member"], ["CK", "Houthem Formation"]],
  };
  function drawWellColumns() {
    const half = 0.45; cx.save(); clipPlot();
    for (const wl of (meta.wells || []).filter((w) => !state.hiddenWells.has(w.name) && WELL_GROUPS[w.name])) {
      const gs = WELL_GROUPS[wl.name], last = wl.path[wl.path.length - 1];
      gs.forEach(([code, unit], i) => {
        const top = wl.tops.find((t) => t.unit === unit); if (!top) return;
        const nxt = i + 1 < gs.length ? wl.tops.find((t) => t.unit === gs[i + 1][1]) : last;
        const t0 = Math.max(top.twt, meta.t_min), k0 = top.km, t1 = nxt.twt, k1 = nxt.km;
        cx.beginPath(); cx.moveTo(X(k0 - half), Y(t0)); cx.lineTo(X(k0 + half), Y(t0)); cx.lineTo(X(k1 + half), Y(t1)); cx.lineTo(X(k1 - half), Y(t1)); cx.closePath();
        cx.globalAlpha = 0.45; cx.fillStyle = GROUPS[code].color; cx.fill(); cx.globalAlpha = 1;
        const xm = (X(k0) + X(k1)) / 2, ym = (Y(t0) + Y(t1)) / 2;
        if (Y(t1) - Y(t0) > 14 && X(k0 + half) - X(k0 - half) > 26) { cx.font = "700 11px Barlow, Arial, sans-serif"; cx.textAlign = "center"; cx.textBaseline = "middle"; cx.fillStyle = "#10151a"; cx.fillText(code, xm, ym); }
      });
    }
    cx.restore();
  }

  /* ---------- geologic background: structural domains and events along the line ---------- */
  const DOMAINS = [
    { km: [0.0, 16.5], name: "Roer Valley Graben", color: "#f2c94c" },
    { km: [16.5, 18.5], name: "Peel Boundary Fault zone (approx.)", color: "#e76f51" },
    { km: [18.5, 48.5], name: "Peel Block (horst) and Venlo Block", color: "#8ec9ea" },
  ];
  function drawGeologyOverlay() {
    const y = MARGIN.t - 7;
    for (const d of DOMAINS) {
      const a = Math.max(X(d.km[0]), MARGIN.l), b = Math.min(X(d.km[1]), W - MARGIN.r); if (b <= a) continue;
      cx.fillStyle = d.color; cx.fillRect(a, y, b - a, 6);
      topLabel((a + b) / 2, d.name, d.color, "#10151a");
    }
    const f = state.geoFocus; if (!f) return;
    cx.save(); clipPlot(); cx.strokeStyle = "#ffd166"; cx.lineWidth = 2.5; cx.setLineDash([8, 5]);
    cx.strokeRect(X(f.km[0]), Y(f.t[0]), X(f.km[1]) - X(f.km[0]), Y(f.t[1]) - Y(f.t[0])); cx.restore();
  }

  function drawSomerenLimits() {
    const [a, b] = meta.someren.km; cx.save(); clipPlot();
    cx.setLineDash([8, 6]); cx.lineWidth = 1.8; cx.strokeStyle = "rgba(118,183,178,.95)";
    for (const k of [a, b]) { cx.beginPath(); cx.moveTo(X(k), MARGIN.t); cx.lineTo(X(k), H - MARGIN.b); cx.stroke(); }
    cx.restore();
    const xm = (Math.max(X(a), MARGIN.l) + Math.min(X(b), W - MARGIN.r)) / 2;
    if (X(b) > MARGIN.l && X(a) < W - MARGIN.r) topLabel(xm, "Someren exploration license (approx.)", "#76b7b2", "#10201f");
  }

  function runs(n, ok) { const out = []; let a = -1; for (let i = 0; i <= n; i++) { if (i < n && ok(i)) { if (a < 0) a = i; } else if (a >= 0) { out.push([a, i - 1]); a = -1; } } return out; }

  function drawUnits(withNames) {
    const km = meta.horizons.km; cx.save(); clipPlot();
    for (const u of meta.units) {
      const base = hz(u.base), top = u.top ? hz(u.top) : null;
      const ref = u.top || u.base, near = (i) => topKm(ref) != null && Math.abs(km[i] - topKm(ref)) <= PICK_HALF_KM;
      for (const [a, b] of runs(km.length, (i) => near(i) && base.twt[i] != null && (!top || top.twt[i] != null))) {
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
      const ok = (k) => Math.abs(km[k] - topKm(u.top || u.base)) <= PICK_HALF_KM && base.twt[k] != null && (!top || top.twt[k] != null);
      let i = -1; for (let k = km.length - 1; k >= 0; k--) if (km[k] < view().kmB - 0.3 && ok(k)) { i = k; break; }
      if (i < 0) continue;
      let a = i; while (a > 0 && ok(a - 1)) a--;
      if (X(km[i]) - X(Math.max(km[a], view().kmA)) < 60) continue;   // name only where the shaded pick is wide enough on screen
      const y = ((top ? Y(top.twt[i]) : Y(meta.t_min + 0.08)) + Y(base.twt[i])) / 2, xk = X(km[i]), w = cx.measureText(u.name).width + 12;
      cx.fillStyle = "rgba(239,229,200,.92)"; cx.fillRect(xk - w, y - 10, w, 20);
      cx.fillStyle = u.reservoir ? "#9d0208" : "#1f1d18"; cx.fillText(u.name, xk - 6, y);
    }
    cx.restore();
  }

  function drawHorizons(thin) {
    const km = meta.horizons.km; cx.save(); clipPlot();
    for (const h of horizonsNow()) {
      const valid = (i) => h.twt[i] != null && nearPick(h, i);
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
    cx.setLineDash([]); cx.font = "600 12px Barlow, Arial, sans-serif"; cx.textBaseline = "middle"; cx.textAlign = "left";
    cx.restore();
  }

  const WELL_STYLE = { "CAL-GT-04": { color: "#ffd166", labelDy: 0, zoom: "well", side: "right" }, "CAL-GT-01": { color: "#f4f1de", labelDy: 24, zoom: "well" },
    "ASTEN-GT-02": { color: "#cdb4db", labelDy: 26, zoom: "someren" } };
  const LABELED_TOPS = { "CAL-GT-04": new Set(["Rupel Clay Member", "Houthem Formation", "Zechstein Upper Claystone Formation", "Epen Formation", "Zeeland Formation", "Bosscheveld Formation"]),
    "CAL-GT-01": new Set(["Veldhoven Formation", "Rupel Clay Member", "Houthem Formation", "Zeeland Formation"]),
    "ASTEN-GT-02": new Set(["Kieseloolite Formation", "Breda Formation (Vrijherenberg Member)", "Heksenberg Formation", "Breda Formation (Kakert Member)",
      "Veldhoven Clay Member", "Voort Sand Member", "Boom Clay Member", "Basal Dongen Sand Member", "Houthem Formation"]) };

  function drawWell() {
    const wells = meta.wells || [{ name: meta.well.name, path: meta.well.path, tops: meta.well.tops }];
    cx.save(); clipPlot(); cx.lineCap = "round";
    for (const wl of wells.filter((w) => !state.hiddenWells.has(w.name))) {
      const st = WELL_STYLE[wl.name] || { color: "#ffffff", labelDy: 48 }, p = wl.path;
      for (const [col, w] of [["#000", 5], [st.color, 2.4]]) {
        cx.strokeStyle = col; cx.lineWidth = w; cx.setLineDash(wl.estimated_path && col !== "#000" ? [6, 4] : []);
        cx.beginPath(); p.forEach((q, i) => (i ? cx.lineTo : cx.moveTo).call(cx, X(q.km), Y(q.twt))); cx.stroke();
      }
      cx.setLineDash([]);
      for (const t of wl.tops) {
        if (t.twt < meta.t_min) continue;
        const key = KEY_TOPS.has(t.unit) || LABELED_TOPS[wl.name]?.has(t.unit);
        cx.fillStyle = st.color; cx.globalAlpha = key ? 1 : 0.6;
        cx.beginPath(); cx.arc(X(t.km), Y(t.twt), key ? 4 : 2.5, 0, Math.PI * 2); cx.fill(); cx.globalAlpha = 1;
        // tops of this well that are not tracked horizons: a short tick and, in the well zoom, the name on the left
        if (LABELED_TOPS[wl.name]?.has(t.unit)) {
          cx.strokeStyle = st.color; cx.lineWidth = 2; cx.beginPath(); cx.moveTo(X(t.km) - 10, Y(t.twt)); cx.lineTo(X(t.km) + 10, Y(t.twt)); cx.stroke();
        }
      }
      if (state.showNames) {
        // formation names beside the well, pushed apart so they do not overlap
        const right = st.side === "right", labs = wl.tops.filter((t) => LABELED_TOPS[wl.name]?.has(t.unit) && t.twt >= meta.t_min).sort((a, b) => a.twt - b.twt);
        cx.font = "600 11px Barlow, Arial, sans-serif"; cx.textBaseline = "middle"; let lastY = -Infinity;
        for (const t of labs) {
          const lab = t.unit.replace(" Formation", " Fm").replace(" Member", " Mbr"), tw = cx.measureText(lab).width + 8;
          let y = Y(t.twt); if (y - lastY < 14) y = lastY + 14; lastY = y;
          const x0 = right ? X(t.km) + 14 : X(t.km) - 14 - tw;
          cx.strokeStyle = st.color; cx.lineWidth = 1; cx.beginPath(); cx.moveTo(X(t.km) + (right ? 10 : -10), Y(t.twt)); cx.lineTo(right ? x0 : x0 + tw, y); cx.stroke();
          cx.fillStyle = "rgba(20,24,26,.85)"; cx.fillRect(x0, y - 7, tw, 14);
          cx.fillStyle = st.color; cx.textAlign = "left"; cx.fillText(lab, x0 + 4, y);
        }
      }
      // karst zones and the fault zone recorded on the mud log
      for (const ev of wl.events || []) {
        if (ev.kind === "fault" && state.showFault) {
          const seg = p.filter((q) => q.md >= ev.md_top && q.md <= ev.md_base);
          cx.strokeStyle = "#e63946"; cx.lineWidth = 7; cx.globalAlpha = 0.85; cx.beginPath();
          [{ km: ev.km_top, twt: ev.twt_top }, ...seg, { km: ev.km_base, twt: ev.twt_base }].forEach((q, i) => (i ? cx.lineTo : cx.moveTo).call(cx, X(q.km), Y(q.twt)));
          cx.stroke(); cx.globalAlpha = 1;

        } else if (ev.kind === "karst" && state.showKarst) {
          const xk = X(ev.km_top), yk = Y(ev.twt_top); cx.fillStyle = "#4cc9f0"; cx.strokeStyle = "#000"; cx.lineWidth = 1;
          cx.beginPath(); cx.moveTo(xk + 7, yk); cx.lineTo(xk, yk - 5); cx.lineTo(xk - 7, yk); cx.lineTo(xk, yk + 5); cx.closePath(); cx.fill(); cx.stroke();
        }
      }
      const off = p.map((q) => q.offset_m);
      const dist = Math.max(...off) - Math.min(...off) < 50 ? `${(off[0] / 1000).toFixed(1)} km` : `${(Math.min(...off) / 1000).toFixed(1)}–${(Math.max(...off) / 1000).toFixed(1)} km`;
      const label = `${wl.name}${wl.year ? ` (drilled ${wl.year})` : ""}, ${dist} from the line${wl.estimated_path ? ", path estimated" : ""}`;
      topLabel(X(p[0].km), label, st.color);
    }
    cx.restore();
  }

  function drawTraceMarker() {
    const x = X(state.traceKm); if (x < MARGIN.l || x > W - MARGIN.r) return;
    cx.save(); cx.strokeStyle = "rgba(200,54,45,.9)"; cx.lineWidth = 1.5; cx.setLineDash([3, 3]);
    cx.beginPath(); cx.moveTo(x, MARGIN.t); cx.lineTo(x, H - MARGIN.b); cx.stroke();
    cx.restore(); topLabel(x, "Trace shown at right", "#c8362d", "#fff");
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
    if (depthAxis || state.zoom === "someren") {
      cx.textAlign = "left";
      const someren = state.zoom === "someren", axis = someren ? meta.someren.depth_axis.axis : meta.well.depth_axis;
      for (const d of axis) { if (d.depth % 500 || d.twt < v.tA || d.twt > v.tB) continue; const y = Y(d.twt); cx.fillRect(W - MARGIN.r, y, 5, 1); cx.fillText(String(d.depth), W - MARGIN.r + 8, y); }
      cx.save(); cx.translate(W - 12, (MARGIN.t + H - MARGIN.b) / 2); cx.rotate(Math.PI / 2); cx.textAlign = "center"; cx.fillText(someren ? `Depth below NAP at ${meta.someren.depth_axis.km} km, from migration velocities (m)` : `Depth below NAP at ${meta.well.name} (m)`, 0, 0); cx.restore();
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
    (state.attrLut === "default" ? a.lut : meta.luts[state.attrLut].lut).forEach((col, i) => { g.fillStyle = `rgb(${col})`; g.fillRect(i / 256 * c.width, 0, c.width / 256 + 1, 20); });
    g.fillStyle = "#1f1d18"; g.font = "12px Barlow, Arial, sans-serif"; g.textBaseline = "top";
    g.textAlign = "left"; g.fillText(fmt(a.min), 0, 24); g.textAlign = "right"; g.fillText(fmt(a.max), c.width, 24); g.textAlign = "center"; g.fillText(a.unit, c.width / 2, 24);
    $("#attrMeasures").textContent = a.measures; $("#attrGeology").textContent = a.geology; $("#attrSource").textContent = a.source;
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

  /* ---------- state changes ---------- */
  const run = () => state.runs[state.current] || null;

  async function refresh() {
    const r = run(), key = state.stage <= 2 ? "" : state.stage === 3 ? `a:${state.attr}:${state.attrLut}` : `s:${r ? r.id : "none"}`;
    if (key !== overlayKey) { overlay = key && !key.endsWith("none") ? await buildOverlay() : null; overlayKey = key; }
    drawSomGrid($("#somGrid")); drawSomGrid($("#somGridVerdict")); drawShapGlobal(); drawShapSample();
    draw();
  }

  function setZoom(z) { state.zoom = z; if (z !== "custom") state.geoFocus = null; $$("[data-zoom]").forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.zoom === z))); }
  function setStage(n) {
    state.stage = n;
    if (n === 6) setZoom("someren");
    $$(".tag").forEach((b) => b.setAttribute("aria-current", String(+b.dataset.stage === n)));
    $$(".card").forEach((c) => (c.hidden = +c.dataset.for !== n));
    refresh();
  }

  /* 2D color bar: bilinear blend of four corner colors across the neuron grid */
  const CORNERS = [[44, 123, 182], [215, 25, 28], [255, 217, 47], [26, 152, 80]];   // top-left, top-right, bottom-left, bottom-right
  function neuronColors(side) {
    const out = [];
    for (let r = 0; r < side; r++) for (let c = 0; c < side; c++) {
      const u = side > 1 ? c / (side - 1) : 0.5, v = side > 1 ? r / (side - 1) : 0.5;
      out.push([0, 1, 2].map((i) => Math.round((1 - u) * (1 - v) * CORNERS[0][i] + u * (1 - v) * CORNERS[1][i] + (1 - u) * v * CORNERS[2][i] + u * v * CORNERS[3][i])));
    }
    return out;
  }

  function drawSomGrid(c, path) {
    const g = c.getContext("2d"), w = c.width, pad = 10; g.fillStyle = "#fffaf0"; g.fillRect(0, 0, w, c.height);
    const r = run();
    if (!r) { g.fillStyle = "#5a5446"; g.font = "13px Barlow, Arial, sans-serif"; g.textAlign = "center"; g.fillText("No SOM trained yet", w / 2, c.height / 2); return; }
    const cols = neuronColors(r.side), cell = (w - 2 * pad) / r.side, maxHit = Math.max(...r.hits);
    for (let k = 0; k < r.side * r.side; k++) {
      const x = pad + (k % r.side) * cell, y = pad + Math.floor(k / r.side) * cell;
      g.fillStyle = `rgb(${cols[k]})`; g.fillRect(x + 1, y + 1, cell - 2, cell - 2);
      if (!path) { const rad = Math.sqrt(r.hits[k] / maxHit) * cell * 0.35; g.fillStyle = "rgba(20,20,20,.55)"; g.beginPath(); g.arc(x + cell / 2, y + cell / 2, Math.max(rad, r.hits[k] > 0 ? 1.5 : 0), 0, Math.PI * 2); g.fill(); }
    }
    if (path) {
      const P = (p) => [pad + (p[0] + 0.5) * cell, pad + (p[1] + 0.5) * cell];
      let cur = path.base.slice();
      g.lineWidth = 2.5; g.font = "600 11px Barlow, Arial, sans-serif"; g.textBaseline = "middle";
      const order = path.phi.map((v, j) => [j, Math.hypot(v[0], v[1])]).sort((a, b) => b[1] - a[1]);
      g.fillStyle = "#fff"; g.strokeStyle = "#000"; const [bx, by] = P(cur); g.beginPath(); g.arc(bx, by, 6, 0, Math.PI * 2); g.fill(); g.stroke();
      for (const [j, mag] of order) {
        const nxt = [cur[0] + path.phi[j][0], cur[1] + path.phi[j][1]], [x0, y0] = P(cur), [x1, y1] = P(nxt);
        g.strokeStyle = "#000"; g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
        const ang = Math.atan2(y1 - y0, x1 - x0); if (mag * cell > 6) { g.fillStyle = "#000"; g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - 8 * Math.cos(ang - 0.4), y1 - 8 * Math.sin(ang - 0.4)); g.lineTo(x1 - 8 * Math.cos(ang + 0.4), y1 - 8 * Math.sin(ang + 0.4)); g.fill(); }
        if (mag * cell > 14) { const lab = shortName(r.features[j]), tw = g.measureText(lab).width + 6, mx = (x0 + x1) / 2, my = (y0 + y1) / 2; g.fillStyle = "rgba(255,250,240,.9)"; g.fillRect(mx - tw / 2, my - 7, tw, 14); g.fillStyle = "#1f1d18"; g.textAlign = "center"; g.fillText(lab, mx, my); }
        cur = nxt;
      }
      const [fx, fy] = P(path.final); g.fillStyle = "#ffd166"; g.strokeStyle = "#000"; g.beginPath(); g.arc(fx, fy, 7, 0, Math.PI * 2); g.fill(); g.stroke();
    }
  }

  function drawRedundancy() {
    const r = run(), el = $("#redundancy");
    if (!r) { el.textContent = ""; return; }
    const pairs = [];
    r.features.forEach((a, i) => r.features.forEach((b, j) => { if (j > i && Math.abs(r.corr[i][j]) >= 0.8) pairs.push(`${meta.attributes[a].label} and ${meta.attributes[b].label} (r = ${Math.max(-1, Math.min(1, r.corr[i][j])).toFixed(2)})`); }));
    el.innerHTML = pairs.length ? `<span class="warn">Correlation of 0.8 or more:</span> ${pairs.join("; ")}.` : "No pair of the chosen attributes correlates at 0.8 or more.";
  }

  function drawRunLog() {
    const el = $("#runLog"); el.innerHTML = "";
    if (!state.runs.length) { el.innerHTML = '<p class="small">No runs yet.</p>'; return; }
    state.runs.forEach((r, i) => {
      const b = document.createElement("button"); b.setAttribute("aria-pressed", String(i === state.current));
      const top = r.importance ? r.features.map((f, j) => [f, r.importance[j]]).sort((a, c) => c[1] - a[1]).slice(0, 2).map(([f]) => shortName(f)).join(", ") : "SHAP running";
      b.innerHTML = `Run ${i + 1}: ${r.features.length} attributes, ${r.side * r.side} neurons<small>${r.window.km[0].toFixed(1)}–${r.window.km[1].toFixed(1)} km, ${r.window.t[0].toFixed(2)}–${r.window.t[1].toFixed(2)} s</small><small>${r.features.map(shortName).join(", ")}</small><small>Largest SHAP: ${top}</small>`;
      b.addEventListener("click", () => { state.current = i; state.sample = null; state.explained = null; drawRunLog(); drawRedundancy(); refresh(); });
      el.append(b);
    });
  }

  function drawShapGlobal() {
    const c = $("#shapGlobal"), r = run();
    $("#shapRunLabel").textContent = r ? `Run ${state.current + 1}: ${r.features.length} attributes, ${r.side * r.side} neurons` : "Train a SOM in stage 3 first.";
    if (!r || !r.importance) {
      const g = c.getContext("2d"); g.fillStyle = "#fffaf0"; g.fillRect(0, 0, c.width, c.height);
      g.fillStyle = "#5a5446"; g.font = "13px Barlow, Arial, sans-serif"; g.textAlign = "center"; g.fillText(r ? "Computing SHAP values…" : "", c.width / 2, c.height / 2); return;
    }
    const order = r.features.map((f, j) => [f, r.importance[j]]).sort((a, b) => b[1] - a[1]);
    c.height = Math.max(90, 26 + 18 * order.length);
    hbars(c, order.map(([f]) => shortName(f)), order.map(([, v]) => v), { min: 0, max: 0.3, zeroLine: false, colors: order.map(() => "#5a5446"), valueFmt: (v) => v.toFixed(3) });
  }

  function drawShapSample() {
    const r = run(), e = state.explained, local = $("#shapLocal");
    if (!r || !e || e.run !== r.id) {
      $("#sampleTitle").textContent = "Click the section to explain a sample";
      drawSomGrid($("#shapPath"));
      const g = local.getContext("2d"); g.fillStyle = "#fffaf0"; g.fillRect(0, 0, local.width, local.height); return;
    }
    const k = r.bmu[e.index];
    $("#sampleTitle").textContent = `${state.sample.km.toFixed(2)} km, ${state.sample.t.toFixed(2)} s: neuron row ${Math.floor(k / r.side) + 1}, column ${k % r.side + 1}`;
    drawSomGrid($("#shapPath"), e);
    const span = Math.max(r.side - 1, 1), order = e.phi.map((v, j) => [j, Math.hypot(v[0], v[1]) / span]).sort((a, b) => b[1] - a[1]);
    local.height = Math.max(90, 26 + 18 * order.length);
    hbars(local, order.map(([j]) => shortName(r.features[j])), order.map(([, v]) => v), { min: 0, max: 0.4, zeroLine: false, colors: order.map(() => "#be2d28"), valueFmt: (v) => v.toFixed(3), title: "Distance moved, fraction of map width" });
  }

  /* ---------- SOM builder ---------- */
  let worker = null;
  function requestExplain() {
    const r = run(); if (!r || !state.sample || !worker || state.current !== state.runs.length - 1) { if (r && state.current !== state.runs.length - 1) $("#sampleTitle").textContent = "Samples can be explained for the most recent run"; return; }
    const g = meta.grid, gi = Math.round((state.sample.km - g.km_min) / (g.km_max - g.km_min) * (g.nx - 1)), j = Math.min(g.nt - 1, Math.round((state.sample.t - meta.t_min) / g.dt));
    const w = run().window.win; if (gi < w.i0 || gi > w.i1 || j < w.j0 || j > w.j1) { $("#sampleTitle").textContent = "That sample is outside the SOM window"; return; }
    $("#sampleTitle").textContent = "Computing SHAP values for the sample…";
    worker.postMessage({ type: "explain", index: gi * g.nt + j });
  }

  function wireGeology() {
    $$("[data-geo]").forEach((b) => b.addEventListener("click", () => {
      const [k0, k1, t0, t1] = b.dataset.geo.split(",").map(Number);
      ZOOMS.custom = [Math.max(0, k0 - 1.5), Math.min(48.5, k1 + 1.5)]; setZoom("custom");
      state.geoFocus = { km: [k0, k1], t: [t0, t1] };
      $$("[data-geo]").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); draw();
    }));
    $("#geoColumns").addEventListener("change", (e) => { state.showGeoColumns = e.target.checked; draw(); });
    const leg = $("#groupLegend");
    for (const [code, g] of Object.entries(GROUPS)) leg.insertAdjacentHTML("beforeend", `<div class="grp"><span style="background:${g.color}">${code}</span>${g.name}</div>`);
  }

  function wireWellChips() {
    const box = $("#wellChips"), wells = meta.wells || [];
    for (const w of wells) {
      const b = document.createElement("button"); b.textContent = w.name; b.setAttribute("aria-pressed", "true");
      b.style.setProperty("--chip", (WELL_STYLE[w.name] || {}).color || "#fff");
      b.addEventListener("click", () => { state.hiddenWells.has(w.name) ? state.hiddenWells.delete(w.name) : state.hiddenWells.add(w.name); b.setAttribute("aria-pressed", String(!state.hiddenWells.has(w.name))); draw(); });
      box.append(b);
    }
    for (const [key, text, color] of [["showKarst", "Karst zones ◆", "#4cc9f0"]]) {
      const c = document.createElement("button"); c.textContent = text; c.setAttribute("aria-pressed", "true"); c.style.setProperty("--chip", color);
      c.addEventListener("click", () => { state[key] = !state[key]; c.setAttribute("aria-pressed", String(state[key])); draw(); });
      box.append(c);
    }
    const n = document.createElement("button"); n.textContent = "Formation names"; n.setAttribute("aria-pressed", "true");
    n.addEventListener("click", () => { state.showNames = !state.showNames; n.setAttribute("aria-pressed", String(state.showNames)); draw(); });
    box.append(n);
  }

  function wireBuilder() {
    const box = $("#attrChecks"), groups = {};
    for (const [k, a] of Object.entries(meta.attributes)) (groups[a.family] ??= []).push([k, a]);
    for (const [fam, list] of Object.entries(groups)) {
      box.append(Object.assign(document.createElement("div"), { className: "fam", textContent: fam }));
      for (const [k, a] of list) {
        const l = document.createElement("label");
        l.innerHTML = `<input type="checkbox" value="${k}"> ${a.label}`; box.append(l);
      }
    }
    $("#runSom").addEventListener("click", async () => {
      const feats = $$("#attrChecks input:checked").map((x) => x.value);
      if (feats.length < 2) { $("#progress").hidden = false; $("#progressText").textContent = "Choose at least two attributes."; return; }
      const side = +$("#neurons").value, g = meta.grid;
      const kmRange = { full: [meta.km_min, meta.km_max], someren: ZOOMS.someren, well: ZOOMS.well, view: ZOOMS[state.zoom] }[$("#somArea").value];
      let tTop = +$("#somTop").value, tBase = +$("#somBase").value; if (tBase <= tTop + 0.1) tBase = tTop + 0.1;
      const toI = (k) => Math.max(0, Math.min(g.nx - 1, Math.round((k - g.km_min) / (g.km_max - g.km_min) * (g.nx - 1))));
      const toJ = (t) => Math.max(0, Math.min(g.nt - 1, Math.round((t - meta.t_min) / g.dt)));
      const win = { i0: toI(kmRange[0]), i1: toI(kmRange[1]), j0: toJ(tTop), j1: toJ(tBase) };
      $("#runSom").disabled = true; state.busy = true; $("#progress").hidden = false; $("#progressText").textContent = "Loading attributes";
      const attrs = [];
      for (const f of feats) { const d = await loadBin(`attr_${f}.bin`, Uint8Array); attrs.push({ key: f, data: d.slice(), min: meta.attributes[f].min, max: meta.attributes[f].max }); }
      worker?.terminate(); worker = new Worker("js/som-worker.js");
      const rec = { id: Date.now(), features: feats, side, bmu: null, hits: null, corr: null, importance: null,
        window: { km: [Math.max(kmRange[0], meta.km_min), Math.min(kmRange[1], meta.km_max)], t: [tTop, tBase], win } };
      worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === "progress") { $("#progressBar").style.width = `${Math.round(m.frac * 100)}%`; $("#progressText").textContent = m.stage; }
        if (m.type === "map") {
          Object.assign(rec, { bmu: m.bmu, hits: m.hits, corr: m.corr }); state.runs.push(rec); state.current = state.runs.length - 1;
          state.sample = null; state.explained = null; $("#runSom").disabled = false; state.busy = false;
          drawRunLog(); drawRedundancy(); refresh();
        }
        if (m.type === "importance") { rec.importance = m.importance; $("#progress").hidden = true; drawRunLog(); drawShapGlobal(); }
        if (m.type === "explain") { state.explained = { run: rec.id, ...m }; drawShapSample(); }
      };
      worker.postMessage({ type: "run", attrs, nx: meta.grid.nx, nt: meta.grid.nt, side, seed: 7, win }, attrs.map((a) => a.data.buffer));
    });
  }

  function wire() {
    const aSel = $("#attrSelect"), groups = {};
    for (const [k, a] of Object.entries(meta.attributes)) (groups[a.family] ??= []).push([k, a]);
    for (const [fam, list] of Object.entries(groups)) {
      const og = document.createElement("optgroup"); og.label = fam;
      for (const [k, a] of list) og.append(Object.assign(document.createElement("option"), { value: k, textContent: a.label }));
      aSel.append(og);
    }
    aSel.value = state.attr; aSel.addEventListener("change", () => { state.attr = aSel.value; drawColorbar(); refresh(); });
    const lSel = $("#attrLut");
    lSel.append(Object.assign(document.createElement("option"), { value: "default", textContent: "Default for this attribute" }));
    for (const [k, v] of Object.entries(meta.luts)) lSel.append(Object.assign(document.createElement("option"), { value: k, textContent: v.label }));
    lSel.addEventListener("change", () => { state.attrLut = lSel.value; drawColorbar(); refresh(); });
    const sSel = $("#seisMap");
    for (const [k, v] of Object.entries(SEIS_MAPS)) sSel.append(Object.assign(document.createElement("option"), { value: k, textContent: v.label }));
    sSel.addEventListener("change", () => { state.seisMap = sSel.value; baseImg = raster(meta.section.nx, meta.nt, grayAt); draw(); });

    $$(".tag").forEach((b) => b.addEventListener("click", () => setStage(+b.dataset.stage)));
    $$("[data-zoom]").forEach((b) => b.addEventListener("click", () => { setZoom(b.dataset.zoom); draw(); }));
    for (const [id, key] of [["#showWell", "showWell"], ["#showUnits", "showUnits"]]) $(id).addEventListener("change", (e) => { state[key] = e.target.checked; draw(); });
    const hBoxes = [$("#showHorizons"), ...$$(".syncHorizons")];
    hBoxes.forEach((box) => box.addEventListener("change", (e) => { state.showHorizons = e.target.checked; hBoxes.forEach((b) => (b.checked = state.showHorizons)); draw(); }));
    $("#hideWellControl").addEventListener("click", (e) => {
      state.hideControl = !state.hideControl; e.target.setAttribute("aria-pressed", String(state.hideControl));
      e.target.textContent = state.hideControl ? "Show well control" : "Hide all well control"; draw();
    });
    for (const [id, key] of [["#attrOpacity", "attrOpacity"], ["#somOpacity", "somOpacity"], ["#verdictOpacity", "verdictOpacity"]])
      $(id).addEventListener("input", (e) => { state[key] = +e.target.value; draw(); });

    const idle = "Move over the section to read position and values. Click to show a trace.";
    cv.addEventListener("mousemove", (e) => {
      const r = cv.getBoundingClientRect(), km = invX(e.clientX - r.left), t = invY(e.clientY - r.top), v = view();
      if (km < v.kmA || km > v.kmB || t < meta.t_min || t > meta.t_max) { $("#readout").textContent = idle; return; }
      const g = meta.grid, gi = Math.round((km - g.km_min) / (g.km_max - g.km_min) * (g.nx - 1)), j = Math.min(g.nt - 1, Math.round((t - meta.t_min) / g.dt));
      let txt = `${km.toFixed(2)} km, ${t.toFixed(3)} s`;
      const attr = cache[`attr_${state.attr}.bin`], rr = run();
      if (state.stage === 3 && attr && gi >= 0 && gi < g.nx) { const a = meta.attributes[state.attr]; txt += `, ${a.label} ${fmt(a.min + attr[gi * g.nt + j] / 255 * (a.max - a.min))} ${a.unit}`; }
      if (state.stage >= 4 && rr && gi >= 0 && gi < g.nx && rr.bmu[gi * g.nt + j] !== 255) { const k = rr.bmu[gi * g.nt + j]; txt += `, neuron row ${Math.floor(k / rr.side) + 1}, column ${k % rr.side + 1}`; }
      $("#readout").textContent = txt;
    });
    cv.addEventListener("click", (e) => {
      const r = cv.getBoundingClientRect(), km = invX(e.clientX - r.left), t = invY(e.clientY - r.top), v = view();
      if (km < v.kmA || km > v.kmB || t < meta.t_min || t > meta.t_max) return;
      if (state.stage === 1 && PICK_MODE) return pickAt(km, t, e.shiftKey);
      if (state.stage === 1) { state.traceKm = km; drawWiggle(); draw(); }
      if (state.stage === 5 && run()) { state.sample = { km, t }; requestExplain(); draw(); }
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
    wire(); wireWellChips(); wireGeology(); wireBuilder(); setupPickMode(); drawColorbar(); drawWiggle(); resize(); refresh();
  }
  init();
})();
