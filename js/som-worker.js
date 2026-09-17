/* Self-organizing map and SHAP values, run in a Web Worker so the page stays responsive.

   SOM: Kohonen (1982). Prototypes start on the plane of the first two principal components and are
   trained online with a Gaussian neighborhood that shrinks during training.

   SHAP: the quantity explained is the sample's position on the SOM grid (column, row), taken as the
   average grid position of all neurons weighted by exp(-squared distance / tau). That position is what the
   2D color bar shows, so each attribute's SHAP value is how far it moves the sample across the map.
   Values are estimated by sampling random attribute orderings with a random background sample
   (Strumbelj and Kononenko, 2014), which converges to the Shapley values of Lundberg and Lee (2017). */
"use strict";

let S = null;   // state of the current run

function rng(seed) { // mulberry32, so the same settings give the same result
  return () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

self.onmessage = (e) => {
  const m = e.data;
  if (m.type === "run") run(m);
  if (m.type === "explain") explain(m.index);
};

function run({ attrs, nx, nt, side, seed }) {
  const M = attrs.length, N = side * side, n = nx * nt, rand = rng(seed);
  const post = (stage, frac) => self.postMessage({ type: "progress", stage, frac });

  // dequantize and z-score over the whole window
  post("Preparing attributes", 0);
  const X = new Float32Array(n * M), mean = new Float64Array(M), sd = new Float64Array(M);
  attrs.forEach((a, j) => {
    const scale = (a.max - a.min) / 255; let s = 0, s2 = 0;
    for (let i = 0; i < n; i++) { const v = a.min + a.data[i] * scale; X[i * M + j] = v; s += v; s2 += v * v; }
    mean[j] = s / n; sd[j] = Math.sqrt(Math.max(s2 / n - mean[j] ** 2, 1e-12));
    for (let i = 0; i < n; i++) X[i * M + j] = (X[i * M + j] - mean[j]) / sd[j];
  });

  // training sample
  const nTrain = 20000, train = new Int32Array(nTrain);
  for (let k = 0; k < nTrain; k++) train[k] = Math.floor(rand() * n);

  // correlation between the chosen attributes (for the redundancy panel)
  const corr = Array.from({ length: M }, () => new Float64Array(M));
  for (const i of train) for (let a = 0; a < M; a++) for (let b = a; b < M; b++) corr[a][b] += X[i * M + a] * X[i * M + b];
  for (let a = 0; a < M; a++) for (let b = a; b < M; b++) { corr[a][b] /= nTrain; corr[b][a] = corr[a][b]; }

  // principal components by power iteration with deflation
  const pcs = [], lams = [], C = corr.map((r) => Float64Array.from(r));
  for (let p = 0; p < Math.min(2, M); p++) {
    let v = new Float64Array(M).fill(1 / Math.sqrt(M)), lam = 0;
    for (let it = 0; it < 200; it++) {
      const w = new Float64Array(M);
      for (let a = 0; a < M; a++) for (let b = 0; b < M; b++) w[a] += C[a][b] * v[b];
      lam = Math.hypot(...w) || 1e-9; v = w.map((x) => x / lam);
    }
    let big = 0; for (let a = 0; a < M; a++) if (Math.abs(v[a]) > Math.abs(v[big])) big = a;
    if (v[big] < 0) v = v.map((x) => -x);
    pcs.push(v); lams.push(lam);
    for (let a = 0; a < M; a++) for (let b = 0; b < M; b++) C[a][b] -= lam * v[a] * v[b];
  }
  if (pcs.length < 2) { pcs.push(new Float64Array(M)); lams.push(0); }

  // prototypes on the PC plane
  const W = new Float32Array(N * M);
  for (let r = 0; r < side; r++) for (let c = 0; c < side; c++) {
    const u = side > 1 ? (c / (side - 1)) * 2 - 1 : 0, v = side > 1 ? (r / (side - 1)) * 2 - 1 : 0;
    for (let j = 0; j < M; j++) W[(r * side + c) * M + j] = 2 * (u * Math.sqrt(lams[0]) * pcs[0][j] + v * Math.sqrt(lams[1]) * pcs[1][j]);
  }

  // online training
  const iters = Math.max(8000, 600 * N), sig0 = Math.max(side / 2, 1), sig1 = 0.5, lr0 = 0.5, lr1 = 0.02;
  for (let it = 0; it < iters; it++) {
    const f = it / iters, sigma = sig0 * (sig1 / sig0) ** f, lr = lr0 * (lr1 / lr0) ** f, i = train[Math.floor(rand() * nTrain)];
    let best = 0, bd = Infinity;
    for (let k = 0; k < N; k++) { let d = 0; for (let j = 0; j < M; j++) { const q = X[i * M + j] - W[k * M + j]; d += q * q; } if (d < bd) { bd = d; best = k; } }
    const br = Math.floor(best / side), bc = best % side, rad = Math.ceil(3 * sigma);
    for (let r = Math.max(0, br - rad); r <= Math.min(side - 1, br + rad); r++)
      for (let c = Math.max(0, bc - rad); c <= Math.min(side - 1, bc + rad); c++) {
        const h = lr * Math.exp(-((r - br) ** 2 + (c - bc) ** 2) / (2 * sigma * sigma)), k = r * side + c;
        for (let j = 0; j < M; j++) W[k * M + j] += h * (X[i * M + j] - W[k * M + j]);
      }
    if (it % 2000 === 0) post("Training the SOM", f);
  }

  // classify every sample
  const bmu = new Uint8Array(n), hits = new Float64Array(N), qe = []; const W2 = new Float32Array(N);
  for (let k = 0; k < N; k++) { let s = 0; for (let j = 0; j < M; j++) s += W[k * M + j] ** 2; W2[k] = s; }
  for (let i = 0; i < n; i++) {
    let best = 0, bd = Infinity;
    for (let k = 0; k < N; k++) { let dot = 0; for (let j = 0; j < M; j++) dot += X[i * M + j] * W[k * M + j]; const d = W2[k] - 2 * dot; if (d < bd) { bd = d; best = k; } }
    bmu[i] = best; hits[best]++;
    if (i % 200000 === 0) post("Classifying the section", i / n);
  }

  // softness of the explained grid position: median squared distance to the best-matching neuron
  for (let s = 0; s < 2000; s++) { const i = train[s]; let bd = Infinity; for (let k = 0; k < N; k++) { let d = 0; for (let j = 0; j < M; j++) d += (X[i * M + j] - W[k * M + j]) ** 2; bd = Math.min(bd, d); } qe.push(bd); }
  qe.sort((a, b) => a - b);
  const tau = Math.max(qe[qe.length >> 1], 1e-3);

  S = { X, W, M, N, side, tau, train, rand, nTrain };
  self.postMessage({ type: "map", bmu, hits: Array.from(hits, (h) => h / n), corr: corr.map((r) => Array.from(r)), qe: Math.sqrt(tau) });

  // global SHAP importance on 400 random samples
  const nS = 400, P = 8, imp = new Float64Array(M), impX = new Float64Array(M), impY = new Float64Array(M);
  for (let s = 0; s < nS; s++) {
    const { phi } = shapFor(Math.floor(rand() * n), P);
    for (let j = 0; j < M; j++) { imp[j] += Math.hypot(phi[j][0], phi[j][1]); impX[j] += Math.abs(phi[j][0]); impY[j] += Math.abs(phi[j][1]); }
    if (s % 50 === 0) post("Computing SHAP values", s / nS);
  }
  const span = Math.max(side - 1, 1);   // report in fractions of the map width
  self.postMessage({ type: "importance", importance: Array.from(imp, (v) => v / nS / span) });
}

function position(z) { // soft grid position of attribute vector z
  const { W, M, N, side, tau } = S; let best = Infinity; const d = new Float64Array(N);
  for (let k = 0; k < N; k++) { let s = 0; for (let j = 0; j < M; j++) s += (z[j] - W[k * M + j]) ** 2; d[k] = s; if (s < best) best = s; }
  let wsum = 0, px = 0, py = 0;
  for (let k = 0; k < N; k++) { const w = Math.exp(-(d[k] - best) / tau); wsum += w; px += w * (k % side); py += w * Math.floor(k / side); }
  return [px / wsum, py / wsum];
}

function shapFor(index, P) {
  const { X, M, train, rand, nTrain } = S, x = X.subarray(index * M, index * M + M);
  const phi = Array.from({ length: M }, () => [0, 0]); let base = [0, 0];
  const order = Array.from({ length: M }, (_, j) => j);
  for (let p = 0; p < P; p++) {
    const b = train[Math.floor(rand() * nTrain)], z = Float64Array.from(X.subarray(b * M, b * M + M));
    for (let j = M - 1; j > 0; j--) { const k = Math.floor(rand() * (j + 1)); [order[j], order[k]] = [order[k], order[j]]; }
    let prev = position(z); base[0] += prev[0]; base[1] += prev[1];
    for (const j of order) { z[j] = x[j]; const cur = position(z); phi[j][0] += cur[0] - prev[0]; phi[j][1] += cur[1] - prev[1]; prev = cur; }
  }
  for (let j = 0; j < M; j++) { phi[j][0] /= P; phi[j][1] /= P; }
  return { phi, base: [base[0] / P, base[1] / P], final: position(x) };
}

function explain(index) {
  if (!S) return;
  const r = shapFor(index, 400);
  self.postMessage({ type: "explain", index, ...r });
}
