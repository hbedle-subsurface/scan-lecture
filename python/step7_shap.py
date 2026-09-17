"""Step 7: SHAP values for the SOM class memberships.

The SOM assigns each sample to its nearest prototype. To give SHAP (Lundberg and Lee, 2017) a smooth
quantity to explain, each sample gets a soft membership in every class: exp(-squared distance) to each
prototype, summed over the prototypes in a class and normalized to sum to 1.

Shapley values are computed exactly by evaluating all 2^M attribute coalitions (M = 3 to 7 attributes).
Attributes outside a coalition are replaced by 20 k-means background samples and averaged
(interventional SHAP). For each sample the stored values explain the membership of the class the SOM
assigned; base value + sum of SHAP values = that membership.
Grid: every 4th attribute trace (80 m) and every 5th sample (10 ms).
"""
import json
from itertools import combinations
from math import factorial
import numpy as np
from sklearn.cluster import KMeans
import config as C

TRACE_STEP, SAMPLE_STEP, N_BACKGROUND = 4, 5, 16

def memberships(X, W, group, K):
    d2 = (X * X).sum(1)[:, None] + (W * W).sum(1)[None, :] - 2 * X @ W.T
    e = np.exp(-(d2 - d2.min(1, keepdims=True)))
    m = np.stack([e[:, group == k].sum(1) for k in range(K)], 1)
    return m / m.sum(1, keepdims=True)

def main(only=None):
    A = np.load(C.WORK / "step4_attributes.npz"); S = np.load(C.WORK / "step5_som.npz"); info = json.load(open(C.WORK / "step5_som.json"))
    t0, t1 = int(C.T_MIN / C.DT), int(C.T_MAX / C.DT)
    out, meta = {}, {}
    for name, pinfo in info.items():
        if only and name != only: continue
        feats = pinfo["features"]; M = len(feats); K = len(pinfo["class_fraction"])
        W = S[name + "_prototypes"].astype(np.float32); group = S[name + "_prototype_class"]
        mu, sd = np.array(pinfo["mean"]), np.array(pinfo["std"])
        full = np.stack([A[f][:, t0:t1] for f in feats], -1)
        Xz_all = ((full.reshape(-1, M) - mu) / sd).astype(np.float32)
        bg = KMeans(N_BACKGROUND, n_init=5, random_state=0).fit(Xz_all[np.random.default_rng(1).choice(len(Xz_all), 50000, replace=False)]).cluster_centers_
        grid = full[::TRACE_STEP, ::SAMPLE_STEP]; nx, nt = grid.shape[:2]
        X = ((grid.reshape(-1, M) - mu) / sd).astype(np.float32); bg = bg.astype(np.float32)
        cls = S[name][::TRACE_STEP, ::SAMPLE_STEP].reshape(-1).astype(int)
        # v[mask] = mean over background of membership, attributes in mask taken from the sample
        masks = [tuple(i for i in range(M) if (m >> i) & 1) for m in range(2 ** M)]
        v = np.zeros((len(X), 2 ** M, K), np.float32)
        for mi, keep in enumerate(masks):
            sel = np.zeros(M, bool); sel[list(keep)] = True
            for s in range(0, len(X), 4000):
                xb = np.where(sel, X[s:s + 4000, None, :], bg[None, :, :])           # (n, B, M)
                v[s:s + 4000, mi] = memberships(xb.reshape(-1, M), W, group, K).reshape(-1, N_BACKGROUND, K).mean(1)
        phi = np.zeros((len(X), M, K), np.float32)
        for i in range(M):
            others = [j for j in range(M) if j != i]
            for r in range(M):
                wgt = factorial(r) * factorial(M - r - 1) / factorial(M)
                for sub in combinations(others, r):
                    m0 = sum(1 << j for j in sub); phi[:, i] += wgt * (v[:, m0 | (1 << i)] - v[:, m0])
        idx = np.arange(len(X))
        own = phi[idx, :, cls]                                   # SHAP for the assigned class
        base = v[:, 0, :].mean(0)                                # identical for every sample
        fx = v[idx, -1, cls]
        err = np.abs(base[cls] + own.sum(1) - fx).max()
        # importance: mean |SHAP| of the assigned class, per class and overall
        per_class = [np.abs(own[cls == k]).mean(0).round(4).tolist() if (cls == k).any() else [0.0] * M for k in range(K)]
        overall = np.abs(own).mean(0).round(4).tolist()
        out[name] = np.clip(np.round(own / 1.0 * 127), -127, 127).astype(np.int8).reshape(nx, nt, M)
        out[name + "_membership"] = np.round(fx * 255).astype(np.uint8).reshape(nx, nt)
        meta[name] = dict(nx=nx, nt=nt, trace_step=TRACE_STEP, sample_step=SAMPLE_STEP, scale=1.0,
                          base=base.round(4).tolist(), importance_by_class=per_class, importance_overall=overall)
        print(f"{name}: grid {nx}x{nt}, additivity error {err:.2e}, overall mean |SHAP|",
              dict(zip(feats, np.round(overall, 3))))
        np.savez(C.WORK / f"step7_shap_{name}.npz", **{k: v for k, v in out.items() if k.startswith(name)})
        json.dump(meta[name], open(C.WORK / f"step7_shap_{name}.json", "w"), indent=1)

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
