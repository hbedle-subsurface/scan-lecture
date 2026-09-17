"""Step 5: self-organizing maps for four attribute presets.

Attributes are z-scored over the 15-40 km, 0.15-1.70 s window. A 10 x 10 SOM (Kohonen, 1982) is trained
on 60,000 random samples, its prototype vectors are grouped into 8 classes with k-means, and every
sample takes the class of its best-matching prototype. Classes are numbered by increasing mean
two-way time so colors are comparable between presets.
"""
import json
import numpy as np
from minisom import MiniSom
from sklearn.cluster import KMeans
import config as C

PRESETS = {
    "amplitude_impedance": ["rms_amplitude", "relative_acoustic_impedance", "amplitude_volume_transform"],
    "frequency_continuity": ["instantaneous_frequency", "coherence", "dip_variability"],
    "combined": ["rms_amplitude", "relative_acoustic_impedance", "amplitude_volume_transform", "instantaneous_frequency", "coherence", "dip_variability"],
    "combined_far_near": ["rms_amplitude", "relative_acoustic_impedance", "amplitude_volume_transform", "instantaneous_frequency", "coherence", "dip_variability", "far_minus_near"],
}
N_CLASSES = 8

def main():
    A = np.load(C.WORK / "step4_attributes.npz"); t0, t1 = int(C.T_MIN / C.DT), int(C.T_MAX / C.DT)
    nx = A["coherence"].shape[0]; tgrid = np.broadcast_to(np.arange(t0, t1) * C.DT, (nx, t1 - t0)).ravel()
    out, info = {}, {}
    for name, feats in PRESETS.items():
        X = np.stack([A[f][:, t0:t1].ravel() for f in feats], 1).astype(float)
        mu, sd = X.mean(0), X.std(0); Xz = (X - mu) / sd
        idx = np.random.default_rng(0).choice(len(Xz), 60000, replace=False)
        som = MiniSom(10, 10, len(feats), sigma=2.0, learning_rate=0.5, random_seed=1)
        som.pca_weights_init(Xz[idx]); som.train_random(Xz[idx], 50000)
        W = som.get_weights().reshape(-1, len(feats))
        group = KMeans(N_CLASSES, n_init=20, random_state=0).fit(W).labels_
        lab = np.empty(len(Xz), int)
        for s in range(0, len(Xz), 40000):
            lab[s:s + 40000] = group[((Xz[s:s + 40000, None] - W[None]) ** 2).sum(2).argmin(1)]
        order = np.argsort([tgrid[lab == k].mean() for k in range(N_CLASSES)]); remap = np.empty(N_CLASSES, int); remap[order] = np.arange(N_CLASSES)
        lab = remap[lab]; group = remap[group]
        out[name] = lab.reshape(nx, t1 - t0).astype(np.uint8)
        out[name + "_prototypes"] = W; out[name + "_prototype_class"] = group
        info[name] = dict(features=feats, mean=mu.tolist(), std=sd.tolist(),
                          class_means_z=[Xz[lab == k].mean(0).round(3).tolist() for k in range(N_CLASSES)],
                          class_fraction=[float((lab == k).mean()) for k in range(N_CLASSES)])
        print(name, "class fractions", np.round(info[name]["class_fraction"], 2))
    np.savez(C.WORK / "step5_som.npz", **out)
    json.dump(info, open(C.WORK / "step5_som.json", "w"), indent=1)

if __name__ == "__main__":
    main()
