"""Step 3: track six horizons from the well-tie seeds, 30-39 km along the line.

Each horizon starts at the reflection nearest the projected top (within +-20 ms) and follows local
reflector dip from a structure tensor, snapping to the same-polarity extremum at each trace.
Where the Zeeland top cannot be followed, and along the whole Bosscheveld top, the horizon is placed at the time thickness
measured at the well below the horizon above and snapped to the nearest matching reflection.
Those segments are flagged as interpolated.
"""
import json
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d, median_filter
import config as C

HORIZONS = ["Rupel Clay Member", "Houthem Formation", "Zechstein Upper Claystone Formation",
            "Epen Formation", "Zeeland Formation", "Bosscheveld Formation"]

def main():
    s1 = np.load(C.WORK / "step1_read.npz"); well = json.load(open(C.WORK / "step2_well.json"))
    km_all = s1["sec_km"]; c0, c1 = np.searchsorted(km_all, C.HORIZON_KM)
    d = s1["full"][c0:c1].astype(float); km = km_all[c0:c1]; n, nt = d.shape
    a = d / (np.sqrt(uniform_filter1d(d ** 2, 150, axis=1)) + 1e-6)
    sm = gaussian_filter(a, 1); gx = np.gradient(sm, axis=0); gt = np.gradient(sm, axis=1)
    Jxx, Jtt, Jxt = (gaussian_filter(v, (8, 6)) for v in (gx * gx, gt * gt, gx * gt))
    p = np.clip(-Jxt / (Jtt + 1e-9), -3, 3)
    l1 = 0.5 * (Jxx + Jtt + np.sqrt((Jxx - Jtt) ** 2 + 4 * Jxt ** 2)); l2 = 0.5 * (Jxx + Jtt - np.sqrt((Jxx - Jtt) ** 2 + 4 * Jxt ** 2))
    linearity = (l1 - l2) / (l1 + l2 + 1e-9)
    tops = {t["unit"]: t for t in well["tops"]}

    def track(col, s0, sign, snap):
        h = np.full(n, np.nan); q = np.full(n, np.nan); h[col] = s0; q[col] = 1
        for step in (1, -1):
            s, c = float(s0), col
            while 0 <= c + step < n:
                s += step * p[c, int(np.clip(round(s), 0, nt - 1))]; c += step
                lo, hi = int(s) - snap, int(s) + snap + 1
                if 0 < lo and hi < nt:
                    w = sign * d[c, lo:hi]; j = w.argmax()
                    if 0 < j < len(w) - 1: s = lo + j
                h[c] = s; q[c] = linearity[c, int(np.clip(round(s), 0, nt - 1))]
        return h * C.DT, q

    out, seeds = {}, {}
    for name in HORIZONS:
        t = tops[name]; col = int(np.abs(km - t["km"]).argmin()); sp = int(t["twt"] / C.DT)
        w = d[col, sp - 10:sp + 11]; j = int(np.abs(w).argmax()); s0 = sp - 10 + j; sign = float(np.sign(w[j]))
        h, q = track(col, s0, sign, 3 if t["twt"] < 1.0 else 2)
        out[name] = dict(t=h, tracked=q > 0.5, sign=sign, col=col)
        seeds[name] = dict(seed_twt=s0 * C.DT, predicted_twt=t["twt"], polarity=int(sign))

    def constrain(name, above_t, sign, everywhere=False):
        raw = out[name]["t"]; col = out[name]["col"]; thick = raw[col] - above_t[col]
        bad = (raw - above_t) < 0.15
        if everywhere: bad[:] = True
        for rng in (range(col, n), range(col, -1, -1)):
            for c in rng:
                if bad[c]:
                    if rng.step == 1: bad[c:] = True
                    else: bad[:c + 1] = True
                    break
        guide = above_t + thick; snapped = np.empty(n)
        for c in range(n):
            s = int(round(guide[c] / C.DT)); w = sign * d[c, s - 5:s + 6]; snapped[c] = (s - 5 + w.argmax()) * C.DT
        t = np.where(bad, median_filter(snapped, 9), raw)
        t = uniform_filter1d(median_filter(t, 31), 9)
        out[name]["t"] = t; out[name]["tracked"] &= ~bad

    constrain("Zeeland Formation", out["Epen Formation"]["t"], out["Zeeland Formation"]["sign"])
    constrain("Bosscheveld Formation", out["Zeeland Formation"]["t"], out["Bosscheveld Formation"]["sign"], everywhere=True)
    res = dict(km=km.tolist(), horizons={k: dict(twt=np.round(v["t"], 4).tolist(), tracked=v["tracked"].astype(int).tolist())
                                         for k, v in out.items()}, seeds=seeds)
    json.dump(res, open(C.WORK / "step3_horizons.json", "w"))
    for k, v in seeds.items():
        print(f"{k:38s} predicted {v['predicted_twt']:.3f}  seed {v['seed_twt']:.3f}  polarity {v['polarity']:+d}  tracked {out[k]['tracked'].mean():.0%}")

if __name__ == "__main__":
    main()
