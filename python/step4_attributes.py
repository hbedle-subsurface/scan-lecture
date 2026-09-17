"""Step 4: compute attributes on the 20 m grid, and read the attributes computed in AASPI.

Every attribute is averaged over a 140 m x 62 ms window so it describes seismic facies character
rather than individual reflections.
"""
import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt
from scipy.ndimage import uniform_filter, uniform_filter1d, gaussian_filter, map_coordinates
import config as C

def main():
    s1 = np.load(C.WORK / "step1_read.npz")
    f = s1["full"][:: C.ATT_STEP // C.SEC_STEP].astype(float)
    nx, nt = f.shape; W = C.ATT_WINDOW; A = {}
    an = hilbert(f, axis=1); env = np.abs(an)
    x, y = f, np.imag(an)
    fi = (x * np.gradient(y, axis=1) - y * np.gradient(x, axis=1)) / (x * x + y * y + 1e-9) / (2 * np.pi * C.DT)
    A["envelope"] = uniform_filter(env, W)
    A["instantaneous_frequency"] = np.clip(uniform_filter(fi * env ** 2, W) / (uniform_filter(env ** 2, W) + 1e-9), 0, 125)
    A["sweetness"] = A["envelope"] / np.sqrt(np.clip(A["instantaneous_frequency"], 5, None))

    def band(lo, hi):
        sos = butter(4, [lo, hi], btype="band", fs=1 / C.DT, output="sos")
        return np.sqrt(np.clip(uniform_filter(np.abs(hilbert(sosfiltfilt(sos, f, axis=1), axis=1)) ** 2, W), 0, None))
    A["spectral_ratio"] = np.log((band(45, 65) + 1e-6) / (band(10, 20) + 1e-6))

    sm = gaussian_filter(f, 1); gx = np.gradient(sm, axis=0); gt = np.gradient(sm, axis=1)
    Jxx, Jtt, Jxt = (gaussian_filter(v, (3, 5)) for v in (gx * gx, gt * gt, gx * gt))
    p = np.clip(-Jxt / (Jtt + 1e-9), -6, 6)                           # samples per 20 m trace
    A["apparent_dip"] = uniform_filter(np.abs(p), W) * (C.DT * 1000) / 20 * 100   # ms per 100 m
    A["dip_variability"] = np.sqrt(np.clip(uniform_filter(p ** 2, W) - uniform_filter(p, W) ** 2, 0, None))
    cc, tt = np.meshgrid(np.arange(nx), np.arange(nt), indexing="ij")
    num = np.zeros_like(f); den = np.zeros_like(f)
    for k in range(-2, 3):
        u = map_coordinates(f, [np.clip(cc + k, 0, nx - 1), tt + p * k], order=1, mode="nearest")
        num += u; den += u * u
    A["coherence"] = uniform_filter(num ** 2, W) / (5 * uniform_filter(den, W) + 1e-9)

    en, ef = (np.abs(hilbert(s1[k].astype(float), axis=1)) for k in ("near", "far"))
    A["far_minus_near"] = uniform_filter(ef - en, W) / (uniform_filter(ef + en, W) + 1e-9)
    # AASPI attributes: matched to the 20 m grid by CDP number. RMS amplitude receives the same time-only
    # gain as the seismic; the other two are normalized by their own time trend (below).
    # Relative acoustic impedance is multiplied by -1 because in this dataset an increase in acoustic
    # impedance is a negative number, so integrating the trace gives impedance with reversed sign.
    for key, fname in C.AASPI.items():
        vol = np.load(C.AASPI_DIR / fname, mmap_mode="r")            # (CDPs from FIRST_CDP, samples)
        cdp_index = s1["cdp"][s1["att_idx"]] - C.FIRST_CDP
        v = np.asarray(vol[cdp_index, :nt], dtype=float)
        if key == "relative_acoustic_impedance": v = -v
        if key == "rms_amplitude":
            v = v * s1["gain"][None, :nt]
        else:
            # integration and the AVT rotation change the frequency content, so these two receive their own
            # time-only normalization (median RMS at each time) instead of the seismic gain
            rms_t = np.median(np.sqrt(uniform_filter1d(v ** 2, 100, axis=1)), axis=0) + 1e-9
            v = v / uniform_filter1d(rms_t, 50)[None, :]
        # RMS amplitude is averaged like the other attributes. Relative acoustic impedance and AVT are
        # band-limited, zero-mean traces, so they are averaged laterally over 7 traces and 5 samples (10 ms).
        A[key] = uniform_filter(v, W if key == "rms_amplitude" else (W[0], 5))

    # amplitude of each stack, with the seismic time gain, averaged laterally over 7 traces only
    A["full_stack_amplitude"] = uniform_filter(f, (W[0], 1))
    for k in ("near", "mid", "far"):
        A[f"{k}_stack_amplitude"] = uniform_filter(s1[k].astype(float), (W[0], 1))

    np.savez(C.WORK / "step4_attributes.npz", **{k: np.nan_to_num(v).astype(np.float32) for k, v in A.items()})
    t0, t1 = int(C.T_MIN / C.DT), int(C.T_MAX / C.DT)
    X = np.stack([A[k][:, t0:t1].ravel()[::97] for k in A], 1); R = np.corrcoef(X.T); keys = list(A)
    print("pairs with |r| > 0.8:", [(keys[i], keys[j], round(R[i, j], 2)) for i in range(len(keys)) for j in range(i) if abs(R[i, j]) > 0.8])

if __name__ == "__main__":
    main()
