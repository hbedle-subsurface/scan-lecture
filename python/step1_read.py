"""Step 1: read the 15-40 km window of the full, near and far stacks and apply one time-only gain."""
import numpy as np
from scipy.ndimage import uniform_filter1d
import config as C, segy

def main():
    C.WORK.mkdir(exist_ok=True)
    tr = segy.open_traces(C.STACKS["full"])
    geo = segy.geometry(tr)
    i0, i1 = np.searchsorted(geo["km"], [C.KM_MIN, C.KM_MAX])
    i0 -= i0 % C.ATT_STEP
    sec_idx = np.arange(i0, i1, C.SEC_STEP)
    att_idx = sec_idx[:: C.ATT_STEP // C.SEC_STEP]
    full = segy.read_block(tr, sec_idx, C.N_READ).astype(np.float64)
    # One gain curve for every trace: the inverse of the smoothed median RMS amplitude at each time.
    # Lateral amplitude differences are preserved.
    rms_t = np.median(np.sqrt(uniform_filter1d(full ** 2, 100, axis=1)), axis=0) + 1e-6
    gain = 1 / uniform_filter1d(rms_t, 50)
    out = dict(full=(full * gain).astype(np.float32), gain=gain, sec_idx=sec_idx, att_idx=att_idx,
               sec_km=geo["km"][sec_idx], att_km=geo["km"][att_idx],
               x=geo["x"], y=geo["y"], km_all=geo["km"], cdp=geo["cdp"])
    for k in ["near", "far"]:
        t = segy.open_traces(C.STACKS[k])
        out[k] = (segy.read_block(t, att_idx, C.N_READ) * gain).astype(np.float32)
    np.savez(C.WORK / "step1_read.npz", **out)
    print("section", out["full"].shape, "attribute grid", out["near"].shape)

if __name__ == "__main__":
    main()
