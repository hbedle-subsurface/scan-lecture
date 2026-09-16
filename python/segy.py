"""Minimal SEG-Y reading for the SCAN029 PreSTM files (IBM float, 5001 samples, 240-byte headers)."""
import numpy as np
from config import NS

TRACE_BYTES = 240 + 4 * NS

def open_traces(path):
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=3600)
    return raw[: raw.size // TRACE_BYTES * TRACE_BYTES].reshape(-1, TRACE_BYTES)

def ibm_to_float(b):
    a = b.view(">u4").astype(np.uint64)
    sign = np.where(a >> 31, -1.0, 1.0)
    expo = ((a >> 24) & 0x7F).astype(np.int64) - 64
    frac = (a & 0xFFFFFF) / 16777216.0
    return (sign * frac * 16.0 ** expo).astype(np.float32)

def geometry(traces):
    h = traces[:, :240]
    i4 = lambda byte: h[:, byte - 1:byte + 3].copy().view(">i4").ravel()
    scalar = h[:, 70:72].copy().view(">i2").ravel().astype(float)
    s = np.where(scalar < 0, -1 / scalar, np.where(scalar > 0, scalar, 1))
    x, y = i4(181) * s, i4(185) * s
    km = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))] / 1000
    return dict(cdp=i4(21), x=x, y=y, km=km)

def read_block(traces, idx, n_samples):
    return np.stack([ibm_to_float(traces[i, 240:240 + 4 * n_samples].copy()) for i in idx])
