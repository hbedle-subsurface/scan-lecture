"""Settings shared by every step of the data workflow."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "raw"           # SEG-Y and velocity files from NLOG (not committed)
INPUTS = HERE / "inputs"
WORK = HERE / "work"         # intermediate arrays (not committed)
DATA = ROOT / "data"         # files the web page loads

LINE = "L2EBN2020ASCAN029"
STACKS = {k: RAW / f"{LINE}_PreSTM_final_{k}.sgy" for k in ["full", "near", "far"]}
VELOCITY = RAW / f"{LINE}_PreSTM_velocities_migration_ascii.txt"

NS, DT = 5001, 0.002          # samples per trace, sample interval (s)
KM_MIN, KM_MAX = 15.0, 40.0   # window along the line
N_READ = 900                  # samples read, 0-1.8 s
T_MIN, T_MAX = 0.15, 1.70     # displayed and classified window (s)
SEC_STEP = 4                  # every 4th CDP (10 m): seismic display and horizon tracking
ATT_STEP = 8                  # every 8th CDP (20 m): attributes and SOM

WELL = dict(name="CAL-GT-04", rd_x=203337.28, rd_y=382807.38, rt_above_nap=31.2)
HORIZON_KM = (30.0, 39.0)     # horizons are tracked only near the well
ATT_WINDOW = (7, 31)          # attribute averaging window: 7 traces x 31 samples (140 m x 62 ms)
