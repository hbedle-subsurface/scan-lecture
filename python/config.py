"""Settings shared by every step of the data workflow."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "raw"           # SEG-Y and velocity files from NLOG (not committed)
INPUTS = HERE / "inputs"
WORK = HERE / "work"         # intermediate arrays (not committed)
DATA = ROOT / "data"         # files the web page loads

LINE = "L2EBN2020ASCAN029"
STACKS = {k: RAW / f"{LINE}_PreSTM_final_{k}.sgy" for k in ["full", "near", "mid", "far"]}
VELOCITY = RAW / f"{LINE}_PreSTM_velocities_migration_ascii.txt"

NS, DT = 5001, 0.002          # samples per trace, sample interval (s)
KM_MIN, KM_MAX = 0.0, 48.5    # window along the line (the whole line)
N_READ = 1501                 # samples read, 0-3.0 s
T_MIN, T_MAX = 0.15, 3.00     # displayed and classified window (s)
SEC_STEP = 4                  # every 4th CDP (10 m): seismic display and horizon tracking
ATT_STEP = 8                  # every 8th CDP (20 m): attributes and SOM
ATT_TSTEP = 2                 # every 2nd sample (4 ms) for exported attributes

WELL = dict(name="CAL-GT-04", rd_x=203337.28, rd_y=382807.38, rt_above_nap=31.2)
# Wells shown on the section. CAL-GT-04 has a deviation survey; CAL-GT-01 has only its end-point offsets and
# vertical depth, so its path is estimated as vertical to a kickoff depth and straight to total depth.
WELLS = [
    dict(name="CAL-GT-04", rd_x=203337.28, rd_y=382807.38, rt_above_nap=31.2,
         tops="cal-gt-04_tops.csv", deviation="cal-gt-04_deviation_stations.txt"),
    dict(name="CAL-GT-01", rd_x=203895.0, rd_y=381599.0, rt_above_nap=33.0,
         tops="cal-gt-01_tops.csv", td_md=2082.0, td_tvd=1856.64, td_dx=-364.82, td_dy=-681.92),
    # ASTEN-GT-02 (1986-1987, vertical geothermal exploration well, TNO). Tops from the TNO geological
    # end-of-well report, depths below rotary table at 30.34 m above NAP. Location from NLOG.
    dict(name="ASTEN-GT-02", rd_x=182010.0, rd_y=376840.0, rt_above_nap=30.34, td_md=1673.0, vertical=True, year=1987,
         tops="asten-02_tops.csv"),
]
WELL_EVENTS = {"CAL-GT-04": "cal-gt-04_events.csv"}
HORIZON_KM = (30.0, 39.0)     # horizons are tracked only near the well
ATT_WINDOW = (7, 31)          # attribute averaging window: 7 traces x 31 samples (140 m x 62 ms)

# Attributes computed in AASPI on the whole line, 0-3 s (CDP 100-19492), read from the VDS files and saved
# as float32 arrays of shape (CDPs, samples) with python/vds_to_npy.py
AASPI_DIR = RAW / "aaspi_0-3s"
AASPI = {"relative_acoustic_impedance": "relative_acoustic_impedance03.npy",
         "rms_amplitude": "rms_amplitude03.npy",
         "amplitude_volume_transform": "avt03.npy"}
FIRST_CDP = 100

# Someren exploration license: approximate crossing of SCAN029, measured on the NLOG license map (Someren_029_line.jpg).
# Replace with the intersection of the license polygon once the coordinates are available.
SOMEREN_KM = (3.3, 16.4)
SOMEREN_TARGET_DEPTH_M = (500, 1500)   # "middeldiepe geothermie" depth range stated by the developer
