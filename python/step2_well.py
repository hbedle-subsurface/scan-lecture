"""Step 2: project CAL-GT-04 onto the line and convert formation tops to two-way time.

Time-depth conversion uses Dix interval velocities from the PreSTM migration velocities at the
velocity location nearest each point of the well path. There is no sonic log or checkshot for this well.
"""
import json, csv
import numpy as np
import config as C

def velocity_functions():
    rows = [l for l in open(C.VELOCITY, encoding="latin1") if l.startswith("V2")]
    cdp = np.array([int(l[15:20]) for l in rows]); t = np.array([int(l[35:40]) for l in rows]) / 1000
    v = np.array([int(l[60:65]) for l in rows]).astype(float)
    return cdp, t, v

def depth_time_table(vcdp, vt, vv, cdp):
    k = np.unique(vcdp)[np.abs(np.unique(vcdp) - cdp).argmin()]
    m = vcdp == k
    tg = np.arange(0, 5, C.DT)
    vrms = np.interp(tg, vt[m], vv[m])
    vint2 = np.gradient(vrms ** 2 * tg) / np.gradient(tg)
    vint = np.sqrt(np.clip(vint2, 1400 ** 2, 7000 ** 2))
    z = np.cumsum(vint * np.r_[0, np.diff(tg)] / 2)   # depth below datum (NAP) for each two-way time
    return z, tg

def main():
    s1 = np.load(C.WORK / "step1_read.npz")
    x, y, km_all, cdp_all = s1["x"], s1["y"], s1["km_all"], s1["cdp"]
    dev = np.loadtxt(C.INPUTS / "cal-gt-04_deviation_stations.txt")   # MD, TVD, dX, dY (m, from rotary table)
    vcdp, vt, vv = velocity_functions()
    W = C.WELL

    def locate(md):
        tvd = np.interp(md, dev[:, 0], dev[:, 1])
        px = W["rd_x"] + np.interp(md, dev[:, 0], dev[:, 2]); py = W["rd_y"] + np.interp(md, dev[:, 0], dev[:, 3])
        d = np.hypot(x - px, y - py); i = int(d.argmin())
        tvdss = tvd - W["rt_above_nap"]
        z, tg = depth_time_table(vcdp, vt, vv, cdp_all[i])
        return dict(tvdss=float(tvdss), km=float(km_all[i]), cdp=int(cdp_all[i]), offset_m=float(d[i]),
                    twt=float(np.interp(tvdss, z, tg)))

    tops = []
    for r in csv.DictReader(open(C.INPUTS / "cal-gt-04_tops.csv")):
        p = locate(float(r["top_md_m"])); p.update(unit=r["unit"], md=float(r["top_md_m"])); tops.append(p)
    path = [dict(md=float(md), **locate(md)) for md in np.arange(0, 3037.2, 20)]
    zone = [t for t in tops if t["unit"] == "Zeeland Formation"][0]
    z, tg = depth_time_table(vcdp, vt, vv, zone["cdp"])
    depth_axis = [dict(depth=int(d), twt=float(np.interp(d, z, tg))) for d in range(0, 5001, 100)]
    # two-way time of the Someren target depth range along the whole line, from the same velocity functions
    band_km, band_top, band_base = [], [], []
    for k in np.unique(vcdp):
        i = int(np.abs(cdp_all - k).argmin())
        zz, tt = depth_time_table(vcdp, vt, vv, k)
        band_km.append(float(km_all[i])); band_top.append(float(np.interp(C.SOMEREN_TARGET_DEPTH_M[0], zz, tt)))
        band_base.append(float(np.interp(C.SOMEREN_TARGET_DEPTH_M[1], zz, tt)))
    mid_km = sum(C.SOMEREN_KM) / 2; i_mid = int(np.abs(km_all - mid_km).argmin())
    zz, tt = depth_time_table(vcdp, vt, vv, cdp_all[i_mid])
    someren_axis = dict(km=round(float(km_all[i_mid]), 2), axis=[dict(depth=int(d), twt=float(np.interp(d, zz, tt))) for d in range(0, 5001, 100)])
    depth_band = dict(km=band_km, top=band_top, base=band_base, depth_m=list(C.SOMEREN_TARGET_DEPTH_M))
    json.dump(dict(tops=tops, path=path, depth_axis=depth_axis, depth_band=depth_band, someren_axis=someren_axis), open(C.WORK / "step2_well.json", "w"), indent=1)
    for t in tops:
        print(f'{t["unit"]:38s} TWT {t["twt"]:.3f} s  km {t["km"]:.2f}  offset {t["offset_m"]:.0f} m')

if __name__ == "__main__":
    main()
