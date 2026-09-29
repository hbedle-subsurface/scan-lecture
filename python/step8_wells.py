"""Step 8: every attribute along the well paths, for the guided exercise.

For CAL-GT-04 and ASTEN-GT-02, the attribute values are read from the exported attribute files (data/attr_*.bin) at
the 20 m trace nearest the projected well path at each 4 ms sample, and stored with the formation of each sample from
the well tops and the measured depth every 100 m, so the page can draw the attributes as tracks along each well.
Runs from the files in data/ alone.
"""
import json
import numpy as np
import config as C


def main():
    meta = json.load(open(C.DATA / "meta.json"))
    g = meta["grid"]
    tg = meta["t_min"] + np.arange(g["nt"]) * g["dt"]
    attrs = {k: np.fromfile(C.DATA / f"attr_{k}.bin", np.uint8).reshape(g["nx"], g["nt"]) for k in meta["attributes"]}
    out = []
    for w in meta["wells"]:
        if w["name"] not in ("CAL-GT-04", "ASTEN-GT-02"): continue
        md = np.array([p["md"] for p in w["path"]]); tw = np.array([p["twt"] for p in w["path"]]); km = np.array([p["km"] for p in w["path"]])
        keep = np.r_[True, np.diff(tw) > 0]; md, tw, km = md[keep], tw[keep], km[keep]
        j = np.where((tg >= max(tw[0], meta["t_min"])) & (tg <= tw[-1]))[0]
        kj = np.interp(tg[j], tw, km)
        i = np.clip(np.round((kj - g["km_min"]) / (g["dx_m"] / 1000)).astype(int), 0, g["nx"] - 1)
        vals = {k: [round(float(A["min"] + v / 255 * (A["max"] - A["min"])), 4) for v in attrs[k][i, j]] for k, A in meta["attributes"].items()}
        tops = sorted(w["tops"], key=lambda t: t["md"])
        units = [dict(name=t["unit"], top_twt=round(float(np.interp(t["md"], md, tw)), 4),
                      base_twt=round(float(np.interp(tops[n + 1]["md"], md, tw)) if n + 1 < len(tops) else float(tw[-1]), 4),
                      top_md=t["md"]) for n, t in enumerate(tops)]
        ticks = [dict(md=int(d), twt=round(float(np.interp(d, md, tw)), 4)) for d in range(0, int(md[-1]) + 1, 100)]
        out.append(dict(name=w["name"], j=[int(x) for x in j], km=[round(float(x), 4) for x in kj], attrs=vals, units=units, depth_ticks=ticks))
        print(f"{w['name']}: {len(j)} samples, {tg[j[0]]:.3f}-{tg[j[-1]]:.3f} s")
    json.dump(dict(wells=out), open(C.DATA / "wells.json", "w"), separators=(",", ":"), ensure_ascii=False)


if __name__ == "__main__":
    main()
