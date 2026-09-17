"""Step 6: write the compact files the web page loads (data/)."""
import json
import numpy as np
from matplotlib import colormaps
import config as C

ATTRIBUTES = {
    "rms_amplitude": dict(label="RMS amplitude", unit="relative", cmap="magma", family="Single-trace",
        measures="Root-mean-square of the reflection amplitude in a window along each trace, averaged over 140 m by 62 ms.",
        geology="Higher values often correspond to stronger impedance contrasts, such as interbedded lithologies or evaporite and carbonate boundaries.",
        source="Standard amplitude statistic; see Chopra and Marfurt (2007)."),
    "relative_acoustic_impedance": dict(label="Relative acoustic impedance", unit="relative", cmap="RdBu_r", family="Single-trace",
        measures="Running integral of the zero-phase trace, which approximates band-limited changes in acoustic impedance. Sign set so that higher impedance is positive.",
        geology="Layers with higher impedance than their surroundings, such as tight carbonate or cemented sandstone below shale, often appear as positive values.",
        source="Trace integration after Becquey, Lavergne and Willm (1979)."),
    "amplitude_volume_transform": dict(label="Amplitude volume transform (AVT)", unit="relative", cmap="RdBu_r", family="Single-trace",
        measures="RMS amplitude in a short window, rotated by -90 degrees in phase, giving a band-limited trace that follows reflection-amplitude packages.",
        geology="Often used to outline bodies with a distinct amplitude character, such as channel fills or carbonate buildups.",
        source="Bulhões and Amorim (2005)."),
    "envelope": dict(label="Envelope", unit="relative", cmap="magma", family="Single-trace",
        measures="Magnitude of the complex trace, averaged in the window. It is independent of phase.",
        geology="Often used to map reflection strength and bright spots.",
        source="Taner, Koehler and Sheriff (1979)."),
    "instantaneous_frequency": dict(label="Instantaneous frequency", unit="Hz", cmap="viridis", family="Single-trace",
        measures="Rate of change of instantaneous phase, weighted by envelope in the window.",
        geology="Often responds to bed thickness and tuning. It also decreases with travel time as higher frequencies are attenuated.",
        source="Taner, Koehler and Sheriff (1979)."),
    "sweetness": dict(label="Sweetness", unit="relative", cmap="magma", family="Single-trace",
        measures="Envelope divided by the square root of instantaneous frequency.",
        geology="Originally used to highlight thick, high-amplitude sand bodies in clastic sections.",
        source="Radovich and Oliveros (1998); Hart (2008)."),
    "spectral_ratio": dict(label="Spectral ratio (45-65 Hz / 10-20 Hz)", unit="log ratio", cmap="viridis", family="Single-trace",
        measures="Natural log of the high-band envelope divided by the low-band envelope.",
        geology="Varies with bed thickness and attenuation. Like instantaneous frequency, it trends lower with travel time.",
        source="Band-limited decomposition after Partyka, Gridley and Lopez (1999)."),
    "apparent_dip": dict(label="Apparent dip", unit="ms per 100 m", cmap="cividis", family="Geometric",
        measures="Magnitude of reflector dip along the line, from the local structure tensor.",
        geology="Highlights tilted strata, folds and fault drag. On a 2D line it is dip in the line direction only.",
        source="Structure-tensor dip after Bakker (2002) and Randen et al. (2000)."),
    "dip_variability": dict(label="Dip variability", unit="samples per trace", cmap="cividis", family="Geometric",
        measures="Standard deviation of local dip in the window.",
        geology="Higher where reflector orientation changes over short distances, as around faults, flexures or chaotic facies.",
        source="Local statistic of structure-tensor dip."),
    "coherence": dict(label="Coherence (semblance)", unit="0-1", cmap="gray", family="Geometric",
        measures="Similarity of five neighboring traces compared along local dip.",
        geology="Low values often mark faults, fractures and disrupted or chaotic reflections.",
        source="Marfurt et al. (1998)."),
    "full_stack_amplitude": dict(label="Full stack amplitude", unit="relative", cmap="RdBu_r", family="Stack amplitude",
        measures="Amplitude of the full stack after the time-only gain, averaged over 140 m along the line but not in time.",
        geology="Sign and strength of reflections, which follow changes in acoustic impedance across bed boundaries.",
        source="PreSTM full stack, SCAN029."),
    "near_stack_amplitude": dict(label="Near stack amplitude", unit="relative", cmap="RdBu_r", family="Stack amplitude",
        measures="Amplitude of the near-angle stack, with the same gain and lateral averaging as the full stack.",
        geology="Small incidence angles, where amplitude depends mostly on the change in acoustic impedance.",
        source="PreSTM near-angle stack, SCAN029."),
    "mid_stack_amplitude": dict(label="Mid stack amplitude", unit="relative", cmap="RdBu_r", family="Stack amplitude",
        measures="Amplitude of the mid-angle stack, with the same gain and lateral averaging as the full stack.",
        geology="Intermediate incidence angles.",
        source="PreSTM mid-angle stack, SCAN029."),
    "far_stack_amplitude": dict(label="Far stack amplitude", unit="relative", cmap="RdBu_r", family="Stack amplitude",
        measures="Amplitude of the far-angle stack, with the same gain and lateral averaging as the full stack.",
        geology="Larger incidence angles, where amplitude also depends on the change in shear-wave velocity.",
        source="PreSTM far-angle stack, SCAN029."),
    "far_minus_near": dict(label="Far minus near envelope", unit="normalized", cmap="RdBu_r", family="AVO",
        measures="Difference between far- and near-angle-stack envelopes divided by their sum.",
        geology="Changes in amplitude with angle depend on elastic contrasts. In the Carboniferous here the near and far stacks correlate poorly, so noise contributes strongly.",
        source="Amplitude-versus-angle concept after Shuey (1985) and Rutherford and Williams (1989)."),
}
UNITS = [
    dict(name="North Sea Group", top=None, base="Houthem Formation", color="#d9b75f"),
    dict(name="Chalk Group and underlying sandstone", top="Houthem Formation", base="Zechstein Upper Claystone Formation", color="#8fb996"),
    dict(name="Zechstein", top="Zechstein Upper Claystone Formation", base="Epen Formation", color="#a98cc4"),
    dict(name="Epen Formation (Namurian shale)", top="Epen Formation", base="Zeeland Formation", color="#7f9fb3"),
    dict(name="Zeeland Formation (Dinantian carbonate, geothermal reservoir)", top="Zeeland Formation", base="Bosscheveld Formation", color="#d1495b", reservoir=True),
]
HORIZON_STYLE = {
    "Rupel Clay Member": dict(label="Top Rupel Clay", color="#e8894a"),
    "Houthem Formation": dict(label="Top Chalk (Houthem Fm)", color="#e3c567"),
    "Zechstein Upper Claystone Formation": dict(label="Top Zechstein", color="#b892ff"),
    "Epen Formation": dict(label="Top Carboniferous (Epen Fm)", color="#8ecae6"),
    "Zeeland Formation": dict(label="Top Zeeland Fm", color="#ff5a5f"),
    "Bosscheveld Formation": dict(label="Top Bosscheveld Fm", color="#c9184a"),
}

def lut(name):
    return (colormaps[name](np.linspace(0, 1, 256))[:, :3] * 255).round().astype(int).tolist()

def main():
    C.DATA.mkdir(exist_ok=True)
    s1 = np.load(C.WORK / "step1_read.npz"); well = json.load(open(C.WORK / "step2_well.json"))
    hz = json.load(open(C.WORK / "step3_horizons.json")); A = np.load(C.WORK / "step4_attributes.npz")
    t0, t1 = int(C.T_MIN / C.DT), int(C.T_MAX / C.DT)

    sec = s1["full"][:, t0:t1]; clip = float(np.percentile(np.abs(sec), 99.0))
    np.clip(np.round(sec / clip * 127), -127, 127).astype(np.int8).tofile(C.DATA / "section.bin")

    attrs = {}
    for k, m in ATTRIBUTES.items():
        v = A[k][:, t0:t1:C.ATT_TSTEP]; lo, hi = (float(x) for x in np.percentile(v, [1, 99]))
        if m["cmap"] == "RdBu_r": hi = max(abs(lo), abs(hi)); lo = -hi
        np.clip(np.round((v - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8).tofile(C.DATA / f"attr_{k}.bin")
        attrs[k] = dict(m, min=round(lo, 4), max=round(hi, 4), lut=lut(m["cmap"]))
    meta = dict(
        line="SCAN029 (L2EBN2020ASCAN029)", km_min=float(s1["sec_km"][0]), km_max=float(s1["sec_km"][-1]),
        t_min=t0 * C.DT, t_max=t1 * C.DT, dt=C.DT, nt=t1 - t0,
        section=dict(file="section.bin", nx=int(sec.shape[0]), dx_m=10),
        grid=dict(nx=int(A["coherence"].shape[0]), dx_m=20, km_min=float(s1["att_km"][0]), km_max=float(s1["att_km"][-1]),
                  nt=len(range(t0, t1, C.ATT_TSTEP)), dt=C.DT * C.ATT_TSTEP),
        wells=[dict(name=w["name"], estimated_path=w["estimated_path"], kickoff_md=w["kickoff_md"], events=w["events"],
                    position_from_map=w["position_from_map"], year=w["year"],
                    path=[dict(md=round(q["md"], 1), km=round(q["km"], 4), twt=round(q["twt"], 4), offset_m=round(q["offset_m"])) for q in w["path"]],
                    tops=[dict(unit=t["unit"], md=t["md"], tvdss=round(t["tvdss"], 1), km=round(t["km"], 4), twt=round(t["twt"], 4), offset_m=round(t["offset_m"])) for t in w["tops"]])
               for w in well["wells"]],
        someren=dict(km=list(C.SOMEREN_KM), depth_band=well["depth_band"], depth_axis=well["someren_axis"]),
        well=dict(name=C.WELL["name"],
                  path=[dict(km=round(p["km"], 4), twt=round(p["twt"], 4), offset_m=round(p["offset_m"])) for p in well["path"]],
                  tops=[dict(unit=t["unit"], md=t["md"], tvdss=round(t["tvdss"], 1), km=round(t["km"], 4), twt=round(t["twt"], 4), offset_m=round(t["offset_m"])) for t in well["tops"]],
                  depth_axis=well["depth_axis"]),
        horizons=dict(km=[round(x, 4) for x in hz["km"]], items=[dict(unit=u, **HORIZON_STYLE[u], **hz["horizons"][u]) for u in HORIZON_STYLE]),
        units=UNITS, attributes=attrs,
    )
    json.dump(meta, open(C.DATA / "meta.json", "w"), separators=(",", ":"))
    print("clip", clip, "files:", sorted(p.name for p in C.DATA.iterdir()))

if __name__ == "__main__":
    main()
