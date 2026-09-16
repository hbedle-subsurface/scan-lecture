# The SCAN029 case

An interactive board for a 45-minute colloquium on seismic attributes, unsupervised seismic facies classification, and explaining what a machine learning model relies on. It uses one open 2D seismic line from the SCAN geothermal program in the Netherlands and the CAL-GT-04 geothermal well near Venlo.

The page loads precomputed data only, so it runs on GitHub Pages or any static web server.

## Stages

1. **Line and well.** The PreSTM full stack from 15 to 40 km along SCAN029, the CAL-GT-04 well path projected onto the line, formation tops, six horizons tracked from the well tie, and formation shading. Well control can be hidden to view the section as if no well existed. Clicking the section shows the trace at that location.
2. **Attributes.** Nine attributes overlaid on the seismic with fixed color scales.
3. **SOM facies.** Class maps from self-organizing maps trained on four attribute presets, with the mean attribute values of each class.
4. **SHAP.** In progress.
5. **Verdict.** SOM classes shown with the well control returned.

## Running locally

The page fetches binary files, so it has to be served over http rather than opened as a file:

```
python -m http.server
```

then open `http://localhost:8000`.

## Rebuilding the data

Place these NLOG files in `raw/` (not committed):

- `L2EBN2020ASCAN029_PreSTM_final_full.sgy`
- `L2EBN2020ASCAN029_PreSTM_final_near.sgy`
- `L2EBN2020ASCAN029_PreSTM_final_far.sgy`
- `L2EBN2020ASCAN029_PreSTM_velocities_migration_ascii.txt`

Then:

```
cd python
pip install -r requirements.txt
python build_data.py
```

| Step | Script | Output |
|---|---|---|
| 1 | `step1_read.py` | 15–40 km of the full, near and far stacks with one time-only gain |
| 2 | `step2_well.py` | Well path and tops in two-way time |
| 3 | `step3_horizons.py` | Six horizons, 30–39 km |
| 4 | `step4_attributes.py` | Attributes on a 20 m grid |
| 5 | `step5_som.py` | SOM classes for four presets |
| 6 | `step6_export.py` | `data/meta.json` and binary arrays |

Well inputs are in `python/inputs/`: formation tops from NLOG, and deviation survey stations (a subset of the NLOG survey stations; the full survey can replace the file in the same column format).

## Methods and limitations

**Gain.** The PreSTM stacks have no gain applied after migration. One gain curve, the inverse of the smoothed median RMS amplitude at each time, is applied to every trace, so lateral amplitude differences are preserved.

**Well tie.** CAL-GT-04 has no sonic log, density log or checkshot. Tops are converted to two-way time with Dix interval velocities from the PreSTM migration velocities, and the well path is projected to the nearest CDP using the deviation survey. The reservoir section of the well lies 0.9–1.3 km from the line. The depth scale on the right of the section is valid only at the well.

**Horizons.** Horizons follow local reflector dip from a structure tensor, starting at the reflection nearest each projected top. Solid segments were tracked on reflections. Dashed segments of the Zeeland top, and the whole Bosscheveld top, are placed at the time thickness measured at the well below the horizon above, adjusted to the nearest matching reflection. Those segments are interpretations for display.

**Attributes.** All attributes are averaged over a 140 m by 62 ms window so they describe seismic facies character rather than individual reflections. Geometric attributes on a 2D line measure apparent dip along the line only. Instantaneous frequency and the spectral ratio decrease with travel time as higher frequencies are attenuated, so part of their variation follows depth. In the Carboniferous section the near and far stacks correlate at about 0.2, so the far minus near attribute contains a large noise component there.

**SOM.** Attributes are converted to z-scores over the 15–40 km, 0.15–1.70 s window. A 10 × 10 SOM is trained on 60,000 random samples, its prototype vectors are grouped into 8 classes with k-means, and classes are numbered by increasing mean two-way time.

## Data sources

- Seismic: SCAN 2D line L2EBN2020ASCAN029, acquired 2020 and processed 2021 for EBN and TNO, available through NLOG (nlog.nl).
- Well: CAL-GT-04, Californië Lipzig Gielen Geothermie B.V., formation tops and deviation survey from NLOG.

## References

- Bakker, P., 2002, Image structure analysis for seismic interpretation: PhD thesis, Delft University of Technology.
- Chopra, S., and K. J. Marfurt, 2007, Seismic attributes for prospect identification and reservoir characterization: SEG.
- Hart, B. S., 2008, Channel detection in 3-D seismic data using sweetness: AAPG Bulletin, 92, 733–742.
- Kohonen, T., 1982, Self-organized formation of topologically correct feature maps: Biological Cybernetics, 43, 59–69.
- Marfurt, K. J., R. L. Kirlin, S. L. Farmer, and M. S. Bahorich, 1998, 3-D seismic attributes using a semblance-based coherency algorithm: Geophysics, 63, 1150–1165.
- Partyka, G., J. Gridley, and J. Lopez, 1999, Interpretational applications of spectral decomposition in reservoir characterization: The Leading Edge, 18, 353–360.
- Radovich, B. J., and R. B. Oliveros, 1998, 3-D sequence interpretation of seismic instantaneous attributes from the Gorgon field: The Leading Edge, 17, 1286–1293.
- Randen, T., E. Monsen, C. Signer, A. Abrahamsen, J. O. Hansen, T. Sæter, and J. Schlaf, 2000, Three-dimensional texture attributes for seismic data analysis: SEG Technical Program Expanded Abstracts.
- Rutherford, S. R., and R. H. Williams, 1989, Amplitude-versus-offset variations in gas sands: Geophysics, 54, 680–688.
- Shuey, R. T., 1985, A simplification of the Zoeppritz equations: Geophysics, 50, 609–614.
- Taner, M. T., F. Koehler, and R. E. Sheriff, 1979, Complex seismic trace analysis: Geophysics, 44, 1041–1063.

## License

Content is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Heather Bedle, University of Oklahoma (hbedle@ou.edu, ORCID 0000-0003-3010-0195).
