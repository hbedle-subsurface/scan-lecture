# The SCAN029 case

An interactive board for a 45-minute colloquium on seismic attributes, unsupervised seismic facies classification, and explaining what a machine learning model relies on. It uses one open 2D seismic line from the SCAN geothermal program in the Netherlands and the CAL-GT-04 geothermal well near Venlo.

The page loads precomputed data only, so it runs on GitHub Pages or any static web server.

## Stages

1. **Line and well.** The PreSTM full stack from 15 to 40 km along SCAN029, the CAL-GT-04 well path projected onto the line, formation tops, six horizons tracked from the well tie, and formation shading. Well control can be hidden to view the section as if no well existed. Clicking the section shows the trace at that location.
2. **Attributes.** Eleven attributes overlaid on the seismic with fixed color scales.
3. **Build a SOM.** Any combination of attributes and a SOM of 4 to 100 neurons, trained in the browser. The section is colored by each sample's neuron on a 2D color bar, the neuron grid shows how many samples each neuron holds, pairs of chosen attributes that correlate at 0.8 or more are listed, and every run is kept so runs can be compared.
4. **SHAP.** For the current run, how far each attribute moves samples across the map on average, and, for a clicked sample, the path from the average position to the sample's neuron built from each attribute's SHAP value.
5. **Verdict.** The current SOM with the well control returned.

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

and the three attributes computed in AASPI, exported as SEG-Y on CDP 5500–16500 and 0–2 s (CDP number in byte 21), in `raw/aaspi/`: `relative_acoustic_impedance_hb_1.segy`, `rms_amplitude_hb_1.segy`, `AVT_hb_1.segy`.

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
| 4 | `step4_attributes.py` | Attributes on a 20 m grid, including the AASPI attributes |
| 6 | `step6_export.py` | `data/meta.json` and binary arrays |

Well inputs are in `python/inputs/`: formation tops from NLOG, and deviation survey stations (a subset of the NLOG survey stations; the full survey can replace the file in the same column format).

## Methods and limitations

**Gain.** The PreSTM stacks have no gain applied after migration. One gain curve, the inverse of the smoothed median RMS amplitude at each time, is applied to every trace, so lateral amplitude differences are preserved.

**Well tie.** CAL-GT-04 has no sonic log, density log or checkshot. Tops are converted to two-way time with Dix interval velocities from the PreSTM migration velocities, and the well path is projected to the nearest CDP using the deviation survey. The reservoir section of the well lies 0.9–1.3 km from the line. The depth scale on the right of the section is valid only at the well.

**Horizons.** Horizons follow local reflector dip from a structure tensor, starting at the reflection nearest each projected top. The Rupel Clay and Chalk tops are tracked from 30 to 39 km. Below the Chalk the reflectors are dipping and discontinuous and the tracker drifts across events, so the Zechstein, Epen, Zeeland and Bosscheveld tops are shown only within 1 km of the well tie. Dashed segments of the Zeeland top, and the whole Bosscheveld top, are placed at the time thickness measured at the well below the horizon above.

Interpreted horizons replace the automatic ones. Opening the page with `?pick` at the end of the address adds a pick panel to stage 1: clicks add points to the selected horizon (optionally moved to the nearest trough or peak matching the horizon polarity within ±8 ms), shift-click removes the nearest point, and "Download picks" saves `horizon_picks.json`. Placing that file in `data/` makes the page use the picks, linearly interpolated between points, for every visitor.

**Attributes.** Relative acoustic impedance, RMS amplitude and the amplitude volume transform were computed in AASPI; envelope, instantaneous frequency, sweetness, spectral ratio, apparent dip, dip variability, coherence and far minus near were computed in `step4_attributes.py`. The AASPI attributes RMS amplitude receives the same time-only gain as the seismic; relative acoustic impedance and AVT are each divided by their own median RMS at each time, because integration and phase rotation change how their amplitude varies with time. Relative acoustic impedance is multiplied by −1 because an increase in acoustic impedance is a negative number in this dataset. RMS amplitude and envelope correlate at 1.00 on this line, and sweetness correlates with both at 0.97–0.98, so only RMS amplitude is used in the SOM presets.

Attributes other than relative acoustic impedance and AVT are averaged over a 140 m by 62 ms window so they describe seismic facies character rather than individual reflections. Relative acoustic impedance and AVT are zero-mean band-limited traces, so they are averaged over the same 140 m laterally but only 10 ms vertically. Geometric attributes on a 2D line measure apparent dip along the line only. Instantaneous frequency and the spectral ratio decrease with travel time as higher frequencies are attenuated, so part of their variation follows depth. In the Carboniferous section the near and far stacks correlate at about 0.2, so the far minus near attribute contains a large noise component there.

**SOM.** The SOM (Kohonen, 1982) runs in a Web Worker (`js/som-worker.js`). The chosen attributes are converted to z-scores over the 15–40 km, 0.15–1.70 s window. Prototypes start on the plane of the first two principal components, which keeps the map orientation similar between runs, and are trained on 20,000 random samples with a Gaussian neighborhood that shrinks from half the map width to 0.5 neurons. A fixed random seed makes the same settings give the same map. Every sample is then assigned to its closest prototype. Neuron colors come from a 2D color bar, so neighboring neurons, which hold similar attribute combinations, have similar colors.

**SHAP.** The explained output is a sample's position on the SOM grid, which sets its color: the average grid position of all neurons, weighted by exp(−squared distance / τ), where τ is the median squared distance from a sample to its closest neuron. SHAP values (Lundberg and Lee, 2017) are estimated by sampling random attribute orderings, each with a randomly chosen training sample as the background (Štrumbelj and Kononenko, 2014): 8 orderings per sample for the 400 samples in the global importance, and 400 orderings for a clicked sample. The average position plus the SHAP values gives the sample's position. Attributes that correlate share credit, so each of several near-duplicate attributes can show a small SHAP value while together they have a large effect.

## Data sources

- Seismic: SCAN 2D line L2EBN2020ASCAN029, acquired 2020 and processed 2021 for EBN and TNO, available through NLOG (nlog.nl).
- Well: CAL-GT-04, Californië Lipzig Gielen Geothermie B.V., formation tops and deviation survey from NLOG.

## References

- Becquey, M., M. Lavergne, and C. Willm, 1979, Acoustic impedance logs computed from seismic traces: Geophysics, 44, 1485–1501.
- Bakker, P., 2002, Image structure analysis for seismic interpretation: PhD thesis, Delft University of Technology.
- Bulhões, E. M., and W. N. Amorim, 2005, Princípio da SismoCamada Elementar e sua aplicação à Técnica Volume de Amplitudes (tecVA): 9th International Congress of the Brazilian Geophysical Society.
- Chopra, S., and K. J. Marfurt, 2007, Seismic attributes for prospect identification and reservoir characterization: SEG.
- Hart, B. S., 2008, Channel detection in 3-D seismic data using sweetness: AAPG Bulletin, 92, 733–742.
- Kohonen, T., 1982, Self-organized formation of topologically correct feature maps: Biological Cybernetics, 43, 59–69.
- Lundberg, S. M., and S.-I. Lee, 2017, A unified approach to interpreting model predictions: Advances in Neural Information Processing Systems, 30, 4765–4774.
- Štrumbelj, E., and I. Kononenko, 2014, Explaining prediction models and individual predictions with feature contributions: Knowledge and Information Systems, 41, 647–665.
- Marfurt, K. J., R. L. Kirlin, S. L. Farmer, and M. S. Bahorich, 1998, 3-D seismic attributes using a semblance-based coherency algorithm: Geophysics, 63, 1150–1165.
- Partyka, G., J. Gridley, and J. Lopez, 1999, Interpretational applications of spectral decomposition in reservoir characterization: The Leading Edge, 18, 353–360.
- Radovich, B. J., and R. B. Oliveros, 1998, 3-D sequence interpretation of seismic instantaneous attributes from the Gorgon field: The Leading Edge, 17, 1286–1293.
- Randen, T., E. Monsen, C. Signer, A. Abrahamsen, J. O. Hansen, T. Sæter, and J. Schlaf, 2000, Three-dimensional texture attributes for seismic data analysis: SEG Technical Program Expanded Abstracts.
- Rutherford, S. R., and R. H. Williams, 1989, Amplitude-versus-offset variations in gas sands: Geophysics, 54, 680–688.
- Shuey, R. T., 1985, A simplification of the Zoeppritz equations: Geophysics, 50, 609–614.
- Taner, M. T., F. Koehler, and R. E. Sheriff, 1979, Complex seismic trace analysis: Geophysics, 44, 1041–1063.

## License

Content is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Heather Bedle, University of Oklahoma (hbedle@ou.edu, ORCID 0000-0003-3010-0195).
