"""Run the data workflow: raw SEG-Y, velocity and AASPI files in raw/ -> web files in data/.
The SOM and SHAP values are computed in the browser (js/som-worker.js)."""
import step1_read, step2_well, step3_horizons, step4_attributes, step6_export

for step in (step1_read, step2_well, step3_horizons, step4_attributes, step6_export):
    print(f"--- {step.__name__}")
    step.main()
