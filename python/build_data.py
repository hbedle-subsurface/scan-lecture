"""Run the full data workflow: raw SEG-Y and well files in raw/ -> web files in data/."""
import step1_read, step2_well, step3_horizons, step4_attributes, step5_som, step6_export

for step in (step1_read, step2_well, step3_horizons, step4_attributes, step5_som, step6_export):
    print(f"--- {step.__name__}")
    step.main()
