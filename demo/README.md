# Survprompt demo

A self-contained demo that runs without MSK-CHORD access and without Azure OpenAI
credentials.

> **The cohort here is entirely synthetic.** `data/mskchord/nsclc_dx_1st_seq_OS.csv`
> holds 200 generated records with `SYNTH-####` identifiers and contains no real
> patient data. See [data/mskchord/SYNTHETIC_DATA_NOTICE.txt](data/mskchord/SYNTHETIC_DATA_NOTICE.txt).
> Numbers produced here do not correspond to any result in the manuscript.

## Contents

| File | Purpose |
|---|---|
| `make_demo_cohort.py` | Regenerates the synthetic cohort (fixed seed, reproducible) |
| `data/mskchord/nsclc_dx_1st_seq_OS.csv` | The committed 200-patient synthetic cohort |
| `render_prompt.py` | Prints a clinical vignette and its prompts — no API call |

## Running it

Both baseline commands need `BASE_DIR` pointed at this directory, because the
package resolves input as `$BASE_DIR/data/$DATA_NAME/{cancer}_dx_1st_seq_OS.csv`.

```bash
export BASE_DIR="$PWD/demo"

python -m survprompt.experiments.experiments rsf_baseline --data_name mskchord --cancer_of_interest nsclc
python -m survprompt.experiments.experiments cox_baseline --data_name mskchord --cancer_of_interest nsclc
```

Outputs land under `demo/results/` (git-ignored):

```
demo/results/predictions/mskchord/{rsf,cox}_baseline_nsclc_fullfts_pred_fold{0..4}.csv
demo/results/metrics/cindex/mskchord/{rsf,cox}_baseline_nsclc_fullfts_pred_fold{0..4}.json
demo/results/models/mskchord/{rsf,cox}_baseline_nsclc_fullfts_model_fold{0..4}.joblib
```

The prompt demo needs no environment variables at all:

```bash
python demo/render_prompt.py
python demo/render_prompt.py --prompting_task TTE_OS --patient 7
```

Expected output and runtimes are documented in the main [README](../README.md#demo).

## Regenerating the cohort

```bash
python demo/make_demo_cohort.py
```

The seed is fixed, so this rewrites the committed CSV byte for byte.
