# Survprompt

[![Preprint](https://img.shields.io/badge/Preprint-Coming%20Soon-blue)](#citation)
[![Code License](https://img.shields.io/badge/Code%20License-MIT-green)](LICENSE)
[![Data](https://img.shields.io/badge/MSK%20CHORD-Data-228B22)](https://datacatalog.mskcc.org/dataset/11458)

Survprompt is a Python package for prompting language models to estimate survival outcomes from clinical vignettes derived from structured oncology data. It includes tools for running zero-shot LLM survival prediction experiments and traditional survival-model baselines.

This project is intended for research use. It is not a clinical device and should not be used to make medical decisions.

## Requirements

Survprompt has been developed and tested with Python 3.10 on Ubuntu 24.04.4 LTS. The package
dependencies are pinned to the versions used for the manuscript results in `pyproject.toml`
and are installed with the package.

A conda environment is recommended:

```bash
conda create -n survprompt python=3.10
conda activate survprompt
```

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/microsoft/survprompt.git
cd survprompt
pip install -e .
```

Typical install time is under a minute on a fast connection, longer on a slower link.

## Demo

A worked example on a small **synthetic** cohort, requiring no MSK-CHORD access and no Azure
OpenAI credentials. The 200 records in [`demo/data/mskchord/`](demo/data/mskchord/) are
generated, contain no real patient data, and do not reproduce any manuscript result.

### Survival baselines

Input is resolved as `$BASE_DIR/data/$DATA_NAME/{cancer}_dx_1st_seq_OS.csv`, so point
`BASE_DIR` at the demo directory:

```bash
export BASE_DIR="$PWD/demo"

python -m survprompt.experiments.experiments rsf_baseline --data_name mskchord --cancer_of_interest nsclc
python -m survprompt.experiments.experiments cox_baseline --data_name mskchord --cancer_of_interest nsclc
```

Each command logs its five cross-validation folds and writes predictions, c-index scores and
fitted models under `demo/results/` (15 files per model). The recorded c-indices are:

| Model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean |
|---|---|---|---|---|---|---|
| `rsf_baseline` | 0.612 | 0.672 | 0.755 | 0.695 | 0.649 | **0.677** |
| `cox_baseline` | 0.638 | 0.679 | 0.705 | 0.701 | 0.648 | **0.674** |

Feature column order comes from Python `set` iteration, so the RSF mean moves by about
± 0.005 between processes despite its fixed `random_state`; export `PYTHONHASHSEED=0` to
reproduce the table exactly.

### Prompt construction, without an API call

Renders the clinical vignette and the exact prompts Survprompt would send to a model.
Nothing is transmitted and no credentials are read:

```bash
python demo/render_prompt.py
python demo/render_prompt.py --prompting_task TTE_OS --patient 7
```

It prints the patient's ground-truth outcome, the vignette, and the full system and user
prompts:

```
The patient is a 65 years old White female with a history of smoking. The patient
has been diagnosed with stage 4 non-squamous cell, adenocarcinoma non-small cell
lung cancer (NSCLC). ...
```

### Expected run time

Measured on Ubuntu 24.04 with `OMP_NUM_THREADS=4`.

| Step | Time |
|---|---|
| `rsf_baseline` (5 folds, 1000 trees) | ~26 s |
| `cox_baseline` (5 folds) | ~2 s |
| `demo/render_prompt.py` | ~2 s |
| `demo/make_demo_cohort.py` | < 1 s |

## Configuration

Copy [`.env.example`](.env.example) to `.env` in the repository root and fill it in.

`BASE_DIR` is the important one: input data is read from beneath it, and all results, saved
models and logs are written beneath it. `INPUT_DIR` is accepted by the baseline and ablation
command-line entry points, but note that the `mskchord` data loader resolves input paths from
`BASE_DIR`, not from `INPUT_DIR`.

If `AZURE_OPENAI_API_KEY` is set it is used directly; otherwise the client falls back to
Entra ID credentials from the Azure CLI. Neither is needed for the baselines or the demo.

## Data

MSK-CHORD data access is managed by the dataset provider. Download the data from the
[MSK-CHORD data catalog](https://datacatalog.mskcc.org/dataset/11458) and place the prepared
files where the loader expects them:

```
$BASE_DIR/data/mskchord/{cancer}_dx_1st_seq_OS.csv
```

with `{cancer}` one of `nsclc`, `brca`, `crc`, `panc`, `prostate`.

To run on your own cohort, each row needs `PATIENT_ID`, `entry`, `stop` and `dead` (models
are fitted on `stop - entry`), plus numeric feature columns as grouped in
`survprompt.data.utils.FEATURE_TYPE_TO_COLS`. See
[`demo/make_demo_cohort.py`](demo/make_demo_cohort.py) for a worked example that builds a
valid file from scratch.

## Usage

Run baseline experiments:

```bash
python -m survprompt.experiments.experiments rsf_baseline --data_name mskchord --cancer_of_interest nsclc
python -m survprompt.experiments.experiments cox_baseline --data_name mskchord --cancer_of_interest nsclc
```

Run all configured zero-shot experiments for a specific prompting task:

```bash
python -m survprompt.experiments.experiments zeroshot --data_name mskchord --cancer_of_interest nsclc --prompting_task SURV_PROB
```

Supported prompting tasks are:

- `SURV_PROB`: estimate survival probabilities over time.
- `TTE_OS`: estimate time to death / overall survival.

## Ablation Experiments

Run the public feature-ablation workflow for Survprompt and the RSF baseline:

```bash
python -m survprompt.experiments.interpretability.run_ablation --methods survprompt rsf --data-name mskchord --cancer nsclc --prompting_tasks SURV_PROB
```

Plot the resulting cMAE and c-index bar plots:

```bash
python -m survprompt.plots.ablation_bar_plots --data-name mskchord --cancer nsclc
```

The ablation runner resolves Survprompt runs through the experiment registry in `survprompt.experiments.experiments`.

## Package Layout

- `survprompt.defaults`: shared model names, prompt defaults, and plotting label constants.
- `survprompt.predictor`: LLM predictor implementation, clinical vignette formatting, prompt construction, API calls, and response parsing.
- `survprompt.predict_survival`: command-line and programmatic entrypoint for survival prediction runs.
- `survprompt.predictor_utils`: data loading, feature preparation, patient record formatting, and sampling utilities used by prediction workflows.
- `survprompt.prompt_utils`: Jinja prompt-template loading and rendering helpers.
- `survprompt.configs`: dataclass configuration objects for experiments and predictors, plus prompt templates.
- `survprompt.data`: data preparation utilities.
- `survprompt.experiments`: experiment registry, baseline, zero-shot, and feature-ablation experiment definitions.
- `survprompt.baselines`: Random Survival Forest and Cox baseline model runners.
- `survprompt.evaluation`: metrics for survival, calibration, and censored error, and support for survival-analysis utilities.
- `survprompt.plots`: plotting utilities for experiment metrics, Kaplan-Meier curves, ablations, and error analysis.
- `demo`: synthetic cohort, its generator, and a credential-free prompt rendering example.

## Citation

If you use Survprompt in research, cite the associated manuscript when available.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). This project has adopted the
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/);
see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

Released under the MIT License. See [LICENSE](LICENSE).

Portions of this project are derived from third-party code licensed under the Apache
License, Version 2.0; attribution appears in the header of each affected file.

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause
confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are
subject to those third-party's policies.
