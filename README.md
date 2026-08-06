# Survprompt

[![Preprint](https://img.shields.io/badge/Preprint-Coming%20Soon-blue)](#citation)
[![Code License](https://img.shields.io/badge/Code%20License-MIT-green)](LICENSE)
[![Data](https://img.shields.io/badge/MSK%20CHORD-Data-228B22)](https://datacatalog.mskcc.org/dataset/11458)

Survprompt is a Python package for prompting language models to estimate survival outcomes from clinical vignettes derived from structured oncology data. It includes tools for running zero-shot LLM survival prediction experiments and traditional survival-model baselines.

This project is intended for research use. It is not a clinical device and should not be used to make medical decisions.

## Requirements

Survprompt has been tested with Python 3.10 on Ubuntu. The package dependencies are defined in `pyproject.toml` and are installed with the package.

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

## Configuration

Create a `.env` file in the repository root:

```bash
BASE_DIR="/path/to/survprompt"
INPUT_DIR="/path/to/input/data"

AZURE_OPENAI_ENDPOINT="your_endpoint"
AZURE_OPENAI_API_VERSION="your_api_version"

# Authenticate with either an API key...
AZURE_OPENAI_API_KEY="your_api_key"
# ...or Entra ID via the Azure CLI (`az login`), in which case leave the key unset.
# Override the token scope only if your deployment sits behind a gateway that
# exposes its own application ID URI.
AZURE_OPENAI_TOKEN_SCOPE="https://cognitiveservices.azure.com/.default"
```

`BASE_DIR` should point to the repository root. `INPUT_DIR` should point to the directory containing the prepared MSK-CHORD input files.

If `AZURE_OPENAI_API_KEY` is set it is used directly; otherwise the client falls back
to Entra ID credentials from the Azure CLI.

## Data

MSK-CHORD data access is managed by the dataset provider. Download the data from the MSK-CHORD data catalog and place the prepared files under the directory referenced by `INPUT_DIR`.

The experiment code expects the supported cohort names used by the package, such as `nsclc`, `brca`, `crc`, `panc`, and `prostate`.

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
