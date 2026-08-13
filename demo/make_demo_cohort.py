"""Generate the synthetic NSCLC demo cohort shipped with Survprompt.

The output of this script is committed as ``demo/data/mskchord/nsclc_dx_1st_seq_OS.csv``
so that the demo in the README can be run without access to MSK-CHORD.

THE DATA PRODUCED HERE IS ENTIRELY SYNTHETIC. It is drawn from the simple
generative model below, contains no real patient records, and reproduces neither
the distributions nor the results of MSK-CHORD. It exists only to exercise the
code paths end to end.

The seed is fixed, so re-running this script reproduces the committed CSV exactly.

Usage:
    python demo/make_demo_cohort.py
"""

import os

import numpy as np
import pandas as pd

from survprompt.data.utils import FEATURE_TYPE_TO_COLS

# Cohort size. 200 patients keeps the file small (~130 KB) while staying large
# enough for the 5-fold cross-validation the baselines run: each fold trains on
# 160 patients against 73 candidate features. Smaller cohorts degrade badly --
# at 10 patients the Cox baseline fails to converge outright, and the random
# survival forest (min_samples_leaf=15) cannot split at all.
N_PATIENTS = 200
SEED = 20260811

# Baseline hazard, applied to a mean-centred linear predictor, so that this
# constant sets the cohort median overall survival directly (~1.4 years).
LAMBDA_0 = 1.0 / 750.0

# Log-hazard ratios for the generative model. These are invented, not estimated,
# but are signed in the clinically expected direction so the demo produces a
# discrimination score in a plausible range rather than coin-flip noise.
BETA = {
    "age_per_decade": 0.18,
    "stage_2": 0.30,
    "stage_3": 0.75,
    "stage_4": 1.25,
    "progressed": 0.45,
    "met_per_site": 0.22,
    "smoker": 0.25,
    "TP53": 0.30,
    "KRAS": 0.20,
    "EGFR": -0.45,
    "ALK": -0.35,
    "pdl1_high": -0.25,
}

# Genes that carry a plausible alteration frequency in NSCLC. Everything else in
# the panel is emitted as an all-zero column, mirroring the shape of the real
# per-cancer files (the Cox baseline drops zero-variance columns automatically).
NSCLC_GENE_FREQ = {
    "TP53": 0.46,
    "KRAS": 0.32,
    "EGFR": 0.23,
    "PIK3CA": 0.09,
    "BRAF": 0.07,
    "MET": 0.06,
    "ALK": 0.05,
    "ERBB2": 0.04,
    "PTEN": 0.04,
    "NRAS": 0.02,
    "RET": 0.02,
    "FGFR1": 0.02,
}


def _all_feature_columns():
    """Every feature column the baselines may ask for, taken from the package."""
    cols = []
    for group in ("demographics", "treatment", "stage", "met", "path", "lab", "genomics"):
        for col in FEATURE_TYPE_TO_COLS[group]:
            if col not in cols:
                cols.append(col)
    return cols


def build_cohort(n=N_PATIENTS, seed=SEED):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({"PATIENT_ID": [f"SYNTH-{i:04d}" for i in range(1, n + 1)]})

    # Start every feature column at zero, then fill in the ones that apply to
    # NSCLC. This guarantees the full 73-column feature space is present, which
    # the baselines require when run with features="all".
    for col in _all_feature_columns():
        df[col] = 0

    # -- Demographics ------------------------------------------------------
    df["AGE"] = np.clip(rng.normal(68, 9.5, n).round(), 35, 92).astype(int)
    df["MALE"] = (rng.random(n) < 0.48).astype(int)
    race = rng.choice(["white", "asian", "black", "other"], n, p=[0.72, 0.11, 0.11, 0.06])
    df["WHITE"] = (race == "white").astype(int)
    df["ASIAN"] = (race == "asian").astype(int)
    df["BLACK"] = (race == "black").astype(int)
    # SMOKER is read with math.isnan() during vignette construction, so it must
    # be a float column rather than an integer one.
    df["SMOKER"] = (rng.random(n) < 0.80).astype(float)

    # -- Treatment ---------------------------------------------------------
    df["ANY_PRIOR_TX"] = (rng.random(n) < 0.62).astype(int)

    # -- Stage and progression --------------------------------------------
    stage = rng.choice([1, 2, 3, 4], n, p=[0.14, 0.10, 0.20, 0.56])
    for s in (1, 2, 3, 4):
        df[f"STAGE {s}"] = (stage == s).astype(int)
    df["STAGE_IV_DX"] = (stage == 4).astype(int)
    progressed = np.where(stage == 4, rng.random(n) < 0.55, rng.random(n) < 0.38).astype(int)
    df["progressed"] = progressed
    df["STAGE_I-III_NOPROG"] = ((stage < 4) & (progressed == 0)).astype(int)
    df["STAGE_I-III_PROG"] = ((stage < 4) & (progressed == 1)).astype(int)

    # -- Metastatic sites --------------------------------------------------
    # All eight DMETS_DX_* columns must be present together: vignette rendering
    # indexes every one of them as soon as any single site column exists.
    met_freq = {
        "DMETS_DX_BONE": 0.30, "DMETS_DX_LUNG": 0.28, "DMETS_DX_LYMPH": 0.34,
        "DMETS_DX_BRAIN": 0.17, "DMETS_DX_LIVER": 0.14, "DMETS_DX_PLEURA": 0.16,
        "DMETS_DX_ADRENAL": 0.09, "DMETS_DX_OTHER": 0.11,
    }
    advanced = stage == 4
    for col, freq in met_freq.items():
        # Metastatic disease is concentrated in stage IV patients.
        p = np.where(advanced, freq, freq * 0.15)
        df[col] = (rng.random(n) < p).astype(int)
    n_met_sites = df[list(met_freq)].sum(axis=1).to_numpy()

    # -- Pathology ---------------------------------------------------------
    adeno = (rng.random(n) < 0.68).astype(int)
    df["ADENOCARCINOMA"] = adeno
    df["NONADENOCARCINOMA"] = 1 - adeno
    # SQUAMOUS is also read with math.isnan() during vignette construction.
    df["SQUAMOUS"] = np.where(adeno == 1, 0.0, (rng.random(n) < 0.75).astype(float))
    has_pdl1 = (rng.random(n) < 0.55).astype(int)
    df["HAS_PDL1"] = has_pdl1
    pdl1 = np.where(has_pdl1 == 1, rng.integers(0, 100, n), 0)
    df["PDL1"] = pdl1
    pdl1_high = ((has_pdl1 == 1) & (pdl1 >= 50)).astype(int)

    # -- Tumour markers ----------------------------------------------------
    # CEA is the only marker routinely measured in NSCLC; the prostate, breast
    # and colorectal markers stay at zero, as they do in the real NSCLC file.
    has_cea = (rng.random(n) < 0.35).astype(int)
    df["HAS_CEA"] = has_cea
    df["CEA"] = np.where(has_cea == 1, rng.gamma(2.0, 4.0, n).round(), 0).astype(int)
    df["MAX_CEA"] = (df["CEA"] * rng.uniform(1.0, 2.2, n)).round().astype(int)

    # -- Genomics ----------------------------------------------------------
    for gene, freq in NSCLC_GENE_FREQ.items():
        df[gene] = (rng.random(n) < freq).astype(int)

    # -- Survival outcome --------------------------------------------------
    # Exponential times from a log-linear hazard in the features above.
    log_hazard = (
        BETA["age_per_decade"] * (df["AGE"].to_numpy() - 65) / 10.0
        + BETA["stage_2"] * df["STAGE 2"].to_numpy()
        + BETA["stage_3"] * df["STAGE 3"].to_numpy()
        + BETA["stage_4"] * df["STAGE 4"].to_numpy()
        + BETA["progressed"] * progressed
        + BETA["met_per_site"] * n_met_sites
        + BETA["smoker"] * df["SMOKER"].to_numpy()
        + BETA["TP53"] * df["TP53"].to_numpy()
        + BETA["KRAS"] * df["KRAS"].to_numpy()
        + BETA["EGFR"] * df["EGFR"].to_numpy()
        + BETA["ALK"] * df["ALK"].to_numpy()
        + BETA["pdl1_high"] * pdl1_high
    )
    # Centre the linear predictor so that LAMBDA_0 controls the cohort-level
    # median survival directly, rather than being scaled up by the mean of
    # exp(log_hazard). Without this the whole cohort dies within a few months.
    log_hazard = log_hazard - log_hazard.mean()
    true_time = rng.exponential(1.0 / (LAMBDA_0 * np.exp(log_hazard)))
    censor_time = rng.uniform(90, 3650, n)
    observed = np.minimum(true_time, censor_time)
    dead = (true_time <= censor_time).astype(int)

    # `entry` is days from diagnosis to sequencing. The baselines model
    # stop - entry, which must be strictly positive for the Cox fit to keep the
    # row, so the observed duration is floored at 1 day.
    df["entry"] = rng.integers(0, 400, n)
    duration = np.maximum(observed.round(), 1).astype(int)
    df["stop"] = df["entry"] + duration
    df["dead"] = dead

    # Cancer-type flags. These are not model features -- they are consumed by
    # vignette construction to pick the right clinical description.
    for flag in ("has_nsclc", "has_brca", "has_crc", "has_prostate", "has_panc"):
        df[flag] = 1 if flag == "has_nsclc" else 0

    lead = ["PATIENT_ID", "entry", "stop", "dead"]
    return df[lead + [c for c in df.columns if c not in lead]]


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mskchord")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "nsclc_dx_1st_seq_OS.csv")

    df = build_cohort()
    df.to_csv(out_path, index=False)

    followup = df["stop"] - df["entry"]
    print(f"Wrote {len(df)} synthetic patients to {out_path}")
    print(f"  columns:           {df.shape[1]}")
    print(f"  deaths observed:   {int(df['dead'].sum())} ({df['dead'].mean():.0%})")
    print(f"  median follow-up:  {followup.median():.0f} days")
    print(f"  stage IV at dx:    {int(df['STAGE_IV_DX'].sum())} ({df['STAGE_IV_DX'].mean():.0%})")


if __name__ == "__main__":
    main()
