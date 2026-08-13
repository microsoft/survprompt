"""Render a Survprompt clinical vignette and prompt without calling any API.

This shows the core input Survprompt sends to a language model -- the clinical
vignette built from a patient's structured record, plus the system and user
prompts wrapped around it -- using only local template rendering.

No credentials and no network access are required. Nothing is sent anywhere.

Usage:
    python demo/render_prompt.py
    python demo/render_prompt.py --prompting_task TTE_OS --patient 3
"""

import argparse
import os

import pandas as pd

from survprompt.configs.predictor_config import PredictorConfig
from survprompt.predictor import CancerVignette

DEMO_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "mskchord", "nsclc_dx_1st_seq_OS.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompting_task", type=str, default="SURV_PROB",
                        choices=["SURV_PROB", "TTE_OS"],
                        help="Which prompting task's system prompt to render")
    parser.add_argument("--patient", type=int, default=0,
                        help="Row index of the patient to render (0-based)")
    parser.add_argument("--csv", type=str, default=DEMO_CSV,
                        help="Cohort CSV to read the patient from")
    return parser.parse_args()


def _banner(title):
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}"


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    if not 0 <= args.patient < len(df):
        raise SystemExit(f"--patient must be between 0 and {len(df) - 1} for {args.csv}")
    record = df.iloc[args.patient].to_dict()

    # Loading the config parses the Jinja templates for the chosen task. It does
    # not construct an API client, so this works with no credentials set.
    cfg = PredictorConfig(prompting_task=args.prompting_task)

    vignette = CancerVignette(record).create_clinical_vignette(record)
    system_prompt = cfg.system_instructions.render()
    user_prompt = cfg.user_instructions.render(examples_formatted="", clinical_vignette=vignette)

    print(_banner(f"PATIENT {record['PATIENT_ID']}  (prompting task: {args.prompting_task})"))
    observed = record["stop"] - record["entry"]
    outcome = "died" if record["dead"] == 1 else "censored"
    print(f"Ground truth: {outcome} at {observed:.0f} days after sequencing")

    print(_banner("CLINICAL VIGNETTE"))
    print(vignette)

    print(_banner("SYSTEM PROMPT"))
    print(system_prompt)

    print(_banner("USER PROMPT"))
    print(user_prompt)

    print(_banner("NOTE"))
    print("Rendered locally. No API call was made and no credentials were used.")
    print("To run real predictions, configure Azure OpenAI access as described in the README.")


if __name__ == "__main__":
    main()
