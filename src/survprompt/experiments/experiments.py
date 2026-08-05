import argparse
from survprompt.defaults import DEFAULT_DATA_NAME

from . import baseline_experiments, zero_shot_experiments


EXPERIMENT_TO_NAME = {
    # Zero-shot experiments
    'zeroshot_gpt5':            zero_shot_experiments.ZeroShotGPT5,
    "zeroshot_gpt54_none":      zero_shot_experiments.ZeroShotGPT54,
    "zeroshot_gpt54_medium":    zero_shot_experiments.ZeroShotGPT54Medium,
    "zeroshot_gpt55_none":      zero_shot_experiments.ZeroShotGPT55,
    "zeroshot_gpt55_medium":    zero_shot_experiments.ZeroShotGPT55Medium,
    "zeroshot_gpt56sol_none":   zero_shot_experiments.ZeroShotGPT56Sol,
    "zeroshot_gpt56sol_medium": zero_shot_experiments.ZeroShotGPT56SolMedium,
    "zeroshot_gpt4om":          zero_shot_experiments.ZeroShotGPT4oMini,
    "zeroshot_gpt4o":           zero_shot_experiments.ZeroShotGPT4o,
    "zeroshot_gpt41m":          zero_shot_experiments.ZeroShotGPT41Mini,
    "zeroshot_gpt41":           zero_shot_experiments.ZeroShotGPT41,
    "zeroshot_gpt4om_temp0":    zero_shot_experiments.ZeroShotGPT4oMiniTemp0,
    "zeroshot_gpt4o_temp0":     zero_shot_experiments.ZeroShotGPT4oTemp0,
    "zeroshot_gpt41m_temp0":    zero_shot_experiments.ZeroShotGPT41MiniTemp0,
    "zeroshot_gpt41_temp0":     zero_shot_experiments.ZeroShotGPT41Temp0,
    "zeroshot_o1":              zero_shot_experiments.ZeroShotO1,

    # Baseline experiments
    "rsf_baseline":       baseline_experiments.RSFBaseline,
    "cox_baseline":       baseline_experiments.CoxBaseline,
}

EXPERIMENT_SETS = {
    "zeroshot": [ zero_shot_experiments.ZeroShotGPT5,
                    zero_shot_experiments.ZeroShotGPT54,
                    zero_shot_experiments.ZeroShotGPT54Medium,
                    zero_shot_experiments.ZeroShotGPT55,
                    zero_shot_experiments.ZeroShotGPT55Medium,
                    zero_shot_experiments.ZeroShotGPT56Sol,
                    zero_shot_experiments.ZeroShotGPT56SolMedium,
                    zero_shot_experiments.ZeroShotGPT4oMini,
                    zero_shot_experiments.ZeroShotGPT4o,
                    zero_shot_experiments.ZeroShotGPT41Mini,
                    zero_shot_experiments.ZeroShotGPT41,
                    zero_shot_experiments.ZeroShotO1,
                    zero_shot_experiments.ZeroShotGPT4oMiniTemp0,
                    zero_shot_experiments.ZeroShotGPT4oTemp0,
                    zero_shot_experiments.ZeroShotGPT41MiniTemp0,
                    zero_shot_experiments.ZeroShotGPT41Temp0
                    ],

    "baselines": [baseline_experiments.RSFBaseline,
                    baseline_experiments.CoxBaseline]
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str, help="Name of the experiment, or that of a collection of experiments to run in series.")
    parser.add_argument('--data_name', type=str, default=DEFAULT_DATA_NAME, help="Dataset name")
    parser.add_argument('--cancer_of_interest', nargs='+', default='nsclc', help='Cancer types to include')
    parser.add_argument('--prompting_task', type=str, default='SURV_PROB', choices=['TTE_OS', 'SURV_PROB'], help='Prompting task (eg. predict for median OS or survival)')
    return parser.parse_args()



def experiment():
    args = parse_args()
    if args.exp_name not in EXPERIMENT_TO_NAME and args.exp_name not in EXPERIMENT_SETS:
        print(f"Experiment {args.exp_name} not found.")
        return

    if args.exp_name in EXPERIMENT_SETS:
        for exp in EXPERIMENT_SETS[args.exp_name]:
            experiment = exp(data_name=args.data_name, cancer_of_interest=args.cancer_of_interest, prompting_task=args.prompting_task)
            experiment.run()
    else:
        experiment = EXPERIMENT_TO_NAME[args.exp_name](data_name=args.data_name, cancer_of_interest=args.cancer_of_interest, prompting_task=args.prompting_task)
        experiment.run()


if __name__=='__main__':
    experiment()