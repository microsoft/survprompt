import os
from .base import BaselineExperiment

class RSFBaseline(BaselineExperiment):
    """
    Reproduce the full baseline experiment from the MSK-CHORD paper.
    """
    def __init__(self, **kwargs):
        super().__init__(
            baseline="rsf",
            features="all",
            data_name=kwargs['data_name'],
            cancer_of_interest=kwargs['cancer_of_interest'],
            input_dir=os.getenv("INPUT_DIR"),
            num_folds=5,
            save_outputs=True,
            save_model=True,
            output_fname="rsf_baseline"
        )

class CoxBaseline(BaselineExperiment):
    """
    Cox proportional hazards model baseline experiment
    """
    def __init__(self, **kwargs):
        super().__init__(
            baseline="cox",
            features="all",
            data_name=kwargs['data_name'],
            cancer_of_interest=kwargs['cancer_of_interest'],
            input_dir=os.getenv("INPUT_DIR"),
            num_folds=5,
            save_outputs=True,
            save_model=True,
            output_fname="cox_baseline"
        )
