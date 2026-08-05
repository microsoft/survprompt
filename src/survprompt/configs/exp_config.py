from typing import Optional, List
import omegaconf
from dataclasses import dataclass
import os
from survprompt.defaults import DEFAULT_DATA_NAME

@dataclass
class ExperimentConfig:
    """
    Configuration setup for the experiment (input and output).
    """
    # Dataset name
    data_name: str = DEFAULT_DATA_NAME
    cancers_available: List[str] = ('nsclc', 'brca', 'prostate', 'crc', 'panc')

    # Cancer types to include
    cancer_of_interest: str|List[str] = omegaconf.MISSING

    # Directory containing input data
    input_dir: str = omegaconf.MISSING

    # Base directory
    base_dir: str = omegaconf.MISSING

    # Log directory
    log_dir: str = None

    # Feature sets to include
    features: List[str] = omegaconf.MISSING

    # Number of folds for cross-validation
    num_folds: int = 5

    # Number of samples to predict
    sample_size: int = 1000

    # Fold to use
    fold: int = 0

    # Prediction number to use (ie 0 for first, 1 for second, etc. Used for best of n, ensemble or consistency)
    prediction_number: int = 0

    # Experiment name
    exp_name: str ='zeroshot'

    # Experiment type
    exp_type: str = 'arbitrary_fixed'

    # Re-prompt model only for NaN predictions
    complete_outputs: bool = True

    # Overwrite existing prediction outputs if found
    overwrite_outputs: bool = False

    # Subsets of features to use for prediction e.g. ['STAGE_IV_DX', 'STAGE_I-III_NOPROG']
    subgroup_features_to_test: Optional[List[str]] = None

    # Save performance metrics after generating predictions e.g. ['c-index']
    save_metrics: Optional[List[str]] = ('coverage',)

    # Save logs to file
    save_logs: bool = True

    def __post_init__(self):
        # Process the 'cancer_of_interest' argument to expand 'all', and be of type list
        if (self.cancer_of_interest == "all") or (isinstance(self.cancer_of_interest, list) and "all" in self.cancer_of_interest):
            self.cancer_of_interest = [*self.cancers_available]
        if not isinstance(self.cancer_of_interest, list):
            if self.cancer_of_interest in self.cancers_available:
                self.cancer_of_interest = [self.cancer_of_interest]
            else:
                raise ValueError(f'Invalid cancer type: {self.cancer_of_interest}, available: {self.cancers_available}')

        # Set data file name based on the data name
        if self.data_name == 'mskchord':
            self.input_data_paths = {c: os.path.join(self.base_dir, 'data', self.data_name, f'{c}_dx_1st_seq_OS.csv') for c in self.cancer_of_interest}
