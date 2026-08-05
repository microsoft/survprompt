import os
from typing import TYPE_CHECKING
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Use TYPE_CHECKING to avoid circular import errors with type hints
if TYPE_CHECKING:
    from survprompt.configs.predictor_config import PredictorConfig

def initialize_prompt_config(cfg: 'PredictorConfig'):
    """
    Initializes prompt-related attributes on the PredictorConfig instance.

    This function populates the config with the Jinja2 environment, outcome text,
    and loaded system/user instruction templates based on the prompting task.

    Args:
        cfg: An instance of the PredictorConfig dataclass.
    """
    # Helper function to create a Jinja2 environment
    def _get_jinja_env():
        return Environment(
            loader=FileSystemLoader(cfg.template_dir),
            autoescape=select_autoescape(enabled_extensions=("j2",)),
        )

    # Attach the Jinja environment to the config object
    cfg._env_jinja = _get_jinja_env()

    # Helper function to load templates using the environment
    def _load_template(name: str):
        return cfg._env_jinja.get_template(name)

    # Set attributes based on the prompting task
    cfg.is_o1 = 'o1' in cfg.pred_engine

    if cfg.prompting_task == 'TTE_OS':
        cfg.outcome_text = 'Estimated survival time (days)'

    elif cfg.prompting_task == 'SURV_PROB':
        cfg.num_timepoints = 21
        cfg.outcome_text = "Survival prediction"
        # cfg.outcome_text = f'Estimated {cfg.num_timepoints} survival probabilities (decimals ranging from 0.0 to 1.0) for a range of years (decimals ranging from 0.0 to {cfg.max_num_yrs} years in the future)'
        cfg._env_jinja.globals['max_num_yrs'] = cfg.max_num_yrs
        cfg._env_jinja.globals['num_timepoints'] = cfg.num_timepoints
        
    else:
        raise ValueError(f"Unknown prompting_task: '{cfg.prompting_task}'")

    # Load system and user prompts
    cfg.user_instructions = _load_template('user.j2')
    cfg.system_instructions = _load_template(f"{cfg.prompting_task}/{cfg.system_prompt_path}")

    # Adjust max_tokens based on the system prompt path
    if cfg.system_prompt_path == 'system.j2':
        cfg.max_tokens = 4096
    else:
        cfg.max_tokens = 1024

    # Set the outcome_text as a global variable in the Jinja environment
    if hasattr(cfg, 'outcome_text'):
        cfg._env_jinja.globals['outcome_text'] = cfg.outcome_text
