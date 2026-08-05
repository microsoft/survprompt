import os
from typing import List, Optional
from dataclasses import dataclass

from survprompt.defaults import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_PRED_ENGINE,
    DEFAULT_REASONING_EFFORT,
)
from survprompt.prompt_utils import initialize_prompt_config

@dataclass
class PredictorConfig:
    """
    Configuration setup for the predictor. 
    Reference for params found at https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
    """
    # Prediction engine to use
    pred_engine: str = DEFAULT_PRED_ENGINE

    # Prompting task (eg. predict for median OS)
    prompting_task: str = 'SURV_PROB' # ['TTE_OS']

    # Temperature to use when prompting.
    temperature: float = 0.4

    # Max tokens to use when prompting. For o1 models corresponds to completion tokens.
    max_tokens: int = DEFAULT_MAX_TOKENS

    # Reasoning effort to use when prompting
    reasoning_effort: str = DEFAULT_REASONING_EFFORT

    # Number of requests per minute for OpenAI API
    requests_per_minute: int = 30
    
    # Model considers the results of the tokens with top_p probability mass
    top_p: float = 0.9

    # Model's likelihood to repeat the same line verbatim (positive values penalize new tokens based on their existing frequency in the text so far)
    frequency_penalty: float = 0.0

    # Model's likelihood to talk about new topics (positive values penalize new tokens based on whether they appear in the text so far)
    presence_penalty: float = 0.0

    # Stop sequence
    stop: List[str] | None = None

    # Max retries for the predictor
    max_retries: int = 15

    # Whether to use AsyncAzureOpenAI or not
    use_async: bool = True

    # Reasoning effort for reasoning models (o1, gpt-5*). None = model default (medium for gpt-5*).
    # Known accepted values: "low", "medium", "high", "xhigh", "none".
    reasoning_effort: Optional[str] = DEFAULT_REASONING_EFFORT

    # Prompt template directory
    template_dir: str = os.path.join(os.path.dirname(__file__), 'prompt_templates')

    # System prompt path
    system_prompt_path: str = 'system.j2'

    def __post_init__(self):
        self.max_num_yrs = 10
        # Initialize the prompt configuration
        initialize_prompt_config(self)
