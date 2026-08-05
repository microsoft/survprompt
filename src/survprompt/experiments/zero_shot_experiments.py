from .base import BaseExperiment

class ZeroShotExp(BaseExperiment):
    """Base class for zero-shot experiments."""
    def __init__(self, exp_name: str, **kwargs):
        super().__init__(
            exp_name=exp_name,
            experiment_type="zero_shot",
            **kwargs
        )

class ZeroShotExp(ZeroShotExp):
    """Zero-shot experiment configuration."""
    def __init__(self, **kwargs):
        kwargs.setdefault("prompting_task", "SURV_PROB")
        super().__init__(
            exp_name=f"zeroshot_{kwargs['prompting_task']}",
            **kwargs,
        )

# 1. Temp = 0.4
class ZeroShotGPT4o(ZeroShotExp):
    """GPT-4o model with temperature 0.4."""
    def __init__(self, **kwargs):
        super().__init__(
            pred_engine="gpt-4o_2024-08-06", #"gpt-4o_2024-11-20",
            requests_per_minute=75,
            **kwargs
        )

class ZeroShotGPT4oMini(ZeroShotExp):
    """GPT-4o mini model with temperature 0.4."""
    def __init__(self, **kwargs):
        super().__init__(
            pred_engine="gpt-4o-mini_2024-07-18",
            requests_per_minute=200,
            **kwargs
        )

class ZeroShotGPT41(ZeroShotExp):
    """GPT-4.1 model with temperature 0.4."""
    def __init__(self, **kwargs):
        super().__init__(
            pred_engine="gpt-4.1",#_2025-04-14",
            requests_per_minute=75,
            **kwargs
        )

class ZeroShotGPT41Mini(ZeroShotExp):
    """GPT-4.1 mini model with temperature 0.4."""
    def __init__(self, **kwargs):
        super().__init__(
            pred_engine="gpt-4.1-mini_2025-04-14",
            requests_per_minute=200,
            **kwargs
        )

class ZeroShotGPT4oTemp0(ZeroShotExp):
    """GPT-4o model with temperature 0.0."""
    def __init__(self, **kwargs):
        super().__init__(
            pred_engine="gpt-4o_2024-08-06",#2024-08-06 not _2024-11-20",
            temperature=0.0,
            requests_per_minute=75,
            **kwargs
        )

class ZeroShotGPT4oMiniTemp0(ZeroShotExp):
    """GPT-4o mini model with temperature 0.0."""
    def __init__(self, **kwargs):
        super().__init__(
            pred_engine="gpt-4o-mini_2024-07-18",
            temperature=0.0,
            requests_per_minute=200,
            **kwargs
        )

class ZeroShotGPT41Temp0(ZeroShotExp):
    """GPT-4.1 model with temperature 0.0."""
    def __init__(self, **kwargs):
        super().__init__(
            pred_engine="gpt-4.1", #_2025-04-14
            temperature=0.0,
            requests_per_minute=75,
            **kwargs
        )

class ZeroShotGPT41MiniTemp0(ZeroShotExp):
    """GPT-4.1 mini model with temperature 0.0."""
    def __init__(self, **kwargs):
        super().__init__(
            pred_engine="gpt-4.1-mini_2025-04-14",
            temperature=0.0,
            requests_per_minute=200,
            **kwargs
        )

# 3. Temp not specified, o1/gpt 5 models
class ZeroShotGPT5(ZeroShotExp):
    """GPT-5 model with temperature 0.4."""
    def __init__(self, **kwargs):
        kwargs.setdefault("max_tokens", 1000)
        super().__init__(
            pred_engine="gpt-5_2025-08-07",
            requests_per_minute=75,
            **kwargs
        )

class ZeroShotGPT54(ZeroShotExp):
    """GPT-5.4 model."""
    def __init__(self, **kwargs):
        super().__init__(
            pred_engine="gpt-5.4",
            requests_per_minute=75,
            max_tokens=32000,
            reasoning_effort="none",  # low, medium, or high
            **kwargs
        )

class ZeroShotGPT54Medium(ZeroShotExp):
    """GPT-5.4 model with medium reasoning effort."""
    def __init__(self, **kwargs):
        super().__init__(
            pred_engine="gpt-5.4",
            requests_per_minute=75,
            max_tokens=32000,
            reasoning_effort="medium",  # low, medium, or high
            **kwargs
        )

class ZeroShotGPT55(ZeroShotExp):
    """GPT-5.5 model."""
    def __init__(self, **kwargs):
        super().__init__(
            pred_engine="gpt-5.5_2026-04-24",
            requests_per_minute=75,
            max_tokens=32000,
            reasoning_effort="none",  # low, medium, or high
            **kwargs
        )

class ZeroShotGPT55Medium(ZeroShotExp):
    """GPT-5.5 model with medium reasoning effort."""
    def __init__(self, **kwargs):
        super().__init__(
            pred_engine="gpt-5.5_2026-04-24",
            requests_per_minute=75,
            max_tokens=32000,
            reasoning_effort="medium",  # low, medium, or high
            **kwargs
        )

class ZeroShotGPT56Sol(ZeroShotExp):
    """GPT-5.6 Sol model."""
    def __init__(self, **kwargs):
        super().__init__(
            pred_engine="gpt-5.6-sol_2026-07-09",
            requests_per_minute=75,
            max_tokens=32000,
            reasoning_effort="none",
            **kwargs
        )

class ZeroShotGPT56SolMedium(ZeroShotExp):
    """GPT-5.6 Sol model with medium reasoning effort."""
    def __init__(self, **kwargs):
        super().__init__(
            pred_engine="gpt-5.6-sol_2026-07-09",
            requests_per_minute=75,
            max_tokens=32000,
            reasoning_effort="medium",
            **kwargs
        )

class ZeroShotO1(ZeroShotExp):
    """o1 model with max tokens 1000."""
    def __init__(self, **kwargs):
        super().__init__(
            pred_engine="o1_2024-12-17",
            **kwargs
        )

