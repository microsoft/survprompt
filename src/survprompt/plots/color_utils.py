import re
from typing import List, Tuple, Optional
import pandas as pd
from survprompt.plots.colorbrewer_palettes import colorbrewer


TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 10
GLOBAL_FONT_SIZE = 14

# Add helper to convert rgb( ) strings to hex codes.
def rgb_str_to_hex(rgb_str: str) -> str:
    # extract r, g, b values and return hex format.
    numbers = list(map(int, re.findall(r'\d+', rgb_str)))
    return "#{:02x}{:02x}{:02x}".format(*numbers)

METHODS_ORDER = list(reversed(['None', 'Fixed', 'Representative', 'NN', 'dNN']))
MODELS_ORDER = list(['4o', '', '4o mini', '', '4.1', '', 'o1', 'o1 mini', '', '5', '', '5.4', '', '5.5', '', '5.6 Sol'])
BASELINES_ORDER = ['RSF', 'Cox',]# '5NN', '5cNN', '5dNN']
MODELS = ["4o", "4.1", "5", "o1"]# "4", "o1 mini", "o1 preview"
GPT5_MODELS = ["5", "5.4", "5.5", "5.6 Sol"]

# The headline "Survprompt" series is always drawn in this purple (the last,
# purple swatch of the Set1 palette), regardless of which model backs it.
_SET1 = colorbrewer["Set1"][f"{len(MODELS)}"]
SURVPROMPT_PURPLE = rgb_str_to_hex(_SET1[-1]) if _SET1[-1].startswith("rgb") else _SET1[-1]
def get_model_color(model, method=None, task=None, simplify=None):
    """
    Returns the color for a given model, method, and task combination.
    """
    # Special handling for RSF baseline comparisons - check the full label
    if 'RSF (TTE)' in str(model):
        return 'blue'
    elif 'RSF (Avg)' in str(model):
        return 'red'
    
    # Choose colors based on model type and task
    if model in ['Ground Truth']:
        return "#000000"
    
    if task == "Baseline":
        palette = colorbrewer["Greys"]["3"]
        idx = BASELINES_ORDER.index(model)
        color = palette[idx+1]
        return rgb_str_to_hex(color) if color.startswith("rgb") else color

    # In the headline "baseline vs Survprompt" figures the Survprompt series is
    # always purple, independent of the backing model (RSF / Ground Truth have
    # already returned above, so only the Survprompt model reaches here). The
    # all-models figures use 'Survprompt_appendix' and keep family colors (e.g.
    # GPT-5 stays light blue), so this only affects the 2-model comparisons.
    if simplify == 'Survprompt_final':
        return SURVPROMPT_PURPLE

    # GPT-5 family always uses Blues, regardless of `simplify` mode.
    # Darker shade -> earlier (lower-numbered) version: 5 darkest, 5.6 Sol lightest.
    if model in GPT5_MODELS:
        palette = colorbrewer["Blues"][f"{len(GPT5_MODELS) + 1}"]
        intensity = GPT5_MODELS.index(model)
        idx = max(len(palette) - 1 - intensity, 0)
        color = palette[idx]
        return rgb_str_to_hex(color) if color.startswith("rgb") else color

    if simplify:
        if simplify == 'Survprompt_appendix' or simplify == 'Survprompt_final':
            palette = colorbrewer["Set1"][f"{len(MODELS)}"]
        
        intensity = MODELS.index(model) if model in MODELS else 2
        idx = max(len(palette) - 1 - intensity, 0)
        color = palette[idx]
        return rgb_str_to_hex(color) if color.startswith("rgb") else color

    if model in MODELS:
        if task == 'TTE_OS':
            palette = colorbrewer["Reds"]["7"]
        if task == 'SURV_PROB':
            palette = colorbrewer["Purples"]["7"]
        intensity = MODELS.index(model) if model in MODELS else 2
        idx = max(len(palette) - 1 - intensity, 0)
        color = palette[idx]
        return rgb_str_to_hex(color) if color.startswith("rgb") else color

    return "#000000"

def get_line_style(size: str) -> str:
    # Dashed denotes compact model size only; reasoning effort should not share
    # the same visual encoding as model size.
    return "dashed" if size == "mini" else "solid"

def parse_label(label: str) -> Tuple[str, Optional[str], str]:
    """
    Parse a label into model name, prompting method, size and prompting task
    """
    # Remove parentheses and their contents, and clean up double whitespace
    label = re.sub(r'\s*\([^)]*\)', '', label)
    label = re.sub(r'\s+', ' ', label).strip()

    if ":" in label:
        task, label = label.split(":")
        task = task.strip()
    else:
        task = None

    if "/" in label:
        model, method = label.split("/", 1)
        method = method.strip()
    else:
        model, method = label, None
        model = model.strip()

    tokens = model.strip().split()
    if tokens and tokens[0].upper() == "GPT":
        tokens = tokens[1:]
    size = "full"
    if "mini" in tokens:
        size = "mini"
        tokens.remove("mini")
    if "medium" in tokens:
        size = "medium"
        tokens.remove("medium")
    if "none" in tokens:
        size = "none"
        tokens.remove("none")
    if tokens and tokens[-1].isdigit():
        tokens = tokens[:-1]
    model = " ".join(tokens)
    return model, method, size, task


# Canonical left-to-right ordering for multi-model bar figures, keyed by
# (model, is_mini) as returned by ``parse_label``. The RSF baseline is always
# leftmost. Unknown models sort to the end (stable), preserving discovery order.
CANONICAL_MODEL_ORDER = [
    ("RSF", False),
    ("4o", False),
    ("4o", True),
    ("4.1", False),
    ("4.1", True),
    ("5", False),
    ("o1", False),
    ("5.4", False),
    ("5.5", False),
    ("5.6 Sol", False),
]


def canonical_model_sort_key(label: str) -> int:
    """Position of a model label in ``CANONICAL_MODEL_ORDER`` (unknowns last)."""
    try:
        model, _method, size, _task = parse_label(label)
    except Exception:
        return len(CANONICAL_MODEL_ORDER)
    try:
        return CANONICAL_MODEL_ORDER.index((model, size == "mini"))
    except ValueError:
        return len(CANONICAL_MODEL_ORDER)

