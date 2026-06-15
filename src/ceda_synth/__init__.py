"""ceda-synth — synthetic data generation for Dutch educational datasets."""

__version__ = "0.1.0"

from ceda_synth.core.synthesize import apply_schema_bounds, fit, sample, set_seed
from ceda_synth.core.validate import evaluate

__all__ = ["fit", "sample", "set_seed", "apply_schema_bounds", "evaluate"]
