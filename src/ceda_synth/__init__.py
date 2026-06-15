"""ceda-synth — synthetic data generation for Dutch educational datasets."""

__version__ = "0.1.0"

from ceda_synth.core.synthesize import detect_datetime_format, fit, sample, set_seed
from ceda_synth.core.validate import evaluate

__all__ = ["fit", "sample", "set_seed", "detect_datetime_format", "evaluate"]
