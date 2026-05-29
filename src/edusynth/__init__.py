"""edusynth — synthetic data generation for Dutch educational datasets."""

__version__ = "0.1.0"

from edusynth.synthesize import fit, sample
from edusynth.validate import evaluate

__all__ = ["fit", "sample", "evaluate"]
