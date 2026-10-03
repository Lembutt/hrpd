"""HRPD: Healthy-Referenced Physics-informed Diagnosis.

Synthetic-to-real bearing fault diagnosis with healthy-only target data.
See ``README.md`` for the data layout and CLI usage.
"""

from .config import DATASETS, DatasetSpec, HRPDConfig
from .data import Base5Data, load_base5
from .evaluate import compute_metrics, summarise_seeds
from .features import phys_features, two_channel
from .models import HRPDModel
from .train import train_hrpd

__version__ = "1.0.0"

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "HRPDConfig",
    "Base5Data",
    "load_base5",
    "compute_metrics",
    "summarise_seeds",
    "phys_features",
    "two_channel",
    "HRPDModel",
    "train_hrpd",
    "__version__",
]
