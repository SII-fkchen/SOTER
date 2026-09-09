"""SOTER: medical time-series foundation model built upon MIRA."""

from .models.configuration_soter import SoterConfig
from .models.modeling_soter import SoterForPrediction, SoterModel

__version__ = "1.0.0"
__all__ = ["SoterConfig", "SoterForPrediction", "SoterModel"]
