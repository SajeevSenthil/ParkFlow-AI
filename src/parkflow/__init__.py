"""ParkFlow-AI: spatial-temporal forecasting of parking violations.

Public surface kept small on purpose; orchestration lives in :mod:`parkflow.pipeline`.
"""

from __future__ import annotations

from .config import Config

__version__ = "0.1.0"
__all__ = ["Config", "__version__"]
