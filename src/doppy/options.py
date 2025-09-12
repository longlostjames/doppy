from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BgCorrectionMethod(Enum):
    FIT = "fit"
    MEAN = "mean"
    PRE_COMPUTED = "pre_computed"


class BgFitMethod(Enum):
    LIN = "lin"
    EXP = "exp"
    EXPLIN = "explin"


class OverlappedGatesMode(Enum):
    AUTO = "auto"  # Use header information (default behavior)
    FORCE_OVERLAPPED = "force_overlapped"  # Force overlapped gate processing
    FORCE_COMMON = "force_common"  # Force common gate processing


@dataclass
class OverlappedGatesOptions:
    """Options for overlapped gates processing in product functions.
    
    Parameters
    ----------
    mode : OverlappedGatesMode
        The processing mode for range gates
    custom_div : float, optional
        Custom divisor for overlapped formula (range_gate_length / custom_div)
    custom_mul : float, optional  
        Custom multiplier for overlapped formula (gate_index * custom_mul)
    """
    mode: OverlappedGatesMode = OverlappedGatesMode.AUTO
    custom_div: float | None = None
    custom_mul: float | None = None
    
    def __post_init__(self) -> None:
        if (self.custom_div is not None or self.custom_mul is not None):
            if self.custom_div is None or self.custom_mul is None:
                raise ValueError(
                    "Both custom_div and custom_mul must be provided together"
                )
            if not (0.0 < self.custom_div < 10.0):
                raise ValueError("custom_div must be between 0 and 10")
            if not (0.0 < self.custom_mul < 10.0):
                raise ValueError("custom_mul must be between 0 and 10")
