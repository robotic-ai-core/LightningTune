"""Lightning callbacks for BOHB optimization."""

import warnings

try:
    from .report import BOHBReportCallback, AdaptiveBOHBCallback
except ImportError as e:
    warnings.warn(f"Report callbacks not available: {e}")
    BOHBReportCallback = None
    AdaptiveBOHBCallback = None

try:
    from .tune_pause_callback import TunePauseCallback, TuneResumeCallback
except ImportError as e:
    warnings.warn(f"Pause callbacks not available: {e}")
    TunePauseCallback = None
    TuneResumeCallback = None

__all__ = [
    "BOHBReportCallback",
    "AdaptiveBOHBCallback",
    "TunePauseCallback",
    "TuneResumeCallback",
]