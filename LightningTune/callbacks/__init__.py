"""Lightning callbacks package - Ray Tune callbacks removed, migrated to Optuna."""

# Legacy Ray Tune callbacks have been removed
# Use Optuna callbacks from LightningTune.optuna.callbacks instead

BOHBReportCallback = None
AdaptiveBOHBCallback = None
TunePauseCallback = None
TuneResumeCallback = None

__all__ = [
    "BOHBReportCallback",
    "AdaptiveBOHBCallback", 
    "TunePauseCallback",
    "TuneResumeCallback",
]