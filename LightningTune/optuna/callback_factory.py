from typing import List
import optuna

from .callbacks import OptunaPruningCallback


def build_optuna_callbacks(trial: optuna.Trial, monitor: str) -> List[object]:
    """Create a standard set of Optuna-related callbacks for training.

    - Always attach EnhancedOptunaPruningCallback (validation-end) when available
      without step-based reporting for val metrics.
    - Always attach NaNDetectionCallback to terminate on train-step NaN/Inf quickly.
    - Always attach PruneOnExceptionCallback to free resources on unexpected errors.
    """
    callbacks: List[object] = []

    # Pruning on monitored validation metric
    try:
        from .nan_detection_callback import EnhancedOptunaPruningCallback
        callbacks.append(
            EnhancedOptunaPruningCallback(
                trial,
                monitor=monitor,
                check_nan=True,
                verbose=True,
            )
        )
    except Exception:
        callbacks.append(OptunaPruningCallback(trial, monitor=monitor))

    # Train-step NaN/Inf detection
    try:
        from .nan_detection_callback import NaNDetectionCallback
        callbacks.append(
            NaNDetectionCallback(
                trial,
                monitor=monitor,
                check_train_loss=True,
                check_every_n_steps=10,
                verbose=True,
            )
        )
    except Exception:
        pass

    # Prune on unexpected exceptions (not KeyboardInterrupt)
    try:
        from .callbacks import PruneOnExceptionCallback
        callbacks.append(PruneOnExceptionCallback(trial))
    except Exception:
        pass

    return callbacks


