from __future__ import annotations

from typing import Dict, Any, Optional
import logging


logger = logging.getLogger(__name__)


def create_wandb_logger(
    project: Optional[str],
    study_name: str,
    trial_number: int,
    suggested_params: Optional[Dict[str, Any]],
    sampler_name: str,
    pruner_name: str,
    upload_checkpoints: bool,
):
    """Create a WandbLogger with simplified config and ensure previous runs are closed.

    Returns None if project is None.
    """
    if not project:
        return None

    # Finish any existing run to avoid conflicts
    try:
        import wandb
        if wandb.run is not None:
            logger.info("Finishing previous WandB run before starting new trial")
            wandb.finish(quiet=True)
    except Exception:
        pass

    from lightning.pytorch.loggers import WandbLogger

    wandb_config: Dict[str, Any] = {}

    if suggested_params:
        for key, value in suggested_params.items():
            parts = key.split('.')
            clean_parts = [p for p in parts if p != 'init_args']
            if clean_parts and clean_parts[0] in ['model', 'data', 'trainer']:
                clean_parts = clean_parts[1:]
            clean_parts = [
                p.replace('transformer_hparams', 'transformer')
                 .replace('adapter_hparams', 'adapter')
                 .replace('_hparams', '')
                for p in clean_parts
            ]
            clean_key = '.'.join(clean_parts) if clean_parts else key
            wandb_config[clean_key] = value

    wandb_config['trial_number'] = trial_number
    wandb_config['sampler'] = sampler_name
    wandb_config['pruner'] = pruner_name

    return WandbLogger(
        project=project,
        name=f"{study_name}_trial_{trial_number}",
        config=wandb_config,
        log_model=upload_checkpoints,
    )


def finalize_wandb_logger(logger_obj, status: str = "success") -> None:
    """Finalize WandB logger and finish the underlying wandb run safely."""
    if logger_obj is None:
        return
    try:
        logger_obj.finalize(status)
    except Exception:
        pass
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish(quiet=True)
    except Exception:
        pass


