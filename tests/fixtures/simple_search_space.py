"""Simple search space for testing."""

from LightningTune import SearchSpace

try:
    from ray import tune
except ImportError:
    tune = None


class SimpleSearchSpace(SearchSpace):
    """Simple search space for E2E tests."""
    
    def get_search_space(self):
        if tune:
            return {
                "model.init_args.learning_rate": tune.choice([0.001, 0.01]),
                "model.init_args.hidden_dim": tune.choice([16, 32]),
                "data.init_args.batch_size": tune.choice([16, 32]),
            }
        else:
            return {
                "model.init_args.learning_rate": [0.001, 0.01],
                "model.init_args.hidden_dim": [16, 32],
                "data.init_args.batch_size": [16, 32],
            }
    
    def get_metric_config(self):
        return {"metric": "val_loss", "mode": "min"}