"""TabNet pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from pytorch_tabnet.tab_model import TabNetClassifier
from pipeline_wrapper import PipelineWrapper


class TabNetWrapper(PipelineWrapper):
    """Wrapper for TabNet pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='TabNet', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build TabNet pipeline with QuantileTransformer (no PCA)."""
        return Pipeline([
            ('quantile', QuantileTransformer(output_distribution='normal')),
            ('tabnet', TabNetClassifier(
                seed=self.random_seed,
                verbose=0
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for TabNet."""
        return {
            'tabnet__n_d': lambda trial: trial.suggest_categorical('tabnet__n_d', [8, 16, 32, 64]),
            'tabnet__n_a': lambda trial: trial.suggest_categorical('tabnet__n_a', [8, 16, 32, 64]),
            'tabnet__n_steps': lambda trial: trial.suggest_int('tabnet__n_steps', 3, 7),
            'tabnet__gamma': lambda trial: trial.suggest_float('tabnet__gamma', 1.0, 2.0),
            'tabnet__lambda_sparse': lambda trial: trial.suggest_float('tabnet__lambda_sparse', 1e-6, 1e-3, log=True),
            'tabnet__mask_type': lambda trial: trial.suggest_categorical('tabnet__mask_type', ['sparsemax', 'entmax'])
        }
