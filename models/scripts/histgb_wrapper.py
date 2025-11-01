"""HistGradientBoosting pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from pipeline_wrapper import PipelineWrapper


class HistGBWrapper(PipelineWrapper):
    """Wrapper for HistGradientBoosting pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='HistGB', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build HistGradientBoosting pipeline."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('histgb', HistGradientBoostingClassifier(
                max_iter=1000,
                max_depth=10,
                learning_rate=0.05,
                class_weight='balanced',
                random_state=self.random_seed,
                verbose=0
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for HistGradientBoosting."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'histgb__max_iter': lambda trial: trial.suggest_int('histgb__max_iter', 500, 2000),
            'histgb__max_depth': lambda trial: trial.suggest_int('histgb__max_depth', 5, 15),
            'histgb__learning_rate': lambda trial: trial.suggest_float('histgb__learning_rate', 0.01, 0.3, log=True),
            'histgb__min_samples_leaf': lambda trial: trial.suggest_int('histgb__min_samples_leaf', 10, 50),
            'histgb__l2_regularization': lambda trial: trial.suggest_float('histgb__l2_regularization', 0, 10)
        }
