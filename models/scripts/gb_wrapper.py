"""Gradient Boosting pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from pipeline_wrapper import PipelineWrapper


class GBWrapper(PipelineWrapper):
    """Wrapper for Gradient Boosting pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='GB', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build Gradient Boosting pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('GB', GradientBoostingClassifier(
                random_state=self.random_seed
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for Gradient Boosting."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'GB__n_estimators': lambda trial: trial.suggest_int('GB__n_estimators', 100, 299),
            'GB__learning_rate': lambda trial: trial.suggest_float('GB__learning_rate', 1e-2, 3e-1, log=True),
            'GB__max_depth': lambda trial: trial.suggest_int('GB__max_depth', 3, 7),
            'GB__subsample': lambda trial: trial.suggest_float('GB__subsample', 0.7, 0.9)
        }
