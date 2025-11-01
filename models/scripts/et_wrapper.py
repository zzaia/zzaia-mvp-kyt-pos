"""Extra Trees pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from pipeline_wrapper import PipelineWrapper


class ETWrapper(PipelineWrapper):
    """Wrapper for Extra Trees pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='ET', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build Extra Trees pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('ET', ExtraTreesClassifier(
                random_state=self.random_seed
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for Extra Trees."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'ET__n_estimators': lambda trial: trial.suggest_int('ET__n_estimators', 100, 499),
            'ET__max_depth': lambda trial: trial.suggest_int('ET__max_depth', 10, 24),
            'ET__min_samples_split': lambda trial: trial.suggest_int('ET__min_samples_split', 5, 19),
            'ET__min_samples_leaf': lambda trial: trial.suggest_int('ET__min_samples_leaf', 2, 9),
            'ET__max_features': lambda trial: trial.suggest_categorical('ET__max_features', ['sqrt', 'log2', None]),
            'ET__bootstrap': lambda trial: trial.suggest_categorical('ET__bootstrap', [True, False])
        }
