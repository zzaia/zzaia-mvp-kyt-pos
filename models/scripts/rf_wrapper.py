"""Random Forest pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from pipeline_wrapper import PipelineWrapper


class RFWrapper(PipelineWrapper):
    """Wrapper for Random Forest pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='RF', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build Random Forest pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('RF', RandomForestClassifier(
                random_state=self.random_seed
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for Random Forest."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'RF__n_estimators': lambda trial: trial.suggest_int('RF__n_estimators', 100, 499),
            'RF__max_depth': lambda trial: trial.suggest_int('RF__max_depth', 10, 24),
            'RF__min_samples_split': lambda trial: trial.suggest_int('RF__min_samples_split', 5, 19),
            'RF__min_samples_leaf': lambda trial: trial.suggest_int('RF__min_samples_leaf', 2, 9),
            'RF__max_features': lambda trial: trial.suggest_categorical('RF__max_features', ['sqrt', 'log2', None]),
            'RF__bootstrap': lambda trial: trial.suggest_categorical('RF__bootstrap', [True, False])
        }
