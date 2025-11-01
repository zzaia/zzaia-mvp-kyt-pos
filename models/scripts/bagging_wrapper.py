"""Bagging Classifier pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from pipeline_wrapper import PipelineWrapper


class BaggingWrapper(PipelineWrapper):
    """Wrapper for Bagging Classifier pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='Bag', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build Bagging pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('Bag', BaggingClassifier(
                random_state=self.random_seed
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for Bagging."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'Bag__n_estimators': lambda trial: trial.suggest_int('Bag__n_estimators', 50, 199),
            'Bag__max_samples': lambda trial: trial.suggest_float('Bag__max_samples', 0.6, 0.9),
            'Bag__max_features': lambda trial: trial.suggest_float('Bag__max_features', 0.7, 1.0)
        }
