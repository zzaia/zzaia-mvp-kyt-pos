"""Logistic Regression pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from pipeline_wrapper import PipelineWrapper


class LRWrapper(PipelineWrapper):
    """Wrapper for Logistic Regression pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='LR', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build Logistic Regression pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('LR', LogisticRegression(
                random_state=self.random_seed
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for Logistic Regression."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'LR__C': lambda trial: trial.suggest_float('LR__C', 1e-4, 1e2, log=True),
            'LR__solver': lambda trial: trial.suggest_categorical('LR__solver', ['lbfgs', 'newton-cg', 'sag', 'saga']),
            'LR__penalty': lambda trial: trial.suggest_categorical('LR__penalty', ['l2', None]),
            'LR__max_iter': lambda trial: trial.suggest_categorical('LR__max_iter', [1000, 2000, 5000]),
            'LR__tol': lambda trial: trial.suggest_float('LR__tol', 1e-6, 1e-3, log=True)
        }
