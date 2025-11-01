"""Naive Bayes pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from pipeline_wrapper import PipelineWrapper


class NBWrapper(PipelineWrapper):
    """Wrapper for Naive Bayes pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='NB', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build Naive Bayes pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('NB', GaussianNB())
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for Naive Bayes."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'NB__var_smoothing': lambda trial: trial.suggest_float('NB__var_smoothing', 1e-12, 1e-6, log=True),
            'NB__priors': lambda trial: trial.suggest_categorical('NB__priors', [None])
        }
