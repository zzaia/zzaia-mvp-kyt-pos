"""Support Vector Machine pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from pipeline_wrapper import PipelineWrapper


class SVMWrapper(PipelineWrapper):
    """Wrapper for Support Vector Machine pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='SVM', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build SVM pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('SVM', SVC(
                random_state=self.random_seed,
                probability=True
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for SVM."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'SVM__C': lambda trial: trial.suggest_float('SVM__C', 1e-2, 1e3, log=True),
            'SVM__kernel': lambda trial: trial.suggest_categorical('SVM__kernel', ['rbf', 'poly', 'sigmoid']),
            'SVM__gamma': lambda trial: trial.suggest_categorical('SVM__gamma', ['scale', 'auto'])
        }
