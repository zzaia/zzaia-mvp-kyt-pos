"""Voting Classifier (Hard) pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from pipeline_wrapper import PipelineWrapper


class VotingHardWrapper(PipelineWrapper):
    """Wrapper for Voting Classifier (Hard) pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='Vote-Hard', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build Voting (Hard) pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('voting', VotingClassifier(
                estimators=[
                    ('svm', SVC()),
                    ('knn', KNeighborsClassifier())
                ],
                voting='hard'
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for Voting (Hard)."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'voting__svm__C': lambda trial: trial.suggest_float('voting__svm__C', 1e-2, 1e3, log=True),
            'voting__svm__kernel': lambda trial: trial.suggest_categorical('voting__svm__kernel', ['rbf', 'poly']),
            'voting__svm__gamma': lambda trial: trial.suggest_categorical('voting__svm__gamma', ['scale', 'auto']),
            'voting__knn__n_neighbors': lambda trial: trial.suggest_int('voting__knn__n_neighbors', 3, 20),
            'voting__knn__weights': lambda trial: trial.suggest_categorical('voting__knn__weights', ['uniform', 'distance']),
            'voting__knn__metric': lambda trial: trial.suggest_categorical('voting__knn__metric', ['euclidean', 'manhattan'])
        }
