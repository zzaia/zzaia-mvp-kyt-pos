"""Stacking Classifier pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from pipeline_wrapper import PipelineWrapper


class StackingWrapper(PipelineWrapper):
    """Wrapper for Stacking Classifier pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='Stack', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build Stacking pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('stacking', StackingClassifier(
                estimators=[
                    ('svm', SVC(probability=True)),
                    ('knn', KNeighborsClassifier())
                ],
                final_estimator=LogisticRegression(),
                cv=5,
                passthrough=False
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for Stacking."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'stacking__svm__C': lambda trial: trial.suggest_float('stacking__svm__C', 1e-2, 1e3, log=True),
            'stacking__svm__kernel': lambda trial: trial.suggest_categorical('stacking__svm__kernel', ['rbf', 'poly']),
            'stacking__svm__gamma': lambda trial: trial.suggest_categorical('stacking__svm__gamma', ['scale', 'auto']),
            'stacking__knn__n_neighbors': lambda trial: trial.suggest_int('stacking__knn__n_neighbors', 3, 20),
            'stacking__knn__weights': lambda trial: trial.suggest_categorical('stacking__knn__weights', ['uniform', 'distance']),
            'stacking__knn__metric': lambda trial: trial.suggest_categorical('stacking__knn__metric', ['euclidean', 'manhattan']),
            'stacking__final_estimator__C': lambda trial: trial.suggest_float('stacking__final_estimator__C', 1e-4, 1e2, log=True),
            'stacking__final_estimator__solver': lambda trial: trial.suggest_categorical('stacking__final_estimator__solver', ['lbfgs', 'saga']),
            'stacking__final_estimator__max_iter': lambda trial: trial.suggest_categorical('stacking__final_estimator__max_iter', [1000, 2000])
        }
