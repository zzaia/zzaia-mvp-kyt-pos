"""Bagging with KNN pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from pipeline_wrapper import PipelineWrapper


class BagKNNWrapper(PipelineWrapper):
    """Wrapper for Bagging with KNN pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='Bag-KNN', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build Bagging-KNN pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('bagging', BaggingClassifier(
                estimator=KNeighborsClassifier(),
                n_estimators=10,
                max_samples=0.8,
                max_features=0.8,
                bootstrap=True,
                random_state=self.random_seed
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for Bagging-KNN."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'bagging__estimator__n_neighbors': lambda trial: trial.suggest_int('bagging__estimator__n_neighbors', 3, 20),
            'bagging__estimator__weights': lambda trial: trial.suggest_categorical('bagging__estimator__weights', ['uniform', 'distance']),
            'bagging__estimator__metric': lambda trial: trial.suggest_categorical('bagging__estimator__metric', ['euclidean', 'manhattan', 'minkowski']),
            'bagging__n_estimators': lambda trial: trial.suggest_int('bagging__n_estimators', 10, 99),
            'bagging__max_samples': lambda trial: trial.suggest_float('bagging__max_samples', 0.5, 0.9),
            'bagging__max_features': lambda trial: trial.suggest_float('bagging__max_features', 0.5, 0.9),
            'bagging__bootstrap': lambda trial: trial.suggest_categorical('bagging__bootstrap', [True, False])
        }
