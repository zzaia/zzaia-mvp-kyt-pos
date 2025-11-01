"""K-Nearest Neighbors pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from pipeline_wrapper import PipelineWrapper


class KNNWrapper(PipelineWrapper):
    """Wrapper for K-Nearest Neighbors pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='KNN', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build KNN pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('KNN', KNeighborsClassifier())
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for KNN."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'KNN__n_neighbors': lambda trial: trial.suggest_int('KNN__n_neighbors', 3, 20),
            'KNN__weights': lambda trial: trial.suggest_categorical('KNN__weights', ['uniform', 'distance']),
            'KNN__metric': lambda trial: trial.suggest_categorical('KNN__metric', ['euclidean', 'manhattan', 'minkowski']),
            'KNN__p': lambda trial: trial.suggest_int('KNN__p', 1, 2)
        }
