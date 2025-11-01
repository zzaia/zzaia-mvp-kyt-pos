"""AdaBoost pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from pipeline_wrapper import PipelineWrapper


class AdaWrapper(PipelineWrapper):
    """Wrapper for AdaBoost pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='Ada', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build AdaBoost pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('Ada', AdaBoostClassifier(
                random_state=self.random_seed
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for AdaBoost."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'Ada__n_estimators': lambda trial: trial.suggest_int('Ada__n_estimators', 50, 199),
            'Ada__learning_rate': lambda trial: trial.suggest_float('Ada__learning_rate', 0.5, 1.5),
            'Ada__algorithm': lambda trial: trial.suggest_categorical('Ada__algorithm', ['SAMME'])
        }
