"""CatBoost pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import catboost as cb
from pipeline_wrapper import PipelineWrapper


class CatBoostWrapper(PipelineWrapper):
    """Wrapper for CatBoost pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='CAT', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build CatBoost pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('cat', cb.CatBoostClassifier(
                iterations=500,
                depth=8,
                learning_rate=0.1,
                auto_class_weights='Balanced',
                random_seed=self.random_seed,
                verbose=False
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for CatBoost."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'cat__iterations': lambda trial: trial.suggest_int('cat__iterations', 100, 1000),
            'cat__depth': lambda trial: trial.suggest_int('cat__depth', 4, 10),
            'cat__learning_rate': lambda trial: trial.suggest_float('cat__learning_rate', 0.01, 0.3, log=True),
            'cat__l2_leaf_reg': lambda trial: trial.suggest_int('cat__l2_leaf_reg', 1, 9),
            'cat__border_count': lambda trial: trial.suggest_categorical('cat__border_count', [32, 64, 128, 255]),
            'cat__bagging_temperature': lambda trial: trial.suggest_float('cat__bagging_temperature', 0, 1),
            'cat__random_strength': lambda trial: trial.suggest_float('cat__random_strength', 0, 2)
        }
