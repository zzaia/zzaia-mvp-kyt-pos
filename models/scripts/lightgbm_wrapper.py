"""LightGBM pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
from pipeline_wrapper import PipelineWrapper


class LightGBMWrapper(PipelineWrapper):
    """Wrapper for LightGBM pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='LGB', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build LightGBM pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('lgb', lgb.LGBMClassifier(
                random_state=self.random_seed,
                verbose=-1,
                force_col_wise=True
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for LightGBM."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'lgb__n_estimators': lambda trial: trial.suggest_int('lgb__n_estimators', 100, 1000),
            'lgb__num_leaves': lambda trial: trial.suggest_int('lgb__num_leaves', 31, 255),
            'lgb__max_depth': lambda trial: trial.suggest_int('lgb__max_depth', 3, 9),
            'lgb__learning_rate': lambda trial: trial.suggest_float('lgb__learning_rate', 0.01, 0.3, log=True),
            'lgb__min_child_samples': lambda trial: trial.suggest_int('lgb__min_child_samples', 5, 30),
            'lgb__subsample': lambda trial: trial.suggest_float('lgb__subsample', 0.6, 1.0),
            'lgb__colsample_bytree': lambda trial: trial.suggest_float('lgb__colsample_bytree', 0.6, 1.0),
            'lgb__reg_alpha': lambda trial: trial.suggest_float('lgb__reg_alpha', 0, 1),
            'lgb__reg_lambda': lambda trial: trial.suggest_float('lgb__reg_lambda', 0, 1)
        }
