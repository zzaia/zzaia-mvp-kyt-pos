"""XGBoost pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from pipeline_wrapper import PipelineWrapper


class XGBoostWrapper(PipelineWrapper):
    """Wrapper for XGBoost pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='XGB', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build XGBoost pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('xgb', xgb.XGBClassifier(
                random_state=self.random_seed,
                eval_metric='logloss',
                use_label_encoder=False
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for XGBoost."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'xgb__n_estimators': lambda trial: trial.suggest_int('xgb__n_estimators', 100, 1000),
            'xgb__max_depth': lambda trial: trial.suggest_int('xgb__max_depth', 3, 9),
            'xgb__learning_rate': lambda trial: trial.suggest_float('xgb__learning_rate', 0.01, 0.3, log=True),
            'xgb__min_child_weight': lambda trial: trial.suggest_int('xgb__min_child_weight', 1, 7),
            'xgb__gamma': lambda trial: trial.suggest_float('xgb__gamma', 0, 0.5),
            'xgb__subsample': lambda trial: trial.suggest_float('xgb__subsample', 0.6, 1.0),
            'xgb__colsample_bytree': lambda trial: trial.suggest_float('xgb__colsample_bytree', 0.6, 1.0),
            'xgb__scale_pos_weight': lambda trial: trial.suggest_float('xgb__scale_pos_weight', 1, 20),
            'xgb__reg_alpha': lambda trial: trial.suggest_float('xgb__reg_alpha', 0, 10),
            'xgb__reg_lambda': lambda trial: trial.suggest_float('xgb__reg_lambda', 1, 100)
        }
