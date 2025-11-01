"""CART (Decision Tree) pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from pipeline_wrapper import PipelineWrapper


class CARTWrapper(PipelineWrapper):
    """Wrapper for CART (Decision Tree) pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='CART', random_seed=random_seed)

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build CART pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('CART', DecisionTreeClassifier(
                random_state=self.random_seed
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for CART."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'CART__max_depth': lambda trial: trial.suggest_int('CART__max_depth', 3, 19),
            'CART__min_samples_split': lambda trial: trial.suggest_int('CART__min_samples_split', 10, 49),
            'CART__min_samples_leaf': lambda trial: trial.suggest_int('CART__min_samples_leaf', 5, 19),
            'CART__criterion': lambda trial: trial.suggest_categorical('CART__criterion', ['gini', 'entropy'])
        }
