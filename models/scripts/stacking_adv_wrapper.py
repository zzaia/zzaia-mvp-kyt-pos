"""Advanced Stacking Classifier pipeline wrapper with FNN, TabNet, and XGBoost."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from scikeras.wrappers import KerasClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from pipeline_wrapper import PipelineWrapper


class StackingAdvWrapper(PipelineWrapper):
    """Wrapper for Advanced Stacking Classifier with deep learning ensemble."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='Stack-Adv', random_seed=random_seed)
        tf.random.set_seed(random_seed)

    @staticmethod
    def create_fnn(meta, hidden_dims=[64, 32], learning_rate=0.001, dropout=0.3):
        """
        Create feedforward neural network for ensemble.

        Args:
            meta: Metadata from KerasClassifier
            hidden_dims: List of hidden layer sizes
            learning_rate: Learning rate for optimizer
            dropout: Dropout rate

        Returns:
            Compiled Keras model
        """
        n_features_in_ = meta["n_features_in_"]
        n_classes_ = meta["n_classes_"]

        model = Sequential()

        # Input layer
        model.add(Dense(hidden_dims[0], activation='relu', input_shape=(n_features_in_,)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        # Hidden layers
        for layer_size in hidden_dims[1:]:
            model.add(Dense(layer_size, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        # Output layer
        model.add(Dense(n_classes_, activation='softmax'))

        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """Build advanced stacking pipeline with FNN, TabNet, and XGBoost."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('stacking', StackingClassifier(
                estimators=[
                    ('fnn', KerasClassifier(
                        model=self.create_fnn,
                        epochs=50,
                        batch_size=128,
                        verbose=0,
                        random_state=self.random_seed
                    )),
                    ('xgb', xgb.XGBClassifier(
                        random_state=self.random_seed,
                        eval_metric='logloss',
                        use_label_encoder=False
                    )),
                    ('tabnet', TabNetClassifier(
                        seed=self.random_seed,
                        verbose=0
                    ))
                ],
                final_estimator=LogisticRegression(random_state=self.random_seed, max_iter=2000),
                cv=3,  # Reduced CV for speed with deep learning models
                passthrough=False
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for Advanced Stacking."""
        return {
            # PCA parameters
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),

            # FNN parameters
            'stacking__fnn__model__hidden_dims': lambda trial: trial.suggest_categorical(
                'stacking__fnn__model__hidden_dims', [[64, 32], [128, 64], [64]]
            ),
            'stacking__fnn__model__learning_rate': lambda trial: trial.suggest_float(
                'stacking__fnn__model__learning_rate', 1e-4, 1e-2, log=True
            ),
            'stacking__fnn__model__dropout': lambda trial: trial.suggest_float(
                'stacking__fnn__model__dropout', 0.2, 0.5
            ),
            'stacking__fnn__epochs': lambda trial: trial.suggest_categorical(
                'stacking__fnn__epochs', [30, 50, 100]
            ),

            # XGBoost parameters
            'stacking__xgb__n_estimators': lambda trial: trial.suggest_int('stacking__xgb__n_estimators', 100, 500),
            'stacking__xgb__max_depth': lambda trial: trial.suggest_int('stacking__xgb__max_depth', 3, 7),
            'stacking__xgb__learning_rate': lambda trial: trial.suggest_float('stacking__xgb__learning_rate', 0.01, 0.2, log=True),
            'stacking__xgb__subsample': lambda trial: trial.suggest_float('stacking__xgb__subsample', 0.6, 1.0),
            'stacking__xgb__colsample_bytree': lambda trial: trial.suggest_float('stacking__xgb__colsample_bytree', 0.6, 1.0),
            'stacking__xgb__scale_pos_weight': lambda trial: trial.suggest_float('stacking__xgb__scale_pos_weight', 1, 15),

            # TabNet parameters
            'stacking__tabnet__n_d': lambda trial: trial.suggest_categorical('stacking__tabnet__n_d', [8, 16, 32]),
            'stacking__tabnet__n_a': lambda trial: trial.suggest_categorical('stacking__tabnet__n_a', [8, 16, 32]),
            'stacking__tabnet__n_steps': lambda trial: trial.suggest_int('stacking__tabnet__n_steps', 3, 5),
            'stacking__tabnet__gamma': lambda trial: trial.suggest_float('stacking__tabnet__gamma', 1.0, 1.5),

            # Meta-learner (Logistic Regression) parameters
            'stacking__final_estimator__C': lambda trial: trial.suggest_float('stacking__final_estimator__C', 1e-4, 1e2, log=True),
            'stacking__final_estimator__solver': lambda trial: trial.suggest_categorical('stacking__final_estimator__solver', ['lbfgs', 'saga']),

            # Threshold parameter
            'threshold': lambda trial: trial.suggest_float('threshold', 0.1, 0.9)
        }
