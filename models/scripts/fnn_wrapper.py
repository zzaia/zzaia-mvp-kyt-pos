"""Feedforward Neural Network pipeline wrapper."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from pipeline_wrapper import PipelineWrapper


class FNNWrapper(PipelineWrapper):
    """Wrapper for Feedforward Neural Network pipeline."""

    def __init__(self, random_seed: int = 42):
        super().__init__(name='FNN', random_seed=random_seed)
        tf.random.set_seed(random_seed)

    @staticmethod
    def create_feedforward_nn(meta, hidden_dims=[128, 64], learning_rate=0.001, dropout=0.3):
        """
        Create feedforward neural network.

        This is a static method to ensure proper pickling and parameter passing
        with KerasClassifier and scikit-learn.

        Args:
            meta: Metadata from KerasClassifier containing n_features_in_ and n_classes_
            hidden_dims: List of hidden layer sizes
            learning_rate: Learning rate for optimizer
            dropout: Dropout rate for regularization

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
        """Build FNN pipeline with StandardScaler and PCA."""
        return Pipeline([
            ('std', StandardScaler()),
            ('pca', PCA(n_components=n_pca_components)),
            ('fnn', KerasClassifier(
                model=FNNWrapper.create_feedforward_nn,
                hidden_dims=[128, 64],
                learning_rate=0.001,
                dropout=0.3,
                epochs=100,
                batch_size=256,
                verbose=0,
                random_state=self.random_seed
            ))
        ])

    def get_param_distributions(self) -> dict:
        """Get Optuna hyperparameter distributions for FNN."""
        return {
            'pca__n_components': lambda trial: trial.suggest_float('pca__n_components', 0.90, 0.99),
            'pca__whiten': lambda trial: trial.suggest_categorical('pca__whiten', [True, False]),
            'pca__svd_solver': lambda trial: trial.suggest_categorical('pca__svd_solver', ['auto', 'full']),
            'fnn__hidden_dims': lambda trial: trial.suggest_categorical('fnn__hidden_dims', [[128, 64], [256, 128], [512, 256, 128], [128, 64, 32]]),
            'fnn__dropout': lambda trial: trial.suggest_float('fnn__dropout', 0.1, 0.5),
            'fnn__learning_rate': lambda trial: trial.suggest_float('fnn__learning_rate', 1e-4, 1e-2, log=True),
            'fnn__epochs': lambda trial: trial.suggest_int('fnn__epochs', 50, 200),
            'fnn__batch_size': lambda trial: trial.suggest_categorical('fnn__batch_size', [128, 256, 512])
        }
