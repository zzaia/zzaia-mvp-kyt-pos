"""Base class for ML pipeline wrappers."""

from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline


class PipelineWrapper(ABC):
    """
    Abstract base class for ML pipeline wrappers.

    Each pipeline wrapper encapsulates:
    - Pipeline construction logic
    - Hyperparameter distribution for Optuna
    - Model-specific configuration
    """

    def __init__(self, name: str, random_seed: int = 42):
        """
        Initialize pipeline wrapper.

        Args:
            name: Short name for the pipeline (e.g., 'XGB', 'CAT')
            random_seed: Random seed for reproducibility
        """
        self.name = name
        self.random_seed = random_seed
        self._pipeline = None

    @abstractmethod
    def build_pipeline(self, n_pca_components: float = 0.95) -> Pipeline:
        """
        Construct the sklearn pipeline.

        Args:
            n_pca_components: PCA variance to retain (0-1)

        Returns:
            Configured sklearn Pipeline
        """
        pass

    @abstractmethod
    def get_param_distributions(self) -> dict:
        """
        Get Optuna hyperparameter distributions.

        Returns:
            Dictionary mapping parameter names to Optuna suggest functions
        """
        pass

    @property
    def pipeline(self) -> Pipeline:
        """Get the pipeline (builds if not already built)."""
        if self._pipeline is None:
            self._pipeline = self.build_pipeline()
        return self._pipeline

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
