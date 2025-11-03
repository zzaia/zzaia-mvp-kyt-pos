"""
Training Manager Module

Encapsulates all training logic for machine learning models with Optuna optimization,
checkpoint management, and Azure fallback support.

Usage:
    from training_manager import TrainingManager

    manager = TrainingManager(
        checkpoint_dir="./models/checkpoints",
        n_trials=10,
        patience_ratio=0.2,
        timeout_seconds=7200,
        n_jobs=1,
        random_seed=42
    )

    trainingModels = manager.train_models(
        pipeline_wrappers=wrappers,
        param_distributions=params,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        scorer=scorer,
        aml_scorer=aml_scorer,
        n_pca_components=0.95
    )
"""

import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import joblib
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


class EarlyStoppingCallback:
    """
    Stop optimization after N trials without improvement or timeout.

    Attributes:
        patience: Number of trials without improvement before stopping
        timeout_seconds: Maximum time in seconds before stopping
        trials_without_improvement: Counter for trials without improvement
        best_value: Best score achieved so far
        start_time: Timestamp when optimization started
        is_timed_out: Flag indicating if optimization was stopped by timeout
    """

    def __init__(self, patience: int, timeout_seconds: Optional[float] = None):
        """
        Initialize early stopping callback.

        Args:
            patience: Number of trials without improvement before stopping
            timeout_seconds: Maximum time in seconds (None for no timeout)
        """
        self.patience = patience
        self.timeout_seconds = timeout_seconds
        self.trials_without_improvement = 0
        self.best_value = None
        self.start_time = None
        self.is_timed_out = False

    def start_timer(self):
        """Start the timeout timer."""
        self.start_time = time.time()

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """
        Check stopping conditions after each trial.

        Args:
            study: Optuna study object
            trial: Completed trial
        """
        # Check timeout
        if self.timeout_seconds and self.start_time:
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.timeout_seconds:
                self.is_timed_out = True
                study.stop()
                return

        # Only process completed trials
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        # Initialize best value on first trial
        if self.best_value is None:
            self.best_value = study.best_value
            self.trials_without_improvement = 0
            return

        # Check for improvement
        if study.best_value > self.best_value:
            self.best_value = study.best_value
            self.trials_without_improvement = 0
        else:
            self.trials_without_improvement += 1

        # Stop if patience exceeded
        if self.trials_without_improvement >= self.patience:
            study.stop()


class TrainingManager:
    """
    Manages model training with Optuna optimization and checkpoint persistence.

    Handles:
    - Hyperparameter optimization using Optuna
    - Checkpoint management (load/save models, studies, metadata)
    - Early stopping with patience and timeout
    - Azure blob storage fallback for model loading
    - Cross-validation and scoring
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        n_trials: int,
        patience_ratio: float,
        timeout_seconds: float,
        n_jobs: int,
        random_seed: int
    ):
        """
        Initialize training manager.

        Args:
            checkpoint_dir: Directory for saving/loading checkpoints
            n_trials: Number of Optuna trials per model
            patience_ratio: Ratio of trials for early stopping patience
            timeout_seconds: Maximum time per model training
            n_jobs: Number of parallel jobs for cross-validation
            random_seed: Random seed for reproducibility
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.n_trials = n_trials
        self.patience = int(patience_ratio * n_trials)
        self.timeout_seconds = timeout_seconds
        self.n_jobs = n_jobs
        self.random_seed = random_seed

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Suppress Optuna warnings
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def load_checkpoint(self, model_name: str) -> Optional[Tuple[Any, optuna.study.Study, Dict]]:
        """
        Load model checkpoint if exists.

        Args:
            model_name: Name of the model to load

        Returns:
            Tuple of (pipeline, study, metadata) if checkpoint exists, None otherwise
        """
        model_path = self.checkpoint_dir / f"{model_name}.pkl"
        study_path = self.checkpoint_dir / f"{model_name}.study.pkl"
        metadata_path = self.checkpoint_dir / f"{model_name}.metadata.json"

        # Check if all checkpoint files exist
        if not (model_path.exists() and study_path.exists() and metadata_path.exists()):
            return None

        try:
            # Load checkpoint files
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            trained_pipe = joblib.load(model_path)
            study = joblib.load(study_path)

            meta_score_mean = metadata['cv_score_mean']
            meta_score_std = metadata['cv_score_std']
            meta_actual_trials = metadata['actual_trials']
            meta_ntrials = metadata['n_trials']
            meta_is_timed_out = metadata['is_timed_out']
            threshold = metadata.get('optimal_threshold', 0.5)
            cv_scores = np.array(metadata['cv_scores'])

            print(f"Loading checkpoint {model_name}... ✅ {meta_score_mean:.4f} (±{meta_score_std:.4f}) [{meta_actual_trials}/{meta_ntrials}] is timed out: {meta_is_timed_out}]")

            return model_name, cv_scores, trained_pipe, study, threshold

        except Exception as e:
            warnings.warn(f"Failed to load checkpoint for {model_name}: {e}")
            return None

    def save_checkpoint(
        self,
        model_name: str,
        pipe: Any,
        study: optuna.study.Study,
        early_stopping: EarlyStoppingCallback,
        aml_scorer: Any
    ) -> None:
        """
        Save model checkpoint with metadata.

        Args:
            model_name: Name of the model
            pipe: Trained pipeline
            study: Optuna study object
            early_stopping: Early stopping callback with training info
            aml_scorer: AML scorer instance for metric equation
        """
        # Calculate statistics
        threshold = study.best_params.get('threshold', 0.5)
        cv_scores = np.array(study.best_trial.user_attrs['cv_scores'])
        actual_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"✅ {study.best_value:.4f} (±{cv_scores.std():.4f}) [{actual_trials}/{self.n_trials}] is timed out: {early_stopping.is_timed_out}]")

        # Create metadata
        metadata = {
            'model_name': model_name,
            'model_class': pipe.named_steps[list(pipe.named_steps.keys())[-1]].__class__.__name__,
            'cv_score_mean': float(cv_scores.mean()),
            'cv_score_std': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist(),
            'cv_score_type': aml_scorer.metric_equation,
            'trained_at': datetime.now().isoformat(),
            'actual_trials': actual_trials,
            'n_trials': self.n_trials,
            'patience': self.patience,
            'timeout_seconds': self.timeout_seconds,
            'is_timed_out': early_stopping.is_timed_out,
            'best_params': study.best_params,
            'random_seed': self.random_seed,
            'optimal_threshold': threshold
        }

        # Save files
        model_path = self.checkpoint_dir / f"{model_name}.pkl"
        study_path = self.checkpoint_dir / f"{model_name}.study.pkl"
        metadata_path = self.checkpoint_dir / f"{model_name}.metadata.json"

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        joblib.dump(pipe, model_path, compress=3)
        joblib.dump(study, study_path, compress=3)

        return model_name, cv_scores, pipe, study, threshold
        

    def train_models(
        self,
        pipeline_wrappers: List[Any],
        param_distributions: Dict[str, Dict],
        X_train: Any,
        y_train: Any,
        cv: Any,
        scorer: Any,
        aml_scorer: Any,
        n_pca_components: float,
        azure_client: Optional[Any] = None
    ) -> List[Tuple[str, np.ndarray, Any, optuna.study.Study, float]]:
        """
        Train all models with Optuna optimization and checkpoint management.

        Args:
            pipeline_wrappers: List of pipeline wrapper instances
            param_distributions: Dictionary of parameter distributions per model
            X_train: Training features
            y_train: Training labels
            cv: Cross-validation splitter
            scorer: Sklearn scorer object
            aml_scorer: AML scorer instance for creating objectives
            n_pca_components: Number of PCA components to keep
            azure_client: Optional Azure client for downloading checkpoints

        Returns:
            List of checkpoint tuples: (model_name, cv_scores, pipeline, study, threshold)
        """
        training_models = []

        print(f"🔍 Training {len(pipeline_wrappers)} models (patience={self.patience}, timeout={self.timeout_seconds/3600:.1f}h)")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print("-" * 60)

        for wrapper in pipeline_wrappers:
            name = wrapper.name

            # Try to load from checkpoint
            checkpoint = self.load_checkpoint(name)

            if checkpoint is not None:
                training_models.append(checkpoint)
                continue

            # No checkpoint found - train from scratch
            print(f"Training {name}...", end=" ", flush=True)

            # Build pipeline
            pipe = wrapper.build_pipeline(n_pca_components)

            # Create Optuna study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.random_seed),
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
            )

            # Create objective function
            objective = aml_scorer.create_objective(
                name, pipe, param_distributions, X_train, y_train, cv, scorer
            )

            # Setup early stopping
            early_stopping = EarlyStoppingCallback(
                patience=self.patience,
                timeout_seconds=self.timeout_seconds
            )
            early_stopping.start_timer()

            # Run optimization
            study.optimize(objective, n_trials=self.n_trials, callbacks=[early_stopping])

            # Train final model with best parameters
            pipeline_params = {k: v for k, v in
            study.best_params.items() if k != 'threshold'}
            pipe.set_params(**pipeline_params)
            pipe.fit(X_train, y_train)

            checkpoint = self.save_checkpoint(name, pipe, study, early_stopping, aml_scorer)
            training_models.append(checkpoint)

        print("-" * 60)
        print(f"✅ {len(training_models)} models ready")

        return training_models

    def load_models_from_checkpoint(
        self,
        azure_client: Optional[Any] = None
    ) -> List[Tuple[str, np.ndarray, Any, optuna.study.Study, float]]:
        """
        Load all trained models from checkpoints.

        If checkpoints don't exist locally and azure_client is provided,
        attempts to download from Azure blob storage.

        Args:
            azure_client: Optional Azure client for downloading checkpoints

        Returns:
            List of checkpoint tuples: (model_name, cv_scores, pipeline, study, threshold)
        """
        training_models = []

        # Check if local checkpoints exist
        if not self.checkpoint_dir.exists() or not any(self.checkpoint_dir.iterdir()):
            if azure_client:
                print("📥 Downloading from Azure...")
                # Extract model name from checkpoint dir path
                model_name = self.checkpoint_dir.name
                parent_dir = self.checkpoint_dir.parent

                success = azure_client.download_documents(
                    "models",
                    model_name,
                    base_path=str(parent_dir.parent)
                )

                if not success:
                    print("❌ No models available")
                    return training_models
            else:
                print("❌ No local checkpoints and no Azure client provided")
                return training_models

        # Load all model checkpoints using load_checkpoint()
        for model_file in sorted(self.checkpoint_dir.glob('*.pkl')):
            # Skip study files (they'll be loaded with their corresponding models)
            if '.study' in model_file.name:
                continue

            model_name = model_file.stem
            checkpoint = self.load_checkpoint(model_name)

            if checkpoint is not None:
                training_models.append(checkpoint)
            else:
                warnings.warn(f"Failed to load checkpoint for {model_name}")

        source = "Azure" if azure_client else "checkpoints"
        print(f"✅ Loaded {len(training_models)} models from {source}")

        return training_models 
