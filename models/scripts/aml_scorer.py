"""
Anti-Money Laundering (AML) Scorer Module

This module provides a composite scoring system for imbalanced fraud detection,
combining Matthews Correlation Coefficient (MCC) with cost-sensitive scoring
to balance detection accuracy and business impact in financial AML applications.

Usage:
    from aml_scorer import AMLScorer
    from sklearn.metrics import make_scorer

    scorer = AMLScorer(cost_fp=1, cost_fn=10, mcc_weight=0.4, cost_weight=0.6)
    composite_scorer = make_scorer(scorer.score)
"""

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import cross_val_score


class AMLScorer:
    """
    Anti-Money Laundering scorer for imbalanced fraud detection.

    Combines Matthews Correlation Coefficient (MCC) with cost-sensitive scoring
    to balance detection accuracy and business impact.
    """

    def __init__(self, cost_fp=1, cost_fn=10, mcc_weight=0.4, cost_weight=0.6):
        """
        Initialize AML scorer.

        Args:
            cost_fp: Cost of false positive (blocking legitimate transaction)
            cost_fn: Cost of false negative (missing illicit transaction)
            mcc_weight: Weight for MCC component (0-1)
            cost_weight: Weight for cost component (0-1)
        """
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.mcc_weight = mcc_weight
        self.cost_weight = cost_weight

    @property
    def metric_equation(self):
        """Get the metric equation as a formatted string."""
        return (
            f"AML Score = {self.mcc_weight} × MCC + {self.cost_weight} × Cost Score\n"
            f"where:\n"
            f"  MCC = Matthews Correlation Coefficient\n"
            f"  Cost Score = 1 - (FP × {self.cost_fp} + FN × {self.cost_fn}) / (N × {self.cost_fn})\n"
            f"  FP = False Positives (licit flagged)\n"
            f"  FN = False Negatives (illicit missed)\n"
            f"  N = Total samples"
        )

    def score(self, y_true, y_pred):
        """
        Calculate AML composite score.

        Args:
            y_true: True labels (0=licit, 1=illicit)
            y_pred: Predicted labels (0=licit, 1=illicit)

        Returns:
            float: Composite score (0-1 range, higher is better)
        """
        # MCC: Handles imbalance naturally
        mcc = matthews_corrcoef(y_true, y_pred)

        # Cost-sensitive component
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        total_cost = (fp * self.cost_fp) + (fn * self.cost_fn)
        max_cost = len(y_true) * self.cost_fn
        cost_score = 1 - (total_cost / max_cost)

        # Weighted combination
        return self.mcc_weight * mcc + self.cost_weight * cost_score

    def cross_val_score_with_threshold(self, pipeline, X, y, cv, threshold):
        """
        Custom cross-validation with threshold-aware predictions.

        Args:
            pipeline: Sklearn pipeline to evaluate
            X: Training features
            y: Training labels
            cv: Cross-validation splitter
            threshold: Classification threshold for probability conversion

        Returns:
            Array of fold scores
        """
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            # Split data
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Train and predict
            pipeline.fit(X_train_fold, y_train_fold)
            y_proba = pipeline.predict_proba(X_val_fold)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

            # Calculate score
            fold_score = self.score(y_val_fold, y_pred)
            scores.append(fold_score)

        return np.array(scores)

    def create_objective(self, model_name, pipeline, param_dist, X_train, y_train, cv, scorer):
        """
        Create Optuna objective function for hyperparameter optimization.

        Args:
            model_name: Name of the model being optimized
            pipeline: Sklearn pipeline to optimize
            param_dist: Dictionary of hyperparameter distributions
            X_train: Training features
            y_train: Training labels
            cv: Cross-validation splitter
            scorer: Sklearn scorer object

        Returns:
            Callable objective function for Optuna
        """
        def objective(trial):
            # Get parameter suggestions by calling lambdas with trial
            params = {}
            for param_name, suggest_fn in param_dist[model_name].items():
                params[param_name] = suggest_fn(trial)

            # Add threshold parameter (check if defined in param_dist, else use default)
            threshold_key = 'threshold'
            if threshold_key in param_dist[model_name]:
                threshold = param_dist[model_name][threshold_key](trial)
            else:
                threshold = trial.suggest_float('threshold', 0.1, 0.9)

            # Set pipeline parameters (exclude threshold as it's not a pipeline param)
            pipeline.set_params(**params)

            # Perform custom cross-validation with threshold
            scores = self.cross_val_score_with_threshold(pipeline, X_train, y_train, cv, threshold)

            # Store fold scores in trial user attributes for later retrieval
            trial.set_user_attr('cv_scores', scores.tolist())
            trial.set_user_attr('threshold', threshold)

            # Return mean score
            return scores.mean()

        return objective
