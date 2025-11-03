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
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, matthews_corrcoef, average_precision_score


class AMLScorer:
    """
    Anti-Money Laundering scorer for imbalanced fraud detection.

    Combines Matthews Correlation Coefficient (MCC) with cost-sensitive scoring
    to balance detection accuracy and business impact.
    """

    def __init__(self, cost_fp=1, cost_fn=10, cost_tn=0, cost_tp=0, mcc_weight=0.3, cost_weight=0.5, prauc_weight=0.2):
        """
        Initialize AML scorer.

        Args:
            cost_fp: Cost of false positive (blocking legitimate transaction)
            cost_fn: Cost of false negative (missing illicit transaction)
            cost_tn: Cost of true negative (correctly allowing legitimate transaction)
            cost_tp: Cost of true positive (correctly blocking illicit transaction)
            mcc_weight: Weight for MCC component (0-1)
            cost_weight: Weight for cost component (0-1)
            prauc_weight: Weight for PR-AUC component (0-1)
        """
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.cost_tn = cost_tn
        self.cost_tp = cost_tp
        self.mcc_weight = mcc_weight
        self.cost_weight = cost_weight
        self.prauc_weight = prauc_weight

    @property
    def metric_equation(self):
        """Get the metric equation as a formatted string."""
        return (
            f"AML Score = {self.mcc_weight} × MCC + {self.cost_weight} × Cost Score + {self.prauc_weight} × PR-AUC\n"
            f"\n"
            f"Components:\n"
            f"  MCC (Matthews Correlation Coefficient):\n"
            f"    - Threshold-dependent: Evaluates classification quality at specific threshold\n"
            f"    - Measures: Correlation between predictions and actuals using all 4 confusion matrix values\n"
            f"    - Range: -1 (worst) to +1 (perfect), 0 = random\n"
            f"    - Key strength: Balanced metric, reliable for imbalanced data\n"
            f"\n"
            f"  Cost Score:\n"
            f"    - Threshold-dependent: Business impact at specific threshold\n"
            f"    - Measures: Financial cost of classification errors\n"
            f"    - Formula: 1 - Total Cost / Max Cost\n"
            f"    - Total Cost = TN × {self.cost_tn} + TP × {self.cost_tp} + FP × {self.cost_fp} + FN × {self.cost_fn}\n"
            f"    - Key strength: Incorporates asymmetric business costs (FN >> FP)\n"
            f"\n"
            f"  PR-AUC (Precision-Recall Area Under Curve):\n"
            f"    - Threshold-independent: Evaluates performance across ALL thresholds\n"
            f"    - Measures: Model's discriminative ability and probability calibration quality\n"
            f"    - Range: 0 to 1 (1 = perfect)\n"
            f"    - Key strength: Ensures robustness if threshold needs adjustment\n"
            f"\n"
            f"Key Takeaways:\n"
            f"  • MCC: 'How good is this specific prediction?' (threshold-dependent quality)\n"
            f"  • Cost Score: 'What is the business impact?' (domain-specific evaluation)\n"
            f"  • PR-AUC: 'How good is this model overall?' (threshold-independent capability)\n"
            f"  • Together: Comprehensive evaluation of classification quality, business value, and model robustness\n"
            f"\n"
            f"Confusion Matrix:\n"
            f"  TN = True Negatives (correctly identified licit)\n"
            f"  TP = True Positives (correctly identified illicit)\n"
            f"  FP = False Positives (licit flagged as illicit)\n"
            f"  FN = False Negatives (illicit missed)\n"
            f"  N = Total samples\n"
            f"  Max Cost = N × {self.cost_fn} (worst case: all FN)"
        )

    def score(self, y_true, y_pred, y_proba):
        """
        Calculate AML composite score.

        Args:
            y_true: True labels (0=licit, 1=illicit)
            y_pred: Predicted labels (0=licit, 1=illicit)
            y_proba: Probability scores for positive class (for PR-AUC)

        Returns:
            float: Composite score (0-1 range, higher is better)
        """
        # MCC: Handles imbalance naturally
        mcc = matthews_corrcoef(y_true, y_pred)

        # Cost-sensitive component
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Total cost includes all confusion matrix outcomes
        total_cost = (tn * self.cost_tn) + (tp * self.cost_tp) + (fp * self.cost_fp) + (fn * self.cost_fn)
        max_cost = len(y_true) * self.cost_fn  # Worst case: all false negatives
        cost_score = 1 - (total_cost / max_cost)
        prauc = average_precision_score(y_true, y_proba)

        return self.mcc_weight * mcc + self.cost_weight * cost_score + self.prauc_weight * prauc

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

            # Calculate score with probabilities for PR-AUC
            fold_score = self.score(y_val_fold, y_pred, y_proba)
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

            # Clone pipeline to avoid fitted state issues (especially CatBoost)
            pipeline_clone = clone(pipeline)

            # Set pipeline parameters (exclude threshold as it's not a pipeline param)
            pipeline_params = {k: v for k, v in params.items() if k != 'threshold'}
            pipeline_clone.set_params(**pipeline_params)

            # Perform custom cross-validation with threshold
            scores = self.cross_val_score_with_threshold(pipeline_clone, X_train, y_train, cv, threshold)

            # Store fold scores in trial user attributes for later retrieval
            trial.set_user_attr('cv_scores', scores.tolist())
            trial.set_user_attr('threshold', threshold)

            # Return mean score
            return scores.mean()

        return objective
