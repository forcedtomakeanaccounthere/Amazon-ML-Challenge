"""
Model training and evaluation module for entity matching.
Implements LightGBM classifier with calibration and threshold tuning.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import pickle


class EntityMatcher:
    """
    Entity matching model using LightGBM.
    """
    
    def __init__(self, threshold: float = 0.5, random_state: int = 42):
        """
        Initialize entity matcher.
        
        Args:
            threshold: Decision threshold for matching
            random_state: Random seed for reproducibility
        """
        self.threshold = threshold
        self.random_state = random_state
        self.model = None
        self.calibrated_model = None
        self.feature_names = None
        
    def prepare_training_data(self, features_df: pd.DataFrame, 
                             ground_truth_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with labels.
        
        Args:
            features_df: DataFrame with pair features
            ground_truth_df: DataFrame with ground truth matches
            
        Returns:
            Tuple of (features_df_with_labels, labels)
        """
        # Parse ground truth into set format
        gt_dict = {}
        for _, row in ground_truth_df.iterrows():
            s1_id = row['source1_entity_id']
            matches_str = row['matched_entity_ids']
            
            if pd.isna(matches_str) or str(matches_str).strip() == '':
                gt_dict[s1_id] = set()
            else:
                matches = set([m.strip() for m in str(matches_str).split(',') if m.strip()])
                gt_dict[s1_id] = matches
        
        # Add labels to features
        labels = []
        for _, row in features_df.iterrows():
            s1_id = row['source1_entity_id']
            cand_id = row['candidate_entity_id']
            
            true_matches = gt_dict.get(s1_id, set())
            is_match = 1 if cand_id in true_matches else 0
            labels.append(is_match)
        
        labels = pd.Series(labels, index=features_df.index)
        
        print(f"Training data prepared:")
        print(f"  Total pairs: {len(labels):,}")
        print(f"  Positive (matches): {labels.sum():,} ({100*labels.mean():.2f}%)")
        print(f"  Negative (non-matches): {(~labels.astype(bool)).sum():,} ({100*(1-labels.mean()):.2f}%)")
        
        return features_df, labels
    
    def train(self, features_df: pd.DataFrame, labels: pd.Series, 
             val_size: float = 0.2) -> Dict[str, any]:
        """
        Train the matching model.
        
        Args:
            features_df: DataFrame with pair features
            labels: Series with match labels
            val_size: Validation split size
            
        Returns:
            Dictionary with training metrics
        """
        # Identify feature columns (exclude ID columns)
        feature_cols = [c for c in features_df.columns if c not in ['source1_entity_id', 'candidate_entity_id']]
        self.feature_names = feature_cols
        
        X = features_df[feature_cols]
        y = labels
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\\nTraining LightGBM model...")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Validation samples: {len(X_val):,}")
        
        # Train LightGBM with class weights to handle imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        self.model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # Calibrate probabilities
        print("\\nCalibrating probabilities...")
        self.calibrated_model = CalibratedClassifierCV(
            self.model, method='isotonic', cv='prefit'
        )
        self.calibrated_model.fit(X_val, y_val)
        
        # Evaluate on validation set
        val_proba = self.calibrated_model.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= self.threshold).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics = {
            'precision': precision_score(y_val, val_pred),
            'recall': recall_score(y_val, val_pred),
            'f1': f1_score(y_val, val_pred),
        }
        
        print(f"\\n✓ Model training complete")
        print(f"  Validation Precision: {metrics['precision']:.4f}")
        print(f"  Validation Recall: {metrics['recall']:.4f}")
        print(f"  Validation F1: {metrics['f1']:.4f}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\\nTop 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.1f}")
        
        return metrics
    
    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict match probabilities for candidate pairs.
        
        Args:
            features_df: DataFrame with pair features
            
        Returns:
            Array of match probabilities
        """
        if self.calibrated_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X = features_df[self.feature_names]
        return self.calibrated_model.predict_proba(X)[:, 1]
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict matches for candidate pairs.
        
        Args:
            features_df: DataFrame with pair features
            
        Returns:
            Array of binary predictions
        """
        proba = self.predict_proba(features_df)
        return (proba >= self.threshold).astype(int)
    
    def tune_threshold(self, features_df: pd.DataFrame, labels: pd.Series,
                      beta: float = 0.5) -> Tuple[float, Dict[str, float]]:
        """
        Tune decision threshold to maximize F_beta score.
        
        Args:
            features_df: DataFrame with validation features
            labels: Series with validation labels
            beta: Beta parameter for F_beta score (0.5 for F_0.5)
            
        Returns:
            Tuple of (best_threshold, metrics_at_best_threshold)
        """
        print(f"\\nTuning threshold for F_{beta} score...")
        
        proba = self.predict_proba(features_df)
        
        # Try different thresholds
        thresholds = np.arange(0.1, 1.0, 0.05)
        best_score = 0.0
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in thresholds:
            pred = (proba >= threshold).astype(int)
            
            tp = ((pred == 1) & (labels == 1)).sum()
            fp = ((pred == 1) & (labels == 0)).sum()
            fn = ((pred == 0) & (labels == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # F_beta score
            if precision + recall > 0:
                f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
            else:
                f_beta = 0.0
            
            if f_beta > best_score:
                best_score = f_beta
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'f_beta': f_beta,
                    'precision': precision,
                    'recall': recall,
                }
        
        self.threshold = best_threshold
        
        print(f"\\n✓ Best threshold: {best_threshold:.3f}")
        print(f"  F_{beta}: {best_metrics['f_beta']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall: {best_metrics['recall']:.4f}")
        
        return best_threshold, best_metrics
    
    def save(self, filepath: str):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'calibrated_model': self.calibrated_model,
                'threshold': self.threshold,
                'feature_names': self.feature_names,
                'random_state': self.random_state
            }, f)
        print(f"\\n✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.calibrated_model = data['calibrated_model']
        self.threshold = data['threshold']
        self.feature_names = data['feature_names']
        self.random_state = data['random_state']
        
        print(f"✓ Model loaded from {filepath}")
        print(f"  Threshold: {self.threshold:.3f}")
        print(f"  Features: {len(self.feature_names)}")


def calculate_f_beta_score(y_true: pd.Series, y_pred: pd.Series, beta: float = 0.5) -> Dict[str, float]:
    """
    Calculate F_beta score and component metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        beta: Beta parameter for F_beta
        
    Returns:
        Dictionary with metrics
    """
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall > 0:
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    else:
        f_beta = 0.0
    
    return {
        'f_beta': f_beta,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }
