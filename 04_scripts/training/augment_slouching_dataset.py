"""
augment_slouching_dataset.py — Slouching Data Augmentation Module

Provides methods to augment feature vectors for minority/confused classes (slouching, leaning_forward)
to improve classifier recall and decision boundary precision.

Strategies:
  1. Gaussian Jitter: Perturbs existing feature vectors with Gaussian noise scaled to feature standard deviation.
  2. Intra-class Convex Interpolation (SMOTE-like): Generates synthetic vectors as linear combinations of true samples.
  3. Boundary Jitter: Perturbs samples near the decision boundaries between upright and slouching.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


def augment_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_class_ids: List[int],
    n_samples_per_class: int = 150,
    noise_level: float = 0.015,
    interpolation_ratio: float = 0.5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augments training data for specific class IDs.
    
    Args:
        X_train: np.ndarray of shape (N, D) - Training feature matrix
        y_train: np.ndarray of shape (N,) - Class labels
        target_class_ids: List of class IDs to augment (e.g. [1, 5] for leaning_forward and slouching)
        n_samples_per_class: Number of synthetic samples to generate per target class
        noise_level: Standard deviation multiplier for Gaussian noise relative to feature std
        interpolation_ratio: Ratio of interpolation vs pure noise (0.0 to 1.0)
        random_state: Seed for reproducibility
        
    Returns:
        X_aug, y_aug: Combined original and synthetic training data
    """
    rng = np.random.RandomState(random_state)
    
    X_syn_list = []
    y_syn_list = []
    
    # Compute std per feature across the training set (ignoring NaNs)
    feat_stds = np.nanstd(X_train, axis=0)
    feat_stds = np.where(feat_stds == 0, 1e-6, feat_stds)
    # Replace NaNs in feature std with global mean std
    mean_std = np.nanmean(feat_stds)
    feat_stds = np.nan_to_num(feat_stds, nan=mean_std)
    
    for cls_id in target_class_ids:
        cls_mask = (y_train == cls_id)
        X_cls = X_train[cls_mask]
        
        if len(X_cls) == 0:
            continue
            
        n_synth = n_samples_per_class
        n_interp = int(n_synth * interpolation_ratio)
        n_jitter = n_synth - n_interp
        
        # 1. Intra-class convex interpolation (SMOTE-like)
        if n_interp > 0 and len(X_cls) > 1:
            idx_a = rng.choice(len(X_cls), size=n_interp, replace=True)
            idx_b = rng.choice(len(X_cls), size=n_interp, replace=True)
            # Ensure pairs are not the same sample where possible
            for i in range(n_interp):
                while idx_a[i] == idx_b[i] and len(X_cls) > 1:
                    idx_b[i] = rng.choice(len(X_cls))
                    
            lambdas = rng.uniform(0.15, 0.85, size=(n_interp, 1))
            syn_interp = lambdas * X_cls[idx_a] + (1 - lambdas) * X_cls[idx_b]
            
            # Add small noise to interpolated samples
            noise = rng.normal(0, noise_level * 0.5, size=syn_interp.shape) * feat_stds
            syn_interp += noise
            
            X_syn_list.append(syn_interp)
            y_syn_list.append(np.full(n_interp, cls_id))
            
        # 2. Gaussian Jitter on random existing samples of this class
        if n_jitter > 0:
            idx_base = rng.choice(len(X_cls), size=n_jitter, replace=True)
            base_samples = X_cls[idx_base].copy()
            noise = rng.normal(0, noise_level, size=base_samples.shape) * feat_stds
            syn_jitter = base_samples + noise
            
            X_syn_list.append(syn_jitter)
            y_syn_list.append(np.full(n_jitter, cls_id))
            
    if X_syn_list:
        X_all_syn = np.vstack(X_syn_list)
        y_all_syn = np.concatenate(y_syn_list)
        
        X_aug = np.vstack([X_train, X_all_syn])
        y_aug = np.concatenate([y_train, y_all_syn])
    else:
        X_aug = X_train.copy()
        y_aug = y_train.copy()
        
    return X_aug, y_aug


def calculate_sample_weights(
    y: np.ndarray,
    class_weights: Optional[dict] = None
) -> np.ndarray:
    """
    Computes sample weights to give more importance to specific classes (e.g. slouching).
    
    Args:
        y: Target class array
        class_weights: Dict mapping class_id to weight multiplier (default: 1.0)
        
    Returns:
        np.ndarray of sample weights
    """
    if class_weights is None:
        return np.ones(len(y), dtype=np.float32)
        
    weights = np.ones(len(y), dtype=np.float32)
    for cls_id, weight in class_weights.items():
        weights[y == cls_id] = float(weight)
        
    return weights
