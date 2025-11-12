"""
Metrics for evaluating medical image segmentation performance.

This module provides functions to compute similarity metrics between
ground truth and predicted segmentations.
"""
import numpy as np
from typing import Union


def numpy_dice(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1.0) -> float:
    """
    Compute the Dice Similarity Coefficient (DSC) between two binary arrays.
    
    The Dice coefficient is a measure of overlap between two binary segmentations.
    It ranges from 0 (no overlap) to 1 (perfect overlap).
    
    Args:
        y_true: Ground truth binary segmentation array
        y_pred: Predicted binary segmentation array
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
        
    Returns:
        Dice Similarity Coefficient as a float between 0 and 1
        
    Example:
        >>> gt = np.array([1, 1, 0, 0])
        >>> pred = np.array([1, 0, 1, 0])
        >>> numpy_dice(gt, pred)
        0.4
    """
    intersection = y_true.flatten() * y_pred.flatten()
    return (2.0 * intersection.sum() + smooth) / (y_true.sum() + y_pred.sum() + smooth)
