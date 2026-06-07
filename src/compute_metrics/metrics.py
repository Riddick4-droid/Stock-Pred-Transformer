#evaluation metrics. functions to compute standard regression and classification metrics
import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (returns percentage, e.g., 5.0 for 5%)."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def directional_accuracy(y_true_dir: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """
    Directional accuracy (percentage of correct direction predictions).
    Parameters
    y_true_dir : np.ndarray of shape (N,)
        True direction labels (0=Down, 1=Flat, 2=Up).
    y_pred_probs : np.ndarray of shape (N, 3)
        Predicted class probabilities from softmax.
    """
    pred_dir = np.argmax(y_pred_probs, axis=1)
    return np.mean(pred_dir == y_true_dir) * 100