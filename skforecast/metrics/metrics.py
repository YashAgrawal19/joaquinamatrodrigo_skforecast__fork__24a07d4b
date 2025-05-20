################################################################################
#                                metrics                                       #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Callable
import numpy as np
import pandas as pd
import inspect
from functools import wraps
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_log_error,
)

def _handle_nan_metric(func: Callable) -> Callable:
    """
    Wrap a metric function to handle NaN values by ignoring them.
    """
    @wraps(func)
    def wrapper(y_true, y_pred, *args, **kwargs):
        # Convert inputs to numpy arrays if they're pandas Series
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Find valid indices (not NaN in either array)
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        
        # If all values are NaN, return NaN
        if not np.any(valid_mask):
            return np.nan
        
        # Filter out NaN values
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        # Handle y_train if it's in kwargs
        if 'y_train' in kwargs and kwargs['y_train'] is not None:
            y_train = np.asarray(kwargs['y_train'])
            if isinstance(y_train, list):
                # For list of time series, filter NaN from each series
                y_train = [np.asarray(x)[~np.isnan(x)] for x in y_train]
            else:
                y_train = y_train[~np.isnan(y_train)]
            kwargs['y_train'] = y_train
        
        return func(y_true_valid, y_pred_valid, *args, **kwargs)
    
    return wrapper

def _get_metric(metric: str) -> Callable:
    """
    Get the corresponding scikit-learn function to calculate the metric.

    Parameters
    ----------
    metric : str
        Metric used to quantify the goodness of fit of the model.

    Returns
    -------
    metric : Callable
        scikit-learn function to calculate the desired metric.

    """
    allowed_metrics = [
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_log_error",
        "mean_absolute_scaled_error",
        "root_mean_squared_scaled_error",
    ]

    if metric not in allowed_metrics:
        raise ValueError((f"Allowed metrics are: {allowed_metrics}. Got {metric}."))

    metrics = {
        "mean_squared_error": mean_squared_error,
        "mean_absolute_error": mean_absolute_error,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "mean_squared_log_error": mean_squared_log_error,
        "mean_absolute_scaled_error": mean_absolute_scaled_error,
        "root_mean_squared_scaled_error": root_mean_squared_scaled_error,
    }

    # First wrap the metric to handle NaN values, then add y_train argument
    metric = _handle_nan_metric(metrics[metric])
    metric = add_y_train_argument(metric)

    return metric


def add_y_train_argument(func):
    """
    Add `y_train` argument to a function if it is not already present.

    Parameters
    ----------
    func : callable
        Function to which the argument is added.

    Returns
    -------
    wrapper : callable
        Function with `y_train` argument added.
    """
    sig = inspect.signature(func)
    
    if "y_train" in sig.parameters:
        return func

    new_params = list(sig.parameters.values()) + [
        inspect.Parameter("y_train", inspect.Parameter.KEYWORD_ONLY, default=None)
    ]
    new_sig = sig.replace(parameters=new_params)

    @wraps(func)
    def wrapper(*args, y_train=None, **kwargs):
        return func(*args, **kwargs)
    
    wrapper.__signature__ = new_sig
    
    return wrapper


def mean_absolute_scaled_error(
    y_true: Union[pd.Series, np.array],
    y_pred: Union[pd.Series, np.array],
    y_train: Union[list, pd.Series, np.array],
) -> float:
    """
    Mean Absolute Scaled Error (MASE)
    MASE is a scale-independent error metric that measures the accuracy of
    a forecast. It is the mean absolute error of the forecast divided by the
    mean absolute error of a naive forecast in the training set. The naive
    forecast is the one obtained by shifting the time series by one period.
    If y_train is a list of numpy arrays or pandas Series, it is considered
    that each element is the true value of the target variable in the training
    set for each time series. In this case, the naive forecast is calculated
    for each time series separately.

    Parameters
    ----------
    y_true : pd.Series, np.array
        True values of the target variable.
    y_pred : pd.Series, np.array
        Predicted values of the target variable.
    y_train : list, pd.Series, np.array
        True values of the target variable in the training set. If list, it
        is consider that each element is the true value of the target variable
        in the training set for each time series.

    Returns
    -------
    float
        MASE value.
    """
    if not isinstance(y_true, (pd.Series, np.ndarray)):
        raise TypeError("y_true must be a pandas Series or numpy array")
    if not isinstance(y_pred, (pd.Series, np.ndarray)):
        raise TypeError("y_pred must be a pandas Series or numpy array")
    if not isinstance(y_train, (list, pd.Series, np.ndarray)):
        raise TypeError("y_train must be a list, pandas Series or numpy array")
    if isinstance(y_train, list):
        for x in y_train:
            if not isinstance(x, (pd.Series, np.ndarray)):
                raise TypeError(
                    "When `y_train` is a list, each element must be a pandas Series "
                    "or numpy array"
                )
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must have at least one element")

    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Find valid indices (not NaN in either array)
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    # If all values are NaN, return NaN
    if not np.any(valid_mask):
        return np.nan
    
    # Filter out NaN values
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    if isinstance(y_train, list):
        # Filter NaN from each training series
        y_train_filtered = [np.asarray(x)[~np.isnan(x)] for x in y_train]
        naive_forecast = np.concatenate([np.diff(x) for x in y_train_filtered])
        # Remove any NaN that might have been created by diff
        naive_forecast = naive_forecast[~np.isnan(naive_forecast)]
        if len(naive_forecast) == 0:
            return np.nan
        mase = np.mean(np.abs(y_true_valid - y_pred_valid)) / np.mean(np.abs(naive_forecast))
    else:
        y_train = np.asarray(y_train)
        y_train_valid = y_train[~np.isnan(y_train)]
        if len(y_train_valid) < 2:  # Need at least 2 points to calculate diff
            return np.nan
        naive_forecast = np.diff(y_train_valid)
        mase = np.mean(np.abs(y_true_valid - y_pred_valid)) / np.mean(np.abs(naive_forecast))

    return mase


def root_mean_squared_scaled_error(
    y_true: Union[pd.Series, np.array],
    y_pred: Union[pd.Series, np.array],
    y_train: Union[list, pd.Series, np.array],
) -> float:
    """
    Root Mean Squared Scaled Error (RMSSE)
    RMSSE is a scale-independent error metric that measures the accuracy of
    a forecast. It is the root mean squared error of the forecast divided by
    the root mean squared error of a naive forecast in the training set. The
    naive forecast is the one obtained by shifting the time series by one period.
    If y_train is a list of numpy arrays or pandas Series, it is considered
    that each element is the true value of the target variable in the training
    set for each time series. In this case, the naive forecast is calculated
    for each time series separately.

    Parameters
    ----------
    y_true : pd.Series, np.array
        True values of the target variable.
    y_pred : pd.Series, np.array
        Predicted values of the target variable.
    y_train : list, pd.Series, np.array
        True values of the target variable in the training set. If list, it
        is consider that each element is the true value of the target variable
        in the training set for each time series.

    Returns
    -------
    float
        RMSSE value.
    """

    if not isinstance(y_true, (pd.Series, np.ndarray)):
        raise TypeError("y_true must be a pandas Series or numpy array")
    if not isinstance(y_pred, (pd.Series, np.ndarray)):
        raise TypeError("y_pred must be a pandas Series or numpy array")
    if not isinstance(y_train, (list, pd.Series, np.ndarray)):
        raise TypeError("y_train must be a list, pandas Series or numpy array")
    if isinstance(y_train, list):
        for x in y_train:
            if not isinstance(x, (pd.Series, np.ndarray)):
                raise TypeError(
                    "When `y_train` is a list, each element must be a pandas Series "
                    "or numpy array"
                )
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must have at least one element")

    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Find valid indices (not NaN in either array)
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    # If all values are NaN, return NaN
    if not np.any(valid_mask):
        return np.nan
    
    # Filter out NaN values
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    if isinstance(y_train, list):
        # Filter NaN from each training series
        y_train_filtered = [np.asarray(x)[~np.isnan(x)] for x in y_train]
        naive_forecast = np.concatenate([np.diff(x) for x in y_train_filtered])
        # Remove any NaN that might have been created by diff
        naive_forecast = naive_forecast[~np.isnan(naive_forecast)]
        if len(naive_forecast) == 0:
            return np.nan
        rmsse = np.sqrt(np.mean((y_true_valid - y_pred_valid) ** 2)) / np.sqrt(np.mean(naive_forecast ** 2))
    else:
        y_train = np.asarray(y_train)
        y_train_valid = y_train[~np.isnan(y_train)]
        if len(y_train_valid) < 2:  # Need at least 2 points to calculate diff
            return np.nan
        naive_forecast = np.diff(y_train_valid)
        rmsse = np.sqrt(np.mean((y_true_valid - y_pred_valid) ** 2)) / np.sqrt(np.mean(naive_forecast ** 2))
    
    return rmsse
