import pandas as pd
import numpy as np


def fill_nan(fill_nan_method, feature):
    """
    (Inner function) 
    fill feature's nan value
    Input:
        fill_nan_method: 'minus', 'plus', 'mean', 'drop', 'max', 'min'
        feature: single column dataframe
        label: label dataframe, only useful for 'drop' method
    Return:
        trans_feature: nan-filled feature
        trans_label: (optional output)
    """
    if fill_nan_method == 'minus':
        value = feature.min()-1
    elif fill_nan_method == 'mean':
        value = feature.mean()
    elif fill_nan_method == 'max':
        value = feature.max()
    elif fill_nan_method == 'min':
        value = feature.min()
    elif fill_nan_method == 'plus':
        value = feature.max()+1
    else:
        raise ValueError('Wrong fill nan method.')
    trans_feature = feature.fillna(value)
    return trans_feature, value