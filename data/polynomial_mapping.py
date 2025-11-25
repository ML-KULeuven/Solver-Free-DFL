#!/usr/bin/env python
# coding: utf-8

import numpy as np


def genData(num_data, num_features, num_variables, deg=1, noise_width=0, seed=135):
    """
    Args:
        num_data (int): number of data points
        num_features (int): dimension of features
        num_variables (int, int): number of variables
        deg (int): data polynomial degree
        noise_width (float): half witdth of data random noise
        seed (int): random seed

    Returns:
       tuple: data features (np.ndarray), costs (np.ndarray)

    """
    # Positive integer parameter
    if type(deg) is not int:
        raise ValueError("deg = {} should be int.".format(deg))
    if deg <= 0:
        raise ValueError("deg = {} should be positive.".format(deg))

    # Set seed
    np.random.seed(seed)

    # Number of data points
    n = num_data

    # Dimension of features
    p = num_features

    # Dimension of the cost vector
    d = num_variables

    # Random matrix parameter B
    B = np.random.binomial(1, 0.5, (d, p))

    # Feature vectors
    x = np.random.normal(0, 1, (n, p))

    # Cost vectors
    c = np.zeros((n, d))
    for i in range(n):
        # Cost without noise
        ci = (np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 3) ** deg + 1
        # Rescale
        ci /= 3.5 ** deg
        # Noise
        epsilon = np.random.uniform(1 - noise_width, 1 + noise_width, d)
        ci *= epsilon
        c[i, :] = ci

    return x, c
