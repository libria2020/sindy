import numpy as np
from sklearn.linear_model import ridge_regression


def stls(x, y, alpha, threshold, iterations=10):
    Xi = ridge_regression(x, y, alpha)

    for i in range(iterations):
        small_indexes = np.abs(Xi) < threshold
        Xi[small_indexes] = 0

        big_indexes = ~small_indexes

        Xi[big_indexes] = ridge_regression(x[:, big_indexes], y, alpha)

    return Xi  # Plot u and u_dot
