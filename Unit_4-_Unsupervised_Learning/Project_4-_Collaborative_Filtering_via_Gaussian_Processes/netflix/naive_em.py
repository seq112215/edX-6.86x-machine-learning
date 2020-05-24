"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def gaussian(x: np.ndarray, mean: np.ndarray, var: float) -> float:
    """Computes the probablity of vector x under a normal distribution

    Args:
        x: (d, ) array holding the vector's coordinates
        mean: (d, ) mean of the gaussian
        var: variance of the gaussian

    Returns:
        float: the probability
    """
    d = len(x)
    log_prob = -d / 2.0 * np.log(2 * np.pi * var)
    log_prob -= 0.5 * ((x - mean)**2).sum() / var
    return np.exp(log_prob)


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    """ My solution:
    n, d = X.shape
    K = mixture.mu.shape[0]

    numerator = np.zeros((n, K))
    for i in range(n):
        for j in range(K):
            numerator[i, j] = mixture.p[j] / (2 * np.pi * mixture.var[j]) ** (d / 2) * \
                         np.exp(- np.sum((X[i] - mixture.mu[j])**2) / (2 * mixture.var[j]))
    denominator = np.sum(numerator, axis=1)
    post = np.asarray([numerator[i, :] / denominator[i] for i in range(n)])

    ln_like = float(np.sum(np.log(denominator)))

    return post, ln_like
    """
    # Instructor's solution: (same)
    n, _ = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))

    ll = 0
    for i in range(n):
        for j in range(K):
            likelihood = gaussian(X[i], mixture.mu[j], mixture.var[j])
            post[i, j] = mixture.p[j] * likelihood
        total = post[i, :].sum()
        post[i, :] = post[i, :] / total
        ll += np.log(total)

    return post, ll


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    """ My solution:
    n, d = X.shape
    K = post.shape[1]

    n_hat = np.sum(post, axis=0)
    p_hat = n_hat / n
    mu_hat = np.asarray(
        [np.divide((post.T @ X)[j], n_hat[j]) for j in range(K)])
    var_hat = np.asarray([(post.T @ np.sum((X - mu_hat[j]) ** 2, axis=1))[j] /
                          (d * n_hat[j]) for j in range(K)])

    return GaussianMixture(mu=mu_hat, var=var_hat, p=p_hat)
    """
    # Instructor's solution: (same)
    n, d = X.shape
    _, K = post.shape

    n_hat = post.sum(axis=0)
    p = n_hat / n

    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        # Computing mean
        mu[j, :] = (X * post[:, j, None]).sum(axis=0) / n_hat[j]
        # Computing variance
        sse = ((mu[j] - X) ** 2).sum(axis=1) @ post[:, j]
        var[j] = sse / (d * n_hat[j])

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    # Instructor's solution
    prev_ll = None
    ll = None
    while prev_ll is None or ll - prev_ll > 1e-6 * np.abs(ll):
        prev_ll = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, ll
