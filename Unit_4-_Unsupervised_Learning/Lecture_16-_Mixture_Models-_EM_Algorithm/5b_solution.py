import numpy as np

x = np.asarray([0.2, -0.9, -1., 1.2, 1.8])
mu = np.asarray([-3, 2])
sigma2 = np.asarray([4, 4])
p = np.asarray([0.5, 0.5])

def posterior(x, mu, sigma2, p):
    numerator1 = p[0] / np.sqrt(2 * np.pi * sigma2[0]) * np.exp(-(x - mu[0])**2./(2 * sigma2[0]))
    numerator2 = p[1] / np.sqrt(2 * np.pi * sigma2[1]) * np.exp(-(x - mu[1])**2./(2 * sigma2[1]))

    denominator = numerator1 + numerator2

    return np.asarray([numerator1 / denominator, numerator2 / denominator])


print(posterior(x, mu, sigma2, p))


def p_hat(x, mu, sigma2, p):
    post = posterior(x, mu, sigma2, p)
    n = len(x)

    return np.sum(post, axis=1) / n


print(p_hat(x, mu, sigma2, p))


def mu_hat(x, mu, sigma2, p):
    post = posterior(x, mu, sigma2, p)
    phat = p_hat(x, mu, sigma2, p)
    n = len(x)

    return np.sum(x * post, axis=1) / (n * phat)


print(mu_hat(x, mu, sigma2, p))


def sigma2_hat(x, mu, sigma2, p):
    post = posterior(x, mu, sigma2, p)
    phat = p_hat(x, mu, sigma2, p)
    muhat = mu_hat(x, mu, sigma2, p)
    n = len(x)
    k = len(mu)

    sigma2hat = np.zeros((k, n))
    for j in range(k):
        for i in range(n):
            sigma2hat[j, i] = (x[i] - muhat[j]) ** 2

    return np.sum(sigma2hat * post, axis=1) / (n * phat)


print(sigma2_hat(x, mu, sigma2, p))
