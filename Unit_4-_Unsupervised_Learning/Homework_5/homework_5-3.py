import numpy as np

pi1, pi2, mu1, mu2, s12, s22 = 0.5, 0.5, 6, 7, 1, 4
theta = [pi1, pi2, mu1, mu2, s12, s22]

x = [-1, 0, 4, 5, 6]


def log_likelihood(theta, x):
    pi1, pi2, mu1, mu2, s12, s22 = theta
    n = len(x)

    sum = 0
    for i in range(n):
        sum += np.log(pi1 / np.sqrt(2 * np.pi * s12) *
                      np.exp(-(x[i] - mu1) ** 2 / (2 * s12)) +
                      pi2 / np.sqrt(2 * np.pi * s22) *
                      np.exp(-(x[i] - mu2) ** 2 / (2 * s22))
                      )

    return sum


def main():
    print(log_likelihood(theta, x))


if __name__ == "__main__":
    main()
