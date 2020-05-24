import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

mixture, post = common.init(X, K,  seed)
mixture, post, ln_like = em.run(X, mixture, post)

print(mixture)
