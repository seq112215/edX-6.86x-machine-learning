import numpy as np
import kmeans
import common
import naive_em
import em


X = np.loadtxt("toy_data.txt")
K = [1, 2, 3, 4]
seed = [0, 1, 2, 3, 4]


def run_kmeans(X, plot=False):
    """ My solution:
    for i in range(len(K)):
        for j in range(len(seed)):
            mixture, post = common.init(X, K[i], seed[j])
            mixture, post, cost = kmeans.run(X, mixture, post)
            print("K = {}, seed = {}, cost = {}".format(K[i], seed[j], cost))
            if plot:
                common.plot(X, mixture, post, "K={}, seed={}".format(K[i], seed[j]))
    """
    # Instructor's solution:
    for K in range(1, 5):
        min_cost = None
        best_seed = None
        for seed in range(0, 5):
            mixture, post = common.init(X, K, seed)
            mixture, post, cost = kmeans.run(X, mixture, post)
            if min_cost is None or cost < min_cost:
                min_cost = cost
                best_seed = seed

        mixture, post = common.init(X, K, best_seed)
        mixture, post, cost = kmeans.run(X, mixture, post)
        title = "K-means for K=, seed=, cost=".format(K, best_seed,
                                                      min_cost)
        print(title)
        common.plot(X, mixture, post, title)

# My solution:
def run_naive_em(X, plot=False):
    max_bic = None
    for i in range(len(K)):
        max_ln_like = None
        best_seed = None
        for j in range(len(seed)):
            mixture, post = common.init(X, K[i], seed[j])
            mixture, post, ln_like = naive_em.run(X, mixture, post)
            if max_ln_like is None or ln_like > max_ln_like:
                max_ln_like = ln_like
                best_seed = seed[j]
            if plot:
                common.plot(X, mixture, post, "K={}, seed={}".format(K[i], seed[j]))

        mixture, post = common.init(X, K[i], best_seed)
        mixture, post, ln_like = naive_em.run(X, mixture, post)
        bic = common.bic(X, mixture, ln_like)
        if max_bic is None or bic > max_bic:
            max_bic = bic
        print("K = {}, Max ln(likelihood) = {}, Best seed = {}, Max BIC = {}".
              format(K[i], max_ln_like, best_seed, max_bic))


# Instructor's solution:
def run_with_bic():
    max_bic = None
    for K in range(1, 5):
        max_ll = None
        best_seed = None
        for seed in range(0, 5):
            mixture, post = common.init(X, K, seed)
            mixture, post, ll = naive_em.run(X, mixture, post)
            if max_ll is None or ll > max_ll:
                max_ll = ll
                best_seed = seed

        mixture, post = common.init(X, K, best_seed)
        mixture, post, ll = naive_em.run(X, mixture, post)
        bic = common.bic(X, mixture, ll)
        if max_bic is None or bic > max_bic:
            max_bic = bic
        title = "EM for K={}, seed={}, ll={}, bic={}".format(K, best_seed, ll, bic)
        print(title)
        common.plot(X, mixture, post, title)


def run_em(X, plot=False):
    max_bic = None
    for i in range(len(K)):
        max_ln_like = None
        best_seed = None
        for j in range(len(seed)):
            mixture, post = common.init(X, K[i], seed[j])
            mixture, post, ln_like = em.run(X, mixture, post)
            if max_ln_like is None or ln_like > max_ln_like:
                max_ln_like = ln_like
                best_seed = seed[j]
            if plot:
                common.plot(X, mixture, post, "K={}, seed={}".format(K[i], seed[j]))

        mixture, post = common.init(X, K[i], best_seed)
        mixture, post, ln_like = em.run(X, mixture, post)
        bic = common.bic(X, mixture, ln_like)
        if max_bic is None or bic > max_bic:
            max_bic = bic
        print("K = {}, Max ln(likelihood) = {}, Best seed = {}, Max BIC = {}".
              format(K[i], max_ln_like, best_seed, max_bic))


def main():
    # run_kmeans(X, plot=False)
    """
    Visually judging by the plots, K = 2 seems to be the best for this data
    This does not, however, yield the lowest cost.

    For each K, select the seed that results in the lowest cost after running kmeans:
    Cost K=1: 5462.297452340001   ;    seed = any
    Cost K=2: 1684.9079502962372  ;    seed = any
    Cost K=3: 1329.5948671544297  ;    seed = 3, 4
    Cost K=4: 1035.4998265394659  ;    seed = 4
    """

    # run_naive_em(X, plot=False)
    """
    The maximum likelihood and BIC scores for each K using seeds 0, 1, 2, 3, 4 are:
    K = 1, seed = 0   ; ln(like) = -1307.2234317600935  ;   bic = -1315.505623136887
    K = 2, seed = 2   ; ln(like) = -1175.7146293666797  ;   bic = -1195.03979821832
    K = 3, seed = 0   ; ln(like) = -1138.8908996872672  ;   bic = -1169.2589347355095
    K = 4, seed = 4   ; ln(like) = -1138.601175699485   ;   bic = -1169.2589347355095

    Solution has for K=4 the bic=-1180.0121. 
    Not sure how, since all other numbers are the same.
    
    From these, the best K is 3.
    """
    # run_with_bic()

    # X = np.loadtxt("netflix_incomplete.txt")
    # run_em(X, plot=False)


if __name__ == "__main__":
    main()
