import numpy as np

x = np.array([[0, -6],
              [4, 4],
              [0, 0],
              [-5, 2]])

z = np.array([[-5, 2],
              [0, -6]])

k = 2
n, d = x.shape


def closest_kmedoids(x, z, ord):
    """
    Assigns data x to cluster whose center z is closest, and the associated cost
        using k-medoids.

    :param x:
    :type x:
    :param z:
    :type z:
    :return:
    :rtype: dictionary, float
    """
    cost = 0
    clusters = {0: [], 1: []}
    for i in range(n):
        difference = np.subtract(x[i], z)
        norm = np.linalg.norm(difference, ord=ord, axis=1)
        minimum = np.amin(norm)
        cost += np.sum(minimum)

        cluster_member = np.where(norm == minimum)[0][0]
        clusters[cluster_member].append(x[i])

    return clusters, cost


def closest_kmeans(x, z, ord):
    """

    :param x:
    :type x:
    :param z:
    :type z:
    :return:
    :rtype:
    """
    cost = 0
    clusters = {0: [], 1: []}
    for i in range(n):
        difference = np.subtract(x[i], z)
        norm_sq = np.linalg.norm(difference, ord=ord, axis=1)
        minimum = np.amin(norm_sq)
        cost += np.sum(minimum)

        cluster_member = np.where(norm_sq == minimum)[0][0]
        clusters[cluster_member].append(x[i])

    centers = {0: [], 1: []}
    for i in centers.keys():
        centers[i] = np.average(clusters[i], axis=0)

    return clusters, cost, centers


def main():
    print(closest_kmedoids(x, z, ord=1))
    print(closest_kmedoids(x, z, ord=2))
    print(closest_kmeans(x, z, ord=1))


if __name__ == "__main__":
    main()
