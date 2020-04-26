import numpy as np
from utilities import hyperboloidDist, generatePoints, randTheta, plot_poincare_clustering
from gradDescentHyperbolic import hyperGradDescent
from lossAlgorithms import HyperGradLoss


def _closest_mean(query, means):
    distances = [hyperboloidDist(query, means[k]) for k in range(means.shape[0])]
    return np.argmin(distances)


class KMeans:
    def __init__(self, x, k, init='random'):
        self.x = x
        self.k = k
        if init == 'random':
            self.means = self._init_rand_means()
        elif init == '++':
            self.means = self._init_pp_means()
        else:
            raise NotImplementedError

    def _init_rand_means(self):
        rand_nums = np.array([randTheta(self.x.shape[1] - 1) for _ in range(self.k)])
        print(rand_nums.shape)
        return rand_nums.reshape((self.k, self.x.shape[1]))

    def _init_pp_means(self):
        # Choose initial cluster center index
        cluster_centers = [np.random.randint(0, self.x.shape[0])]

        while len(cluster_centers) < self.k:
            # First, get distances from each data point to each cluster
            d_xs = []
            for i in range(self.x.shape[0]):
                if i in cluster_centers:
                    d_xs.append(0)
                    continue
                distances = [hyperboloidDist(self.x[i], self.x[clu_idx]) for clu_idx in cluster_centers]
                d_xs.append(min(distances) ** 2)
            # Now, normalize the d_xs so they define a probability distribution
            pr_x = [d_xs[idx] / sum(d_xs) for idx in range(len(d_xs))]

            # Finally, sample the new cluster center and add it to the working set
            idx_list = list(range(self.x.shape[0]))
            cluster_centers.append(np.random.choice(idx_list, p=pr_x))

        means = np.zeros((self.k, self.x.shape[1]))
        counter = 0
        for clu_idx in cluster_centers:
            means[counter, :] = self.x[clu_idx]
            counter += 1

        return means

    def _iter(self):
        """ Do one iteration of the KMeans clustering algorithm """
        # First, classify points based on current means
        cluster_assignments = []
        for idx1 in range(self.x.shape[0]):
            cluster_assignments.append(_closest_mean(self.x[idx1], self.means))

        cluster_assignments = np.array(cluster_assignments)
        # Second, update means to be averages of their clusters
        for k_idx in range(self.k):
            cluster_points = self.x[np.where(cluster_assignments == k_idx)]
            # If any points belong to the current mean, update it
            if len(cluster_points) > 0:
                # generate initial random theta value of proper dimension
                theta = randTheta(self.x.shape[1] - 1)
                _, centroid_list = hyperGradDescent(HyperGradLoss, theta, 100, 0.1, cluster_points, False)
                # get the converged centroid of the points
                self.means[k_idx] = np.reshape(centroid_list[-1], (self.x.shape[1],))
                # self.means[k_idx] = np.mean(cluster_points, axis=0)

    def _eval_obj(self):
        """ Evaluate and return the KMeans objective function """
        obj = 0
        # First, classify points
        cluster_assignments = []
        for idx1 in range(self.x.shape[0]):
            cluster_assignments.append(_closest_mean(self.x[idx1], self.means))

        cluster_assignments = np.array(cluster_assignments)
        # Second, sum up squared norms
        for k_idx in range(self.k):
            cluster_points = self.x[np.where(cluster_assignments == k_idx)]
            for x_idx in range(len(cluster_points)):
                obj += hyperboloidDist(self.means[k_idx], cluster_points[x_idx]) ** 2

        return obj

    def fit(self, max_iter=200):
        for idx in range(max_iter):
            previous_means = self.means.copy()
            self._iter()
            sum_of_norms = 0

            # Check for convergence
            for k_idx in range(self.k):
                sum_of_norms += np.linalg.norm(previous_means[k_idx] - self.means[k_idx])
            if sum_of_norms == 0:
                break
        print(self._eval_obj())
        print(idx)


def main():
    x = generatePoints(8, 2)
    print(x)
    print('Mean and variance of each feature:', x.mean(axis=0), x.var(axis=0))
    # k_means = KMeans(x, 5, init='++')
    k_means = KMeans(x, 4, init='random')

    k_means.fit()
    means = k_means.means
    print(means)
    plot_poincare_clustering(x, means)


if __name__ == '__main__':
    main()
