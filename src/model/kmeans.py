import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed


def _parallel_compute_distance(x, cluster):
    n_samples = x.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((x[i] - cluster) ** 2, axis=0))
    return dis_mat


class batch_KMeans(object):

    def __init__(self, h_size, n_prototypes, n_jobs):
        self.n_features = h_size
        self.n_clusters = n_prototypes
        self.clusters = np.zeros((self.n_clusters, self.n_features))
        self.count = 100 * np.ones(self.n_clusters)  # serve as learning rate
        self.n_jobs = n_jobs

    def _compute_dist(self, x):
        dis_mat = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_distance)(x, self.clusters[i])
            for i in range(self.n_clusters))
        dis_mat = np.hstack(dis_mat)

        return dis_mat

    def init_cluster(self, x):
        """ Generate initial clusters using sklearn.Kmeans and kmeans++ initialization """
        model = KMeans(n_clusters=self.n_clusters,
                       n_init=20, init='k-means++')
        model.fit(x)
        self.clusters = model.cluster_centers_  # copy cluster centers

    def update_cluster(self, x, cluster_idx):
        """ Update clusters in Kmeans on a batch of data """
        n_samples = x.shape[0]
        for i in range(n_samples):
            self.count[cluster_idx] += 1
            eta = 1.0 / self.count[cluster_idx]
            updated_cluster = ((1 - eta) * self.clusters[cluster_idx] +
                                    eta * x[i])
            self.clusters[cluster_idx] = updated_cluster

    def update_assign(self, x):
        """ Assign samples in `x` to clusters """
        dis_mat = self._compute_dist(x)

        return np.argmin(dis_mat, axis=1)

    def get_cluster_prototype(self, x, cluster_idx):
        """Return the prototype representative of the cluster"""
        dist_mat = _parallel_compute_distance(x, self.clusters[cluster_idx])

        return x[np.argmin(dist_mat, axis=0)]
