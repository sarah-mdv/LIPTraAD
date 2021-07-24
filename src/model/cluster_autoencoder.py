import logging
import time

import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.model.misc import compute_squared_distances

from src.model.standard_autoencoder import StandardAutoencoderModel
from src.model.ae_modules import ClusterAutoencoder
from src.misc import output_hidden_states
from src.preprocess.dataloader import (
    DataSet,
    Random
)
from src.model.misc import print_model_parameters

LOGGER = logging.getLogger(__name__)


class ClusterAutoencoderModel(StandardAutoencoderModel):
    def __init__(self, results_dir):
        super().__init__(results_dir)
        self.name = "ClusterAE"

    def build_model(self, nb_classes, nb_measures, h_size,
                    h_drop, i_drop, mean, stds, n_prototypes):
        self.ae_model = ClusterAutoencoder(nb_classes=nb_classes, nb_measures=nb_measures, h_size=h_size,
                                h_drop=h_drop, i_drop=i_drop, n_clusters=n_prototypes)
        self.n_clusters = n_prototypes
        self.cluster_model = KMeans(n_clusters=self.n_clusters, init="k-means++")
        setattr(self.ae_model, 'mean', mean)
        setattr(self.ae_model, 'stds', stds)

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.ae_model.to(self.device)

    def set_encoder(self, encoder_model):
        self.ae_model = torch.load(encoder_model)
        self.ae_model.to(self.device)


    def fit_clusters(self, dataset:Random, data, fold ):
        self.ae_model.freeze_encoder()

        batch_x = self.collect_hidden_states(dataset)

        # Only get time points that exist (not imputed values, check cat mask)
        mask = (batch_x.DX_mask == 1) & ~(batch_x.TP == 0.)
        true_pts = batch_x[mask]
        hiddens = true_pts.iloc[:, -(self.ae_model.h_size + 1):-1].to_numpy()
        assert hiddens.shape[1] == self.ae_model.h_size

        self.cluster_model = self.cluster_model.fit(hiddens)

        # Output all hidden states with their clusters


        cluster_centroids = self.get_cluster_centroids(true_pts)

        true_pts["Clusters"] = self.cluster_model.labels_
        output_hidden_states(true_pts, data, self.results_dir, fold, first_data_point = 0)
        cluster_hidden = (tup[2] for tup in cluster_centroids)
        cluster_hidden = self.cluster_model.cluster_centers_
        cluster_hidden = np.array(np.vstack(cluster_hidden))
        self.ae_model.set_clusters(cluster_hidden)

        # Reset optimizer
        self.optimizer = torch.optim.Adam(
            self.ae_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_cluster_centroids(self, batch_x):
        centers = self.cluster_model.cluster_centers_
        LOGGER.debug(centers)
        cluster_list = [None]*self.n_clusters
        for cl in range(self.n_clusters):
            # Only get time points that exist (not imputed values, check cat mask)
            mask = (np.array(self.cluster_model.labels_) == cl)
            assigned = batch_x[mask]
            dist_mat = self._parallel_compute_distance(assigned.iloc[:, -(self.ae_model.h_size + 1):-1].to_numpy(),
                                                       np.array(centers[cl]))
            closest_dp = np.argmax(dist_mat)
            LOGGER.debug(dist_mat[closest_dp])
            rid = assigned.iloc[closest_dp, 0]
            tp = assigned.iloc[closest_dp, 1]
            hidden = assigned.iloc[closest_dp, -(self.ae_model.h_size + 1):-1]
            cluster_list[cl] = [int(rid), int(tp), hidden]
        return cluster_list

    def save_model(self, results_dir):
        super().save_model()
        t = time.localtime()
        current_time = time.strftime("%H%M%S", t)
        out = results_dir / "{}_{}.pt".format(self.name, current_time)
        torch.save(self.ae_model, out)
        LOGGER.info("Model {} saved to {}".format(self.name, out))
        return out

    def _parallel_compute_distance(self, x, cluster):
        squared_distances = np.add.outer((x * x).sum(axis=-1), (cluster * cluster).sum(axis=-1)) - 2 * np.dot(x,
                                                                                                    cluster.T)
        sim_mat = np.exp(np.negative(squared_distances))
        return sim_mat
