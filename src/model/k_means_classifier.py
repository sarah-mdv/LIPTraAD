import logging
import time

import numpy as np
import torch
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path

from src.model.classifier import Classifier
from src.model.dnc import DCN
from src.model.kmeans import batch_KMeans
from src.preprocess.dataloader_nguyen import (
    DataSet,
    Random,
    Sorted
)
from src import misc

from src.model.misc import (
    ent_loss,
    mae_loss,
    to_cat_seq,
)

LOGGER = logging.getLogger(__name__)


class RNNPrototypeClassifier(Classifier):
    def __init__(self, n_prototypes):
        super().__init__("RNNPro")
        self.rnn_encoder = None
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.beta = 1  # coefficient of the clustering term
        self.n_prototypes = n_prototypes
        self.hidden_size = 0

    def build_model(self, encoder_model, h_size, n_jobs, beta=1):
        self.beta = beta
        self.hidden_size = h_size
        #Load pretrained encoding module
        self.rnn_encoder = torch.load(encoder_model)
        self.rnn_encoder.to(self.device)

        #Load k-means clustering
        self.kmeans = batch_KMeans(self.hidden_size, self.n_prototypes, n_jobs)


    """
    Initialize clusters in self.kmeans after pre-training
    """
    def init_clusters(self, train_data:Random):
        LOGGER.debug("Initializing kmeans++ clusters with {} clusters".format(self.n_prototypes))
        batch_x = []
        for batch in train_data:
            latent_x = self.rnn_encoder(to_cat_seq(batch['cat']), batch['val'], latent=True)
            batch_x.append(latent_x[len(latent_x) - 1].detach().cpu().numpy())
        batch_x = np.vstack(batch_x)
        LOGGER.debug(batch_x.shape)
        self.kmeans.init_cluster(batch_x)

    def fit(self, data: DataSet, epochs: int):
        return 0
        # TODO

    def build_optimizer(self, lr, weight_decay):
        super().build_optimizer(lr, weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr, weight_decay=weight_decay)

    def predict(self, data: DataSet):
        return 0
        # TODO

    def run(self, data: DataSet, epochs: int, predict=False, seed=0):
        return 0
        # TODO
