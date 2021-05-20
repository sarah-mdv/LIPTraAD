import logging

import numpy as np
import torch
from torch.nn.parameter import Parameter

from src.model.ae_modules import PrototypeAutoencoder

from src.model.standard_autoencoder import StandardAutoencoderModel

from src.model.misc import (
    diversity_loss,
    interpretability_loss
)

from src.preprocess.dataloader import (
    DataSet,
    Random
)

LOGGER = logging.getLogger(__name__)


class PrototypeAutoencoderModel(StandardAutoencoderModel):
    def __init__(self, results_dir):
        super().__init__(results_dir)
        self.name = "PrototypeAE"

    def build_model(self, nb_classes, nb_measures, h_size,
                    h_drop, i_drop, mean, stds, n_prototypes):
        self.ae_model = PrototypeAutoencoder(nb_classes=nb_classes, nb_measures=nb_measures, h_size=h_size,
                                h_drop=h_drop, i_drop=i_drop, n_prototypes=n_prototypes)
        setattr(self.ae_model, 'mean', mean)
        setattr(self.ae_model, 'stds', stds)

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.ae_model.to(device)

    def reg_loss(self, hidden, div_lambda=1, c_lamda=1, e_lambda=1):
        diversity = diversity_loss(self.ae_model.prototype.prototypes)
        clustering_reg = interpretability_loss(self.ae_model.prototype.prototypes, hidden)
        evidence_reg = interpretability_loss(hidden, self.ae_model.prototype.prototypes)
        tot_loss = div_lambda * diversity + c_lamda * clustering_reg + e_lambda * evidence_reg
        return tot_loss / self.ae_model.prototype.n_prototypes


    def train_epoch(self, dataset:Random, epoch, w_ent=1, w_reg=0.5, projection_step=4):
        if not epoch == 0 and epoch%projection_step == 0:
            self.project_prototypes(dataset)
        return super().train_epoch(dataset, epoch, w_ent, w_reg)


    def project_prototypes(self, dataset:DataSet):
        (rids, tps, prototypes) = self.get_prototypes(dataset)
        self.ae_model.prototype.prototypes = Parameter(torch.from_numpy(prototypes))
        LOGGER.debug("RIDS {} TPs {}".format(rids, tps))


    def get_prototypes(self, dataset:Random):
        batch_x = self.collect_hidden_states(dataset)
        prototype_list = [None]*self.ae_model.prototype.n_prototypes
        rids = np.zeros((self.ae_model.prototype.n_prototypes, 1))
        tps = np.zeros((self.ae_model.prototype.n_prototypes, 1))
        # Only get time points that exist (not imputed values, check cat mask)
        mask = (batch_x.DX_mask == 1) & ~(batch_x.TP == 0.)
        true_pts = batch_x[mask]
        hiddens = true_pts.iloc[:, -self.ae_model.h_size:].to_numpy()
        LOGGER.debug(hiddens)
        for i, p in enumerate(self.ae_model.prototype.prototypes):
            dist_mat = self._parallel_compute_distance(hiddens, p.detach().numpy())
            closest_dp = np.argmin(dist_mat)
            rids[i] = true_pts.RID.iloc[closest_dp]
            tps[i] = true_pts.TP.iloc[closest_dp]
            prototype_list[i] = [hiddens[closest_dp]]
        #prototype_hidden = (tup for tup in prototype_list)
        prototype_hidden = np.vstack(prototype_list)
        prototype_hidden = np.array(prototype_hidden)
        assert len(rids) == len(tps) == len(prototype_hidden) == self.ae_model.prototype.n_prototypes
        return (rids, tps, prototype_hidden)

    def _parallel_compute_distance(self, x, cluster):
        n_samples = x.shape[0]
        dis_mat = np.zeros((n_samples, 1))
        for i in range(n_samples):
            dis_mat[i] += np.sqrt(np.sum((x[i] - cluster) ** 2, axis=0))
        return dis_mat