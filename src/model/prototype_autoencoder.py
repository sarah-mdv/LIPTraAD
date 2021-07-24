import logging

import numpy as np
import pandas as pd
import torch
from torch.nn.parameter import Parameter

from src.model.ae_modules import (
    PrototypeAutoencoder,
    PrototypeTransitionAutoencoder
)

from src.model.standard_autoencoder import StandardAutoencoderModel

from src.model.misc import (
    diversity_loss,
    interpretability_loss,
compute_squared_distances
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
                    h_drop, i_drop, mean, stds, n_prototypes, n_transition_prototypes):
        self.ae_model = PrototypeAutoencoder(nb_classes=nb_classes, nb_measures=nb_measures, h_size=h_size,
                                h_drop=h_drop, i_drop=i_drop, n_prototypes=n_prototypes,
                                             n_transition_prototypes=n_transition_prototypes)
        setattr(self.ae_model, 'mean', mean)
        setattr(self.ae_model, 'stds', stds)

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.ae_model.to(device)

    def reg_loss(self, hidden, div_lambda=0.005, c_lamda=0.01, e_lambda=0.01):
        diversity = diversity_loss(self.ae_model.prototype.prototypes)
        hidden = hidden.reshape(-1, self.ae_model.h_size)
        clustering_reg, evidence_reg = interpretability_loss(hidden,
                                                             self.ae_model.prototype.prototypes.data)
        #LOGGER.debug("Diversity {} clustering {} evidence {}".format(diversity, clustering_reg, evidence_reg))
        diversity = div_lambda * diversity / self.ae_model.prototype.n_prototypes
        clustering_reg = c_lamda * clustering_reg / hidden.shape[0]
        evidence_reg = e_lambda * evidence_reg /self.ae_model.prototype.n_prototypes
        # LOGGER.debug("div {} clu {} ev {}".format(diversity, clustering_reg, evidence_reg))
        #evidence_reg = torch.tensor([0.0])
        return diversity, clustering_reg, evidence_reg

    def train_epoch(self, dataset:Random, epoch, w_ent=1, w_reg=0.5, projection_step=10):
        # if epoch == 0:
        #     self.project_prototypes(dataset)
        if not epoch ==0 and epoch%projection_step == 0:
            self.project_prototypes(dataset)
        t = super().train_epoch(dataset, epoch, w_ent, w_reg)
        return t

    def fit(self, data: DataSet, epochs: int, seed: int, fold, w_ent=1, w_reg=1, hidden=False):
        super().fit(data, epochs, seed, w_ent, w_reg, hidden)
        p = pd.DataFrame(self.ae_model.prototype.prototypes.detach().numpy())
        p.to_csv((self.results_dir / "prototype_hiddens_{}.csv".format(fold)), index=False)
        rids, tps, prototypes = self.get_prototypes(data.train)
        p = pd.DataFrame(np.concatenate((rids,tps,prototypes),axis=1))
        p.to_csv(self.results_dir / "prototype_ids_{}.csv".format(fold), index=False)


    def project_prototypes(self, dataset:DataSet):
        (rids, tps, prototypes) = self.get_prototypes(dataset)
        self.ae_model.prototype.prototypes.data = torch.from_numpy(prototypes)
        if len(np.unique(rids)) != self.ae_model.prototype.n_prototypes:
            LOGGER.debug("There is a duplicate prototype")
        LOGGER.debug("RIDS {}".format(rids.squeeze()))
        LOGGER.debug("TPs {}".format(tps.squeeze()))


    def get_prototypes(self, dataset:Random):
        batch_x = self.collect_hidden_states(dataset)
        prototype_list = [None]*self.ae_model.prototype.n_prototypes
        rids = np.zeros((self.ae_model.prototype.n_prototypes, 1))
        tps = np.zeros((self.ae_model.prototype.n_prototypes, 1))
        # Only get time points that exist (not imputed values, check cat mask)
        mask = (batch_x.DX_mask == 1) & ~(batch_x.TP == 0.)
        trans_mask = mask & (batch_x.DXCHANGE)
        true_pts = batch_x[mask]
        true_pts_trans = batch_x[trans_mask]
        hiddens = true_pts.iloc[:, -(self.ae_model.h_size +1):-1]
        hiddens_trans = true_pts_trans.iloc[:, -(self.ae_model.h_size +1):-1]
        for i, p in enumerate(self.ae_model.prototype.prototypes):
            # Select prototypes from transitioners if
            h = hiddens_trans if i < self.ae_model.prototype.n_transition_prototypes else hiddens
            pts = true_pts_trans if i < self.ae_model.prototype.n_transition_prototypes else true_pts
            dist_mat = compute_squared_distances(torch.tensor(h.values), p.clone().unsqueeze(0))
            closest_dp = torch.argmin(dist_mat).item()
            rids[i] = pts.RID.iloc[closest_dp]
            tps[i] = pts.TP.iloc[closest_dp]
            prototype_list[i] = [h.iloc[closest_dp,:]]
        #prototype_hidden = (tup for tup in prototype_list)
        prototype_hidden = np.vstack(prototype_list)
        prototype_hidden = np.array(prototype_hidden)
        assert len(rids) == len(tps) == len(prototype_hidden) == self.ae_model.prototype.n_prototypes
        return (rids, tps, prototype_hidden)

class PrototypeTransitionAutoencoderModel(PrototypeAutoencoderModel):
    def __init__(self, results_dir):
        super().__init__(results_dir)
        self.name = "PrototypeTransitionAE"

    def build_model(self, nb_classes, nb_measures, h_size,
                    h_drop, i_drop, mean, stds, n_prototypes, n_transition_prototypes):
        self.ae_model = PrototypeTransitionAutoencoder(nb_classes=nb_classes, nb_measures=nb_measures, h_size=h_size,
                                             h_drop=h_drop, i_drop=i_drop, n_prototypes=n_prototypes,
                                             n_transition_prototypes=n_transition_prototypes)
        setattr(self.ae_model, 'mean', mean)
        setattr(self.ae_model, 'stds', stds)

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
