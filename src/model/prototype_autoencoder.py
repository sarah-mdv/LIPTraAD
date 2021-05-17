import logging

import torch

from src.model.ae_modules import PrototypeAutoencoder

from src.model.standard_autoencoder import StandardAutoencoderModel

from src.model.misc import (
    diversity_loss,
    interpretability_loss
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