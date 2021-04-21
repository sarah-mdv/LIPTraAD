import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

from src.model.autoencoder import Autoencoder
from src.model.ae_modules import StandardAutoencoder
from src.preprocess.dataloader import (
    DataSet,
    Random,
)
from src import misc

from src.model.misc import(
    ent_loss,
    mae_loss,
    to_cat_seq,
)

LOGGER = logging.getLogger(__name__)

class StandardAutoencoderModel(Autoencoder):
    def __init__(self, results_dir):
        super().__init__("StandardAE")
        self.results_dir = results_dir


    def build_model(self, nb_classes, nb_measures, h_size,
                    h_drop, i_drop, nb_layers, mean, stds):
        self.model = StandardAutoencoder(nb_classes=nb_classes, nb_measures=nb_measures, h_size=h_size,
                                nb_layers=nb_layers, h_drop=h_drop, i_drop=i_drop)
        setattr(self.model, 'mean', mean)
        setattr(self.model, 'stds', stds)

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)

    def build_optimizer(self, lr, weight_decay):
        super().build_optimizer(lr, weight_decay)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, dataset:Random, w_ent=1.):
        """
        Train a recurrent model for 1 epoch
        Args:
            dataset: training data
            w_ent: weight given to entropy loss
        Returns:
            cross-entropy loss of epoch
            mean absolute error (MAE) loss of epoch
        """

        self.model.train()
        total_ent = total_mae = 0
        for batch in dataset:
            torch.autograd.set_detect_anomaly(True)
            if len(batch['tp']) == 1:
                continue

            self.optimizer.zero_grad()

            pred_cat, pred_val = self.model(to_cat_seq(batch['cat']), batch['val'])

            mask_cat = batch['cat_msk']
            mask_val = batch['val_msk']
            assert mask_cat.sum() > 0
            """
            Are we learning the identity function at t or not
            Look at the predictions at t

            Check the gradients from the hidden state

            Check with synthetic data on the model, does it learn simple patterns,
            With deterministic output so that we know it should be learned

            """

            batch_ent = batch_mae = 0
            for i in range(len(pred_cat)):
                curr_cat_mask = np.full(mask_cat.shape, False)
                curr_cat_mask[-(i+1):, :, :] = True
                curr_val_mask = np.full(mask_val.shape, False)
                curr_val_mask[-(i+1):, :, :] = True

                shifted_cat_mask = np.roll(mask_cat, len(mask_cat) - (i+1))
                shifted_val_mask = np.roll(mask_val, len(mask_val) - (i+1))

                assert shifted_cat_mask.shape == mask_cat.shape == curr_cat_mask.shape
                assert shifted_val_mask.shape == mask_val.shape == curr_val_mask.shape

                curr_cat_mask = curr_cat_mask & shifted_cat_mask
                curr_val_mask = curr_val_mask & shifted_val_mask

                true_cat = np.roll(batch['true_cat'], len(batch['true_cat']) - (i+1))
                true_val = np.roll(batch['true_val'], len(batch['true_cat']) - (i+1))

                batch_ent += ent_loss(pred_cat[i], true_cat, curr_cat_mask)
                batch_mae += mae_loss(pred_val[i].clone(), true_val, curr_val_mask)
            total_loss = batch_mae + w_ent * batch_ent
            total_loss.backward()
            self.optimizer.step()

            batch_size = mask_cat.shape[1]
            total_ent += batch_ent.item() * batch_size
            total_mae += batch_mae.item() * batch_size

        return total_ent / len(dataset.subjects), total_mae / len(dataset.subjects)



    def fit(self, data: DataSet, epochs: int, seed:int, hidden=False):
        """
        Fit the autoencoder model on the data
        Return the hidden states if hidden is True
        """
        super().fit(data, epochs)
        t = datetime.now().strftime("%H:%M:%S")

        np.random.seed(seed)
        torch.manual_seed(seed)

        start = time.time()
        loss_table = {}
        try:
            for i in range(epochs):
                loss = self.train_epoch(data.train)
                loss_table[i] = loss
                log_info = (i + 1, epochs, misc.time_from(start)) + loss
                LOGGER.info('Epoch: %d/%d %s ENT %.3f, MAE %.3f' % log_info)
        except KeyboardInterrupt:
            LOGGER.error('Early exit')
        df = pd.DataFrame.from_dict(loss_table, orient='index', columns=["ENT", "MAE"])
        df.to_csv(self.results_dir / Path("{}_loss.csv".format(t)))
        LOGGER.debug("End of fitting")

    def predict(self, data: DataSet):
        pass

    def save_model(self):
        super().save_model()
        out = self.results_dir / Path("{}.pt".format(self.name))
        torch.save(self.model, out)
        LOGGER.info("Model {} saved to {}".format(self.name, out))
        return out

    def run(self, data: DataSet, epochs: int, predict=False, seed=0):
        LOGGER.debug("Not implemented sorry")
        pass