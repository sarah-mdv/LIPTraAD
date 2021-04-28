import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

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
    roll_mask
)

LOGGER = logging.getLogger(__name__)

class StandardAutoencoderModel(Autoencoder):
    def __init__(self, results_dir):
        super().__init__("StandardAE")
        self.results_dir = results_dir
        self.writer = SummaryWriter()


    def build_model(self, nb_classes, nb_measures, h_size,
                    h_drop, i_drop, mean, stds):
        self.model = StandardAutoencoder(nb_classes=nb_classes, nb_measures=nb_measures, h_size=h_size,
                                h_drop=h_drop, i_drop=i_drop)
        setattr(self.model, 'mean', mean)
        setattr(self.model, 'stds', stds)

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)


    def build_optimizer(self, lr, weight_decay):
        super().build_optimizer(lr, weight_decay)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, dataset:Random, epoch, w_ent=4):
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

            pred_cat, pred_val, hf, hb = self.model(torch.from_numpy(to_cat_seq(batch['cat'])),
                                                    torch.from_numpy(batch['val']),latent=True)


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

                curr_cat_mask = roll_mask(mask_cat, i)
                curr_val_mask = roll_mask(mask_val, i)

                true_cat = np.roll(batch['true_cat'], len(batch['true_cat']) - (i+1), axis=0)
                true_val = np.roll(batch['true_val'], len(batch['true_val']) - (i+1), axis=0)

                ent = ent_loss(pred_cat[i].clone(), true_cat, curr_cat_mask)
                mae = mae_loss(pred_val[i].clone(), true_val, curr_val_mask)
                batch_ent += ent
                batch_mae += mae
            total_loss = w_ent * batch_ent + batch_mae
            total_loss.backward()
            self.optimizer.step()

            batch_size = mask_cat.shape[1]
            total_ent += batch_ent.item() * batch_size
            total_mae += batch_mae.item() * batch_size

        total_ent = (total_ent / len(dataset.subjects))
        total_mae = (total_mae / len(dataset.subjects))
        self.writer.add_scalar("Loss", total_loss, epoch)
        self.writer.add_scalar("ENT", total_ent, epoch)
        self.writer.add_scalar("MAE", total_mae, epoch)

        self.writer.add_histogram("in.bias", self.model.fw_cell.W.bias, epoch)
        self.writer.add_histogram("in_weight", self.model.fw_cell.W.weight, epoch)
        self.writer.add_histogram("in_weight.grad", self.model.fw_cell.W.weight.grad, epoch)

        self.writer.add_histogram("fw_cells.bias", self.model.fw_cell.bias_hh, epoch)
        self.writer.add_histogram("fw_cells.weight_uh", self.model.fw_cell.weight_uh, epoch)
        self.writer.add_histogram("fw_cells.weight_uh.grad", self.model.fw_cell.weight_uh.grad, epoch)

        self.writer.add_histogram("bw_cells.bias", self.model.bw_cell.bias_hh, epoch)
        self.writer.add_histogram("bw_cells.weight_uh", self.model.bw_cell.weight_uh, epoch)
        self.writer.add_histogram("bw_cells.weight_uh.grad", self.model.bw_cell.weight_uh.grad, epoch)

        self.writer.add_histogram("out_cat.bias", self.model.hid2category.bias, epoch)
        self.writer.add_histogram("out_cat.weight", self.model.hid2category.weight, epoch)
        self.writer.add_histogram("out_cat.weight.grad", self.model.hid2category.weight.grad, epoch)

        # self.writer.add_histogram("out_val.bias", self.model.hid2measures.bias, epoch)
        # self.writer.add_histogram("out_val.weight", self.model.hid2measures.weight, epoch)
        # self.writer.add_histogram("out_val.weight.grad", self.model.hid2measures.weight.grad, epoch)

        return total_ent, total_mae



    def fit(self, data: DataSet, epochs: int, seed:int, hidden=False):
        """
        Fit the autoencoder model on the data
        Return the hidden states if hidden is True
        """
        super().fit(data, epochs)
        n_data = next(data.train)
        self.writer.add_graph(self.model, (torch.from_numpy(to_cat_seq(n_data['cat']).astype(float)),
                                           torch.from_numpy(n_data['true_val'].astype(float))))

        t = datetime.now().strftime("%H:%M:%S")

        np.random.seed(seed)
        torch.manual_seed(seed)

        start = time.time()
        loss_table = {}
        try:
            for i in range(epochs):
                loss = self.train_epoch(data.train, i)
                loss_table[i] = loss
                log_info = (i + 1, epochs, misc.time_from(start)) + loss
                LOGGER.info('Epoch: %d/%d %s ENT %.3f, MAE %.3f' % log_info)
        except KeyboardInterrupt:
            LOGGER.error('Early exit')
        df = pd.DataFrame.from_dict(loss_table, orient='index', columns=["ENT", "MAE"])
        df.to_csv(self.results_dir / Path("{}_loss.csv".format(t)))
        LOGGER.debug("End of fitting")
        self.writer.close()

    def predict(self, data: DataSet, fold_n:int, outdir):
        super().predict(data)
        self.model.eval()
        testdata = data.test

        ret = {}
        dx = []
        viscode = []
        RID = []
        dx_true = []

        for data in testdata:
            rid = data['rid']

            mask_cat = data["dx_mask"].squeeze()
            mask_val = data["val_mask"]


            icat = torch.from_numpy(np.asarray([misc.to_categorical(c, 3) for c in data['cat']]))
            ival = torch.from_numpy(data['val'][:, None, :])
            ocat, oval = self.model(icat, ival)

            all_tp = data['tp'].squeeze(axis=1)

            for i in range(len(all_tp)):
                # Only save the time points that have ground truth
                if not mask_cat[i]:
                    continue

                pred = ocat[i]

                curr_cat_mask = np.full(mask_cat.shape, False)
                curr_cat_mask[-(i+1):] = True

                shifted_cat_mask = np.roll(mask_cat, len(mask_cat) - (i+1), axis= 0)

                assert shifted_cat_mask.shape == mask_cat.shape == curr_cat_mask.shape


                curr_cat_mask = curr_cat_mask & shifted_cat_mask

                tp = np.roll(all_tp, len(mask_cat) - (i+1), axis=0)[curr_cat_mask]
                rids = np.repeat(rid, len(tp))

                true_cat = np.roll(data['dx_truth'], len(data['dx_truth']) - (i+1), axis=0)

                pred = pred.reshape(pred.size(0) * pred.size(1), -1)
                curr_cat_mask = curr_cat_mask.reshape(-1, 1)

                o_true = pred.new_tensor(true_cat.reshape(-1, 1)[curr_cat_mask], dtype=torch.long)
                o_pred = pred[pred.new_tensor(
                    curr_cat_mask.squeeze(1).astype(np.uint8), dtype=torch.bool)]
                assert o_pred.shape[0] == tp.shape[0]


                dx.append(o_pred.detach().numpy())
                viscode.append(tp)
                RID.append(rids)
                dx_true.append(o_true.detach().numpy())

        ret["DX"] = np.concatenate(dx, axis=0)
        ret["VISCODE"] = np.concatenate(viscode)
        ret["RID"] = np.concatenate(RID)
        ret["DX_true"] = np.concatenate(dx_true)

        assert len(ret["DX"]) == len(ret["VISCODE"]) == len(ret["RID"])

        out = "{}_prediction.csv".format(str(str(fold_n)))

        return self.output_preds(ret, outdir / Path(out))

    def output_preds(self, res, out):
        table = pd.DataFrame()
        table["RID"] = res["RID"]
        table["Forcast Month"] = res["VISCODE"]
        diag = res["DX"]
        table['CN relative probability'] = diag[:, 0]
        table['MCI relative probability'] = diag[:, 1]
        table['AD relative probability'] = diag[:, 2]
        table['DX_pred'] = np.argmax(diag, axis=1)
        table['DX_true'] = res["DX_true"]
        table.to_csv(out, index=False)

        LOGGER.info("Predictions output in file {}".format(out))
        return table

    def save_model(self):
        super().save_model()
        out = self.results_dir / Path("{}.pt".format(self.name))
        torch.save(self.model, out)
        LOGGER.info("Model {} saved to {}".format(self.name, out))
        return out

    def run(self, data: DataSet, epochs: int, predict=False, seed=0):
        LOGGER.debug("Not implemented sorry")
        pass