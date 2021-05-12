import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

import torch
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
    roll_mask,
write_board,
copy_model_params,
check_model_params
)

LOGGER = logging.getLogger(__name__)

class StandardAutoencoderModel(Autoencoder):
    def __init__(self, results_dir):
        super().__init__("StandardAE")
        self.results_dir = results_dir



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

        writer_name = "runs/lr= {}, weight_decay={}".format(lr, weight_decay)
        self.writer = SummaryWriter(writer_name)


    def train_epoch(self, dataset:Random, epoch, w_ent=1):
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
        total_ent = total_mae = total_loss = 0
        j = 0
        for batch in dataset:
            if len(batch['tp']) == 1:
                continue

            self.optimizer.zero_grad()

            pred_cat, pred_val = self.model(torch.from_numpy(to_cat_seq(batch['cat'])),
                                                    torch.from_numpy(batch['val']),latent=False)

            mask_cat = batch['cat_msk']
            mask_val = batch['val_msk']
            assert mask_cat.sum() > 0

            batch_ent = batch_mae = 0
            nb_tp = len(pred_cat)
            batch_cat_n_measurements = 0
            batch_val_n_measurements = 0
            for i in range(nb_tp):

                curr_cat_mask = roll_mask(mask_cat, i)
                curr_val_mask = roll_mask(mask_val, i)

                true_cat = np.roll(batch['true_cat'], len(batch['true_cat']) - (i+1), axis=0)
                true_val = np.roll(batch['true_val'], len(batch['true_val']) - (i+1), axis=0)

                cat_n_measurements = np.sum(curr_cat_mask)
                val_n_measurements = np.sum(curr_val_mask)
                batch_cat_n_measurements += cat_n_measurements
                batch_val_n_measurements += val_n_measurements
                # LOGGER.debug("Categorical number of measurements {}".format(cat_n_measurements))
                # LOGGER.debug("Value number of measurements {}".format(val_n_measurements))

                ent = ent_loss(pred_cat[i].clone(), true_cat, curr_cat_mask)
                mae = mae_loss(pred_val[i].clone(), true_val, curr_val_mask)

                batch_ent += ent * cat_n_measurements
                batch_mae += mae * val_n_measurements
            batch_ent = batch_ent / batch_cat_n_measurements
            batch_mae = batch_mae / batch_val_n_measurements
            loss = w_ent * batch_ent + batch_mae
            loss.backward()
            self.optimizer.step()

            batch_size = mask_cat.shape[1]
            total_loss += loss.item() *batch_size
            total_ent += batch_ent.item() * batch_size
            total_mae += batch_mae.item() * batch_size
            j +=1

        total_ent = (total_ent / len(dataset.subjects))
        total_mae = (total_mae / len(dataset.subjects))

        write_board(self.writer, self.model, epoch, total_loss, total_ent, total_mae)

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
        val = []
        val_true = []

        for d in testdata:
            rid = d['rid']
            mask_cat = d["dx_mask"].squeeze(1)
            mask_val = d["val_mask"]

            icat = torch.from_numpy(np.asarray([misc.to_categorical(c, 3) for c in d['cat']]))
            ival = torch.from_numpy(d['val'][:, None, :])
            ocat, oval = self.model(icat, ival)

            all_tp = d['tp'].squeeze(axis=1)
            try:
                for i in range(len(all_tp)):
                    # Only save the time points that have ground truth
                    if not mask_cat[i]:
                        continue

                    curr_cat_mask = np.full(mask_cat.shape, False)
                    curr_cat_mask[-(i+1):] = True
                    curr_val_mask = np.full(mask_val.shape, False)
                    curr_val_mask[-(i+1):] = True

                    shifted_cat_mask = np.roll(mask_cat, len(mask_cat) - (i+1), axis= 0)
                    shifted_val_mask = np.roll(mask_val, len(mask_cat) - (i+1), axis= 0)

                    assert shifted_cat_mask.shape == mask_cat.shape == curr_cat_mask.shape
                    assert shifted_val_mask.shape == mask_val.shape == curr_val_mask.shape


                    curr_cat_mask = curr_cat_mask & shifted_cat_mask
                    curr_val_mask = curr_val_mask & shifted_val_mask


                    tp = np.roll(all_tp, len(mask_cat) - (i+1), axis=0)[curr_cat_mask]
                    rids = np.repeat(rid, len(tp))

                    true_cat = np.roll(d['dx_truth'], len(d['dx_truth']) - (i+1), axis=0)
                    true_val = np.roll(d['val'][:, :], len(d['val']) - (i+1), axis=0)

                    # Mask cat values
                    pred = ocat[i]

                    pred = pred.reshape(pred.size(0) * pred.size(1), -1)
                    curr_cat_mask = curr_cat_mask.reshape(-1, 1)
                    o_true = pred.new_tensor(true_cat.reshape(-1, 1)[curr_cat_mask], dtype=torch.long)
                    n_mask = pred.new_tensor(curr_cat_mask.squeeze(1).astype(np.uint8), dtype=torch.bool)
                    o_pred = pred[n_mask]

                    # Mask continuous variables
                    pred = oval[i].squeeze(1)
                    invalid = ~curr_val_mask
                    true_val[invalid] = 0
                    indices = pred.new_tensor(invalid.astype(np.uint8), dtype=torch.uint8)
                    pred[indices] = 0

                    o_true_val = true_val[torch.tensor(true_val).new_tensor(
                        curr_cat_mask.squeeze(1).astype(np.uint8), dtype=torch.bool)]
                    o_pred_val = pred[pred.new_tensor(
                        curr_cat_mask.squeeze(1).astype(np.uint8), dtype=torch.bool)]
                    assert o_pred.shape[0] == tp.shape[0]


                    viscode.append(tp)
                    RID.append(rids)
                    dx.append(o_pred.detach().numpy())
                    dx_true.append(o_true.detach().numpy())
                    val_true.append(o_true_val)
                    val.append(o_pred_val.detach().numpy())
            except IndexError:
                LOGGER.debug("Index error for mask {}".format(mask_cat))
                continue
        ret["VISCODE"] = np.concatenate(viscode)
        ret["RID"] = np.concatenate(RID)
        ret["DX"] = np.concatenate(dx, axis=0)
        ret["DX_true"] = np.concatenate(dx_true)
        ret["Val"] = np.concatenate(val, axis=0)
        ret["Val_true"] = np.concatenate(val_true, axis=0)

        assert len(ret["DX"]) == len(ret["VISCODE"]) == len(ret["RID"])

        out = "{}_prediction.csv".format(str(str(fold_n)))

        return self.output_preds(ret, testdata.attributes, outdir / Path(out), data.mean, data.std)

    def output_preds(self, res, fields, out, mean, std):
        t1 = pd.DataFrame(res["Val"], columns=fields) * std + mean
        true_fields = [field + "_true" for field in fields]
        t2 = pd.DataFrame(res["Val_true"], columns=fields) * std + mean
        t2.columns = true_fields
        table = pd.DataFrame()
        table["RID"] = res["RID"]
        table["Forcast Month"] = res["VISCODE"]
        diag = res["DX"]
        table['CN relative probability'] = diag[:, 0]
        table['MCI relative probability'] = diag[:, 1]
        table['AD relative probability'] = diag[:, 2]
        table['DX_pred'] = np.argmax(diag, axis=1)
        table['DX_true'] = res["DX_true"]
        table = pd.concat([table, t1, t2], axis=1)
        table.to_csv(out, index=False)

        bac = balanced_accuracy_score(table["DX_pred"], table['DX_true'])
        LOGGER.info("The BAC is {}".format(bac))

        LOGGER.info("Predictions output in file {}".format(out))
        return bac

    def collect_hidden_states(self, dataset:DataSet):
        self.model.eval()
        hidden_val = {}
        batch_n = 0
        batch_x = []
        for batch in dataset.train:
            batch_data = {}
            batch_data["mask_cat"] = batch['cat_msk'][:]
            batch_data["true_cat"] = batch['true_cat'][:]
            batch_data["tp"] = batch["tp"][:]

            if not batch_data["mask_cat"].sum() > 0:
                continue

            # latent_x shape: (n_tp, batch_size, hidden_size)
            latent_x = self.model(torch.from_numpy(to_cat_seq(batch['cat'])),
                                                    torch.from_numpy(batch['val']),latent=True)
            if len(latent_x) != 0:
                latent_x = torch.from_numpy(latent_x)
                rids = torch.empty((latent_x.shape[0], 1, 1))
                # Attach rids to hidden states so real values can be retrieved
                for i in range(len(batch["rids"])):
                    rid = torch.full((latent_x.shape[0], 1, 1), batch["rids"][i])
                    rids = rid if i == 0 else torch.cat((rids, rid), dim=1)
                mask = torch.from_numpy(batch_data["mask_cat"].astype(np.float32))
                diags = torch.from_numpy(batch_data["true_cat"].astype(np.float32))
                tps = torch.from_numpy(batch_data["tp"].astype(np.float32))
                latent_x = torch.cat((mask, latent_x[:, :, :]), dim=2)
                latent_x = torch.cat((diags, latent_x[:, :, :]), dim=2)
                latent_x = torch.cat((tps, latent_x[:, :, :]), dim=2)
                latent_x = torch.cat((rids, latent_x[:, :, :]), dim=2)
                # LOGGER.debug("latent full {}".format(latent_x))
                batch_data["hidden"] = latent_x
                # Transform to (batch_size x n_tp) x hidden_size
                latent_x = torch.transpose(latent_x, 0, 1)
                latent_x = torch.flatten(latent_x, start_dim=0, end_dim=1)
                batch_x.append(latent_x)
            hidden_val[batch_n] = batch_data
            batch_n += 1
        batch_x = torch.cat(batch_x, dim=0)
        columns = ["RID", "TP", "DX", "DX_mask"] + ["hidden_" + str(i) for i in range(self.model.h_size)]
        hidden_states = pd.DataFrame(batch_x.numpy(), columns=columns)
        return hidden_states


    def save_model(self):
        super().save_model()
        out = self.results_dir / Path("{}.pt".format(self.name))
        torch.save(self.model, out)
        LOGGER.info("Model {} saved to {}".format(self.name, out))
        return out

    def run(self, data: DataSet, epochs: int, predict=False, seed=0):
        LOGGER.debug("Not implemented sorry")
        pass