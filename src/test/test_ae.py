import unittest
import torch
import logging
import time
import os

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


from model.ae_modules import StandardAutoencoder
from model.misc import (
    to_cat_seq,
    ent_loss,
    mae_loss,
roll_mask
)

from src import misc
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)

class AETest(unittest.TestCase):

    def setUp(self):
        self.in_features = 1
        self.batch_size = 64
        self.len = 100
        self.test_len = 1000
        self.seq_len = 4
        self.nb_classes = 2
        self.nb_measures = 1
        self.h_size = 128
        self.h_drop = 0.4
        self.i_drop = 0.1
        self.lr = 0.0001
        self.weight_decay = 0.01
        self.epochs =10

        self.writer = SummaryWriter()


    def test_Autoencoder(self):

        #Generate toy dataset of size nb_batches x nb_tp x batch_size, nb_measures
        train_cat = np.random.randint(self.nb_classes, size=(self.len, self.seq_len, self.batch_size, 1))
        test_cat = np.random.randint(self.nb_classes, size=(self.test_len, self.seq_len, 1, 1))

        rng = np.random.default_rng(seed=42)
        train_val = rng.random((self.len, self.seq_len, self.batch_size, 1))
        test_val = rng.random((self.test_len, self.seq_len, 1))


        #Create model
        model = StandardAutoencoder(nb_classes=self.nb_classes, nb_measures=self.nb_measures, h_size = self.h_size,
                                    h_drop=self.h_drop, i_drop=self.i_drop, )


        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.writer.add_graph(model, (torch.from_numpy(to_cat_seq(train_cat[0], 2).astype(float)),
                                           torch.from_numpy(train_val[0].astype(float))))

        t = datetime.now().strftime("%H:%M:%S")

        np.random.seed(3)
        torch.manual_seed(3)

        start = time.time()
        loss_table = {}
        for i in range(self.epochs):
            loss = self.train_epoch(model, train_cat, train_val, i)
            loss_table[i] = loss
            log_info = (i + 1, self.epochs, misc.time_from(start)) + loss
            print('Epoch: %d/%d %s ENT %.3f, MAE %.3f' % log_info)

        self.predict(model, test_cat, test_val, "/Documents/Studies/Chalmers/Thesis/LIPTraAD/src/logs/")

    def train_epoch(self, model, cat, val, epoch, w_ent=1.):
        model.train()
        total_ent = total_mae = total_loss = 0
        for i in range(cat.shape[0]):
            torch.autograd.set_detect_anomaly(True)

            self.optimizer.zero_grad()

            cat_in = cat[i]
            val_in = val[i]

            pred_cat, pred_val, hf, hb = model(torch.from_numpy(to_cat_seq(cat_in, self.nb_classes)),
                                                    torch.from_numpy(val_in), latent=True)
            """
            Are we learning the identity function at t or not
            Look at the predictions at t

            Check the gradients from the hidden state

            Check with synthetic data on the model, does it learn simple patterns,
            With deterministic output so that we know it should be learned

            """

            batch_ent = batch_mae = 0
            for i in range(len(pred_cat)):
                curr_cat_mask = np.full(cat_in.shape, False)
                curr_cat_mask[-(i+1):, :, :] = True
                curr_val_mask = np.full(val_in.shape, False)
                curr_val_mask[-(i+1):, :, :] = True

                assert cat_in.shape == curr_cat_mask.shape
                assert val_in.shape == curr_val_mask.shape

                true_cat = np.roll(cat_in, len(cat_in) - (i+1))
                true_val = np.roll(val_in, len(val_in) - (i+1))

                batch_ent += ent_loss(pred_cat[i].clone(), true_cat, curr_cat_mask)
                batch_mae += mae_loss(pred_val[i].clone(), true_val, curr_val_mask)
            batch_loss = batch_ent * w_ent + batch_mae
            batch_loss.backward()
            self.optimizer.step()

            batch_size = cat_in.shape[1]
            total_ent += batch_ent.item() * batch_size
            total_mae += batch_mae.item() * batch_size
            total_loss += batch_loss * batch_size
        self.writer.add_scalar("Loss", total_loss/ (batch_size * cat.shape[0]), epoch)
        self.writer.add_scalar("ENT", total_ent / (batch_size * cat.shape[0]), epoch)
        self.writer.add_scalar("MAE", total_mae / (batch_size * cat.shape[0]), epoch)

        self.writer.add_histogram("fw_cells.bias", model.fw_cell.bias_hh, epoch)
        self.writer.add_histogram("fw_cells.weight_uh", model.fw_cell.weight_uh, epoch)
        self.writer.add_histogram("fw_cells.weight_uh.grad", model.fw_cell.weight_uh.grad, epoch)

        self.writer.add_histogram("bw_cells.bias", model.bw_cell.bias_hh, epoch)
        self.writer.add_histogram("bw_cells.weight_uh", model.bw_cell.weight_uh, epoch)
        self.writer.add_histogram("bw_cells.weight_uh.grad", model.bw_cell.weight_uh.grad, epoch)

        self.writer.add_histogram("out_cat.bias", model.hid2category.bias, epoch)
        self.writer.add_histogram("out_cat.weight", model.hid2category.weight, epoch)
        self.writer.add_histogram("out_cat.weight.grad", model.hid2category.weight.grad, epoch)

            # self.writer.add_histogram("out_val.bias", model.hid2measures.bias, epoch)
            # self.writer.add_histogram("out_val.weight", model.hid2measures.weight, epoch)
            # self.writer.add_histogram("out_val.weight.grad", model.hid2measures.weight.grad, epoch)


        return total_ent / (batch_size * cat.shape[0]), total_mae / (batch_size*cat.shape[0])

    def predict(self, model, cat_data, val_data, outdir):
        model.eval()

        ret = {}
        dx = []
        viscode = []
        RID = []
        dx_true = []

        for j in range(self.test_len):
            rid = j
            cat = cat_data[j]
            val = val_data[j]
            icat = torch.from_numpy(np.asarray([misc.to_categorical(c, self.nb_classes) for c in cat]))
            ival = torch.from_numpy(val[:, None, :])
            ocat, oval = model(icat, ival)

            all_tp = np.arange(self.seq_len)


            for i in range(self.seq_len):

                pred = ocat[i]

                curr_cat_mask = np.full(icat.shape, False)
                curr_cat_mask[-(i+1):] = True


                assert icat.shape == curr_cat_mask.shape

                tp_mask = curr_cat_mask[:,:,0].squeeze()

                tp = np.roll(all_tp, len(icat) - (i+1))[tp_mask]
                rids = np.repeat(rid, len(tp))

                true_cat = np.roll(cat, len(cat) - (i+1))


                pred = pred.reshape(pred.size(0) * pred.size(1), -1)
                curr_cat_mask = curr_cat_mask.squeeze(1).astype(np.uint8)
                #curr_cat_mask = curr_cat_mask.reshape(-1, 1)

                o_true = pred.new_tensor(true_cat.reshape(-1, 1)[tp_mask], dtype=torch.long)
                o_pred = pred[pred.new_tensor(curr_cat_mask, dtype=torch.bool)].reshape(i+1, -1)
                #assert o_pred.shape[0] == tp.shape[0]


                dx.append(o_pred.detach().numpy())
                viscode.append(tp)
                RID.append(rids)
                dx_true.append(o_true.detach().numpy())

        ret["DX"] = np.concatenate(dx, axis=0)
        ret["VISCODE"] = np.concatenate(viscode)
        ret["RID"] = np.concatenate(RID)
        ret["DX_true"] = np.concatenate(dx_true)

        assert len(ret["DX"]) == len(ret["VISCODE"]) == len(ret["RID"])

        out = outdir + "test_prediction.csv"

        return self.output_preds(ret, Path(out))

    def output_preds(self, res, out):
        table = pd.DataFrame()
        table["RID"] = res["RID"]
        table["Forcast Month"] = res["VISCODE"]
        diag = res["DX"]
        table['0 relative probability'] = diag[:, 0]
        table['1 relative probability'] = diag[:, 1]
        table['DX_pred'] = np.argmax(diag, axis=1)
        table['DX_true'] = res["DX_true"]
        SRC_DIR = Path(__file__).parent.resolve()

        outpath = SRC_DIR/out
        print(outpath)

        table.to_csv(Path("test_prediction.csv"), index=False)

        LOGGER.info("Predictions output in file {}".format(SRC_DIR/out))
        return table

if __name__ == '__main__':
    unittest.main()