import logging

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.cluster import KMeans

from src.model.classifier import Classifier
from src.model.prototype import FixedPrototype
from src.preprocess.dataloader import (
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
        self.prototypes = {}
        self.batch_x = []
        self.batch_similarity = []
        self.prototype = None

    def build_model(self, encoder_model, h_size, beta=1):
        self.beta = beta
        self.hidden_size = h_size

        #Load pretrained encoding module
        self.rnn_encoder = torch.load(encoder_model)
        self.rnn_encoder.to(self.device)

        self.model = KMeans(n_clusters=self.n_prototypes, init="k-means++")

    """
    Initialize clusters in self.kmeans after pre-training
    """
    def fit(self, train_data: Random, lr=0.05, weight_decay=1e-5, epochs=5, hidden=True, outdir=""):
        LOGGER.debug("Initializing kmeans++ clusters with {} clusters".format(self.n_prototypes))
        #Store hidden_val length = n_batches
        hidden_val = self.collect_hidden_states(train_data)
        self.batch_x = torch.cat(self.batch_x, dim=0)


        # self.batch has dims (nb_patients x respective traj lens), (rid + DX + mask + hiddenstate)
        self.model = self.model.fit(self.batch_x[:, 4:])

        prototype_hidden = self.get_prototypes()

        self.prototype =FixedPrototype(self.hidden_size, self.n_prototypes, 3, prototype_hidden)
        optimizer = torch.optim.Adam(self.prototype.parameters(), lr=lr, weight_decay=weight_decay)

        for i in range(epochs):
            tot_ent = 0
            tot_dp = 0
            for batch in hidden_val.values():
                optimizer.zero_grad()
                preds = self.prototype(batch["hidden"][:, :, 4:])
                ent = ent_loss(preds, batch["true_cat"], batch["mask_cat"])
                ent.backward()
                batch_size = len(batch["true_cat"])
                tot_ent += ent * batch_size
                tot_dp += batch_size
                optimizer.step()
            LOGGER.debug("Epoch {}/{} ENT loss: {}".format(i + 1, epochs, tot_ent/tot_dp))

        return self.hidden_state_frame() if hidden else None
        # for i, (prototype, tp, hidden) in enumerate(prototype_list):
        #     seq = train_data.data[prototype]['input']
        #     dict = {}
        #     dict["rid"] = prototype
        #     dict["hidden"] = hidden
        #     for k, mask in train_data.mask.items():
        #         dict[k] = seq[:tp, mask]
        #     self.prototypes[i] = dict

    def collect_hidden_states(self, train_data):
        hidden_val = {}
        batch_n = 0
        for batch in train_data:
            batch_data = {}
            batch_data["mask_cat"] = batch['cat_msk'][1:]
            batch_data["true_cat"] = batch['true_cat'][1:]
            batch_data["tp"] = batch["tp"][1:]

            assert batch_data["mask_cat"].sum() > 0

            # latent_x shape: (n_tp, batch_size, hidden_size)
            latent_x = self.rnn_encoder(to_cat_seq(batch['cat']), batch['val'], latent=True)

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
                self.batch_x.append(latent_x)
            hidden_val[batch_n] = batch_data
            batch_n += 1
        return hidden_val

    def hidden_state_frame(self):
        columns = ["RID", "TP", "DX", "DX_mask"] + ["hidden_" + str(i) for i in range(self.hidden_size)]
        hidden_states = pd.DataFrame(self.batch_x.numpy(), columns=columns)
        hidden_states["cluster"] = self.model.labels_
        return hidden_states

    """Return list of prototype RIDs"""
    def get_prototypes(self):
        centers = self.model.cluster_centers_
        prototype_list = [None]*self.n_prototypes
        for pr in range(self.n_prototypes):
            # Only get time points that exist (not imputed values, check cat mask)
            mask = (np.array(self.model.labels_) == pr) & (self.batch_x.numpy()[:, 3] == 1)
            assigned = self.batch_x[mask]
            dist_mat = self._parallel_compute_distance(assigned[:, 4:].numpy(), centers[pr])
            closest_dp = np.argmin(dist_mat)
            rid = assigned[closest_dp, 0]
            tp = assigned[closest_dp, 1]
            hidden = assigned[closest_dp, 4:]
            prototype_list[pr] = [int(rid), int(tp), hidden]
        prototype_hidden = (tup[2] for tup in prototype_list)
        prototype_hidden = np.vstack(prototype_hidden)
        prototype_hidden = np.array(prototype_hidden)
        return prototype_hidden


    def _parallel_compute_distance(self, x, cluster):
        n_samples = x.shape[0]
        dis_mat = np.zeros((n_samples, 1))
        for i in range(n_samples):
            dis_mat[i] += np.sqrt(np.sum((x[i] - cluster) ** 2, axis=0))
        return dis_mat

    def build_optimizer(self, lr, weight_decay):
        super().build_optimizer(lr, weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr, weight_decay=weight_decay)

    def predict(self, dataset:DataSet, fold_n:int, outdir):
        LOGGER.info("Predicting labels for fold {}".format(fold_n))
        data = dataset.test

        self.prototype.eval()
        ret = {"subjects": data.subjects}
        dx = []
        viscode = []
        RID = []
        dx_true = []

        for batch in data:
            rid = batch['rid']

            #Only save the time points that have ground truth
            all_tp = batch['tp'].squeeze(axis=1)[batch["dx_mask"]][1:]
            rids = np.repeat(rid, len(all_tp))
            icat = np.asarray(
                [misc.to_categorical(c, 3) for c in batch['cat']])
            ival = batch['val'][:, None, :]
            with torch.no_grad():
                latent_x = self.rnn_encoder(icat, ival, latent=True)
            if len(latent_x) == 0:
                continue
            preds = self.prototype(torch.from_numpy(latent_x))
            preds = preds[batch["dx_mask"][1:],0,:].detach().numpy()
            dx.append(preds)
            viscode.append(all_tp)
            RID.append(rids)
            true = batch["cat"][batch["dx_mask"]][1:]
            dx_true.append(true)
            assert preds.shape[0] == all_tp.shape[0]

        ret["DX"] = np.concatenate(dx, axis=0)
        ret["VISCODE"] = np.concatenate(viscode)
        ret["RID"] = np.concatenate(RID)
        ret["DX_true"] = np.concatenate(dx_true)
        assert len(ret["DX"]) == len(ret["VISCODE"]) == len(ret["RID"])

        out = "{}_{}_prediction.csv".format(str(self.n_prototypes), str(fold_n))

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

    def predict_single_prototype(self, dataset: DataSet, fold_n:int, outdir):
        #Here predict using kmeans, then return final diagnosis of prototype
        LOGGER.info("Predicting labels for fold {}".format(fold_n))
        test_batch = []
        data = dataset.test
        for batch in data:

            rid = batch['rid']
            all_tp = batch['tp'].squeeze(axis=1)
            start = misc.month_between(dataset.pred_start[rid], dataset.baseline[rid])
            mask = all_tp < start
            icat = np.asarray(
                [misc.to_categorical(c, 3) for c in batch['cat'][mask]])
            ival = batch['val'][:, None, :][mask]
            with torch.no_grad():
                latent_x = self.rnn_encoder(icat, ival, latent=True)

            if len(latent_x) != 0:
                rid = np.repeat(batch["rid"], len(latent_x))
                #Mark the timepoint to situate prototype
                tp = batch["tp"].flatten()[1:(len(icat))]
                latent_x = np.concatenate((tp[:, np.newaxis], latent_x), axis=1)
                latent_x = np.concatenate((rid[:, np.newaxis], latent_x), axis=1)
                test_batch.append(latent_x)

        test_batch = np.vstack(test_batch)
        test_batch = np.array(test_batch)
        cl_assignment = self.model.predict(torch.from_numpy(test_batch[:, 2:]))
        pred = pd.DataFrame({"RID":test_batch[:,0], "VISCODE":test_batch[:,1],
                             "DX_pred":np.zeros(len(test_batch[:,0]))}, dtype=int)
        for i, cl in enumerate(cl_assignment):
            pred.at[i, "DX_pred"] = self.prototypes[cl]["cat"][-1]
        out = str(self.n_prototypes) + "_" + str(fold_n) + "_prediction.csv"
        outpath = outdir / Path(out)
        LOGGER.info("The predictions have been output at {}".format(outpath))
        pred.to_csv(outpath)
        dataset.prediction = pred

    def run(self, data: DataSet, epochs: int, predict=False, seed=0):
        return 0
        # TODO
