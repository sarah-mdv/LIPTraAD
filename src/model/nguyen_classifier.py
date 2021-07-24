import logging
import time

import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path

from src.model.classifier import Classifier
from src.model.rnn_modules import MinimalRNN
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

class RNNClassifier(Classifier):
    def __init__(self):
        super().__init__("RNN")

    def build_model(self, nb_classes, nb_measures, h_size,
                    h_drop, i_drop, nb_layers, mean, stds):
        self.model = MinimalRNN(nb_classes=nb_classes, nb_measures=nb_measures, h_size=h_size,
                                nb_layers=nb_layers, h_drop=h_drop, i_drop=i_drop)
        setattr(self.model, 'mean', mean)
        setattr(self.model, 'stds', stds)

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)

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
            if len(batch['tp']) == 1:
                continue

            self.optimizer.zero_grad()
            pred_cat, pred_val = self.model(to_cat_seq(batch['cat']), batch['val'])

            mask_cat = batch['cat_msk'][1:]
            assert mask_cat.sum() > 0

            ent = ent_loss(pred_cat, batch['true_cat'][1:], mask_cat)
            mae = mae_loss(pred_val, batch['true_val'][1:], batch['val_msk'][1:])
            total_loss = mae + w_ent * ent

            total_loss.backward()
            self.optimizer.step()
            batch_size = mask_cat.shape[1]
            total_ent += ent.item() * batch_size
            total_mae += mae.item() * batch_size

        return total_ent / len(dataset.subjects), total_mae / len(dataset.subjects)

    def fit(self, data:DataSet, epochs: int, seed=0):
        super().fit(data, epochs)
        date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

        np.random.seed(seed)
        torch.manual_seed(seed)

        start = time.time()
        loss_table = {}
        try:
            for i in range(epochs):
                loss = self.train_epoch(data.train)
                log_info = (i + 1, epochs, misc.time_from(start)) + loss
                loss_table[i] = loss
                LOGGER.info('Epoch: %d/%d %s ENT %.3f, MAE %.3f' % log_info)
        except KeyboardInterrupt:
            LOGGER.error('Early exit')
        df = pd.DataFrame.from_dict(loss_table, orient='index', columns=["ENT", "MAE"])
        df.to_csv(misc.LOG_DIR / Path(date + "_loss.csv"))


    def build_optimizer(self, lr, weight_decay):
        super().build_optimizer(lr, weight_decay)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def save_model(self, results_dir):
        super().save_model()
        t = time.localtime()
        current_time = time.strftime("%H%M%S", t)
        out = results_dir / Path("{}_{}.pt".format(self.name, current_time))
        torch.save(self.model, out)
        LOGGER.info("Model {} saved to {}".format(self.name, out))
        return out

    def predict_subject(self, cat_seq, value_seq, time_seq):
        """
        Predict Alzheimer’s disease progression for a subject
        Args:
            model: trained pytorch model
            cat_seq: sequence of diagnosis [nb_input_timpoints, nb_classes]
            value_seq: sequence of other features [nb_input_timpoints, nb_features]
            time_seq: months from baseline [nb_output_timpoints, nb_features]
        nb_input_timpoints <= nb_output_timpoints
        Returns:
            out_cat: predicted diagnosis
            out_val: predicted features
        """
        in_val = np.full((len(time_seq),) + value_seq.shape[1:], np.nan)
        in_val[:len(value_seq)] = value_seq

        in_cat = np.full((len(time_seq),) + cat_seq.shape[1:], np.nan)
        in_cat[:len(cat_seq)] = cat_seq

        with torch.no_grad():
            out_cat, out_val = self.model(in_cat, in_val)
        out_cat = out_cat.cpu().numpy()
        out_val = out_val.cpu().numpy()

        assert out_cat.shape[1] == out_val.shape[1] == 1

        return out_cat, out_val

    def predict(self, dataset:DataSet):
        """
        Predict Alzheimer’s disease progression using a trained model
        Args:
            model: trained pytorch model
            dataset: test data
            pred_start (dictionary): the date at which prediction begins
            duration (dictionary): how many months into the future to predict
            baseline (dictionary): the baseline date
        Returns:
            dictionary which contains the following key/value pairs:
                subjects: list of subject IDs
                DX: list of diagnosis prediction for each subject
                ADAS13: list of ADAS13 prediction for each subject
                Ventricles: list of ventricular volume prediction for each subject
        """
        super().predict(dataset)
        self.model.eval()
        testdata = dataset.test #testdata of type Sorted
        ret = {'subjects': testdata.subjects}
        ret['DX'] = []  # 1. likelihood of NL, MCI, and Dementia
        ret['ADAS13'] = []  # 2. (best guess, upper and lower bounds on 50% CI)
        ret['Ventricles'] = []  # 3. (best guess, upper and lower bounds on 50% CI)
        ret['dates'] = misc.make_date_col(
            [dataset.pred_start[s] for s in testdata.subjects], dataset.duration)

        col = ['ADAS13', 'Ventricles', 'ICV']
        indices = misc.get_index(list(testdata.value_fields()), col)
        mean = self.model.mean[col].values.reshape(1, -1)
        stds = self.model.stds[col].values.reshape(1, -1)

        for data in testdata:
            rid = data['rid']
            all_tp = data['tp'].squeeze(axis=1)
            start = misc.month_between(dataset.pred_start[rid], dataset.baseline[rid])
            assert np.all(all_tp == np.arange(len(all_tp)))
            mask = all_tp < start
            itime = np.arange(start + dataset.duration)
            icat = np.asarray(
                [misc.to_categorical(c, 3) for c in data['cat'][mask]])
            ival = data['val'][:, None, :][mask]

            ocat, oval = self.predict_subject(icat, ival, itime)
            oval = oval[-dataset.duration:, 0, indices] * stds + mean

            ret['DX'].append(ocat[-dataset.duration:, 0, :])
            ret['ADAS13'].append(misc.add_ci_col(oval[:, 0], 1, 0, 85))
            ret['Ventricles'].append(
                misc.add_ci_col(oval[:, 1] / oval[:, 2], 5e-4, 0, 1))
        return ret

    def run(self, data: DataSet, epochs: int, out, predict=False, seed=0):
        self.fit(data, epochs, seed)

        outpath = misc.LOG_DIR / Path(out)
        if predict:
            data.prediction = misc.build_pred_frame(self.predict(data), outpath)
            LOGGER.info("Predictions have been output at {}".format(outpath))


