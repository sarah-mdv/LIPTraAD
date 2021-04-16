import logging

import numpy as np
import torch
import torch.nn as nn

from src.model.cells import (
    MinimalRNNCell,
    MinimalDecoderRNNCell
)

LOGGER = logging.getLogger(__name__)

class AutoencoderModel(nn.Module):
    """
    RNN Autoencoder base class
    """

    def __init__(self, celltype, nb_classes, nb_measures, h_size, **kwargs):
        super(AutoencoderModel, self).__init__()

        self.h_ratio = 1. - kwargs['h_drop']
        self.i_ratio = 1. - kwargs['i_drop']
        self.h_size = h_size

        self.hid2category = nn.Linear(h_size, nb_classes)
        self.hid2measures = nn.Linear(h_size, nb_measures)

        self.cells = nn.ModuleList()
        self.cells.append(celltype(nb_classes + nb_measures, h_size))

    def init_hidden_state(self, batch_size):
        dev = next(self.parameters()).device
        state = []
        for cell in self.cells:
            state.append(torch.zeros(batch_size, cell.hidden_size, device=dev))
        return state

    def dropout_mask(self, batch_size):
        dev = next(self.parameters()).device
        i_mask = torch.ones(
            batch_size, self.hid2measures.out_features, device=dev)
        r_mask = [
            torch.ones(batch_size, cell.hidden_size, device=dev)
            for cell in self.cells
        ]

        if self.training:
            i_mask.bernoulli_(self.i_ratio)
            for mask in r_mask:
                mask.bernoulli_(self.h_ratio)

        return i_mask, r_mask

    """
        Perform forwards prediction  and if latent=True return the hidden state at each time point
        Returns n_tp x batch_size x hidden_size
        """
    def forward(self, _cat_seq, _val_seq, latent=False):
        out_cat_seq, out_val_seq = [], []

        hidden = self.init_hidden_state(_val_seq.shape[1])
        masks = self.dropout_mask(_val_seq.shape[1])

        cat_seq = _cat_seq.copy()
        val_seq = _val_seq.copy()
        hidden_batch = []
        for i, j in zip(range(len(val_seq)), range(1, len(val_seq))):
            o_cat, o_val, hidden = self.predict(cat_seq[i], val_seq[i], hidden,
                                                masks)

        for i, j in zip(range(len(val_seq)), range(1, len(val_seq))):
            o_cat, o_val, hidden = self.predict(cat_seq[i], val_seq[i], hidden,
                                                masks)
            out_cat_seq.append(o_cat)
            out_val_seq.append(o_val)

            h = hidden[len(hidden) - 1].detach().cpu().numpy()
            hidden_batch.append(h)
        if len(hidden_batch) != 0:
            hidden_batch = np.array(hidden_batch)
        if latent:
            return hidden_batch
        return hidden_batch if latent else torch.stack(out_cat_seq), torch.stack(out_val_seq)
