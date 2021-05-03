import logging

import numpy as np
import torch
import torch.nn as nn

from src.model.cells import (
    MinimalRNNCell,
    MinimalDecoderRNNCell
)

LOGGER = logging.getLogger(__name__)

def jozefowicz_init(forget_gate):
    """
    Initialize the forget gaste bias to 1
    Args:
        forget_gate: forget gate bias term
    References: https://arxiv.org/abs/1602.02410
    """
    forget_gate.data.fill_(1)

class Autoencoder(nn.Module):
    """
    RNN Autoencoder base class
    """

    def __init__(self, fw_celltype, bw_celltype, nb_classes, nb_measures, h_size, **kwargs):
        super(Autoencoder, self).__init__()

        self.h_ratio = 1. - kwargs['h_drop']
        self.i_ratio = 1. - kwargs['i_drop']
        self.h_size = h_size

        self.hid2category = nn.Linear(h_size, nb_classes)
        self.hid2measures = nn.Linear(h_size, nb_measures)

        self.fw_cell = fw_celltype(nb_classes + nb_measures, h_size)

        self.bw_cell = bw_celltype(h_size)

    def init_hidden_state(self, batch_size):
        dev = next(self.parameters()).device

        return torch.zeros(batch_size, self.fw_cell.hidden_size, device=dev)

    def dropout_mask(self, batch_size):
        dev = next(self.parameters()).device
        i_mask = torch.ones(
            batch_size, self.hid2measures.out_features, device=dev)
        r_mask = torch.ones(batch_size, self.fw_cell.hidden_size, device=dev)
        b_mask = torch.ones(batch_size, self.bw_cell.hidden_size, device=dev)

        if self.training:
            i_mask.bernoulli_(self.i_ratio)
            r_mask.bernoulli_(self.h_ratio)
            b_mask.bernoulli_(self.h_ratio)

        return i_mask, r_mask, b_mask

    """
        Perform forwards prediction  and if latent=True return the hidden state at each time point
        Returns n_tp x batch_size x hidden_size
        """
    def forward(self, _cat_seq, _val_seq, latent=False):
        out_cat_seq, out_val_seq = [], []

        hidden = self.init_hidden_state(_val_seq.shape[1])

        cat_seq = _cat_seq.clone().requires_grad_(True).float()
        val_seq = _val_seq.clone().requires_grad_(True).float()
        hidden_fw_batch = []
        hidden_bw_batch = []
        sequence_len = len(val_seq)
        for i in range(sequence_len):
            masks = self.dropout_mask(_val_seq.shape[1])
            #o_cal, and o_val should have shapes tp x batch_size x #c or #v respectively
            o_cat, o_val, hidden, hidden_bw = self.predict(cat_seq[i], val_seq[i], hidden,
                                                masks, sequence_len)
            out_cat_seq.append(o_cat)
            out_val_seq.append(o_val)

            h_fw = hidden.detach().cpu()
            hidden_fw_batch.append(h_fw)
            h_bw = hidden_bw.detach().cpu()
            hidden_bw_batch.append(h_bw)
        # if len(hidden_fw_batch) != 0:
        #     hidden_fw_batch = np.array(hidden_fw_batch)
        #
        # if len(hidden_bw_batch) != 0:
        #     hidden_bw_batch = np.array(hidden_bw_batch)

        return (torch.stack(out_cat_seq), torch.stack(out_val_seq), hidden_fw_batch, hidden_bw_batch) if latent else (torch.stack(out_cat_seq), torch.stack(out_val_seq))


class StandardAutoencoder(Autoencoder):
    """Standard non-prototype Autoencoder"""
    def __init__(self, **kwargs):
        super(StandardAutoencoder, self).__init__(MinimalRNNCell, MinimalDecoderRNNCell, **kwargs)
        jozefowicz_init(self.fw_cell.bias_hh)
        jozefowicz_init(self.bw_cell.bias_hh)


    def predict(self, i_cat, i_val, hid, masks, sequence_len):
        out_cat_seq, out_val_seq = [], []

        i_mask, r_mask, b_mask = masks
        in_comb = torch.cat([hid.new(i_cat), hid.new(i_val) * i_mask],
                        dim=-1)

        next_hidden = self.fw_cell(in_comb, hid * r_mask)

        o_cat = nn.functional.softmax(self.hid2category(next_hidden), dim=-1)
        o_val = self.hid2measures(next_hidden)

        out_cat_seq.append(o_cat)
        out_val_seq.append(o_val)

        next_bw_hidden = next_hidden.clone()

        for i in range(sequence_len - 1):
            next_bw_hidden = self.bw_cell(next_bw_hidden)

            o_cat = nn.functional.softmax(self.hid2category(next_bw_hidden), dim=-1)
            o_val = self.hid2measures(next_bw_hidden)

            out_cat_seq.append(o_cat)
            out_val_seq.append(o_val)


        # Return a chronologically ordered sequence (like input)
        return torch.flip(torch.stack(out_cat_seq), [0]), torch.flip(torch.stack(out_val_seq), [0]), next_hidden, next_bw_hidden
