#!/usr/bin/env python
# Adapted from code written by Minh Nguyen and CBIG under MIT license:

import logging

import math
import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


class MinimalRNNCell(nn.Module):
    """A Minimal RNN cell .
    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`

    Inputs: input, hidden
        - input of shape `(batch, input_size)`: input features
        - hidden of shape `(batch, hidden_size)`: initial hidden state

    Outputs: h'
        - h' of shape `(batch, hidden_size)`: next hidden state
    """

    def __init__(self, input_size, hidden_size):
        super(MinimalRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Linear(input_size, hidden_size)

        self.weight_uh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_uz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.weight_uh.data.uniform_(-stdv, stdv)
        self.weight_uz.data.uniform_(-stdv, stdv)

        self.bias_hh.data.uniform_(stdv)

    def forward(self, input, hx):
        z = torch.tanh(self.W(input))
        u = torch.addmm(self.bias_hh, hx, self.weight_uh)
        u = torch.addmm(u, z, self.weight_uz)
        u = torch.sigmoid(u)
        return u * hx + (1 - u) * z


class LssCell(nn.Module):
    """A Linear State-space cell .
    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`

    Inputs: input, hidden
        - input of shape `(batch, input_size)`: input features
        - hidden of shape `(batch, hidden_size)`: initial hidden state

    Outputs: h'
        - h' of shape `(batch, hidden_size)`: next hidden state
    """

    def __init__(self, input_size, hidden_size):
        super(LssCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_i2h = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_h2h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.weight_i2h.data.uniform_(-stdv, stdv)
        self.weight_h2h.data.uniform_(-stdv, stdv)

        self.bias.data.uniform_(stdv)

    def forward(self, input, hx):
        h = torch.addmm(self.bias, input, self.weight_i2h)
        h = torch.addmm(h, hx, self.weight_h2h)
        return h


class MinimalDecoderRNNCell(nn.Module):
    """A Minimal Decoder RNN cell .
    Args:
        hidden_size: The number of features in the hidden state `h`

    Inputs: input, hidden
        - hidden of shape `(batch, hidden_size)`: initial hidden state

    Outputs: h'
        - h' of shape `(batch, hidden_size)`: next hidden state
    """

    def __init__(self, hidden_size):
        super(MinimalDecoderRNNCell, self).__init__()
        self.hidden_size = hidden_size

        self.weight_uh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.weight_uh.data.uniform_(-stdv, stdv)

        self.bias_hh.data.uniform_(stdv)

    def forward(self, hx):
        u = torch.addmm(self.bias_hh, hx, self.weight_uh)
        u = torch.sigmoid(u)
        return u
