import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import math
import logging
from src.model.misc import compute_squared_distances

LOGGER = logging.getLogger(__name__)


"""
This class learns only the prediction layer, but the prototypes are fixed beforehand (used for k-means similarity)
"""
class FixedPrototype(nn.Module):
    def __init__(self, hidden_size, n_prototypes, nb_classes, prototypes):
        super(FixedPrototype, self).__init__()

        self.hidden_size = hidden_size
        self.n_prototypes = n_prototypes
        self.prototypes = torch.from_numpy(prototypes)
        self.linear = nn.Linear(n_prototypes, nb_classes)

    """
    input: X hidden states of each patient for every time point (
    """
    def forward(self, X):
        all_similarities = []
        for encoded_sequence in X.reshape(-1, self.hidden_size):
            squared_distances = torch.pow(encoded_sequence - self.prototypes, 2).sum(1)
            similarities = torch.exp(torch.neg(squared_distances))
            all_similarities.append(similarities.unsqueeze(0))
        batch_size = X.shape[0]
        similarities = torch.cat(all_similarities, dim=0).reshape(batch_size, -1, self.n_prototypes)
        out = nn.functional.softmax(self.linear(similarities), dim=-1)
        return out


class Prototype(nn.Module):
    def __init__(self, hidden_size, n_prototypes):
        super(Prototype, self).__init__()

        self.hidden_size = hidden_size  # hidden_size
        self.n_prototypes = n_prototypes  # n_prototypes
        self.n_transition_prototypes = 0

        k = 1 / math.sqrt(hidden_size)
        random_tensor = torch.rand(n_prototypes, hidden_size, dtype=torch.float32, requires_grad=True)
        self.prototypes = Parameter(k + ((-k) - k) * random_tensor)  # what is this multiplication

    # Takes (batch_size x hidden_size) and returns (batch_size x n_prototypes)
    def forward(self, X, transitions=None):
        squared_distances = compute_squared_distances(X, self.prototypes)
        out = torch.exp(torch.neg(squared_distances))
        return out

class TransitionPrototype(Prototype):
    def __init__(self, hidden_size, n_prototypes, n_transition_prototypes):
        super(TransitionPrototype, self).__init__(hidden_size, n_prototypes)

        self.n_transition_prototypes = n_transition_prototypes

        self.transition_mask = torch.zeros((self.n_prototypes), dtype=torch.float32)
        self.transition_mask[:self.n_transition_prototypes] = 1.0
        LOGGER.debug(self.transition_mask)

        self.mask = torch.zeros((self.n_prototypes), dtype=torch.float32)
        self.mask[self.n_transition_prototypes:] = 1.0
        LOGGER.debug(self.mask)


    # Takes (batch_size x hidden_size) and returns (batch_size x n_prototypes)
    def forward(self, X, transitions=None):
        # Assumes batch_first = True
        all_similarities = []
        for encoded_sequence, t in zip(X, transitions):
            squared_distances = torch.pow(encoded_sequence - self.prototypes, 2).sum(1)
            sim = torch.exp(torch.neg(squared_distances))
            #If the embedded sequence is a transition sequence, only compare to transition prototypes
            mask = self.transition_mask if t else self.mask
            similarities = sim * mask
            all_similarities.append(similarities.unsqueeze(0))
        batch_size = X.shape[0]
        out = torch.cat(all_similarities, dim=0).reshape(batch_size, self.n_prototypes)
        return out
