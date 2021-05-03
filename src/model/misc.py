import logging

import argparse
import json
import time
import pickle
import itertools

import numpy as np
import pandas as pd
import torch

import src.misc as misc

LOGGER = logging.getLogger(__name__)


def ent_loss(pred, true, mask):
    """
    Calculate cross-entropy loss
    Args:
        pred: predicted probability distribution,
              [nb_timpoints, nb_subjects, nb_classes]
        true: true class, [nb_timpoints, nb_subjects, 1]
        mask: timepoints to evaluate, [nb_timpoints, nb_subjects, 1]
    Returns:
        cross-entropy loss
    """
    assert isinstance(pred, torch.Tensor)
    assert isinstance(true, np.ndarray) and isinstance(mask, np.ndarray)
    nb_subjects = true.shape[1]


    pred = pred.reshape(pred.size(0) * pred.size(1), -1)
    mask = mask.reshape(-1, 1)

    o_true = pred.new_tensor(true.reshape(-1, 1)[mask], dtype=torch.long)
    o_pred = pred[pred.new_tensor(
        mask.squeeze(1).astype(np.uint8), dtype=torch.bool)]
    return torch.nn.functional.cross_entropy(
        o_pred, o_true, reduction='sum') / nb_subjects


def mae_loss(pred, true, mask):
    """
    Calculate mean absolute error (MAE)
    Args:
        pred: predicted values, [nb_timpoints, nb_subjects, nb_features]
        true: true values, [nb_timpoints, nb_subjects, nb_features]
        mask: values to evaluate, [nb_timpoints, nb_subjects, nb_features]
    Returns:
        MAE loss
    """
    assert isinstance(pred, torch.Tensor)
    assert isinstance(true, np.ndarray) and isinstance(mask, np.ndarray)
    nb_subjects = true.shape[1]

    invalid = ~mask
    true[invalid] = 0
    indices = pred.new_tensor(invalid.astype(np.uint8), dtype=torch.uint8)
    assert pred.shape == indices.shape
    pred[indices] = 0

    return torch.nn.functional.l1_loss(
        pred, pred.new(true), reduction='sum') / nb_subjects


def to_cat_seq(labels, nb_classes=3):
    """
    Return one-hot representation of a sequence of class labels
    Args:
        labels: [nb_subjects, nb_timpoints]
    Returns:
        [nb_subjects, nb_timpoints, nb_classes]
    """
    return np.asarray([misc.to_categorical(c, nb_classes) for c in labels])

def roll_mask(mask, shift):
    curr_cat_mask = np.full(mask.shape, False)
    curr_cat_mask[-(shift + 1):, :, :] = True

    shifted_cat_mask = np.roll(mask, mask.shape[0] - (shift + 1), axis=0)

    assert shifted_cat_mask.shape == mask.shape == curr_cat_mask.shape

    curr_cat_mask = curr_cat_mask & shifted_cat_mask

    return curr_cat_mask

def write_board(writer, model, epoch, loss, ent, mae):
    writer.add_scalar("Loss", loss, epoch)
    writer.add_scalar("ENT", ent, epoch)
    writer.add_scalar("MAE", mae, epoch)

    for name, weight in model.named_parameters():
        writer.add_histogram(name, weight, epoch)
        writer.add_histogram("{}.grad".format(name), weight.grad, epoch)


def print_model_parameters(model):
    for parameter in model.parameters():
        print(parameter.shape)
        print(parameter)

def copy_model_params(model):
    params = []
    for parameter in model.parameters():
        params.append(parameter.clone().detach())
    return params

def check_model_params(model, params):
    for b, a in zip(params, model.parameters()):
        assert (b.data != a.data).any()
