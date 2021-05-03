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

class TestMisc(unittest.TestCase):

    def setUp(self):
        self.nb_tp = 5
        self.nb_subs = 3
        self.shape_cat = (self.nb_tp, self.nb_subs, 3)
        self.shape_mask = (self.nb_tp, self.nb_subs, 1)
        self.shape_val = (self.nb_tp, self.nb_subs, 2)

        self.pred = torch.zeros(self.shape_cat)
        self.pred[:, :, 2:3] = 1.

        self.true = torch.zeros(self.shape_mask)
        self.true[2:, :, :] = 2.

        self.mask = torch.zeros(self.shape_mask, dtype=torch.bool)
        self.mask[2:,1:,:] = True

        self.false = torch.zeros(self.shape_mask, dtype=torch.bool)
        self.all_true = torch.ones(self.shape_mask, dtype=torch.bool)

        self.roll = [None]*5
        self.roll[0] = torch.zeros(self.shape_mask, dtype=torch.bool)
        self.roll[0][4:,:,:] = True

        self.roll[1] = torch.zeros(self.shape_mask, dtype=torch.bool)
        self.roll[1][3:, :, :] = True

        self.roll[2] = torch.zeros(self.shape_mask, dtype=torch.bool)
        self.roll[2][2:, :, :] = True

        self.roll[3] = torch.zeros(self.shape_mask, dtype=torch.bool)
        self.roll[3][1:, :, :] = True

        self.roll[4] = torch.zeros(self.shape_mask, dtype=torch.bool)
        self.roll[4][0:, :, :] = True

    def test_mask_roll(self):
        assert self.nb_tp == len(self.pred)
        for i in range(len(self.pred)):
            rolled = roll_mask(self.all_true, i)
            assert torch.eq(torch.from_numpy(rolled), self.roll[i]).all()

    def test_false_roll(self):
        for i in range(self.nb_tp):
            rolled = roll_mask(self.false, i)
            assert not rolled.any()

if __name__ == '__main__':
    unittest.main()