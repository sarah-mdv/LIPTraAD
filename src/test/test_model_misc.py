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

    def test_mask_roll(self):
        assert self.nb_tp == len(self.pred)
        print(self.mask)
        for i in range(len(self.pred)):
            rolled = roll_mask(self.mask, i)
            print(rolled)


if __name__ == '__main__':
    unittest.main()