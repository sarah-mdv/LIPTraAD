import logging
import time

import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path

from src.model.autoencoder import Autoencoder
from src.model.ae_modules import AutoencoderModel
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

class StandardAutoencoderModel(AutoencoderModel):
    def __init__(self):
        super.__init__("StandardAE")

    def build_model(self, **kwargs) -> nn.Module:

        pass

    def build_optimizer(self, lr, weight_decay):

        pass

    def fit(self, data: DataSet, epochs: int):

        if not self.model:
            LOGGER.error("ERROR: self.model is None. Did you call build_model()?")
            exit(1)
        elif not self.optimizer:
            LOGGER.error("ERROR: self.optimizer is None. Did you call build_optimizer()?")
            exit(1)

    def predict(self, data: DataSet):

        pass

    def save_model(self):

        pass

    def run(self, data: DataSet, epochs: int, predict=False, seed=0):

        pass