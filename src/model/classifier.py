import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import pandas as pd
import torch.nn as nn

from ignite.handlers import ModelCheckpoint

from src.misc import DATA_DIR
from src.preprocess.dataloader_nguyen import DataSet

LOGGER = logging.getLogger(__name__)


class BaseClass:
    def __init__(self) -> None:
        pass


class Classifier(BaseClass):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.model = None # type: nn.Module
        self.score = None
        self.max_length = None
        self.cudnn = False
        self.optimizer = None

    def build_model(self, **kwargs) -> nn.Module:
        """
        Build the model which is later called by fit method to train on given data.
        The final version of the model should be returned.
        :return: Model
        """
        pass

    def build_optimizer(self, lr, weight_decay):
        """
        Build the optimizer that should be used during training
        :param lr: Learning rate for the optimizer
        :param weight_decay:
        :return:
        """

    def create_weight_save_callback(self, path: Optional[str] = None) -> ModelCheckpoint:
        """
        Create an Ignite callback to save the model after each epoch.
        Needs to be registered when starting to train the model.
        :param path: path were to save the model weights
        :return: ModelCheckpoint callback
        """

        if not path:
            checkpoint_path = DATA_DIR / Path(
                f"{self.name}-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}-cp.ckpt"
            )
        else:
            checkpoint_path = Path(path)

        if not checkpoint_path.parent.exists():
            checkpoint_path.parent.mkdir()

        # Create a callback that saves the model's weights
        return ModelCheckpoint(dirname=checkpoint_path.parent())

    def fit(self, data:DataSet, epochs: int):
        """
        Start the training of the model with trainings data
        The methods build_model and build_optimizer should be called before this method
        :param x_train: x data
        :param y_train: labels
        :param epochs: number of epochs to train for.
        :return: None
        """
        if not self.model:
            LOGGER.error("ERROR: self.model is None. Did you call build_model()?")
            exit(1)
        elif not self.optimizer:
            LOGGER.error("ERROR: self.optimizer is None. Did you call build_optimizer()?")
            exit(1)

    def predict(self, data: DataSet):
        """
        Given unknown input data predict the labels for this.
        :param data:
        :return:
        """
        pass

    def save_model(self):
        """
        Save the entire model to disk (if possible)
        :return:
        """
        pass

    def write_predictions(self, pred_X, output_path):
        """
        Write predictions into csv file which conforms which with kaggle requirements.
        :param pred_X:
        :param output_path:
        :return:
        """
        LOGGER.info(f"Creating predictions...")
        pred_Y = self.predict(pred_X)

        columns = ["Id", "Prediction"]
        df = pd.DataFrame(columns=columns)
        for idx, pred in enumerate(pred_Y):
            df.loc[idx] = idx + 1, -1 if pred < 1 else 1

        LOGGER.info(f"Writing submission to {output_path}")
        with output_path.open("w") as f:
            df.to_csv(f, index=False, header=True)

    def run(self, data: DataSet, epochs: int, predict=False, seed=0):
        """
        Run the model (train, test, predict)
        :param data: Dataset containing training, validation, and test data
        :param predict: Whether to predict test data labels and write submissions file
        :return:
        """
        pass
