import logging
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



from src.preprocess.dataloader import DataSet

LOGGER = logging.getLogger(__name__)


class BaseClass:
    def __init__(self) -> None:
        pass


class Autoencoder(BaseClass):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.ae_model = None # type: nn.Module
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
        pass

    def build_summary_writer(self, lr, weight_decay, batch_size):
        datetime_now = datetime.now().strftime("%d%m-%H%M%S")

        writer_name = "runs/lr= {}, weight_decay={}, batch_size={}, name={} date={}".format(lr, weight_decay,
                                                                                            batch_size,
                                                                                    self.name, datetime_now)
        self.writer = SummaryWriter(writer_name)


    def reg_loss(self):
        return 0

    def fit(self, data:DataSet, epochs:int):
        """
        Start the training of the model with trainings data
        The methods build_model and build_optimizer should be called before this method
        :param x_train: x data
        :param epochs: number of epochs to train for.
        :return: None
        """
        if not self.ae_model:
            LOGGER.error("ERROR: self.ae_model is None. Did you call build_model()?")
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

    def run(self, data: DataSet, epochs: int, predict=False, seed=0):
        """
        Run the model (train, test, predict)
        :param data: Dataset containing training, validation, and test data
        :param predict: Whether to predict test data labels and write submissions file
        :return:
        """
        pass
