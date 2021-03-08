from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from src.preprocess.dataloader import DataSet
from src.model.nguyen_classifier import RNNClassifier

import logging

"""
This class serves as a wrapper for different classifiers
It simplifies the interface between the caller and the different modules
"""

LOGGER = logging.getLogger(__name__)

class BaselineRunner(ABC):
    def __init__(self):
        self.classifier = None


    @property
    def requires_embedding(self) -> bool:
        """
        Whether an embedding is needed for this classifier.
        :return: bool
        """
        return True

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: name which will be used for selecting the runner on the CLI
        """
        pass

    @abstractmethod
    def run(
        self,
        data: DataSet,
        validation,
        args
    ) -> None:
        """
        Setup the classifier module and run it here.
        :return: None
        """
        pass

#TODO: Add runners for all other classifiers


class RNNRunner(BaselineRunner):
    @property
    def name(self):
        return "rnn"

    def run(
        self,
        fold_dataset: DataSet,
        validation,
        args
    ):
        self.classifier = RNNClassifier()
        self.pred_output = args.out
        self.classifier.build_model(nb_classes=3, nb_measures=len(fold_dataset.train.value_fields()),
                                   h_size=args.h_size, i_drop=args.i_drop,
                                   h_drop=args.h_drop, nb_layers=args.nb_layers, mean=fold_dataset.mean,
                                   stds=fold_dataset.std)
        self.classifier.build_optimizer(args.lr, args.weight_decay)
        self.classifier.run(fold_dataset, epochs=args.epochs, out=self.pred_output)


REGISTERED_BASELINE_RUNNERS = [
    RNNRunner(),
]

RUNNERS_BY_NAME = {
    runner.name: runner for runner in REGISTERED_BASELINE_RUNNERS
}  # type:Dict[str, BaselineRunner]


def run_from_name(
    classifier_name: str,
    data:DataSet,
    validation,
    args
):
    RUNNERS_BY_NAME[classifier_name].run(
        data,
        validation,
        args
    )

