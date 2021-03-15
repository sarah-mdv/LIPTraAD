from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from src.preprocess.dataloader_nguyen import DataSet
from src.model.nguyen_classifier import RNNClassifier
from src.model.k_means_classifier import RNNPrototypeClassifier

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
            dataset: DataSet,
            validation_dataset,
            kwargs
    ) -> None:
        """
        Setup the classifier module and run it here.
        :return: None
        """
        pass


# TODO: Add runners for all other classifiers


class RNNRunner(BaselineRunner):
    @property
    def name(self):
        return "rnn"

    def run(
            self,
            fold_dataset: DataSet,
            validation_dataset,
            kwargs
    ):
        self.classifier = RNNClassifier()
        print(kwargs)
        self.pred_output = kwargs.out
        self.classifier.build_model(h_drop=kwargs.h_drop, h_size=kwargs.h_size,
                                    i_drop=kwargs.i_drop, nb_classes=3, nb_layers=kwargs.nb_layers,
                                    nb_measures=len(fold_dataset.train.value_fields()),
                                    mean=fold_dataset.mean, stds=fold_dataset.std)
        self.classifier.build_optimizer(kwargs.lr, kwargs.weight_decay)
        self.classifier.run(fold_dataset, epochs=kwargs.epochs, out=self.pred_output, predict=True)


class RNNProRunner(BaselineRunner):
    @property
    def name(self):
        return "rnnpro"

    def run(
            self,
            fold_dataset: DataSet,
            validation,
            args
    ):
        # Train the RNN classifier as a separate class
        encoder_model = ""
        if not encoder_model:
            encoder = RNNClassifier()
            encoder.build_model(nb_classes=3, nb_measures=len(fold_dataset.train.value_fields()),
                                h_size=args.h_size, i_drop=args.i_drop,
                                h_drop=args.h_drop, nb_layers=args.nb_layers, mean=fold_dataset.mean,
                                stds=fold_dataset.std)
            encoder.build_optimizer(args.lr, args.weight_decay)
            LOGGER.info('========== Start pretraining ==========')
            encoder.run(fold_dataset, epochs=args.pre_epochs, out=args.out)
            LOGGER.info('========== End pretraining ==========\n')
            encoder_model = encoder.save_model()
        self.classifier = RNNPrototypeClassifier(args.n_prototypes)
        self.classifier.build_model(encoder_model=encoder_model, h_size=args.h_size,
                                    n_jobs=args.n_jobs)
        self.classifier.init_clusters(fold_dataset.train)


REGISTERED_BASELINE_RUNNERS = [
    RNNRunner(),
    RNNProRunner()
]

RUNNERS_BY_NAME = {
    runner.name: runner for runner in REGISTERED_BASELINE_RUNNERS
    }  # type:Dict[str, BaselineRunner]


def run_from_name(
        classifier_name: str,
        dataset: DataSet,
        validation_dataset,
        args
):
    RUNNERS_BY_NAME[classifier_name].run(
        dataset,
        validation_dataset,
        args
    )
