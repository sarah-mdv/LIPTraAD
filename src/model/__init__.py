from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from src.preprocess.dataloader_nguyen import DataSet
from src.model.nguyen_classifier import RNNClassifier
from src.model.k_means_classifier import RNNPrototypeClassifier
from src.misc import load_feature

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
            fold,
            results_dir,
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
            fold,
            results_dir,
            kwargs
    ):
        fold_dataset = DataSet(fold, kwargs.validation, kwargs.data,
                               load_feature(kwargs.features), strategy=kwargs.strategy, batch_size=kwargs.batch_size)
        self.classifier = RNNClassifier()
        self.pred_output = kwargs.out
        self.classifier.build_model(h_drop=kwargs.h_drop, h_size=kwargs.h_size,
                                    i_drop=kwargs.i_drop, nb_classes=3, nb_layers=kwargs.nb_layers,
                                    nb_measures=len(fold_dataset.train.value_fields()),
                                    mean=fold_dataset.mean, stds=fold_dataset.std)
        self.classifier.build_optimizer(kwargs.lr, kwargs.weight_decay)
        self.classifier.run(fold_dataset, epochs=kwargs.epochs, out=self.pred_output, predict=True)
        return fold_dataset


class RNNProRunner(BaselineRunner):
    @property
    def name(self):
        return "rnnpro"

    def run(
            self,
            fold,
            results_dir,
            kwargs
    ):
        # Train the RNN classifier as a separate class
        encoder_model = kwargs.encoder_model
        if not encoder_model:
            LOGGER.info('\n========== Start pretraining ==========')
            LOGGER.debug("Generate dataset for encoder pretraining")
            fold_dataset = DataSet(fold, kwargs.validation, kwargs.data,
                                   load_feature(kwargs.features), fold_n=fold[3], strategy=kwargs.strategy,
                                   batch_size=kwargs.batch_size)
            encoder = RNNClassifier()
            encoder.build_model(nb_classes=3, nb_measures=len(fold_dataset.train.value_fields()),
                                h_size=kwargs.h_size, i_drop=kwargs.i_drop,
                                h_drop=kwargs.h_drop, nb_layers=kwargs.nb_layers, mean=fold_dataset.mean,
                                stds=fold_dataset.std)
            encoder.build_optimizer(kwargs.lr, kwargs.weight_decay)
            encoder.run(fold_dataset, epochs=kwargs.pre_epochs, out=kwargs.out)
            LOGGER.info('\n========== End pretraining ==========')
            encoder_model = encoder.save_model()
        else:
            encoder_model = kwargs.encoder_model
        #New dataset with batch size of 1 so that we get 1 to 1 prototype to hidden state mapping
        kmeans_dataset = DataSet(fold, kwargs.validation, kwargs.data,
                                   load_feature(kwargs.features), fold_n=fold[3], strategy=kwargs.strategy,
                                   batch_size=kwargs.batch_size)
        self.classifier = RNNPrototypeClassifier(kwargs.n_prototypes)
        self.classifier.build_model(encoder_model=encoder_model, h_size=kwargs.h_size,
                                    n_jobs=kwargs.n_jobs)
        #TODO add args here for learning rate and weight decay
        self.classifier.fit(kmeans_dataset.train, outdir=results_dir)
        #self.classifier.output_prototypes(n_fold)
        res_table = self.classifier.predict(kmeans_dataset, fold[3], results_dir)
        return kmeans_dataset

REGISTERED_BASELINE_RUNNERS = [
    RNNRunner(),
    RNNProRunner()
]

RUNNERS_BY_NAME = {
    runner.name: runner for runner in REGISTERED_BASELINE_RUNNERS
    }  # type:Dict[str, BaselineRunner]


def run_from_name(
        classifier_name: str,
        fold,
        results_dir,
        kwargs
):
    RUNNERS_BY_NAME[classifier_name].run(
        fold,
        results_dir,
        kwargs
    )
