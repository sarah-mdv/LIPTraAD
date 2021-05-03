from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from src.preprocess.dataloader import DataSet
from src.model.nguyen_classifier import RNNClassifier
from src.model.k_means_classifier import RNNPrototypeClassifier
from src.model.standard_autoencoder import StandardAutoencoderModel
from src.misc import load_feature, output_hidden_states

import logging

"""
This class serves as a wrapper for different classifiers
It simplifies the interface between the caller and the different modules
"""

LOGGER = logging.getLogger(__name__)


class BaselineRunner(ABC):
    def __init__(self):
        self.model = None

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
        LOGGER.debug("Running basic RNN model")
        fold_dataset = DataSet(fold, kwargs.validation, kwargs.data,
                               load_feature(kwargs.features), fold_n=fold[3],
                               strategy=kwargs.strategy, batch_size=kwargs.batch_size)
        self.model = RNNClassifier()
        self.pred_output = kwargs.out
        self.model.build_model(h_drop=kwargs.h_drop, h_size=kwargs.h_size,
                                    i_drop=kwargs.i_drop, nb_classes=kwargs.n_classes, nb_layers=kwargs.nb_layers,
                                    nb_measures=len(fold_dataset.train.value_fields()),
                                    mean=fold_dataset.mean, stds=fold_dataset.std)
        self.model.build_optimizer(kwargs.lr, kwargs.weight_decay)
        self.model.run(fold_dataset, epochs=kwargs.epochs, out=self.pred_output, predict=True)
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
        LOGGER.debug("Running prototype-similarity RNN model")
        # Train the RNN classifier as a separate class
        encoder_model = kwargs.encoder_model
        fold_dataset = DataSet(fold, kwargs.validation, kwargs.data,
                               load_feature(kwargs.features), fold_n=fold[3], strategy=kwargs.strategy,
                               batch_size=kwargs.batch_size)
        if not encoder_model:
            LOGGER.info('\n============================ Start pretraining ============================')
            LOGGER.debug("Generating dataset for encoder pretraining")
            encoder = RNNClassifier()
            encoder.build_model(nb_classes=kwargs.n_classes, nb_measures=len(fold_dataset.train.value_fields()),
                                h_size=kwargs.h_size, i_drop=kwargs.i_drop,
                                h_drop=kwargs.h_drop, nb_layers=kwargs.nb_layers, mean=fold_dataset.mean,
                                stds=fold_dataset.std)
            encoder.build_optimizer(kwargs.lr, kwargs.weight_decay)
            encoder.run(fold_dataset, epochs=kwargs.pre_epochs, out=kwargs.out)
            LOGGER.info('\n============================ End pretraining ============================')
            encoder_model = encoder.save_model(results_dir)
        else:
            encoder_model = kwargs.encoder_model
        #New dataset with batch size of 1 so that we get 1 to 1 prototype to hidden state mapping
        self.model = RNNPrototypeClassifier(kwargs.n_prototypes)
        self.model.build_model(encoder_model=encoder_model, h_size=kwargs.h_size,
                                    n_jobs=kwargs.n_jobs)
        hidden_states = self.model.fit(fold_dataset.train, hidden=kwargs.inter_res,epochs=kwargs.epochs,
        outdir=results_dir)
        if kwargs.inter_res:
            output_hidden_states(hidden_states, kwargs.data, results_dir, fold[3])
        #self.model.output_prototypes(n_fold)
        res_table = self.model.predict(fold_dataset, fold[3], results_dir)
        return fold_dataset


class AutoencoderRunner(BaselineRunner):
    @property
    def name(self):
        return "ae"

    def run(
            self,
            fold,
            results_dir,
            kwargs
    ):
        LOGGER.debug("Running basic autoencoder model")
        fold_dataset = DataSet(fold, kwargs.validation, kwargs.data,
                               load_feature(kwargs.features), fold_n=fold[3], strategy=kwargs.strategy,
                               batch_size=kwargs.batch_size)
        self.model = StandardAutoencoderModel(results_dir)
        self.model.build_model(nb_classes=kwargs.n_classes, nb_measures=len(fold_dataset.train.value_fields()),
                                h_size=kwargs.h_size, i_drop=kwargs.i_drop,
                                h_drop=kwargs.h_drop, mean=fold_dataset.mean,
                                stds=fold_dataset.std)
        self.model.build_optimizer(kwargs.lr, kwargs.weight_decay)
        self.model.fit(fold_dataset, kwargs.epochs, kwargs.seed, hidden=kwargs.inter_res)
        self.model.predict(fold_dataset, fold[3], results_dir)
        return 0


REGISTERED_BASELINE_RUNNERS = [
    RNNRunner(),
    RNNProRunner(),
    AutoencoderRunner()
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
