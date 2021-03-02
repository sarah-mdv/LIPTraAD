from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import logging

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
        embedding,
        training_files,
        testing_files,
        prediction_file,
        epochs: int,
        proxy: bool,
    ) -> None:
        """
        Setup the classifier module and run it here.
        :return: None
        """
        pass

