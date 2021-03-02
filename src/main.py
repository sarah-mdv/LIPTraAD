import logging
import argparse
import json
from datetime import datetime
from pathlib import Path

from src.preprocess.dataloader import (
    STRATEGIES,
    DataSet
)

from src.preprocess.fold_generator import FoldGen

from src.misc import (
    _setup_logger,
    load_feature,
    LOG_DIR
)


def main(args):
    #Setup results dir
    datetime_now = datetime.now().strftime("%d%m%y-%H%M%S")
    results_dir = LOG_DIR / "results" / datetime_now
    results_dir.mkdir(parents=True)

    (results_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    #Set program-wide logging level
    _setup_logger(logging._nameToLevel[args.debug_level.upper()])
    LOGGER = logging.getLogger("src.main")

    #Setup fold iterator
    folds = FoldGen(args.seed, args.data, args.features, args.nbfolds, results_dir)

    #Generate datasets over folds and perform training and evaluation
    for fold in folds:
        fold_dataset = DataSet(fold, args.validation, args.data,
                                load_feature(args.features), strategy=args.strategy, batch_size=args.batch_size)
        fold_dataset.gen_data()

    return 0


def get_args():
    parser = argparse.ArgumentParser()

    #Set logging level
    parser.add_argument(
        "--debug-level",
        "-d",
        choices=["error", "warning", "info", "debug"],
        default="info",
    )

    #Fold generator arguments
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--nbfolds', type=int, required=True)

    #Dataloader arguments
    parser.add_argument('--strategy', choices=STRATEGIES.keys(), required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--out')
    parser.add_argument('--validation', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
