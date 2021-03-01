import logging
import argparse
import json
from src.preprocessing import modules as fill_modules
from src.preprocessing.imputation import FillerModule
from datetime import datetime
from pathlib import Path

from src.preprocess.fold_generator import FoldGen

fill_dict = {module.name: module for module in fill_modules}

from src.misc import (
    _setup_logger,
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
    folds = FoldGen(args.seed, args.datafile, args.features, args.nbfolds, results_dir)

    for fold in folds:
        mask_frame, val_frame, test_frame = fold
    #Load data
    data_dict = {}

    #Get the imputation model selected
    filler_module = fill_dict[args.filler]
    LOGGER.info(f"Performing imputation with {args.filler}")

    #Perform imputation on data
    data_dict = filler_module.run(data_dict)

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

    #Set the type of imputation
    parser.add_argument(
        "--filler",
        dest="filler",
        action="store_true",
        choices=fill_dict.keys(),
        help="Type of filling that will be done on the data."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
