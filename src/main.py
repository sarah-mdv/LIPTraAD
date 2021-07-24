import logging
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd

from src.model import run_from_name
from src.preprocess.fold_generator import FoldGen

from src.misc import (
    _setup_logger,
    load_feature,
    LOG_DIR,
)
from src.preprocess.dataloader import (
    STRATEGIES,
    DataSet
)


def main(args):
    #Setup results dir
    datetime_now = datetime.now().strftime("%d%m%y-%H%M%S_lṛ_{}_wd_{}_bs_{}_n_prototypes_{}_model_{}".format(args.lr,
                                                                                              args.weight_decay,
                                                                                    args.batch_size,
                                                                                                              args.n_prototypes, args.model))
    results_dir = LOG_DIR / "results" / datetime_now
    results_dir.mkdir(parents=True)

    (results_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    #Set program-wide logging level
    _setup_logger(logging._nameToLevel[args.debug_level.upper()])
    LOGGER = logging.getLogger("src.main")

    if not args.reuse_folder:
        #Setup fold iterator
        folds = FoldGen(args.seed, args.data, args.features, args.folds, results_dir)

        #Generate datasets over folds and perform training and evaluation
        fold_bac = 0
        bac = []
        for i, fold in enumerate(folds):
            b = run_from_name(args.model, fold, results_dir, args)
            fold_bac += b[0]
            bac.append(b)

        mean_bac = fold_bac / args.folds

        bac = pd.DataFrame(np.concatenate(np.vstack(bac)).reshape(args.folds, -1))
        bac.to_csv(results_dir / "score_record.csv", index=False)
        LOGGER.info("BAC over {} folds: {}".format(args.folds, mean_bac))
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
    parser.add_argument('--cache', action='store_true')

    #Fold generator arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help="Seed for generating random data throughout the pipeline"
    )
    parser.add_argument(
        '--data',
        required=True,
        help='Path to the ADNI dataset (.csv format)'
        )
    parser.add_argument('--features',
                        required=True,
                        help="Path to the file with the set of features to be used for prediction"
                        )
    parser.add_argument('--folds',
                        type=int,
                        required=True,
                        help="Number of folds to create across the data"
                        )

    #Dataloader arguments
    parser.add_argument('--strategy',
                        choices=STRATEGIES.keys(),
                        default="forward",
                        help="Choice of imputation method"
                        )
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help="Batch size (default 128)"
                        )
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--reuse_folder',
                        type=str,
                        default="",
                        help="Add the path of a directory here if you want to reuse previously generated folds"
                        )
    parser.add_argument('--encoder_model',
                        type=str,
                        default="",
                        help="Load an already trained encoder model from memory"
                        )

    #Model arguments
    parser.add_argument('--model',
                        '-m',
                        required=True,
                        help="Name of the classifier model to use"
                        )
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help="Number of epochs to train"
                        )
    parser.add_argument('--pre_epochs',
                        type=int,
                        default=10,
                        help="Number of epochs for pretraining the encoder layer"
                        )
    parser.add_argument('--lr',
                        type=float,
                        required=True,
                        default=0.001,
                        help="Learning rate"
                        )
    parser.add_argument('--w_ent',
                        type=float,
                        default=1.,
                        help="Entropy weight in loss"
                        )
    parser.add_argument('--w_reg',
                        type=float,
                        default=1.,
                        help="Entropy weight in loss"
                        )
    parser.add_argument('--nb_layers',
                        type=int,
                        default=1,
                        help="Number of hidden layers in the model"
                        )
    parser.add_argument('--h_size',
                        type=int,
                        default=128,
                        help="Hidden size"
                        )
    parser.add_argument('--i_drop',
                        type=float,
                        default=.1)
    parser.add_argument('--h_drop',
                        type=float,
                        default=.4)
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.01)
    parser.add_argument('--n_prototypes',
                        type=int,
                        default=10)
    parser.add_argument('--n_t_prototypes',
                        type=int,
                        default=0)
    parser.add_argument('--inter_res',
                        action="store_true",
                        help="Flag indicating whether to store intermediate values (train, test sets)")
    parser.add_argument('--out',
                        required=True,
                        help="Output file for prediction")
    parser.add_argument('--beta',
                        type=float,
                        default=.5)
    parser.add_argument('--n_classes',
                        type=int,
                        default=3)

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
