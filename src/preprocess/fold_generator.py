#!/usr/bin/env python
# Adapted from code written by Minh Nguyen and CBIG under MIT license:
"""
The purpose of this module is to generate train, test, and validation sets
"""
import argparse
import os.path as path
import numpy as np
import pandas as pd
import logging
import src.misc as misc

LOGGER = logging.getLogger(__name__)

class FoldGen(object):
    def __init__(self, seed, datafile, features, nbfolds, outdir):
        self.seed = seed
        self.datafile = datafile
        self.features = features
        self.nbfolds = nbfolds
        self.outdir = outdir
        self.curr_fold = 0
        self.folds = []
        self.leftover = []
        self.data = []

        self.create_folds()

    def create_folds(self):
        np.random.seed(self.seed)

        columns = ['RID', 'DXCHANGE', 'EXAMDATE']

        features = misc.load_feature(self.features)
        self.data = pd.read_csv(
            self.datafile,
            usecols=columns + features,
            converters=misc.CONVERTERS)
        self.data['has_data'] = ~self.data[features].isnull().apply(np.all, axis=1)

        """ Generate *nb_folds* cross-validation folds from *data """
        subjects = np.unique(self.data.RID)
        has_2tp = np.array([np.sum(self.data.RID == rid) >= 2 for rid in subjects])

        potential_targets = np.random.permutation(subjects[has_2tp])
        self.folds = np.array_split(potential_targets, self.nbfolds)

        self.leftover = [subjects[~has_2tp]]

    def __iter__(self):
        return self

    def __len__(self):
        return self.nbfolds

    def __next__(self):
        if self.curr_fold == self.nbfolds:
            self.curr_fold = 0
            raise StopIteration()

        mask_frame, val_frame, test_frame = gen_fold(
            self.data, self.nbfolds, self.curr_fold, self.folds, self.leftover, self.outdir)
        self.curr_fold += 1
        return mask_frame, val_frame, test_frame, self.curr_fold - 1


#This function is leftover from Nguyen, I don't think it's useful though
def split_by_median_date(data, subjects):
    """
    Split timepoints in two halves, use first half to predict second half
    Args:
        data (Pandas data frame): input data
        subjects: list of subjects (unique RIDs)
    Return:
        first_half (ndarray): boolean mask, rows used as input
        second_half (ndarray): boolean mask, rows to predict
    """
    first_half = np.ones(data.shape[0], int)
    second_half = np.zeros(data.shape[0], int)
    for rid in subjects:
        subj_mask = (data.RID == rid) & data.has_data
        median_date = np.sort(data.EXAMDATE[subj_mask])[subj_mask.sum() // 2]
        first_half[subj_mask & (data.EXAMDATE < median_date)] = 1
        second_half[subj_mask & (data.EXAMDATE >= median_date)] = 1
    return first_half, second_half


def gen_fold(data, nb_folds, test_fold, folds, leftover, outdir=""):
    """ Generate *nb_folds* cross-validation folds from *data """
    #We will end up with nb_train folds = nb_folds - 2, 1 test_fold, 2 val_fold
    LOGGER.info("Generating {} folds across data.".format(nb_folds))
    val_fold = (test_fold + 1) % nb_folds
    train_folds = [
        i for i in range(nb_folds) if (i != test_fold and i != val_fold)
    ]

    train_subj = np.concatenate(
        [folds[i] for i in train_folds] + leftover, axis=0)
    val_subj = folds[val_fold]
    test_subj = folds[test_fold]

    train_mask = (np.in1d(data.RID, train_subj) & data.has_data).astype(int)

    val_mask = (np.in1d(data.RID, val_subj) & data.has_data).astype(int)
    test_mask = (np.in1d(data.RID, test_subj) & data.has_data).astype(int)

    # val_in_timepoints, val_out_timepoints = split_by_median_date(data, val_subj)
    # test_in_timepoints, test_out_timepoints = split_by_median_date(data, test_subj)

    mask_frame = gen_mask_frame(data, train_mask, val_mask,
                                test_mask)

    val_frame = gen_ref_frame(data, val_mask)

    test_frame = gen_ref_frame(data, test_mask)
    # if outdir:
    #     mask_frame.to_csv(path.join(outdir, 'fold%d_mask.csv' % test_fold), index=False)
    #     val_frame.to_csv(path.join(outdir, 'fold%d_val.csv' % test_fold), index=False)
    #     test_frame.to_csv(path.join(outdir, 'fold%d_test.csv' % test_fold), index=False)

    return mask_frame, val_frame, test_frame


def gen_mask_frame(data, train, val, test):
    """
    Create a frame with 3 masks:
        train: timepoints used for training model
        val: timepoints used for validation
        test: timepoints used for testing model
    """
    col = ['RID', 'EXAMDATE']
    ret = pd.DataFrame(data[col], index=range(train.shape[0]))
    ret['train'] = train
    ret['val'] = val
    ret['test'] = test

    return ret


def gen_ref_frame(data, test_timepoint_mask):
    """ Create reference frame which is used to evalute models' prediction """
    columns = [
        'RID', 'CognitiveAssessmentDate', 'Diagnosis', 'ADAS13', 'ScanDate'
    ]
    ret = pd.DataFrame(
        np.nan, index=range(len(test_timepoint_mask)), columns=columns)
    ret[columns] = data[['RID', 'EXAMDATE', 'DXCHANGE', 'ADAS13', 'EXAMDATE']]
    ret['Ventricles'] = data['Ventricles'] / data['ICV']
    ret = ret[test_timepoint_mask == 1]

    # map diagnosis from numeric categories back to labels
    mapping = {
        1: 'CN',
        7: 'CN',
        9: 'CN',
        2: 'MCI',
        4: 'MCI',
        8: 'MCI',
        3: 'AD',
        5: 'AD',
        6: 'AD'
    }
    ret.replace({'Diagnosis': mapping}, inplace=True)
    ret.reset_index(drop=True, inplace=True)

    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--datafile', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--nbfolds', type=int, required=True)
    parser.add_argument('--outdir', required=True)

    args = parser.parse_args()
    folds = FoldGen(args.seed, args.datafile, args.features, args.nbfolds, args.outdir)
    folds.__next__()

if __name__ == '__main__':
    main()
