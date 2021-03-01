"""
The purpose of this module is to load the data, to fill it according to a requested strategy,
and to return batches of the data to the main function
"""

# Written by Minh Nguyen and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
from __future__ import print_function, division
import numpy as np
from scipy.interpolate import interp1d
import argparse
import pickle
import src.misc as misc

from src.misc import LOG_DIR

import logging

LOGGER = logging.getLogger(__name__)


def typecheck(interp_func):
    def func_wrapper(month_true, val_true, val_default, month_interp):
        assert isinstance(month_true, np.ndarray)
        assert isinstance(val_true, np.ndarray)
        assert isinstance(month_interp, np.ndarray)
        assert month_true.dtype == month_interp.dtype == int
        assert month_true.shape == val_true.shape
        assert np.all(month_true[1:] > month_true[:-1]), 'not sorted?'
        assert np.all(month_interp[1:] > month_interp[:-1]), 'not sorted?'
        return interp_func(month_true, val_true, val_default, month_interp)

    return func_wrapper


def mask_and_reference(interp_func):
    """
    A function decorator
    Args:
        interp_func: a function that interpolates data
    Returns:
        func_wrapper: a function that returns the interpolated data (interp)
        and 2 additional arrays having the same shape as interp
            - truth: containing ground truth values, NaN => no ground truth
            - mask: a boolean mask, True => timepoint has ground truth data
    """

    def func_wrapper(month_true, val_true, val_default, month_interp):
        """
        Args:
            month_true (ndarray): months with ground truth
            val_true (ndarray): ground truth values
            val_default (float): value to fill when there is no ground truth
            month_interp (ndarray): months to interpolate
        Returns:
            interp (ndarray): interpolated values
            mask (ndarray): boolean mask
            truth (ndarray): ground truth values
        """
        has_data = ~np.isnan(val_true)

        mask = np.in1d(month_interp, month_true[has_data].astype(np.int))
        truth = np.full(month_interp.shape, np.nan)
        truth[mask] = val_true[has_data]

        interp = interp_func(month_true, val_true, val_default, month_interp)

        return interp, mask, truth

    return func_wrapper


@typecheck
@mask_and_reference
def bl_fill(month_true, val_true, val_default, month_interp):
    """ Model filling
    fill only first timepoints, other timepoints are filled by the model
    """
    has_data = ~np.isnan(val_true)
    valid_x, valid_y = month_true[has_data], val_true[has_data]
    interp = np.full(month_interp.shape, np.nan, np.float)

    # fill first timepoint
    interp[0] = valid_y[0] if sum(has_data) else val_default

    # find timepoints in both valid_x and month_interp
    common_tps = month_interp[np.in1d(month_interp, valid_x)]
    interp[np.in1d(month_interp, common_tps)] = \
        valid_y[np.in1d(valid_x, common_tps)]

    return interp


@typecheck
@mask_and_reference
def ff_fill(month_true, val_true, val_default, month_interp):
    """ Forward filling """
    has_data = ~np.isnan(val_true)
    valid_x, valid_y = month_true[has_data], val_true[has_data]
    interp = np.full(month_interp.shape, np.nan, np.float)

    if len(valid_y) == 0:
        interp[:] = val_default
    elif len(valid_y) == 1:
        interp[:] = valid_y[0]
    else:
        interp_fn = interp1d(
            valid_x, valid_y, kind='previous', fill_value='extrapolate')
        interp[:] = interp_fn(month_interp)

    return interp


@typecheck
@mask_and_reference
def neighbor_fill(month_true, val_true, val_default, month_interp):
    """ Nearest-neighbor filling """
    has_data = ~np.isnan(val_true)
    valid_x, valid_y = month_true[has_data], val_true[has_data]
    interp = np.full(month_interp.shape, np.nan, np.float)

    if len(valid_y) == 0:
        interp[:] = val_default
    elif len(valid_y) == 1:
        interp[:] = valid_y[0]
    else:
        interp_fn = interp1d(
            valid_x, valid_y, kind='nearest', fill_value='extrapolate')
        interp[:] = interp_fn(month_interp)

    return interp


def valid(time_array, tmax, tmin):
    return time_array[(time_array >= tmin) & (time_array <= tmax)]


@typecheck
@mask_and_reference
def ln_fill_partial(month_true, val_true, val_default, month_interp):
    """ Mixed model-linear filling """
    has_data = ~np.isnan(val_true)
    valid_x, valid_y = month_true[has_data], val_true[has_data]
    interp = np.full(month_interp.shape, np.nan, np.float)

    interp[0] = valid_y[0] if sum(has_data) else val_default

    if len(valid_y) == 1:
        # will be different from previous line when valid_x is not the first tp
        interp[np.in1d(month_interp, valid_x[0])] = valid_y[0]
    elif len(valid_y) > 1:
        interp_fn = interp1d(valid_x, valid_y, kind='linear')
        timepoints = valid(month_true, valid_x[-1], valid_x[0]).astype(np.int)
        mask = np.in1d(month_interp, timepoints)
        interp[mask] = interp_fn(month_interp[mask])

    return interp


@typecheck
@mask_and_reference
def ln_fill_full(month_true, val_true, val_default, month_interp):
    """ Linear filling """
    has_data = ~np.isnan(val_true)
    valid_x, valid_y = month_true[has_data], val_true[has_data]
    interp = np.full(month_interp.shape, np.nan, np.float)

    interp[0] = valid_y[0] if sum(has_data) else val_default

    if len(valid_y) == 1:
        # will be different from previous line when valid_x is not the first tp
        interp[np.in1d(month_interp, valid_x[0])] = valid_y[0]
    elif len(valid_y) > 1:
        interp_fn = interp1d(valid_x, valid_y, kind='linear')
        timepoints = valid(month_interp, valid_x[-1],
                           valid_x[0]).astype(np.int)
        interp[np.in1d(month_interp, timepoints)] = interp_fn(timepoints)

    return interp


STRATEGIES = {
    'forward': ff_fill,
    'neighbor': neighbor_fill,
    'mixed': ln_fill_partial,
    'model': bl_fill
}


def extract(frame, strategy, features, defaults):
    """
    Extract and interpolate time series for each subject in data frame
    Args:
        frame (Pandas frame): input data frame
        strategy (string): name of the interpolation strategy
        features (list of strings): list of features
        defaults (dict): contains default value for each feature
    Returns:
        ret (dict): contain 3 arrays for each subject
            - input: input to RNN/LSS
            - mask: boolean mask indicating availability of ground truth
            - truth: ground truth values
        fields (list of strings):
    """
    interp_fn = STRATEGIES[strategy]

    fields = ['Month_bl', 'DX'] + features
    ret = dict()
    for rid, sframe in misc.get_data_dict(frame, features).items():
        #The indices are the visit months here
        xin = sframe.index.values
        #Assert that the visits are unique
        assert len(xin) == len(set(xin)), rid
        xin -= xin[0]
        """
        Default start is 0, default step size is 1
        (this is to accommodate the 1 month step size expected by the TADPOLE challenge)
        """
        xout = np.arange(xin[-1] - xin[0] + 1)

        in_seqs = {'Month_bl': xout}
        mk_seqs = {'Month_bl': np.zeros(len(xout), dtype=bool)}
        th_seqs = {'Month_bl': np.full(len(xout), np.nan)}

        for f in fields[1:]:
            yin = sframe[f].values
            in_seqs[f], mk_seqs[f], th_seqs[f] = interp_fn(
                xin, yin, defaults[f], xout)

        ret[rid] = {'input': np.array([in_seqs[f] for f in fields]).T}
        ret[rid]['mask'] = np.array([mk_seqs[f] for f in fields]).T
        ret[rid]['truth'] = np.array([th_seqs[f] for f in fields]).T

        assert ret[rid]['input'].shape == ret[rid]['mask'].shape == ret[rid][
            'truth'].shape
    return ret, fields


class Sorted(object):
    """
    An dataloader class for test/evaluation
    The subjects are sorted in ascending order according to subject IDs.
    """

    def __init__(self, data, batch_size, attributes):
        self.data = data[0]
        self.fields = np.array(data[1])
        self.batch_size = batch_size
        self.attributes = attributes
        self.subjects = np.sort(np.array(list(self.data.keys())))
        self.idx = 0

        self.mask = {}
        self.mask['tp'] = self.fields == 'Month_bl'
        self.mask['cat'] = self.fields == 'DX'
        self.mask['val'] = np.zeros(shape=self.fields.shape, dtype=bool)
        for field in self.attributes:
            self.mask['val'] |= self.fields == field

        assert not np.any(self.mask['tp'] & self.mask['val']), 'overlap'
        assert not np.any(self.mask['cat'] & self.mask['val']), 'overlap'

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.ceil(len(self.subjects) / self.batch_size))

    # must give a deep copy of the training data !important
    def __next__(self):
        if self.idx == len(self.subjects):
            self.idx = 0
            raise StopIteration()

        rid = self.subjects[self.idx]
        self.idx += 1

        subj_data = {'rid': rid}
        seq = self.data[rid]['input']
        for k, mask in self.mask.items():
            subj_data[k] = seq[:, mask]

        return subj_data

    def value_fields(self):
        return self.fields[self.mask['val']]


def batch(matrices):
    """
    Create a batch of data from subjects' timecourses
    Args:
        matrices (list of ndarray): [nb_timpoints, nb_features]
    Returns:
        (ndarray): [max_nb_timpoint, nb_subjects, nb_features]
    """
    #print(matrices[0])
    maxlen = max(len(m) for m in matrices)
    ret = [
        np.pad(m, [(0, maxlen - len(m)), (0, 0)], 'constant')[:, None, :]
        for m in matrices
    ]
    return np.concatenate(ret, axis=1)


class Random(Sorted):
    """
    An dataloader class for training
    The subjects are shuffled randomly in every epoch.
    """

    def __init__(self, *args, **kwargs):
        super(Random, self).__init__(*args, **kwargs)
        self.rng = np.random.RandomState(seed=0)

    # must give a deep copy of the training data !important
    def __next__(self):
        if self.idx == len(self.subjects):
            self.rng.shuffle(self.subjects)
            self.idx = 0
            raise StopIteration()

        rid_list = self.subjects[self.idx:self.idx + self.batch_size]
        self.idx += len(rid_list)

        input_batch = batch([self.data[rid]['input'] for rid in rid_list])
        mask_batch = batch([self.data[rid]['mask'] for rid in rid_list])
        truth_batch = batch([self.data[rid]['truth'] for rid in rid_list])

        subj_data = {}
        for k, mask in self.mask.items():
            subj_data[k] = input_batch[:, :, mask]
        subj_data['cat_msk'] = mask_batch[:, :, self.mask['cat']]
        subj_data['val_msk'] = mask_batch[:, :, self.mask['val']]
        subj_data['true_cat'] = truth_batch[:, :, self.mask['cat']]
        subj_data['true_val'] = truth_batch[:, :, self.mask['val']]

        return subj_data


class DataSet:
    def __init__(self, fold, validation, datafile, features,
                    strategy=ff_fill, batch_size=128):
        self.fold = fold
        self.validation = validation
        self.datafile = datafile
        self.features = features
        self.strategy = strategy
        self.batch_size = batch_size

        self.baseline = []
        self.pred_start = []
        self.duration = 7 * 12
        self.mean = 0.
        self.std = 0.
        self.VentICVstd = 0.
        self.train = object
        self.test = object

    def get_data(self):
        """
        Generate training/test data batches and save as pickle file
        *args* specify
            mask: path to mask file
            strategy: filling strategy
            spreadsheet: path to TADPOLE data
            batch_size: batch size
            out: path to save pickle file
            validation: evaluate on validation subjects (instead of test subjects)
        """

        ret = {}
        mask_frame, rest = self.fold
        train_mask, pred_mask, pred_mask_frame = misc.get_mask(
            mask_frame, self.validation)
        self.baseline, self.pred_start = misc.get_baseline_prediction_start(
            pred_mask_frame)

        columns = ['RID', 'Month_bl', 'DX'] + self.features
        frame = misc.load_table(self.datafile, columns)

        tf = frame.loc[train_mask, self.features]
        self.mean = tf.mean()
        self.mean = tf.std()
        self.VentICVstd = (tf['Ventricles'] / tf['ICV']).std()

        #Normalize the data
        frame[self.features] = (frame[self.features] - ret['mean']) / ret['stds']
        default_val = {f: 0. for f in self.features}
        default_val['DX'] = 0.

        data = extract(frame[train_mask], self.strategy, self.features,
                                  default_val)
        self.train = Random(data, self.batch_size, self.features)

        data = extract(frame[pred_mask], self.strategy, self.features,
                                  default_val)
        self.test = Sorted(data, 1, self.features)

        print('train', len(self.train.subjects), 'subjects')
        print('test', len(self.test.subjects), 'subjects')
        print(len(self.features), 'features')

        return ret

    def cache_data(self, outdir):
        LOGGER.debug("Pickle fold {} data at {}".format(self.fold[3], self.outdir))
        with open(outdir, 'wb') as fhandler:
            pickle.dump(self.__dict__, fhandler)

    @classmethod
    def retrieve_cached(cls, cached_data_file):
        LOGGER.debug("Retrieving cached dataloader object from {}".format(cached_data_file))
        data = cls.__new__(cls)
        with open(cached_data_file, 'rb') as fhandler:
            _dict = pickle.load(fhandler)
            data.__dict__ = _dict
        return data


def main(args):
    fields = misc.load_feature(args.features)
    with open(args.out, 'wb') as fhandler:
        d = get_data(args, fields)
        pickle.dump(d, fhandler)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--strategy', required=True)
    parser.add_argument('--spreadsheet', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--validation', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
