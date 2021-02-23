import pandas as pd
import numpy as np
import pickle
from datetime import datetime

CN = ("CN", 0.)
MCI = ("MCI", 1.)
AD = ("AD", 2.)


def str2date(string):
    """ Convert string to datetime object """
    return datetime.strptime(string, '%Y-%m-%d')


def VISCODE_conv(value):
    """Convert visit code to month int"""
    if isinstance(value, str):
        if value.startswith("m"):
            return int(value[1:])
        if value.startswith("bl"):
            return 0
        return int("NaN")


def PET_conv(value):
    '''Convert PET measures from string to float '''
    try:
        return float(value.strip().strip('>'))
    except ValueError:
        return float(np.nan)


def Diagnosis_conv(value):
    '''Convert diagnosis from string to float '''
    if value == 'CN':
        return CN
    if value == 'MCI':
        return MCI
    if value == 'AD':
        return AD
    return float('NaN')


def DX_conv(value):
    '''Convert change in diagnosis from string to float '''
    if isinstance(value, str):
        if value.endswith('Dementia') or value.endswith('AD'):
            return 2.
        if value.endswith('MCI'):
            return 1.
        if value.endswith('CN'):
            return 0.

    return float('NaN')


CONVERTERS = {
    'VISCODE':VISCODE_conv,
    'CognitiveAssessmentDate': str2date,
    'ScanDate': str2date,
    'Forecast Date': str2date,
    'EXAMDATE': str2date,
    'Diagnosis': Diagnosis_conv,
    'DX': DX_conv,
    'DX_bl':DX_conv,
    'PTAU_UPENNBIOMK9_04_19_17': PET_conv,
    'TAU_UPENNBIOMK9_04_19_17': PET_conv,
    'ABETA_UPENNBIOMK9_04_19_17': PET_conv
}


def load_data():
    data_file = "data/TADPOLE/ADNIMERGE.csv"
    merge_data = pd.read_csv(data_file, na_values="", converters=CONVERTERS)
    return merge_data



def has_data_mask(frame):
    """
    Check whether rows has any valid value (i.e. not NaN)
    Args:
        frame: Pandas data frame
    Return:
        (ndarray): boolean mask with the same number of rows as *frame*
        True implies row has at least 1 valid value
    """
    return ~frame.isnull().apply(np.all, axis=1)


def get_data_dict(frame, features):
    """
    From a frame of all subjects, return a dictionary of frames
    The keys are subjects' ID
    The data frames are:
        - sorted by *Month_bl* (which are integers)
        - have empty rows dropped (empty row has no value in *features* list)
    Args:
        frame (Pandas data frame): data frame of all subjects
        features (list of string): list of features
    Return:
        (Pandas data frame): prediction frame
    """
    ret = {}
    frame_ = frame.copy()
    for subj in np.unique(frame_.RID):
        subj_data = frame_[frame_.RID == subj].sort_values('VISCODE')
        subj_data = subj_data[has_data_mask(subj_data[features])]

        #subj_data = subj_data.set_index('VISCODE', drop=True)
        ret[subj] = subj_data.drop(['RID'], axis=1)
    return ret


