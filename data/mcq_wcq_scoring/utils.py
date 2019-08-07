import numpy as np

def get_n_participants(data):
    '''useful function to get number of participants'''
    return np.max(data['id'].values) + 1