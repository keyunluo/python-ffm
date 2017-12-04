# coding: utf-8

try:
    import cPickle as pickle
except ImportError:
    import pickle


def save_data(data, path=None):
    '''
    save data to pickle format
    '''
    if path is None:
        path = './data.pkl'
    pickle.dump(data, open(path, "wb"), protocol=4)
    print("Save Data to %s Successfully" % path)

def load_data(path):
    '''
    load data from pickle to FFMData
    '''
    X, y = pickle.load(open(path, 'rb'))
    return X, y