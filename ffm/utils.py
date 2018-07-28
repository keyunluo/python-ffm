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
    try:
        pickle.dump(data, open(path, "wb"), protocol=4)
    except:
        pickle.dump(data, open(path, "wb"), protocol=2)
    print("Save Data to %s Successfully" % path)

def load_data(path):
    '''
    load data from pickle to FFMData
    '''
    X, y = pickle.load(open(path, 'rb'))
    return X, y

def save_libffm(X, y, path=None):
    '''
    save data to original libffm format
    '''
    if path is None:
        path = './data.libffm'
    convert_func = lambda t: '{}:{}:{}'.format(t[0], t[1], t[2])
    with open(path, 'wt') as f:
        for label, data in zip(y, X):
            line = "%d " % label + " ".join([convert_func(t) for t in data]) + '\n'
            f.write(line)
    print("Save Libffm Data to %s Done." % path)

def load_libffm(path):
    '''
    load original libffm data format
    '''
    X, y = [], []
    with open(path, 'rt') as f:
        for line in f:
            line = str.strip(line)
            line = line.split(' ')
            if len(line) < 2:
                continue
            y.append(int(line[0]))
            l = []
            for t in line[1:]:
                t = t.split(':')
                tp = tuple([int(t[0]), int(t[1]), float(t[2])])
                l.append(tp)
            X.append(l)
    return X, y
