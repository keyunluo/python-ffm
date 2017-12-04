# coding: utf-8

import os
path = os.path.dirname(os.path.abspath(__file__))
libffm_file = [i for i in os.listdir(path) if i.startswith('libffm.')][0]
lib_path = path + '/' + libffm_file

# binding code

import numpy as np
import ctypes
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin

class FFM_Parameter(ctypes.Structure):
    _fields_ = [
        ('eta', ctypes.c_float),
        ('lam', ctypes.c_float),
        ('nr_iters', ctypes.c_int),
        ("k", ctypes.c_int),
        ('normalization', ctypes.c_bool),
        ('auto_stop', ctypes.c_bool), 
    ]

class FFM_Model(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int),
        ("m", ctypes.c_int),
        ("k", ctypes.c_int),
        ("W", ctypes.POINTER(ctypes.c_float)),
        ('normalization', ctypes.c_bool)
    ]

class FFM_Node(ctypes.Structure):
    _fields_ = [
        ("f", ctypes.c_int),
        ("j", ctypes.c_int),
        ("v", ctypes.c_float),
    ]

class FFM_Line(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(FFM_Node)),
        ("label", ctypes.c_float),
        ("size", ctypes.c_int),
    ]

class FFM_Problem(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("num_nodes", ctypes.c_long),

        ("data", ctypes.POINTER(FFM_Node)),
        ("pos", ctypes.POINTER(ctypes.c_long)),
        ("labels", ctypes.POINTER(ctypes.c_float)),
        ("scales", ctypes.POINTER(ctypes.c_float)),

        ("n", ctypes.c_int),
        ("m", ctypes.c_int),
    ]

FFM_Node_ptr = ctypes.POINTER(FFM_Node)
FFM_Line_ptr = ctypes.POINTER(FFM_Line)
FFM_Model_ptr = ctypes.POINTER(FFM_Model)
FFM_Problem_ptr = ctypes.POINTER(FFM_Problem)
FFM_Float_ptr = ctypes.POINTER(ctypes.c_float)

_lib = ctypes.cdll.LoadLibrary(lib_path)

_lib.ffm_convert_data.restype = FFM_Problem_ptr
_lib.ffm_convert_data.argtypes = [FFM_Line_ptr, ctypes.c_int]

_lib.free_ffm_problem.restype = None
_lib.free_ffm_problem.argtypes = [FFM_Problem_ptr]

_lib.free_ffm_float.restype = None
_lib.free_ffm_float.argtypes = [FFM_Float_ptr]

_lib.ffm_init_model.restype = FFM_Model
_lib.ffm_init_model.argtypes = [FFM_Problem_ptr, FFM_Parameter]

_lib.ffm_train_iteration.restype = ctypes.c_float
_lib.ffm_train_iteration.argtypes = [FFM_Problem_ptr, FFM_Model_ptr, FFM_Parameter]

_lib.ffm_predict_array.argtypes = [FFM_Node_ptr, ctypes.c_int, FFM_Model_ptr]
_lib.ffm_predict_array.restype = ctypes.c_float

_lib.ffm_predict_batch.restype = ctypes.POINTER(ctypes.c_float)
_lib.ffm_predict_batch.argtypes = [FFM_Problem_ptr, FFM_Model_ptr]

_lib.ffm_load_model_c_string.restype = FFM_Model
_lib.ffm_load_model_c_string.argtypes = [ctypes.c_char_p]

_lib.ffm_save_model_c_string.argtypes = [FFM_Model_ptr, ctypes.c_char_p]


# some wrapping to make it easier to work with

def wrap_tuples(row):
    size = len(row)
    nodes_array = (FFM_Node * size)()

    for i, (f, j, v) in enumerate(row):
        node = nodes_array[i]
        node.f = f
        node.j = j
        node.v = v

    return nodes_array

def wrap_dataset_init(X, target):
    l = len(target)
    data = (FFM_Line * l)()

    for i, (x, y) in enumerate(zip(X, target)):
        d = data[i]
        nodes = wrap_tuples(x)
        d.data = nodes
        d.label = y
        d.size = nodes._length_

    return data

def wrap_dataset(X, y):
    line_array = wrap_dataset_init(X, y)
    return _lib.ffm_convert_data(line_array, line_array._length_)

# Print Trainning Process
def print_line(data=None, val=True):
    if val:
        if data is None:
            print('%-8s%-16s%-16s%-16s%-8s' %("Iter", "Train_Loss", "Train_Score", "Val_Score", "Best_Iter"))
        else:
            print('%-8d%-16.4f%-16.4f%-16.4f%-8d' %(data[0], data[1], data[2], data[3], data[4]))
    else:
        if data is None:
            print('%-8s%-16s%-16s%-16s%-8s' %("Iter", "Train_Loss", "Train_Score", "Best_Iter"))
        else:
            print('%-8d%-16.4f%-16.4f%-8d' %(data[0], data[1], data[2], data[3]))

class FFMData():
    def __init__(self, X=None, y=None):
        self.data = X
        self.labels = y
        if X is not None and y is not None:
            self._data = wrap_dataset(X, y)
        elif X is not None and y is  None:
            self._data = wrap_dataset(X, [0]*len(X))
        else:
            self._data = None
        self.size = len(X)

    def num_rows(self):
        return self.size

    def __del__(self):
        if self._data is not None:
            _lib.free_ffm_problem(self._data)


class Prediction:
    def __init__(self, pred_ptr=None, data=None):
        if pred_ptr is not None:
            self._pred_ptr = pred_ptr
            size = data.size
            pred_ptr_address = ctypes.addressof(pred_ptr.contents)
            array_cast = (ctypes.c_float * size).from_address(pred_ptr_address)
            self.pred = np.ctypeslib.as_array(array_cast)

    def __del__(self):
        if self._pred_ptr is not None:
            _lib.free_ffm_float(self._pred_ptr)

# FFM model

class FFM(BaseEstimator, ClassifierMixin):
    def __init__(self, eta=0.2, lam=0.00002, k=4, normalization=True, num_iter=None, early_stopping=None, metric=None):
        self._params = FFM_Parameter(eta=eta, lam=lam, k=k, normalization=normalization)
        self.set_params(eta=eta, lam=lam, k=k, normalization=normalization, num_iter=num_iter, early_stopping=early_stopping, metric=metric)
        self._model = None

    def read_model(self, path):
        path_char = ctypes.c_char_p(path.encode())
        model = _lib.ffm_load_model_c_string(path_char)
        self._model = model
        return self

    def save_model(self, path):
        model = self._model
        path_char = ctypes.c_char_p(path.encode())
        _lib.ffm_save_model_c_string(model, path_char)

    def init_model(self, ffm_data):
        params = self._params
        model = _lib.ffm_init_model(ffm_data._data, params)
        self._model = model
        return self

    def iteration(self, ffm_data):
        data = ffm_data._data
        model = self._model
        params = self._params
        loss = _lib.ffm_train_iteration(data, model, params)
        return loss

    def predict(self, ffm_data):
        proba = self.predict_proba(ffm_data)
        y_pred = [round(i) for i in proba]
        return y_pred

    def predict_proba(self, ffm_data):
        data = ffm_data._data
        model = self._model
        pred_ptr = _lib.ffm_predict_batch(data, model)

        prediction = Prediction(pred_ptr, data.contents)
        proba = np.copy(prediction.pred)

        return proba

    def _predict_row(self, nodes):
        n = nodes._length_
        model = self._model
        pred = _lib.ffm_predict_array(nodes, n, model)
        return pred

    def fit(self, X, y=None, num_iter=10, val_data=None, metric='logloss', early_stopping=5, maximum=False):
        '''
        X: feature data or FFMData format
        y: target
        num_iter: number of iterations
        val_data: data for validation, list or FFMData format
        metric: str or self defined function, build_in metrics are 'auc', 'logloss', 'f1', 'accuracy'
        early_stopping: early stopping rounds
        maximum: whether the biger the score, the better the metric
        '''
        # Translate Traing Data
        if isinstance(X, FFMData):
            train_data = X
        else:
            train_data = FFMData(X, y)

        # Init Model
        self.init_model(train_data) 
        self.set_params(num_iter=num_iter, early_stopping=early_stopping, metric=metric)
        
        # Translate Validation Data
        val = True if val_data is not None else False
        if val:
            if not isinstance(val_data, FFMData):
                val_data = FFMData(val_data[0], val_data[1])
        
        # Print Header
        print_line(data=None, val=val) 
        
        # Score Recorder: > or <
        best_model = None
        score_index = -1
        if maximum:
            cmp = lambda x, y: x > y
            score = -np.inf
        else:
            cmp = lambda x, y: x < y
            score = np.inf
        
        # Trainning Process
        for i in range(num_iter):
            self.iteration(train_data)
            train_loss = self.score(train_data, train_data.labels, scoring='logloss')
            train_score = self.score(train_data, train_data.labels, scoring=metric)
       
            if val:
                val_score = self.score(val_data, val_data.labels, scoring=metric)
                if cmp(val_score, score):
                    score = val_score
                    score_index = i
                    best_model = self._model                   
                print_line(data=[i, train_loss, train_score, val_score, score_index], val=val) 
            else:
                if cmp(val_score, score):
                    score = val_score
                    score_index = i
                    best_model = self._model
                print_line(data=[i, train_loss, train_score, score_index], val=val)
                    
            if (i - score_index) >= early_stopping:
                print("Early Stoping At %d Rounds" % score_index)
                break
            
            self._model = best_model
        return self
        
    def score(self, X, y=None, scoring='logloss'):
        if self._model is None:
            raise ValueError("``score`` must be call after fit" )
        if isinstance(X, FFMData):
            val_data = X
        else:
            val_data = FFMData(X, y)
        y_pred = self.predict_proba(val_data)
        y_true = val_data.labels
        if isinstance(scoring, str):           
            if scoring == 'logloss':
                return log_loss(y_true, y_pred)
            elif scoring == 'auc':
                return roc_auc_score(y_true, y_pred)
            elif scoring == 'f1':
                y_pred = [round(i) for i in y_pred]
                return f1_score(y_true, y_pred)
            else: 
                y_pred = [round(i) for i in y_pred]
                return accuracy_score(y_true, y_pred)
        else :
            return scoring(y_true, y_pred)

    
def read_model(path):
    return FFM().read_model(path)

