# coding: utf-8

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from ffm import FFM, FFMFormatPandas

# define metric
def Gini(y_true, y_pred):
    # check and get number of samples
    assert len(y_true) == len(y_pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred * 1. / G_true

# make dataset
train, y = make_classification(n_samples=2000, n_features=5, n_informative=2, n_redundant=2, n_classes=2, random_state=42)

train = pd.DataFrame(train, columns=['int1','int2','float1','s1','s2'])
train['int1'] = train['int1'].map(int) + np.random.randint(0, 8)
train['int2'] = train['int2'].map(int)
train['s1'] = round(np.log(abs(train['s1'] +1 ))).map(str)
train['s2'] = round(np.log(abs(train['s2'] +1 ))).map(str)
train['clicked'] = y

# transform data
categorical=['int1', 'int2', 's1', 's2']
numerical = ['float1']
target = 'clicked'

train_data, val_data = train_test_split(train, test_size=0.2)

ffm_train = FFMFormatPandas()
ffm_train.fit(train, target=target, categorical=categorical, numerical=numerical)
train_data = ffm_train.transform(train_data)
val_data = ffm_train.transform(val_data)

# make model for train
model = FFM(eta=0.1, lam=0.0001, k=4)
model.fit(train_data, num_iter=32, val_data=val_data, metric=Gini, early_stopping=5, maximum=True) 

# predict 
val_proba = model.predict_proba(val_data)