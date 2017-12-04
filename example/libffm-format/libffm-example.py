# coding: utf-8

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from ffm import FFM, FFMData, load_libffm

data_dir = './'

train_X, train_y = load_libffm(data_dir + 'bigdata.tr.txt')
test_X, test_y = load_libffm(data_dir + 'bigdata.te.txt')

train_data = FFMData(train_X, train_y)
test_data = FFMData(test_X, test_y)

model = FFM(eta=0.1, lam=0.0002, k=4)
model.fit(train_data, num_iter=50, val_data=test_data, metric='logloss', early_stopping=5, maximum=False) 

acc = model.score(test_data, scoring='acc')
print("Accuracy Score: ", acc)
f1 = model.score(test_data, scoring='f1')
print("F1 Score: ", f1)