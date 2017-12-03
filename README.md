## Python wrapper for libffm

This is a python wrapped for LibFFM library writen in C++, orignal implementation is made by [Yuchin Juan](https://github.com/guestwalk/libffm) and Python Wrapped by [JRCondeNast](https://github.com/JRCondeNast/libffm-python).

### Features

- [√] Scikit-Learn Compatible API
- [√] Early_Stopping Support
- [√] Evaluation Metric Support
- [x] Pandas/Numpy Data Converter



### Installation 

Installing it(Tested in Ubuntu16.04 and Windows10, both Python2 and Python3):

``` shell
git clone --recursive https://github.com/keyunluo/python-ffm
python3 setup.py install
```
### Usage

``` python

import ffm
    
# prepare the data
# (field, index, value) format
    
X = [[(1, 2, 1), (2, 3, 1), (3, 5, 1)],
    [(1, 0, 1), (2, 3, 1), (3, 7, 1)],
    [(1, 1, 1), (2, 3, 1), (3, 7, 1), (3, 9, 1)],
    [(1, 0, 1), (2, 3, 1), (3, 5, 1)], ]
    
y = [1, 1, 0, 1]
    
ffm_data = ffm.FFMData(X, y)
ffm_data_test = ffm.FFMData(X,y)
    
# train the model for 10 iterations 
model = ffm.FFM(eta=0.1, lam=0.0001, k=4)
model.fit(ffm_data,num_iter=10, val_data=ffm_data_test, scoring='auc', early_stopping=6, maximum=True) 

# pred proba
print(model.predict_proba(ffm_data_test))
    
# save the model
model.save_model('ololo.bin')
    
# load it to reuse the model
model = ffm.read_model('ololo.bin')

# pred label
print(model.predict(ffm_data_test))
```