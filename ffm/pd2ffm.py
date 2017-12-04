# coding: utf-8

import numpy as np 
import pandas as pd 
from .ffm import FFMData
from tqdm import tqdm

class FFMFormatPandas:
    def __init__(self):
        self.field_index = {}
        self.feature_index = {}
        self.numerical = None
        self.categorical = None
        self.target = None

    def fit(self, df, target=None, categorical=None, numerical=None):
        '''
        df: Pandas DataFrame
        target: label column, str
        categorical: categorical columns, list
        numerical: numerical columns, list
        '''
        self.target = target
        self.categorical = categorical
        self.numerical = numerical

        feature_code = 0
        for field_index, col in enumerate(categorical):
            self.field_index[col] = field_index
            vals = df[col].unique()
            for val in vals:
                if pd.isnull(val):
                    continue
                name = '{}_{}'.format(col, val)
                if self.feature_index.get(name, -1) == -1:
                    self.feature_index[name] = feature_code
                    feature_code += 1

        for field_index, col in enumerate(numerical, start=len(categorical)):
            self.field_index[col] = field_index
            self.feature_index[col] = feature_code
            feature_code += 1

        return self

    def fit_transform(self, df, target=None):
        self.fit(df, target)
        return self.transform(df)

    def transform(self, df):
        X, y = [], []
        target = self.target
        for _, row in tqdm(df.iterrows(), total=len(df)):
            feature_tuple = []
            for cat in self.categorical:
                if pd.notnull(row[cat]):
                    feature = '{}_{}'.format(cat, row[cat])
                    feature_tuple.append((self.field_index[cat], self.feature_index[feature], 1))
            for num in self.numerical:
                if pd.notnull(row[num]):
                    feature_tuple.append((self.field_index[num], self.feature_index[num], row[num]))
            
            X.append(feature_tuple)
            y.append(row[self.target])
        return X, y      
        
    def transform_convert(self, df):
        X, y = self.transform(df)
        return FFMData(X, y)   
