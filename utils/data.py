import numpy as np
import pandas as pd

def read_return_data(path='data/auto-mpg.csv'):
    '''reads and returns data as pandas dataframe.'''
    return pd.read_csv(path, sep=',', skipinitialspace=True).replace('?', np.nan).dropna()

def split_dataset(data, frac=0.7):
    '''Split pandas dataframe (data) into train & test set.'''
    train_d = dataset.sample(frac=frac, random_state=0)
    test_d = dataset.drop(train_dataset.index)
    return train_d, test_d

def get_train_data(data):
    X = data.drop(['mpg', 'car name'], axis=1)
    y = data['mpg']
    return X, y
