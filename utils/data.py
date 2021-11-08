import numpy as np
import pandas as pd

def read_return_data(path='data/auto-mpg.csv'):
    '''reads and returns data as pandas dataframe.'''
    data = pd.read_csv(path, sep=',', skipinitialspace=True).replace('?', np.nan).dropna()
    data['origin'] = data['origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    return data

def prepare_data(data):
    '''Prepares the dataset further.'''
    data = data.drop(['car name'], axis=1)
    return pd.get_dummies(data, columns=['origin'])

def split_dataset(data, frac=0.7):
    '''Split pandas dataframe (data) into train & test set.'''
    data = prepare_data(data)
    
    train_d = data.sample(frac=frac, random_state=0)
    test_d = data.drop(train_d.index)
    return train_d, test_d

def get_train_data(data):
    X = data.drop(['mpg', 'car name'], axis=1)
    y = data['mpg']
    return X, y
