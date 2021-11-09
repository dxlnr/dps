import numpy as np
import pandas as pd

def read_return_data(path='data/auto-mpg.csv'):
    '''reads and returns data as pandas dataframe.'''
    return pd.read_csv(path, sep=',', skipinitialspace=True).replace('?', np.nan).dropna()

def prepare_data(data):
    '''Prepares the dataset for training.'''
    # drop categories that are unecessary.
    data = data.drop(['car name'], axis=1)
    # one-hot encodes the 'origin' category.
    data['origin'] = data['origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    data =  pd.get_dummies(data, columns=['origin'])
    # prepare 'horsepower' for training.
    data = adjust_horsepower(data)
    # normalize the dataset & return.
    return normalize(data)

def adjust_horsepower(data):
    data['horsepower'] = data['horsepower'].fillna(data['horsepower'].median())
    data['horsepower'] = data['horsepower'].astype('float64')
    return data

def normalize(data):
    metrics = data.describe().transpose()[['mean', 'std']]
    return (data - metrics['mean']) / metrics['std']

def drop_category(data, cat='mpg'):
    '''drop category that should be learned.'''
    train_data = data.drop([str(cat)], axis=1)
    labels = data[str(cat)]
    return train_data, labels

def calc_real_mpg_value(normalize_res):
    ''' transforms the normalized result into human interpretable number.'''
    #         mean         std
    # mpg     23.445918    7.805007
    return (normalize_res * 7.805007) + 23.445918
