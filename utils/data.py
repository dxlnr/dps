import pandas as pd

def read_return_data(path='data/auto-mpg.csv'):
    '''reads and returns data as pandas dataframe.'''
    return pd.read_csv(path, sep=',', skipinitialspace=True)
