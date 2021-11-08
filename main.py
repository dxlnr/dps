import numpy as np
from utils.data import read_return_data, split_dataset
from model.build import build_model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main():
    '''main function.'''
    dataset = read_return_data()
    print(dataset)

    # X, y = get_train_data(dataset)
    # print(X)
    # print(y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    train_data, test_data = split_dataset(dataset)
    y_train = train_data.pop('mpg')
    y_test = test_data.pop('mpg')

    input_shape = len(train_data.keys())
    train_data = np.asarray(train_data).astype(np.float32)

    # model = LinearRegression()
    # model.fit(X_train, y_train)

    # def norm(x):
    #     train_stats = train_data.describe()
    #     return (x - train_stats['mean']) / train_stats['std']
    #
    # normed_train_data = norm(train_data)
    # normed_test_data = norm(test_data)

    epochs = 1000
    model = build_model(input_shape)
    model.fit(train_data, y_train, epochs=epochs, validation_split = 0.2)


    test_mpg = [1, 2, 3, 2, -2, -1, -2, -1, 0]
    test_result = model.predict([test_mpg])
    print(test_result)


if __name__ == "__main__":
    main()
