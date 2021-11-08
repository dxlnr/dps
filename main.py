import numpy as np
from utils.data import read_return_data, get_train_data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main():
    '''main function.'''
    dataset = read_return_data()
    print(dataset)

    X, y = get_train_data(dataset)
    print(X)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(model.coef_)
    print(model.score(X_train,y_train))
    print("\n")
    print(model.score(X_test,y_test))
    print("\n")

    y_pred = model.predict(X_test)
    print(y_pred)


    test_mpg = np.array([1, 2, 3, 2, -2, -1, -2, -1, 0])


if __name__ == "__main__":
    main()
