from utils.data import read_return_data, prepare_data, drop_category
from model.build import build_model

from sklearn.model_selection import train_test_split

def train(epochs=1000, save=True):
    '''training function.'''
    # read raw dataset.
    dataset = read_return_data()
    # use the adjusted dataset.
    data = prepare_data(dataset)
    # set the category that the model should be trained and inferred on.
    data, labels = drop_category(data, cat="mpg")
    # Split in training and testing.
    train_data, test_data, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=1)

    # Build the model.
    model = build_model(len(train_data.keys()))
    model.fit(train_data, y_train, epochs=epochs, validation_split = 0.2)
    #model.summary()

    #saving the results
    if save:
        model.save('results/model')

    return model
