from utils.data import read_return_data, get_train_data
from model.build import build_model

def train(epochs=100):
    model = build_model()
    model.fit(normed_train_data, train_labels, epochs=epochs, validation_split = 0.2)
