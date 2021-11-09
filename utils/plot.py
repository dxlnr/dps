import matplotlib.pyplot as plt
import seaborn as sb

from utils.data import read_return_data, adjust_horsepower

def inspect_data():
    '''get general overview of data.'''
    data = read_return_data()
    data = adjust_horsepower(data)
    sb.pairplot(data[['mpg', 'cylinders', 'displacement', 'weight']], diag_kind='kde')
    plt.show()

def plot_loss(model):
    '''plot training and validation loss.'''
    plt.plot(model.history.history['loss'], label='loss')
    plt.plot(model.history.history['val_loss'], label='val_loss')
    plt.ylim([0, 0.4])
    plt.xlabel('epoch')
    plt.ylabel('error [mpg]')
    plt.legend()
    plt.grid(True)
    plt.show()
