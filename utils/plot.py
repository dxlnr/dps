import matplotlib.pyplot as plt
import seaborn as sb

from utils.data import read_return_data, adjust_horsepower

def inspect_data():
    data = read_return_data()
    data = adjust_horsepower(data)
    sb.pairplot(data[['mpg', 'cylinders', 'displacement', 'weight']], diag_kind='kde')
    plt.show()
