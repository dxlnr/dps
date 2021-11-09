from model.train import train

from utils.data import *

def main():
    '''main function.'''
    # perform the training step.
    model = train(save=False)

    # testing the model.
    test_mpg = [1, 2, 3, 2, -2, -1, -2, -1, 0]
    test_result = model.predict([test_mpg])

    print("--- resulting mpg: %.1f ---" % round(calc_real_mpg_value(test_result[0][0]), 1))


if __name__ == "__main__":
    main()
