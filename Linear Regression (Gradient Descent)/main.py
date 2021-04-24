import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


class MSE:
    """
    Basic MSELoss implementation for linear regression
    """

    @staticmethod
    def mse(y, y_pred):
        return ((y - y_pred) ** 2).mean()

    @staticmethod
    def grad(X, y, y_pred, l):
        d_m = (-2 / l) * sum(X * (y - y_pred))
        d_c = (-2 / l) * sum(y - y_pred)

        return d_m, d_c


def train(X, y, epochs=1000, lr=0.0001):
    """
    trains linear regression model thru gradient descent instead of least squares
    """
    m, c = 0, 0
    l = float(len(X))

    for i in range(epochs):
        y_pred = m * X + c
        d_m, d_c = MSE.grad(X, y, y_pred, l)
        m = m - (lr * d_m)
        c = c - (lr * d_c)
        print(f'Epoch: {i + 1} - MSE: {MSE.mse(y, y_pred)} - m: {m} c: {c}')

    return X, y_pred, m, c


if __name__ == '__main__':
    df = pd.read_csv('./data/test.csv')
    # df = df[df['Gender'] == 'Male']
    X, y = df.iloc[:, 0], df.iloc[:, 1]

    X, y_pred, m, c = train(X, y)

    plt.scatter(X, y)
    plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='red')
    plt.savefig('linear_regression_grad.png')
    plt.show()


