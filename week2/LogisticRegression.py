import numpy as np


class LogisticRegression:
    @staticmethod
    def __sigmoid(z):
        s = 1 / (1 + np.exp(-z))
        return s

    def __init__(self, dimension):
        self.w = np.zeros((dimension, 1))
        self.dw = np.zeros((dimension, 1))
        self.b = 0
        self.db = 0

    def __propagate(self, X, y):
        m = X.shape[1]

        A = LogisticRegression.__sigmoid(self.w.T.dot(X) + self.b)
        cost = (-1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

        self.dw = (1 / m) * X.dot((A - y).T)
        self.db = (1 / m) * np.sum(A - y)

        return cost

    def fit(self, X_train, y_train, num_iterations, learning_rate, print_cost=False):
        costs = []
        for i in range(num_iterations):
            cost = self.__propagate(X=X_train, y=y_train)

            self.w = self.w - learning_rate * self.dw
            self.b = self.b - learning_rate * self.db

            if i % 100 == 0:
                costs.append(cost)

            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        params = {"w": self.w, "b": self.b}
        grads = {"dw": self.dw, "db": self.db}

        return params, grads, costs

    def predict(self, X_test):
        m = X_test.shape[1]
        y_pred = np.zeros((1, m))

        A = LogisticRegression.__sigmoid(self.w.T.dot(X_test) + self.b)
        for i in range(A.shape[1]):
            if A[0, i] <= 0.5:
                y_pred[0, i] = 0
            else:
                y_pred[0, i] = 1

        return y_pred
