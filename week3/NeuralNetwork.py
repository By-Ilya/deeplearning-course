import numpy as np

from planar_utils import sigmoid

class NeuralNetwork:
    def __init__(self):
        self.m = 0
        self.n_x = 0
        self.n_h = 0
        self.n_y = 0
        self.cost = 0
    
    def __set_layer_sizes(self, X, Y, n_h):
        self.n_x = len(X)
        self.n_h = n_h
        self.n_y = len(Y)
        self.m = Y.shape[1]
    
    def __initialize_parameters(self):
        np.random.seed(2)
        self.W1 = np.random.randn(self.n_h, self.n_x) * 0.01
        self.b1 = np.zeros((self.n_h, 1))
        self.W2 = np.random.randn(self.n_y, self.n_h) * 0.01
        self.b2 = np.zeros((self.n_y, 1))

    def __forward_propagation(self, X):
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = sigmoid(self.Z2)
    
    def __compute_cost(self, Y):
        logprobs = np.multiply(np.log(self.A2), Y) + np.multiply((1 - Y), np.log(1 - self.A2))
        self.cost = float(np.squeeze(-1 / self.m * np.sum(logprobs)))

    def __backward_propagation(self, X, Y):
        self.dZ2 = self.A2 - Y
        self.dW2 = 1 / self.m * self.dZ2.dot(self.A1.T)
        self.db2 = 1 / self.m * np.sum(self.dZ2, axis=1, keepdims=True)
        self.dZ1 = self.W2.T.dot(self.dZ2) * (1 - np.power(self.A1, 2))
        self.dW1 = 1 / self.m * self.dZ1.dot(X.T)
        self.db1 = 1 / self.m * np.sum(self.dZ1, axis=1, keepdims=True)
    
    def __update_parameters(self, learning_rate = 1.2):
        self.W1 = self.W1 - learning_rate * self.dW1
        self.b1 = self.b1 - learning_rate * self.db1
        self.W2 = self.W2 - learning_rate * self.dW2
        self.b2 = self.b2 - learning_rate * self.db2

    def fit(self, X, Y, n_h, num_iterations=10000, learning_rate=1.2, print_cost=False):
        np.random.seed(3)
        self.__set_layer_sizes(X=X, Y=Y, n_h=n_h)
        self.__initialize_parameters()

        for i in range(0, num_iterations):
            self.__forward_propagation(X=X)
            self.__compute_cost(Y=Y)
            self.__backward_propagation(X=X, Y=Y)
            self.__update_parameters(learning_rate=learning_rate)

            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, self.cost))
    
    def predict(self, X):
        self.__forward_propagation(X=X)
        return (self.A2 > 0.5)

