import matplotlib.pyplot as plt
import numpy as np

from planar_utils import load_planar_dataset, visualize_dataset, plot_decision_boundary
from NeuralNetwork import NeuralNetwork

COUNT_LAYERS = 5
NUM_ITERATIONS = 10000

X, Y = load_planar_dataset()
visualize_dataset(X=X, Y=Y)

neural_network = NeuralNetwork()
neural_network.fit(
    X=X, Y=Y,
    n_h=COUNT_LAYERS,
    num_iterations=NUM_ITERATIONS,
    print_cost=True
)

plot_decision_boundary(lambda x: neural_network.predict(X=x.T), X, Y)
plt.title("Decision Boundary for hidden layer size = " + str(COUNT_LAYERS))
plt.show()

predictions = neural_network.predict(X=X)
print ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y,1 - predictions.T)) / float(Y.size) * 100) + '%')