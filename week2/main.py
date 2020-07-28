import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

from lr_utils import load_dataset
from LogisticRegression import LogisticRegression

# Train-test sets parameters
datasets_path = 'datasets/'
train_dataset_name = 'train_catvnoncat.h5'
test_dataset_name = 'test_catvnoncat.h5'

# Logistic regression parameters
num_iterations = 1500
learning_rate = 0.01
print_cost = True

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset(
    train_dataset_path=(datasets_path + train_dataset_name),
    test_dataset_path=(datasets_path + test_dataset_name)
)


# Prepare data and print data statistics
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("\n")

train_set_x_flatten = train_set_x_orig.reshape(m_train, (num_px * num_px * 3)).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, (num_px * num_px * 3)).T

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("\n")

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# Fit model
logistic_regression = LogisticRegression(train_set_x.shape[0])
logistic_regression.fit(
    X_train=train_set_x,
    y_train=train_set_y,
    num_iterations=num_iterations,
    learning_rate=learning_rate,
    print_cost=print_cost
)

# Predict test-train set examples
Y_prediction_test = logistic_regression.predict(X_test=test_set_x)
Y_prediction_train = logistic_regression.predict(X_test=train_set_x)

print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))
print("\n")

my_images_path = 'my_images/'
my_images_names = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']

for image_name in my_images_names:
    file_name = my_images_path + image_name
    image = np.array(ndimage.imread(file_name, flatten=False))
    image = image / 255.
    resized_image = scipy.misc.imresize(
        image,
        size=(num_px, num_px)).reshape((1, num_px * num_px * 3)
    ).T
    image_prediction = logistic_regression.predict(resized_image)
    print(
        "y =", np.squeeze(image_prediction),
        "your algorithm predicts a \"" + classes[int(np.squeeze(image_prediction)),].decode("utf-8") +  "\" picture."
    )
