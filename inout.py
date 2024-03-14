import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import idx2numpy
import numpy as np
from six.moves import cPickle
import platform
import cv2
sns.set(color_codes=True)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)          # suppress messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_mnist():
    X_train = idx2numpy.convert_from_file('MNIST_data/train-images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('MNIST_data/train-labels.idx1-ubyte')
    X_test = idx2numpy.convert_from_file('MNIST_data/t10k-images.idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('MNIST_data/t10k-labels.idx1-ubyte')

    train_images = []                                                   # reshape train images so that the training set
    for i in range(X_train.shape[0]):                                   # is of shape (60000, 1, 28, 28)
        train_images.append(np.expand_dims(X_train[i], axis=0))
    train_images = np.array(train_images)
    train_images = train_images[5000:10000]
    print(train_images[0].shape)
    print(train_images.shape)
    test_images = []                                                    # reshape test images so that the test set
    for i in range(X_test.shape[0]):                                    # is of shape (10000, 1, 28, 28)
        test_images.append(np.expand_dims(X_test[i], axis=0))
    test_images = np.array(test_images)
    test_images = test_images[1000:2000]
    indices = np.random.permutation(train_images.shape[0])              # permute and split training data in
    training_idx, validation_idx = indices[:4500], indices[4500:]     # training and validation sets
    train_images, validation_images = train_images[training_idx, :], train_images[validation_idx, :]
    train_labels, validation_labels = train_labels[training_idx], train_labels[validation_idx]

    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'validation_images': validation_images,
        'validation_labels': validation_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }





def minmax_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


def preprocess(dataset):
    dataset['train_images'] = np.array([minmax_normalize(x) for x in dataset['train_images']])
    dataset['validation_images'] = np.array([minmax_normalize(x) for x in dataset['validation_images']])
    dataset['test_images'] = np.array([minmax_normalize(x) for x in dataset['test_images']])
    return dataset


def plot_accuracy_curve(accuracy_history, val_accuracy_history):
    plt.plot(accuracy_history, 'b', linewidth=3.0, label='Training accuracy')
    plt.plot(val_accuracy_history, 'r', linewidth=3.0, label='Validation accuracy')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Accuracy rate', fontsize=16)
    plt.legend()
    plt.title('Training Accuracy', fontsize=16)
    plt.savefig('training_accuracy.png')
    plt.show()


def plot_learning_curve(loss_history):
    plt.plot(loss_history, 'b', linewidth=3.0, label='Cross entropy')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.title('Learning Curve', fontsize=16)
    plt.savefig('learning_curve.png')
    plt.show()


def plot_sample(image, true_label, predicted_label):
    plt.imshow(image)
    if true_label and predicted_label is not None:
        if type(true_label) == 'int':
            plt.title('True label: %d, Predicted Label: %d' % (true_label, predicted_label))
        else:
            plt.title('True label: %s, Predicted Label: %s' % (true_label, predicted_label))
    plt.show()


def plot_histogram(layer_name, layer_weights):
    plt.hist(layer_weights)
    plt.title('Histogram of ' + str(layer_name))
    plt.xlabel('Value')
    plt.ylabel('Number')
    plt.show()


def to_gray(image_name):
    image = cv2.imread(image_name + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray image', image)
    cv2.imwrite(image_name + '.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
