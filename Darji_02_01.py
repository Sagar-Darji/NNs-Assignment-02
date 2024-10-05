# Kamangar, Farhad
# 1000_123_456
# 2024_10_06
# Assignment_02_01

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

# This function has been converted from Tensorflow to PyTorch by  Islam, SM Mazharul. 2024_09_19
def multi_layer_nn_torch(x_train, y_train, layers, activations, alpha=0.01, batch_size=32, epochs=0, loss_func='mse', val_split=(0.8, 1.0), seed=7321):
    # This function creates and trains a multi-layer neural Network in PyTorch
    # X_train: numpy array of input for training [nof_train_samples,input_dimensions]
    # Y_train: numpy array of desired outputs for training samples [nof_train_samples,output_dimensions]
    # layers: Either a list of integers or alist of numpy weight matrices.
    # If layers is a list of integers then it represents number of nodes in each layer. In this case
    # the weight matrices should be initialized by random numbers.
    # If the layers is given as a list of weight matrices (numpy array), then the given matrices should be used and NO random
    # initialization is needed.
    # activations: list of case-insensitive activations strings corresponding to each layer. The possible activations
    # are, "linear", "sigmoid", "relu".
    # alpha: learning rate
    # epochs: number of epochs for training.
    # loss_func: is a case-insensitive string determining the loss function. The possible inputs are: "svm" , "mse",
    # "ce". Do not use any PyTorch provided methods to compute loss. Implement the equations by yourself.
    # validation_split: a two-element list specifying the normalized start and end point to
    # extract validation set. Use floor in case of non integers.

    # return: This function should return a list containing 3 elements:
        # The first element of the return list should be a list of weight matrices.
        # Each element of the list should be a 2-d numpy array which corresponds to the weight matrix of the
        # corresponding layer (Biases should be included in each weight matrix in the first row).

        # The second element should be a one dimensional list of numbers
        # representing the error after each epoch. You should compute the mean-absolute error between the target and the prediction.
        # Be careful to not mix-up loss-function with error. Each error should
        # be calculated by using the entire validation set while the network is frozen.
        # Frozen means that the weights should not be adjusted while calculating the error.
        # In case of epochs == 0, do not compute error, instead return an empty list.

        # The third element should be a two-dimensional numpy array [nof_validation_samples,output_dimensions]
        # representing the actual output of the network when validation set is used as input.

    # Notes:
    # The data set in this assignment is the transpose of the data set in assignment_01. i.e., each row represents
    # one data sample.
    # The weights in this assignment are the transpose of the weights in assignment_01.
    # Each output weights in this assignment is the transpose of the output weights in assignment_01
    # DO NOT use any other package other than PyTorch and numpy
    # Bias should be included in the weight matrix in the first row.
    # Use steepest descent for adjusting the weights
    # Use minibatch to calculate error and adjusting the weights
    # Do not use any random number seeding. The test case will take care of the random number seeding.
    # Use numpy for weight to initialize weights. Do not use PyTorch weight initialization.
    # Do not use any random method from PyTorch
    # Do not shuffle data
    # i.e., Initialize the weights for each layer by:
    # np.random.seed(seed)
    # np.random.randn()
    # Runtime for all the test cases will be less than 5 seconds

    pass