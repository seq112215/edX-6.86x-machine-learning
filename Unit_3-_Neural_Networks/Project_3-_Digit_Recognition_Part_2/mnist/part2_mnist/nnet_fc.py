#! /usr/bin/env python

import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model
# import _pickle as cPickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")


def main():
    # Load the dataset
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # Split dataset into batches
    batch_size = 32
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #################################
    ## Model specification TODO
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = batch_size, 784, 128, 10

    model = nn.Sequential(
              nn.Linear(D_in, H),
              nn.ReLU(),
              nn.Linear(H, D_out)
            )
    lr=0.1
    momentum=0
    leaky_relu_active = False
    ##################################

    train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    print("Batch size: {}; Learning Rate: {}; Momentum: {}; LeakyReLU: {}; Hidden Dimension: {}".
          format(batch_size, lr, momentum, leaky_relu_active, H))
    print("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))


if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()


"""
RESULTS:

Part 1:

Batch size: 32; Learning Rate: 0.1; Momentum: 0; LeakyReLU: False; Hidden Dimension: 10
Loss on test set:0.26722692017136024 Accuracy on test set: 0.9204727564102564

Batch size: 64; Learning Rate: 0.1; Momentum: 0; LeakyReLU: False; Hidden Dimension: 10
Loss on test set:0.2457950113950154 Accuracy on test set: 0.9298878205128205

Batch size: 32; Learning Rate: 0.01; Momentum: 0; LeakyReLU: False; Hidden Dimension: 10
Loss on test set:0.2788655046158685 Accuracy on test set: 0.9206730769230769

Batch size: 32; Learning Rate: 0.1; Momentum: 0.9; LeakyReLU: False; Hidden Dimension: 10
Loss on test set:0.4557993769188985 Accuracy on test set: 0.8873197115384616

Batch size: 32; Learning Rate: 0.1; Momentum: 0; LeakyReLU: True; Hidden Dimension: 10
Loss on test set:0.26892605330645797 Accuracy on test set: 0.9207732371794872

Part 2:

Batch size: 32; Learning Rate: 0.1; Momentum: 0; LeakyReLU: False; Hidden Dimension: 128
Loss on test set:0.075017019293857 Accuracy on test set: 0.9769631410256411

Batch size: 64; Learning Rate: 0.1; Momentum: 0; LeakyReLU: False; Hidden Dimension: 128
Loss on test set:0.0834942466156999 Accuracy on test set: 0.9748597756410257

Batch size: 32; Learning Rate: 0.01; Momentum: 0; LeakyReLU: False; Hidden Dimension: 128
Loss on test set:0.19775939949800092 Accuracy on test set: 0.9427083333333334

Batch size: 32; Learning Rate: 0.1; Momentum: 0.9; LeakyReLU: False; Hidden Dimension: 128
Loss on test set:0.2584091846490712 Accuracy on test set: 0.9548277243589743

Batch size: 32; Learning Rate: 0.1; Momentum: 0; LeakyReLU: True; Hidden Dimension: 128
Loss on test set:0.07390797881194325 Accuracy on test set: 0.9775641025641025
"""
