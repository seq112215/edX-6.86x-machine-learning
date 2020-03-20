#! /usr/bin/env python

import _pickle as c_pickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model, Flatten

def main():
    # Load the dataset
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # We need to rehape the data back into a 1x28x28 image
    X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))

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

    # print(X_train[0].shape)  # 1x28x28

    #################################
    ## Model specification TODO
    model = nn.Sequential(
        nn.Conv2d(1, 32, (3, 3)),   # 0.
        nn.ReLU(),                  # 1.
        nn.MaxPool2d((2, 2)),       # 2.
        nn.Conv2d(32, 64, (3, 3)),  # 3.
        nn.ReLU(),                  # 4.
        nn.MaxPool2d((2, 2)),       # 5.
        Flatten(),                  # 6.    In: torch.Size([32, 64, 5, 5])  Out: torch.Size([32, 1600])
        nn.Linear(1600, 128),       # 7.
        nn.Dropout(0.5),            # 8.
        nn.Linear(128, 10),         # 9.
    )

    # Model's state_dict:
    # 0.weight  torch.Size([32, 1, 3, 3])
    # 0.bias    torch.Size([32])
    # 3.weight  torch.Size([64, 32, 3, 3])
    # 3.bias    torch.Size([64])
    # 7.weight  torch.Size([128, 1600])
    # 7.bias    torch.Size([128])
    # 9.weight  torch.Size([10, 128])
    # 9.bias    torch.Size([10])

    # Optimizer's state_dict:
    # state    {}
    # param_groups     [{'lr': 0.01, 'momentum': 0.9, 'dampening': 0,
    # 'weight_decay': 0, 'nesterov': True,
    # 'params': [140710194828704, 140710194828784, 140710194829184, 140710194829264,
    #            140710194829824, 140710194829904, 140710194830144, 140710194830224]}]

    ##################################

    train_model(train_batches, dev_batches, model, nesterov=True)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    print("Loss on test set:" + str(loss) + " Accuracy on test set: " + str(accuracy))


if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)
    main()


# Train loss: 0.020535 | Train accuracy: 0.993424
# Val loss:   0.035862 | Val accuracy:   0.990642
# Loss on test set:0.027618716882157147 Accuracy on test set: 0.9908854166666666
