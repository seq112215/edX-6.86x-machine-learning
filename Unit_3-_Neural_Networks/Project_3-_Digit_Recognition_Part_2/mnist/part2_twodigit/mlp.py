import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions

class MLP(nn.Module):

    """ My solution:
    def __init__(self, input_dimension):
        super(MLP, self).__init__()
        self.flatten = Flatten()

        H = 64
        self.linear1 = nn.Linear(input_dimension, H)
        self.linear_out1 = nn.Linear(H, num_classes)
        self.linear_out2 = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        xf = self.flatten(x)  # need to flatten to use linear layers

        xf_out = F.relu(self.linear1(xf))
        out_first_digit = self.linear_out1(xf_out)
        out_second_digit = self.linear_out2(out_first_digit)

        return out_first_digit, out_second_digit
    """

    # Instructor's solution: (uses extra Linear(64, 64) layer then two different
    #   Linear(64, 10) layers, whereas I ran the two digit output one after the other)
    def __init__(self, input_dimension):
        super(MLP, self).__init__()
        self.flatten = Flatten()
        self.linear1 = nn.Linear(input_dimension, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear_first_digit = nn.Linear(64, 10)
        self.linear_second_digit = nn.Linear(64, 10)

    def forward(self, x):
        xf = self.flatten(x)
        out1 = F.relu(self.linear1(xf))
        out2 = F.relu(self.linear2(out1))
        out_first_digit = self.linear_first_digit(out2)
        out_second_digit = self.linear_second_digit(out2)
        return out_first_digit, out_second_digit


def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)
    # print(train_batches[0]['x'].shape, train_batches[0]['y'].shape)
    # batch[i]['x'] is (64, 1, 42, 28) = (batch_size, 1, img_rows, img_cols)
    # batch[i]['y'] is (2, 64) = (num_labels, batch_size)

    # print(len(X_train), len(y_train))  # 36000, 2
    # print(len(X_dev), len(y_dev))  # 4000, 2
    # print(len(train_batches), len(dev_batches), len(test_batches))  # 562, 62, 62

    # Load model
    input_dimension = img_rows * img_cols
    model = MLP(input_dimension)

    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()


"""
Takes less than a second per epoch.

Results:

Epoch 30:
Train | loss1: 0.125550  accuracy1: 0.964858 | loss2: 0.103384  accuracy2: 0.968027
Valid | loss1: 0.286577  accuracy1: 0.918599 | loss2: 0.414554  accuracy2: 0.898942
Test loss1: 0.318301  accuracy1: 0.912046  loss2: 0.379931   accuracy2: 0.901714


Entering into website gives:
    There was a problem running the staff solution (Staff debug: L364)

Many other students have got this error, too.
"""
