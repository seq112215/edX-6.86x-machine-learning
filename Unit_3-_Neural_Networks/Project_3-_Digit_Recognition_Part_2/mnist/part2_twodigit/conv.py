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
nb_epoch = 15
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions


class CNN(nn.Module):

    """ My solution:
    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        # print(input_dimension)  # 1176
        self.conv1 = nn.Conv2d(1, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.maxpool = nn.MaxPool2d(((2, 2)))
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(2880, 128)
        self.dropout = nn.Dropout(0.5)
        self.linear_out1 = nn.Linear(128, num_classes)
        self.linear_out2 = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        # Use same structure as part2_mnist/nnet_cnn.py
        x1 = self.conv1(x)
        x2 = self.relu(x1)
        x3 = self.maxpool(x2)
        x4 = self.conv2(x3)
        x5 = self.relu(x4)
        x6 = self.maxpool(x5)
        x7 = self.flatten(x6)
        # Flatten input size: torch.Size([64, 64, 9, 5])
        # Flatten output size: torch.Size([64, 2880])
        x8 = self.linear(x7)
        x9 = self.dropout(x8)

        out_first_digit = self.linear_out1(x9)
        out_second_digit = self.linear_out2(out_first_digit)
        """
        """
        Model's state_dict:
        conv1.weight     torch.Size([32, 1, 3, 3])
        conv1.bias       torch.Size([32])
        conv2.weight     torch.Size([64, 32, 3, 3])
        conv2.bias       torch.Size([64])
        linear.weight    torch.Size([128, 2880])
        linear.bias      torch.Size([128])
        linear_out1.weight       torch.Size([10, 128])
        linear_out1.bias         torch.Size([10])
        linear_out2.weight       torch.Size([10, 10])
        linear_out2.bias         torch.Size([10])

        Optimizer's state_dict:
        state    {}
        param_groups     [{'lr': 0.01, 'momentum': 0.9, 'dampening': 0,
        'weight_decay': 0, 'nesterov': False,
        'params': [139881607984944, 139881607985024, 139881607985264,
                   139881607985424, 139881607985824, 139881607985904,
                   139881607986144, 139881607986224, 139881607986464, 139881607986544]}]
        """
        """
        return out_first_digit, out_second_digit
        """

        # Instructor's solution: (WAY more compact, same network)
    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        self.linear1 = nn.Linear(input_dimension, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear_first_digit = nn.Linear(64, 10)
        self.linear_second_digit = nn.Linear(64, 10)

        self.encoder = nn.Sequential(
              nn.Conv2d(1, 8, (3, 3)),
              nn.ReLU(),
              nn.MaxPool2d((2, 2)),
              nn.Conv2d(8, 16, (3, 3)),
              nn.ReLU(),
              nn.MaxPool2d((2, 2)),
              Flatten(),
              nn.Linear(720, 128),
              nn.Dropout(0.5),
        )

        self.first_digit_classifier = nn.Linear(128,10)
        self.second_digit_classifier = nn.Linear(128,10)

    def forward(self, x):
        out = self.encoder(x)
        out_first_digit = self.first_digit_classifier(out)
        out_second_digit = self.second_digit_classifier(out)
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

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension)

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
Takes about 30 seconds per epoch. After about 15 epochs, losses bounce back and forth.

Results:

Epoch 30:
Train | loss1: 0.025851  accuracy1: 0.990408 | loss2: 0.038036  accuracy2: 0.986794
Valid | loss1: 0.104929  accuracy1: 0.978579 | loss2: 0.092213  accuracy2: 0.975806
Test loss1: 0.114066  accuracy1: 0.973286  loss2: 0.133247   accuracy2: 0.968246
"""

"""
With only one set of Conv2d, ReLU, MaxPool, get
Train | loss1: 0.050981  accuracy1: 0.981845 | loss2: 0.070573  accuracy2: 0.976201
Valid | loss1: 0.133993  accuracy1: 0.961442 | loss2: 0.135496  accuracy2: 0.961694
Test loss1: 0.173759  accuracy1: 0.959929  loss2: 0.218468   accuracy2: 0.940776
"""
