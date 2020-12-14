import argparse

import numpy as np
import tensorflow
from tensorflow.keras import datasets
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, SpatialDropout2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD


def _fix_seed(SEED=42):
    tensorflow.random.set_seed(SEED)
    np.random.seed(SEED)


def _get_datasets():
    fashion_mnist = datasets.fashion_mnist
    return fashion_mnist.load_data()


def _get_optimizer(args):
    if args.optimizer == 'sgd':
        return SGD(learning_rate=args.lr, momentum=args.momentum)
    else:
        return Adam(learning_rate=args.lr)


def train(args):
    """model compile and learning"""
    (X_train, y_train), (X_test, y_test) = _get_datasets()
    model = get_model(args)

    optimizer = _get_optimizer(args)
    X_train, X_test = X_train[:, :, :, None], X_test[:, :, :, None]
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1)
    model.evaluate(X_test, y_test, batch_size=args.test_batch_size, verbose=1)

    return model


def get_model(args):
    """
    Create a simple model
    """
    hidden_channels = args.hidden_channels
    kernel_size = args.kernel_size
    dropout = args.dropout

    model = Sequential([
        Conv2D(hidden_channels, kernel_size=kernel_size, activation="relu", input_shape=(28, 28, 1)),
        MaxPool2D(2),
        Conv2D(20, kernel_size=kernel_size, activation="relu"),
        SpatialDropout2D(dropout),
        MaxPool2D(2),
        Flatten(),
        Dense(50, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])
    return model


if __name__ == "__main__":
    _fix_seed()
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', type=str, default="sgd",
                        help='optimizer for training.')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='DROP',
                        help='dropout rate (default: 0.5)')
    parser.add_argument('--kernel_size', type=int, default=5, metavar='KERNEL',
                        help='conv2d filter kernel size (default: 5)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--hidden_channels', type=int, default=10,
                        help='number of channels in hidden conv layer')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    args = parser.parse_args()

    train(args)
