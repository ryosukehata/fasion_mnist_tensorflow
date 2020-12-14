from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, SpatialDropout2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD


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


def _get_optimizer(args):
    if args.optimizer == 'sgd':
        return SGD(learning_rate=args.lr, momentum=args.momentum)
    else:
        return Adam(learning_rate=args.lr)
