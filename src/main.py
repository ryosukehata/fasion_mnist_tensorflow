from configs import parser_config
from datasets import _get_datasets
from models import get_model, _get_optimizer
from utils import _fix_seed


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

def main(args):
    if args.train:
        train(args)
    else:
        test(args)

if __name__ == "__main__":
    _fix_seed()
    args = parser_config()
    train(args)
