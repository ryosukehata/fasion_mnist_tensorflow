import os

from tensorflow.python.lib.io import file_io

from src.configs import parser_config
from src.datasets import _get_datasets
from src.models import get_model, _get_optimizer
from src.utils import _fix_seed

CENSUS_MODEL = 'census.hdf5'

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

    if args.job_dir.startswith('gs://'):
        model.save(CENSUS_MODEL)
        copy_file_to_gcs(args.job_dir, CENSUS_MODEL)
    else:
        model.save(os.path.join(args.job_dir, CENSUS_MODEL))

    return model

# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as fp:
            fp.write(input_f.read())

def test(args):
    pass


def main(args):
    if args.train:
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    _fix_seed()
    args = parser_config()
    train(args)
