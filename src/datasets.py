from tensorflow.keras import datasets


def _get_datasets():
    fashion_mnist = datasets.fashion_mnist
    return fashion_mnist.load_data()
