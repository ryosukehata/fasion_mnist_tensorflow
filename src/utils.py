
import numpy as np
import tensorflow


def _fix_seed(SEED=42):
    tensorflow.random.set_seed(SEED)
    np.random.seed(SEED)


