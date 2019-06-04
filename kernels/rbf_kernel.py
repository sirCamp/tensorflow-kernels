import numpy as np
import tensorflow as tf

from kernels.base import BaseKernel

__author__ = "Stefano Campese"

__version__ = "0.1.2"
__maintainer__ = "Stefano Campese"
__email__ = "sircampydevelop@gmail.com"


class RBFKernel(BaseKernel):
    """
    Radial Basis Function kernel
        K(x, y) = e^(-g||x - y||^2)
    where:
        g = gamma
    """

    def __init__(self, gamma=None):
        self._gamma = gamma

    def _compute(self, x, y):
        if self._gamma is None:
            self._gamma = 1. / x.shape[1]

        norms_1 = tf.math.reduce_sum(tf.math.pow(x, tf.constant(2, dtype=x.dtype)), axis=1)
        norms_2 = tf.math.reduce_sum(tf.math.pow(y, tf.constant(2, dtype=y.dtype)), axis=1)

        dot = tf.matmul(x, y, transpose_b=True)

        dist_sq = tf.abs(tf.math.subtract(tf.math.add(tf.reshape(norms_1, shape=(-1, 1)), norms_2),
                                          tf.math.multiply(tf.constant(2, dtype=dot.dtype), dot)))
        # TODO: convert gamma to a tensor
        return tf.math.exp(-self._gamma * dist_sq)

    def dim(self):
        return np.inf
