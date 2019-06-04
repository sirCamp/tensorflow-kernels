import tensorflow as tf

from kernels.base import BaseKernel

__author__ = "Stefano Campese"

__version__ = "0.1.2"
__maintainer__ = "Stefano Campese"
__email__ = "sircampydevelop@gmail.com"


class PolynomialKernel(BaseKernel):
    """
    Polynomial kernel, defined as a power of an affine transformation
        K(x, y) = (a<x, y> + b)^p
    where:
        a = scale
        b = bias
        p = degree
    """

    def __init__(self, scale=1, bias=0, degree=2):
        self._dim = None
        self._scale = scale
        self._bias = bias
        self._degree = degree

    def _compute(self, x, y):
        self._dim = x.shape[1]

        dot = tf.matmul(x, y, transpose_b=True)

        return tf.math.pow(
            tf.math.add(tf.math.multiply(tf.constant(self._scale, dtype=dot.dtype), dot),
                        tf.constant(self._bias, dtype=dot.dtype)), tf.constant(self._degree, dtype=dot.dtype)
        )

    def dim(self):
        return self._dim ** self._degree
