import tensorflow as tf

__author__ = "Stefano Campese"

__version__ = "0.1"
__maintainer__ = "Stefano Campese"
__email__ = "sircampydevelop@gmail.com"


class FourierKernel():
    """
    Fourier kernel,
        K(x, y) = PROD_i (1-q^2)/(2(1-2q cos(x_i-y_i)+q^2))
    """

    def __init__(self, q=0.1):
        self._q = q

    def _compute(self, x, y):
        kernel = tf.ones(
            (x.shape[0], y.shape[0]),
            dtype=tf.dtypes.float32,
            name=None
        )

        for d in range(x.shape[1]):
            col1 = tf.reshape(x[:, d], shape=(-1, 1))
            col2 = tf.reshape(y[:, d], shape=(-1, 1))

            denominator = tf.math.multiply(tf.constant(2, dtype=kernel.dtype),
                                           (
                                               tf.math.add(
                                                   tf.math.subtract(tf.constant(1, dtype=kernel.dtype),
                                                                    tf.math.multiply(
                                                                        tf.constant((2 * self._q), dtype=kernel.dtype),
                                                                        tf.math.cos(tf.math.subtract(col1,
                                                                                                     tf.transpose(
                                                                                                         col2))))),

                                                   tf.constant((self._q ** 2), dtype=kernel.dtype)

                                               )))

            tf.divide(tf.constant((1 - self._q ** 2), dtype=kernel.dtype), denominator)
            kernel = tf.multiply(kernel, tf.divide(tf.constant((1 - self._q ** 2), dtype=kernel.dtype), denominator))

        return kernel

    def dim(self):
        return None
