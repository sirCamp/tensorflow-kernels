import tensorflow as tf

__author__ = "Stefano Campese"

__version__ = "0.1"
__maintainer__ = "Stefano Campese"
__email__ = "sircampydevelop@gmail.com"


class CosineSimilarityKernel():
    """
    Cosine similarity kernel,
        K(x, y) = <x, y> / (||x|| ||y||)
    """

    def _compute(self, x, y):
        self._dim = x.shape[1]

        norms_1 = tf.math.sqrt(tf.reshape(tf.math.reduce_sum(tf.math.pow(x, tf.constant(2, dtype=x.dtype)), axis=1),
                                          shape=(x.shape[0], 1)))
        norms_2 = tf.math.sqrt(tf.reshape(tf.math.reduce_sum(tf.math.pow(y, tf.constant(2, dtype=y.dtype)), axis=1),
                                          shape=(y.shape[0], 1)))

        return tf.math.divide(tf.matmul(x, y, transpose_b=True), tf.matmul(norms_1, norms_2, transpose_b=True))

    def dim(self):
        return self._dim
