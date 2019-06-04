import tensorflow as tf

from kernels.base import BaseKernel

__author__ = "Stefano Campese"

__version__ = "0.1.2"
__maintainer__ = "Stefano Campese"
__email__ = "sircampydevelop@gmail.com"


class SplineKernel(BaseKernel):
    """
    Spline kernel,
        K(x, y) = PROD_i 1 + x_iy_i + x_iy_i min(x_i,y_i)
                           - (x_i+y_i)/2 * min(x_i,y_i)^2
                           + 1/3 * min(x_i, y_i)^3
    """

    def _compute(self, data_1, data_2):

        if not tf.math.greater_equal(tf.math.reduce_min(data_1), tf.constant(0, dtype=data_1.dtype)).eval(
                session=tf.Session()) or not tf.math.greater_equal(tf.math.reduce_min(data_2),
                                                               tf.constant(0, dtype=data_2.dtype)).eval(
                session=tf.Session()):
            raise ValueError('This kernel is a positive kernel!! Your elements must be all >=0')

        kernel = tf.ones(
            (data_1.shape[0], data_2.shape[0]),
            dtype=tf.dtypes.float32,
            name=None
        )

        for d in range(data_1.shape[1]):
            col1 = tf.reshape(data_1[:, d], shape=(-1, 1))
            col2 = tf.reshape(data_2[:, d], shape=(-1, 1))

            c_prod = tf.matmul(col1, col2, transpose_b=True)
            c_sum = tf.math.add(col1, tf.transpose(col2))
            c_min = tf.math.minimum(col1, tf.transpose(col2))

            kernel = tf.math.multiply(kernel,
                                      tf.add(
                                          tf.subtract(
                                              tf.add(
                                                  tf.add(
                                                      tf.constant(1, dtype=kernel.dtype),
                                                      c_prod
                                                  ),
                                                  tf.math.multiply(
                                                      c_prod,
                                                      c_min
                                                  )
                                              ),

                                              tf.math.multiply(
                                                  tf.divide(c_sum, tf.constant(2, dtype=c_sum.dtype)),
                                                  tf.pow(c_min, tf.constant(2, dtype=c_min.dtype))
                                              ),

                                          ),
                                          tf.math.multiply(
                                              tf.divide(
                                                  tf.constant(1, dtype=c_min.dtype),
                                                  tf.constant(3, dtype=c_min.dtype)
                                              ),
                                              tf.pow(
                                                  c_min,
                                                  tf.constant(3, dtype=c_min.dtype)
                                              )
                                          )
                                      )
                                      )
        return kernel

    def dim(self):
        # TODO fix dim
        return None
