import tensorflow as tf

from kernels.base import BaseKernel

tf.enable_eager_execution()
tf.executing_eagerly()

import numpy as np

__author__ = "Stefano Campese"

__version__ = "0.1.2"
__maintainer__ = "Stefano Campese"
__email__ = "sircampydevelop@gmail.com"


class PSpectrumKernel(BaseKernel):
    """
    P-Spectrum  kernel, defined as weighted trasformation of subsequences. usefull for character embedding
        K(x, y) = = <Φp(x ), Φp(y)>.
    where:
        p = the spectrum weight
    """

    def __init__(self, p=2):
        self._dim = None
        self._p = p

    def _compute(self, x, y):
        self._dim = x._rank()
        kernel = np.zeros((tf.size(x), tf.size(y)))

        for l in tf.range(start=0, limit=tf.size(x), delta=1, dtype=None, name='l_range'):
            for m in tf.range(start=0, limit=tf.size(y), delta=1, dtype=None, name='m_range'):

                vx = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                                                        value_dtype=tf.int64,
                                                        default_value=-1)

                vz = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                                                        value_dtype=tf.int64,
                                                        default_value=-1)

                vx_keys = tf.reshape(tf.Variable([], collections=[], dtype=tf.string), (-1, 1))
                vz_keys = tf.reshape(tf.Variable([], collections=[], dtype=tf.string), (-1, 1))

                x_t = tf.gather(x, l)
                x_t_len = tf.strings.length(x_t)
                x_t = tf.string_split([x_t], delimiter='').values

                z_t = tf.gather(y, m)
                z_t_len = tf.strings.length(z_t)
                z_t = tf.string_split([z_t], delimiter='').values

                for i in tf.range(start=0, limit=x_t_len - self._p + 1, delta=1, dtype=None, name='range'):
                    u = tf.string_join(x_t[i:i + self._p], '')
                    vx_keys, r = tf.cond(
                        tf.greater(vx.lookup(u), -1),
                        true_fn=lambda: (vx_keys, tf.add(vx.lookup(u), 1)),
                        false_fn=lambda: (tf.concat([vx_keys, tf.reshape(u, (-1, 1))], axis=0),
                                          tf.constant(1, dtype=tf.int64, name='constant'))
                    )
                    vx.insert(u, r)

                for i in tf.range(start=0, limit=z_t_len - self._p + 1, delta=1, dtype=None, name='range'):
                    u = tf.string_join(z_t[i:i + self._p], '')
                    vz_keys, r = tf.cond(
                        tf.greater(vz.lookup(u), -1),
                        true_fn=lambda: (vz_keys, tf.add(vz.lookup(u), 1)),
                        false_fn=lambda: (
                            tf.concat([vz_keys, tf.reshape(u, (-1, 1))], axis=0), tf.constant(1, dtype=tf.int64))
                    )
                    vz.insert(u, r)

                kk = tf.Variable(0, dtype=tf.int64)
                for i in tf.range(start=0, limit=tf.size(vx_keys), delta=1, dtype=None, name='range'):
                    for j in tf.range(start=0, limit=tf.size(vz_keys), delta=1, dtype=None, name='range'):
                        to_add = tf.cond(
                            tf.greater(vz.lookup(vx_keys[i]), -1),
                            true_fn=lambda: tf.math.multiply(vx.lookup(vx_keys[i]), vz.lookup(vz_keys[j])),
                            false_fn=lambda: tf.constant(0, dtype=tf.int64)
                        )
                        kk = tf.math.add(kk, to_add)
                    kernel[l][m] = kk

        return tf.convert_to_tensor(kernel, dtype=tf.int64)

    def dim(self):
        return self._dim
