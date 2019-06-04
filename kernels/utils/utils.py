import numpy as np
import tensorflow as tf

__author__ = "Stefano Campese"

__version__ = "0.1.2"
__maintainer__ = "Stefano Campese"
__email__ = "sircampydevelop@gmail.com"


def array_to_tensor(array=None, dtype=tf.float32):
    """

    :param array: numpy array to convert to a tensor
    :param dtype: tensor type
    :return: tensor of the specified type
    """
    if array is None:
        raise ValueError("array is None, please put a valid variable")
    return tf.convert_to_tensor(array, dtype=dtype)


def tensor_to_array(tensor=None, dtype=np.float32, session=None):
    """
    This method convert a tensor to a numpy array
    :param tensor: tensor
    :param dtype: numpy type to convert the tensor
    :param session: your session of tensorflow
    :return: numpy array of the specified type
    """

    sess = session
    if tensor is None:
        raise ValueError("tensor is None, please put a valid tensor")

    if sess is None:
        sess = tf.Session()

    if tf.is_numeric_tensor(tensor) and dtype == np.object:
        raise ValueError(
            "type mismatch: you have pass a numeric tensor and you're trying to convert it to a string array")

    result = None
    if not tf.executing_eagerly():
        result = np.array(tensor.eval(session=sess), dtype=dtype)
    else:
        result = np.array(tensor, dtype=dtype)

    return result
