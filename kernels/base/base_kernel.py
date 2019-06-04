from abc import abstractmethod

__author__ = "Stefano Campese"

__version__ = "0.1.2"
__maintainer__ = "Stefano Campese"
__email__ = "sircampydevelop@gmail.com"


class BaseKernel(object):
    """
    Base, abstract kernel class
    """

    def __call__(self, x, y):
        return self._compute(x, y)

    def compute(self, x, y):
        return self._compute(x, y)

    @abstractmethod
    def _compute(self, data_1, data_2):
        """
        The main computation method
        """
        raise NotImplementedError('This is an abstract class')

    @abstractmethod
    def dim(self):
        """
        Returns dimension of the feature space
        """
        raise NotImplementedError('This is an abstract class')
