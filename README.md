# Tensorflow-kernels

![Logo](/doc/img/image.png)

A package with Tensorflow (both CPU and GPU) implementation of most popular Kernels for kernels methods (SVM, MKL...).

Those kernels works with tensor as inputs.

The main idea of this project is to exploit the powerfull of GPUs and modern CPUs on matrix and kernels elaborations.
Actually the implemented kernels are:

+ Linear
+ RBF
+ Polynomial
+ CosineSimilarity
+ Fourier
+ Spline

An Experimental Kernel have been added:
+ PSpectrum

**Attention:** Due to the GPUs usage the precision of decimal numbers may be different, and hence, the results may be slightly differs as well

**Attention 2:** Due to exploit the power of GPUs it's strongly recommended to work with float32 or even in half precision float16.
# Examples: 

A simple example with ```PolynomialKernel```
```python
import numpy as np
import tensorflow as tf
from kernels.polynomial_kernel import PolynomialKernel
from kernels import array_to_tensor, tensor_to_array

n = 2000
p = 1000
a = np.random.random((n, p)).astype(np.float32)
b = np.random.random((n, p)).astype(np.float32)

x = array_to_tensor(a, dtype=tf.float32)
y = array_to_tensor(b, dtype=tf.float32)


poly = PolynomialKernel(scale=1, bias=0, degree=4)
kernel = poly.compute(x, y)

print(tensor_to_array(kernel, dtype=np.float32))


```

A simple example with ```PSpectrumKernel```. 
**Attention:** PSpectrum is still experimental and it exploits *eager computation* in order to work properly. 
Furthermore it maybe won't works with Tensorflow 2.0 since some packages have been removed.

**Attention 2:** Due to the usage of the type ```tf.string``` computation will be shared between GPUs and CPUs.

**Attention 3:** This kernel return tensor with type tf.int64.

```python
import numpy as np
import tensorflow as tf
from kernels.experimental.p_spectrum_kernel import PSpectrumKernel
from kernels import array_to_tensor, tensor_to_array

a = np.array(['aaaaaaaa','bbbbbbb','ccccc','aaaaaaa','cccccc','bbbbbb'])
b = np.array(['aaaaaaaa','bbbbbbb','aaaaaaa','cccccc'])

x = array_to_tensor(a, dtype=tf.string)
y = array_to_tensor(b, dtype=tf.string)

p_spectrum = PSpectrumKernel(p=3)

kernel = p_spectrum.compute(x, y)

print(tensor_to_array(kernel, dtype=np.float32))

```


# Credits:
The idea was born by using methods available here: [https://github.com/gmum/pykernels](https://github.com/gmum/pykernels)
