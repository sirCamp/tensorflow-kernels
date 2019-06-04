from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='tensorflow-kernels',
    packages=find_packages(exclude=['build', '_docs', 'templates']),
    version='0.1.2',
    license="MIT",
    description='A package with Tensorflow (both CPU and GPU) implementation of most popular Kernels for kernels methods (SVM, MKL...).',
    author="Campese Stefano",
    install_requires=[
        "tensorflow",
        "numpy",
    ],
    author_email="stefano.campese.90@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sirCamp/tensorflow-kernels",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
