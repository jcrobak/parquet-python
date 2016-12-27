"""setup.py - build script for parquet-python."""

import os

from setuptools import setup, Extension

from Cython.Build import cythonize

cython_modules = [Extension('fastparquet.speedups',
                            ['fastparquet/speedups.pyx'])]

ext_modules = cythonize(cython_modules)


setup(
    name='fastparquet',
    version='0.0.4',
    description='Python support for Parquet file format',
    ext_modules=ext_modules,
    author='Joe Crobak, Martin Durant',
    author_email='mdurant@continuum.io',
    url='https://github.com/martindurant/fastparquet/',
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    packages=['fastparquet'],
    install_requires=[open('requirements.txt').read().strip().split('\n')],
    long_description=(open('README.rst').read() if os.path.exists('README.rst')
                      else ''),
    package_data={'fastparquet': ['*.thrift']},
    include_package_data=True,
)
