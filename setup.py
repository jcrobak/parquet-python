"""setup.py - build script for parquet-python."""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='parquet',
    version='2.0',
    description='Python support for Parquet file format',
    author='Joe Crobak, Martin Durant',
    author_email='mdurant@continuum.io',
    url='https://github.com/martindurant/parquet-python/',
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    packages=['parquet'],
    install_requires=[open('requirements.txt').read().strip().split('\n')],
    long_description=(open('README.rst').read() if os.path.exists('README.rst')
                      else ''),
    package_data={'parquet': ['*.thrift']},
    include_package_data=True,
)
