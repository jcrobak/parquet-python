try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='parquet',
    version='1.0',
    description='Python support for Parquet file format',
    author='Joe Crobak',
    author_email='joecrow@gmail.com',
    packages=[ 'parquet' ],
    install_requires=[
        'python-snappy',
        'thriftpy>=0.3.6',
    ],
    extras_require={
        ':python_version=="2.7"': [
            "backports.csv",
        ],
    },
    entry_points={
        'console_scripts': [
            'parquet = parquet.__main__:main',
        ]
    },
    package_data={'parquet': ['*.thrift']},
    include_package_data=True,
)
