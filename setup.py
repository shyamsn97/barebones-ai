import io
import os
import re
from os import path

from setuptools import find_packages
from setuptools import setup


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="barebones-ai",
    version="0.1.2",
    url="https://github.com/shyamsn97/mario-gpt",
    license='MIT',

    author="Shyam Sudhakaran",
    author_email="shyamsnair@protonmail.com",

    description="Popular ML algorithms implemented from scratch using numpy",

    long_description=long_description,
    long_description_content_type="text/markdown",

    include_package_data = True,

    packages=find_packages(exclude=('tests',)),

    install_requires=[
        'numpy',
        'matplotlib',
        'tqdm',
        'pandas',
        'seaborn',
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
