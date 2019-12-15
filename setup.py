"""
pyldr - setup configuration.

Setup for deploying as package to PyPi.
"""
from os import path
from setuptools import setup, find_packages

LOCAL_PATH = path.abspath(path.dirname(__file__))

with open(path.join(LOCAL_PATH, 'README.md')) as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='sampleproject',
    version='0.0.1',
    description='Latent Dimensionality Reduction in Python',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/ekrekr/pyldr',
    author='Elias Kassell',
    author_email='eliaskassell@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Machine Learning :: Model Visualization',
        'License :: GNU GPLv3 License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    # TODO: Add keywords
    keywords='sample setuptools development',

    package_dir={'': 'ldr'},
    packages=find_packages(where='ldr'),
    python_requires='>3.5, <4',

    # TODO: Check install requires
    install_requires=[''],

    extras_require={
        'dev': [''],
        'test': ['pytest'],
    },
    package_data={
        'sample': ['package_data.dat'],
    },

    # TODO: Confirm entry points
    entry_points={
        'console_scripts': [
            'sample=sample:main',
        ],
    },

    project_urls={
    },
)
