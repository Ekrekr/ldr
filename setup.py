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
    name='ldr',
    version='0.2',
    description='Latent Dimensionality Reduction in Python',
    long_description=f"{LONG_DESCRIPTION}",
    long_description_content_type='text/markdown',
    url='https://github.com/ekrekr/pyldr',
    author='Elias Kassell',
    author_email='eliaskassell@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],

    # TODO: Add keywords
    keywords='sample setuptools development',

    package_dir={'': 'src'},
    packages=find_packages(),
    python_requires='>3.5',

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

    data_files=[("", ["README.md", "contributing.md"])],

    project_urls={
    },
)
