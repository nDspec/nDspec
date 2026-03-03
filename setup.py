#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['astropy>=5.0.2',
                'corner>=2.2.1',
                'emcee>=3.1.4',               
                'lmfit>=1.3.1',
                'matplotlib>=3.5.1',
                'numpy>=1.24',
                'scipy>=1.10',
                'stingray',
                'pyfftw',
                'tqdm']

test_requirements = [ ]

setup(
    author="Matteo Lucchini",
    author_email='m.lucchini@uva.nl',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.10',
    ],
    description="The nDspec modelling software.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='ndspec',
    name='ndspec',
    #packages=find_packages(include=['ndspec']),
    #test_suite='tests',
    #tests_require=test_requirements,
    url='https://github.com/matteolucchini1/ndspec',
    version='0.1.0',
    zip_safe=False,
)
