#!/usr/bin/env python
from setuptools import setup

try:
    from sphinx.setup_command import BuildDoc
except ImportError:
    BuildDoc = None

long_description = open('README.rst').read()

setup(
    name='scikit-signal',
    version='0.1',
    description='A scikit for signal processing',
    long_description=long_description,
    author='Marc Joliet',
    author_email='marcec@gmx.de',
    url='https://github.com/scikit-signal/scikit-signal',
    keywords=['audio', 'filtering'],
    packages=['filter_banks'],
    license='MIT',
    # "requires" is required by the --requires option
    requires=['numpy', 'scipy'],
    install_requires=['numpy', 'scipy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Other Environment',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Multimedia :: Sound/Audio'
    ],
    cmdclass=({'docs': BuildDoc} if BuildDoc else {})
)
