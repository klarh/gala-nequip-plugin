#!/usr/bin/env python

import os
from setuptools import find_packages, setup

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

version_fname = os.path.join(THIS_DIR, 'gala_nequip_plugin', 'version.py')
with open(version_fname) as version_file:
    exec(version_file.read())

readme_fname = os.path.join(THIS_DIR, 'README.md')
with open(readme_fname) as readme_file:
    long_description = readme_file.read()

setup(name='gala-nequip-plugin',
      author='Matthew Spellings',
      author_email='mspells@vectorinstitute.ai',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
      ],
      description='Plugin to incorporate geometric algebra attention-based models into the nequip potential framework',
      entry_points={
      },
      extras_require={
      },
      install_requires=[
          'geometric-algebra-attention',
          'nequip',
      ],
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(
          include=['gala_nequip_plugin', 'gala_nequip_plugin.*']
      ),
      python_requires='>=3',
      version=__version__
      )
