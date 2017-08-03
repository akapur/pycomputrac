# coding=utf-8
"""Setup pycomputrac"""

from setuptools import setup

setup(name='pycomputrac',
      version='0.1',
      description='Read market data stored in Computrac/Metastock format',
      url='https://github.com/akapur/pycomputrac',
      author='Ashwin Kapur',
      author_email='akapur@amvirk.com',
      license='GPL v2',
      packages=['pycomputrac'],
      zip_safe=False, install_requires=['numpy', 'h5py'])
