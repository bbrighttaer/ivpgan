# Author: bbrighttaer
# Project: IVPGAN for DTI
# Date: 07/06/2019
# Time: 
# File: setup.py.py


from setuptools import setup

setup(
    name='ivpgan',
    version='0.0.1',
    packages=['ivpgan', 'ivpgan.nn', 'ivpgan.nn.tests', 'ivpgan.utils', 'ivpgan.utils.tests',
              'ivpgan.metrics'],
    url='',
    license='MIT',
    author='Brighter Agyemang',
    author_email='brighteragyemang@gmail.com',
    description='',
    install_requires=['torch', 'numpy', 'scikit-optimize', 'padme', 'pandas', 'matplotlib', 'seaborn', 'soek']
)
