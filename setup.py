__author__ = 'singerpp'

from setuptools import setup
from setuptools import find_packages

setup(
    name='hyptrails',
    version='0.4',
    author='Philipp Singer',
    author_email='philipp.singer@gesis.org',
    packages=['hyptrails'],
    license='MIT License',
    url='https://github.com/psinger/HypTrails',
    install_requires=[
      'pathtools==0.6',
    ],
    dependency_links=[
      'https://github.com/psinger/PathTools/archive/master.zip#egg=pathtools-0.6'
    ]
)
