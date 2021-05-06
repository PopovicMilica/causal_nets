from setuptools import setup
from setuptools import find_packages

description = '''Software for estimation of heterogeneous treatment effects
                 and propensity scores.'''
  
setup(
      name='causal_nets',
      version='0.0.1',
      description=description,
      author='Milica Popovic',
      author_email='popovic.v.milica@gmail.com',
      classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'],
      keywords=['deep learning', 'treatment effects', 'causal inference'],
      install_requires=['tensorflow>=2.4.0',
                        'numpy'],
      extras_require={
        'tests': ['pytest', 'scikit-learn']},
      packages=find_packages(exclude=('tests',))
)