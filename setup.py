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
      keywords=['deep learning', 'treatment effects', 'causal inference'],
      install_requires=['numpy==1.16.4',
                        'tensorflow==2.4.0'],
      extras_require={
        'tests': ['scikit-learn']},
      packages=find_packages(exclude=('tests',))
)