from setuptools import setup
long_description = '''Software for estimating treatment
                      effects and propensity scores.'''
	  
setup(name='causal_nets',
      version='0.0.1',
      long_description=long_description,
      author='Milica Popovic',
      author_email='popovic.v.milica@gmail.com',
      packages=['causal_nets'],
      install_requires=['numpy==1.16.4',
                        'tensorflow>=1.14.0'])