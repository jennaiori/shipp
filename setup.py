from setuptools import setup

setup(
   name='shipp',
   version='1.1',
   description='A simple module for sizing hybrid power plants',
   author='Jenna Iori',
   author_email='j.iori@tudelft.nl',
   packages=['shipp'],  #same as name
   install_requires=['numpy',
                     'numpy-financial',
                     'pandas>=2.0',
                     'scipy',
                     'matplotlib',
                     'requests',
                     'pyomo',
                     'entsoe-py'], #external packages as dependencies
   extras_require={
        'test': [
            'pytest']}
)
