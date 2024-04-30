from setuptools import setup

setup(
   name='shipp',
   version='0.0',
   description='A simple module for sizing hybrid power plants',
   author='Jenna Iori',
   author_email='j.iori@tudelft.nl',
   packages=['shipp'],  #same as name
   install_requires=['numpy',
                     'numpy-financial',
                     'pandas',
                     'scipy',
                     'matplotlib',
                     'requests',
                     'pyomo',
                     'ipykernel'], #external packages as dependencies
   extras_require={
        'test': [
            'pytest']}
)
