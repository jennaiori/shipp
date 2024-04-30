from setuptools import setup

setup(
   name='shipp',
   version='0.0',
   description='A simple module for sizing hybrid power plants',
   author='Jenna Iori',
   author_email='j.iori@tudelft.nl',
   packages=['shipp'],  #same as name
   install_requires=['numpy>=1.26.0',
                     'numpy-financial==1.0.0',
                     'pandas>=2.2.0',
                     'scipy==1.11.3',
                     'matplotlib==3.8.1',
                     'requests==2.31.0',
                     'pyomo==6.7.1',
                     'ipykernel==6.26.0',
                     'entsoe-py==0.6.7'], #external packages as dependencies
   extras_require={
        'test': [
            'pytest']}
)
