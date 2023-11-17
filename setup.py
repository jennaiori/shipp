from setuptools import setup

setup(
   name='sizing_opt_hpp',
   version='0.0',
   description='A simple module for sizing hybrid power plants',
   author='Jenna Iori',
   author_email='j.iori@tudelft.nl',
   packages=['sizing_opt_hpp'],  #same as name
   install_requires=['numpy'], #external packages as dependencies
   extras_require={
        'test': [
            'pytest',]}
)
