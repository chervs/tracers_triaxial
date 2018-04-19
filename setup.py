#! /usr/bin/env python

from setuptools import setup

setup(name='Stracers',
      version='0.1',
      description='Code to assign stellar tracers to DM halos',
      author='Nico Garavito-Camargo and Chervin Laporte',
      author_email='jngaravitoc@email.arizona.edu',
      install_requieres=['numpy', 'scipy', 'matplotlib', 'astropy', 'pygadgetreader'],
      packages=['Stracers'],
)
