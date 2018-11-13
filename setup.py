from setuptools import setup

with open("requirements.txt") as f:
    REQUIREMENTS = f.read()

with open("LICENSE") as f:
    LICENSE = f.read()

with open("README.md") as f:
    DESCRIPTION = f.read()

setup(name='bayes_models',
      version='0.1',
      description=DESCRIPTION,
      packages=['bayes_models', ],
      author='Edward Turner',
      license=LICENSE,
      author_email='edward.turnerr@gmail.com',
      install_requires=REQUIREMENTS,
      python_requires='==Python3.7')
