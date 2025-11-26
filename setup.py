from setuptools import setup

with open("requirements.txt", "r") as f:
    required_packages = f.read().splitlines()

setup(name='image_classification',
      version='1.0',
      author='Andrew Mitchell',
      install_requires=[*required_packages]
      )