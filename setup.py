from setuptools import find_packages, setup


setup(
    name="Deepfake-recognition-kaggle",
    author="Peter Todorov, Denis Diackov",
    version="1.0",
    description="Detecting deepfake videos, kaggle competition",
    install_requires=list(open("requirements.txt").readlines()),
    packages=find_packages(),
)
