from setuptools import setup

setup(
    name="csl",
    version="0.1.0",
    author="Luiz Chamon",
    author_email="luiz.chamon@simtech.uni-stuttgart.de",
    packages=["csl", "csl.datasets"],
    url="http://pypi.python.org/pypi/PackageName/",
    license="LICENSE.txt",
    description="A Python package based around pytorch to simplify the definition of constrained learning problems and then solving them.",
    long_description=open("README.md").read(),
    install_requires=[],
)
