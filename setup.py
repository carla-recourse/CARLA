import pathlib
import re

from setuptools import find_packages, setup

VERSIONFILE = "carla/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    VERSION = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="carla-recourse",
    version=VERSION,
    description="A library for counterfactual recourse",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/indyfree/CARLA",
    author="Martin Pawelczyk, Sascha Bielawski, Joannes van den Heuvel, Tobias Richter and Gjergji Kasneci",
    author_email="martin.pawelczyk@uni-tuebingen.de",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=("test",)),
    include_package_data=True,
    install_requires=[
        "lime==0.2.0.1",
        "mip==1.12.0",
        "numpy==1.19.4",
        "pandas==1.1.4",
        "recourse==1.0.0",
        "scikit-learn==0.23.2",
        "tensorflow==1.14.0",
        "torch==1.7.0",
        "torchvision==0.8.1",
        "h5py==2.10.0",
        "dice-ml==0.5",
        "ipython",
        "keras==2.3.0",
    ],
)
