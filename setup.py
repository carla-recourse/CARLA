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
    url="https://github.com/carla-recourse/CARLA",
    author="Martin Pawelczyk, Sascha Bielawski, Johannes van den Heuvel, Tobias Richter and Gjergji Kasneci",
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
        "protobuf==4.23.1",
        "lime==0.2.0.1",
        "mip==1.12.0",
        "numpy==1.23.5",
        "pandas==2.0.1",
        "recourse==1.0.0",
        "scikit-learn==1.2.2",
        "tensorflow==2.12.0",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "h5py==3.8.0",
        "dice-ml==0.9",
        "ipython",
        "keras==2.12.0",
        "xgboost==1.7.5",
        "causalgraphicalmodels==0.0.4",
    ],
)
