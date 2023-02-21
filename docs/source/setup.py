from setuptools import setup

VERSION = "0.0.1"

setup(
    name="carla",
    version=VERSION,
    package_dir={"carla": "carla"},
    install_requires=[
        "sphinx-copybutton",
        "sphinx_gallery",
        "markupsafe==2.0.1",
        "scipy==1.6.2",
        "lime==0.2.0.1",
        "mip==1.12.0",
        "numpy==1.19.4",
        "pandas==1.1.4",
        "scikit-learn==0.23.2",
        "scikit-image==0.19.3",
        "tensorflow==1.14.0",
        "torch==1.7.0",
        "torchvision==0.8.1",
        "h5py==2.10.0",
        "PyWavelets==1.3.0",
        "dice-ml==0.5",
        "ipython==7.22.0",
        "ipykernel==5.5.3",
        "nbconvert==5.6.1",
        "jupyter-client==6.1.12",
        "jupyterlab==3.3.2",
        "jupyterlab-server==2.10.0",
        "jupyter-server==1.15.6",
        "keras==2.3.0",
        "nbsphinx",
        "pillow==9.0.1",
        "matplotlib==3.5.1",
        "networkx==2.5.1",
        "imageio==2.9.0",
        "werkzeug==2.1.2",
    ],
    entry_points="""
                [console_scripts]
                claims-assessment=carla.run:main
            """,
)
