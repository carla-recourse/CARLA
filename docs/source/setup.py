from setuptools import setup

VERSION = "0.0.1"

setup(
    name="carla",
    version=VERSION,
    package_dir={"carla": "carla"},
    install_requires=[
        "scipy==1.6.2",
        "lime==0.2.0.1",
        "mip==1.12.0",
        "numpy==1.19.4",
        "pandas==1.1.4",
        "scikit-learn==0.23.2",
        "tensorflow==1.14.0",
        "torch==1.7.0",
        "torchvision==0.8.1",
        "h5py==2.10.0",
        "dice-ml==0.5",
        "ipython",
        "keras==2.3.0",
        "pillow==9.0.1",
        "matplotlib==3.5.1",
        "networkx==2.5.1",
        "imageio==2.9.0",
    ],
    entry_points="""
                [console_scripts]
                claims-assessment=carla.run:main
            """,
)
