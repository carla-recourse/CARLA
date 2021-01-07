from setuptools import setup

VERSION = "0.0.1"

setup(
    name="cf_benchmark",
    version=VERSION,
    package_dir={"cf_benchmark": "cf_benchmark"},
    install_requires=[
        "lime==0.2.0.1",
        "mip==1.12.0",
        "numpy==1.19.4",
        "pandas==1.1.4",
        "recourse==1.0.0",
        "scikit-learn==0.23.2",
        "tensorflow==2.4.0",
        "torch==1.7.0",
    ],
    entry_points="""
                [console_scripts]
                claims-assessment=cf_benchmark.run:main
            """,
)
