from setuptools import setup, find_packages

setup(
    name="O1NumHess_QC",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "O1NumHess",
    ],
    python_requires=">=3.6",
)
