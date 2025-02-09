from setuptools import setup, find_packages

setup(
    name="pumpprobeFDTD",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
    ],
    python_requires=">=3.10",
    author="Tianchuang Luo",
    description="A package for pump-probe FDTD simulations",
    license="MIT",
    url="https://github.com/Sky-creates/pumpprobeFDTD.git",
)
