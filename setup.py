
from setuptools import setup, find_packages

setup(
    name="einit",
    version="0.1.0",
    author="Alexander Kolpakov (UATX), Michael Werman (HUJI), Judah Levin (UATX)",
    description="Ellipsoid ICP initialization for OpenCV-compatible pipelines",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.15",
        "scipy>=1.0",
    ],
    extras_require={
        "opencv": ["opencv-python-headless>=3.4"],
        "test": ["pytest>=6.0", "opencv-python-headless>=3.4", "matplotlib>=3.0"],
        "docs": ["sphinx>=4.0", "sphinx_rtd_theme>=1.0"],
        "all": ["opencv-python-headless>=3.4", "pytest>=6.0", "matplotlib>=3.0", "sphinx>=4.0", "sphinx_rtd_theme>=1.0"],
    },
    python_requires=">=3.6",
)