from pathlib import Path
from setuptools import setup, find_packages

# Read the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="einit",
    version="0.2.0",
    author="Alexander Kolpakov (UATX), Michael Werman (HUJI), Judah Levin (UATX)",
    description="Ellipsoid ICP initialization for OpenCV-compatible pipelines",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Enables Markdown rendering on PyPI
    packages=find_packages(),
    install_requires=[
        "numpy>=1.15",
        "scipy>=1.0",
    ],
    extras_require={
        "opencv": ["opencv-python-headless>=3.4"],
        "test": ["pytest>=6.0", "open3d>=0.17", "matplotlib>=3.0", "colorama>=0.4"],
        "docs": ["sphinx>=4.0", "sphinx_rtd_theme>=1.0"],
        "all": [
            "open3d>=0.17",
            "pytest>=6.0",
            "matplotlib>=3.0",
            "sphinx>=4.0",
            "sphinx_rtd_theme>=1.0"
        ],
    },
    python_requires=">=3.6",
)
