from pathlib import Path
from setuptools import setup, find_packages

# Read the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="einit",
    version="0.2.0",
    author="Alexander Kolpakov, Michael Werman, Judah Levin",
    author_email="akolpakov@uaustin.org",
    maintainer="Alexander Kolpakov",
    maintainer_email="akolpakov@uaustin.org",
    description="Ellipsoid ICP initialization for OpenCV-compatible pipelines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sashakolpakov/einit",
    project_urls={
        "Documentation": "https://sashakolpakov.github.io/einit/",
        "Repository": "https://github.com/sashakolpakov/einit",
        "Issues": "https://github.com/sashakolpakov/einit/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="computer-vision point-cloud registration icp 3d-alignment ellipsoid",
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
            "sphinx_rtd_theme>=1.0",
            "colorama>=0.4",
        ],
    },
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=False,
)
