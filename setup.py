from setuptools import setup, find_packages

setup(
    name="gdmil",
    version="1.0.0",
    description="Grade-Disentangled MIL for multimodal biochemical "
                "recurrence prediction in prostate cancer",
    author="Dasari Naga Raju",
    author_email="raajuuu1998@gmail.com",
    url="https://github.com/raajuuu1998/gd-mil-bcr",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.23",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "lifelines>=0.27",
        "matplotlib>=3.6",
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
