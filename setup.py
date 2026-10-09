from setuptools import setup, find_packages

setup(
    name="lesion_3d_measurement",
    version="0.1.0",
    packages=["lesion_3d_measurement", "lesion_3d_measurement.src", "lesion_3d_measurement.tests"],
    package_dir={
        "lesion_3d_measurement": ".",
        "lesion_3d_measurement.src": "src",
        "lesion_3d_measurement.tests": "tests"
    },
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
    ],
    python_requires=">=3.10",
)
