from setuptools import setup, find_packages

setup(
    name="product-defect-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automated product defect detection using ResNet-50 and transfer learning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.2.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.13.0",
    ],
)
