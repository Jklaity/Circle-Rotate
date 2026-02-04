from setuptools import setup, find_packages

setup(
    name="circle_rotate",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
        "ultralytics>=8.0.0",
        "lpips>=0.1.4",
    ],
    python_requires=">=3.8",
)
