from setuptools import setup, find_packages

setup(
    name="object-detection-streamlit",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "opencv-python",
        "numpy",
        "pillow", 
        "pandas"
    ]
)
