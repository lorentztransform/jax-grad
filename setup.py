from setuptools import setup, find_packages

setup(
    name="jax-grad",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A from-scratch implementation of automatic differentiation using JAX",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/jax-grad",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
