import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sk-autobots", # Replace with your own username
    version="0.0.2",
    author="Caleb Castleberry",
    author_email="castle.caleb@gmail.com",
    description="Custom data transformers that use the scikit-learn API.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ccastleberry/sk-autobots",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
