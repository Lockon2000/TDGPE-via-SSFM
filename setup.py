import os.path
import setuptools
import codecs


"""
The following section is needed to provide a single-source implementation for package meta data, e.g. version.
"""

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def getVersion(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

def getDescription(rel_path):
    try:
        descriptionIndex = None
        lines = read(rel_path).splitlines()

        for i, line in enumerate(lines):
            if line.startswith('# TDGPE-via-SSFM'):
                descriptionIndex = i + 2    # The description is on the next next line
                break

        return lines[descriptionIndex]
    except:
        raise RuntimeError("Unable to find description string.")


setuptools.setup(
    name="TDGPEviaSSFM",
    version=getVersion("TDGPEviaSSFM/__init__.py"),
    description=getDescription("README.md"),
    url="https://github.com/Lockon2000/TDGPE-via-SSFM",
    author="Mohamed Abdelwahab",
    author_email="m.abdelwahab2@gmail.com",
    license="MIT License",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy", "matplotlib", "PyCav>=1.0.0b3"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3"
    ]
)

