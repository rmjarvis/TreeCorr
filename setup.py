import sys
import os
import glob
import re

from setuptools import setup, find_packages
import setuptools
print("Using setuptools version", setuptools.__version__)

from pybind11.setup_helpers import Pybind11Extension, build_ext

print('Python version = ',sys.version)
py_version = "%d.%d"%sys.version_info[0:2]  # we check things based on the major.minor version.

scripts = ['corr2', 'corr3']
scripts = [ os.path.join('scripts',f) for f in scripts ]

packages = find_packages()
print('packages = ',packages)

sources = glob.glob(os.path.join('src','*.cpp'))
headers = glob.glob(os.path.join('include','*.h'))

def find_pybind11_path():
    import pybind11
    import os
    print('PyBind11 is version ',pybind11.__version__)
    print('Looking for pybind11 header files: ')
    locations = [pybind11.get_include(user=True),
                 pybind11.get_include(user=False),
                 '/usr/include',
                 '/usr/local/include',
                 None]
    for try_dir in locations:
        if try_dir is None:
            # Last time through, raise an error.
            print("Could not find pybind11 header files.")
            print("They should have been in one of the following locations:")
            for l in locations:
                if l is not None:
                    print("   ", l)
            raise OSError("Could not find PyBind11")
        print('  ',try_dir,end='')
        if os.path.isfile(os.path.join(try_dir, 'pybind11/pybind11.h')):
            print('  (yes)')
            pybind11_path = try_dir
            break
        else:
            raise OSError("Could not find PyBind11")
            print('  (no)')

    return pybind11_path

ext = Pybind11Extension("treecorr._treecorr",
                        sources,
                        depends=headers,
                        include_dirs=['include', find_pybind11_path()])

build_dep = ['setuptools>=38', 'numpy>=1.17', 'pybind11>=2.2']
run_dep = ['numpy>=1.17', 'pyyaml', 'LSSTDESC.Coord>=1.1']

with open('README.rst') as file:
    long_description = file.read()

# Read in the treecorr version from treecorr/_version.py
# cf. http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
version_file=os.path.join('treecorr','_version.py')
verstrline = open(version_file, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    treecorr_version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))
print('TreeCorr version is %s'%(treecorr_version))

dist = setup(
    name="TreeCorr",
    version=treecorr_version,
    author="Mike Jarvis",
    author_email="michael@jarvis.net",
    description="Python module for computing 2-point correlation functions",
    long_description=long_description,
    license="BSD License",
    url="https://github.com/rmjarvis/TreeCorr",
    download_url="https://github.com/rmjarvis/TreeCorr/releases/tag/v%s.zip"%treecorr_version,
    packages=['treecorr'],
    package_data={'treecorr' : headers },
    ext_modules=[ext],
    setup_requires=build_dep,
    install_requires=run_dep,
    cmdclass={'build_ext': build_ext},
    scripts=scripts
)
