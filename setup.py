import sys
import os
import glob
from distutils.core import setup, Extension

scripts = ['corr2']
scripts = [ os.path.join('scripts',f) for f in scripts ]

sources = glob.glob(os.path.join('src','*.cpp'))

# If we build with debug, also undefine NDEBUG flag
if "--debug" in sys.argv:
    undef_macros=['NDEBUG']
else:
    undef_macros=None

ext=Extension("treecorr._treecorr",
              sources,
              undef_macros = undef_macros)

setup(name="TreeCorr", 
      version="3.0",
      description="Python module for computing 2-point correlation functions",
      license = "BSD",
      author="Mike Jarvis",
      author_email="michael@jarvis.net",
      url="https://github.com/rmjarvis/TreeCorr",
      packages=['treecorr'],
      ext_modules=[ext],
      scripts=scripts)

