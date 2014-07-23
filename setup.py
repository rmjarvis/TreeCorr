import os
import glob
from distutils.core import setup

scripts = ['corr2']
scripts = [ os.path.join('scripts',f) for f in scripts ]

setup(name="TreeCorr", 
      version="3.0",
      description="Python module for computing 2-point correlation functions",
      license = "BSD",
      author="Mike Jarvis",
      author_email="michael@jarvis.net",
      packages=['treecorr'],
      scripts=scripts)

