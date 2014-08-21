import sys
import os
import glob
try:
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
except ImportError:
    print 'Unable to import setuptools.  Using distutils instead.'
    from distutils.core import setup, Extension
    from distutils.command.build_ext import build_ext
try:
    from sysconfig import get_config_vars
except:
    from distutils.sysconfig import get_config_vars

py_version = "%d.%d"%sys.version_info[0:2]
print 'Python version = ',py_version

scripts = ['corr2']
scripts = [ os.path.join('scripts',f) for f in scripts ]

sources = glob.glob(os.path.join('src','*.cpp'))

# If we build with debug, also undefine NDEBUG flag
if "--debug" in sys.argv:
    undef_macros=['NDEBUG']
else:
    undef_macros=None

copt =  {
    'gcc' : ['-fopenmp','-O3','-ffast-math'],
    'icc' : ['-openmp','-O3'],
    'clang' : ['-O3','-ffast-math'],
}
lopt =  {
    'gcc' : ['-fopenmp'],
    'icc' : ['-openmp'],
    'clang' : [],
}

def get_compiler(cc):
    """Try to figure out which kind of compiler this really is.
    In particular, try to distinguish between clang and gcc, either of which may
    be called cc or gcc.
    """
    cmd = cc + ' --version 2>&1'
    import subprocess
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    lines = p.stdout.readlines()
    if 'clang' in lines[0]:
        # Supposedly, clang will support openmp in version 3.5.  Let's go with that for now...
        # If the version is reports >= 3.5, let's call in gcc, rather than clang to get
        # the gcc -fopenmp flag.
        line = lines[1]
        import re
        match = re.search(r'[0-9]+(\.[0-9]+)+', line)
        if match:
            version = match.group(0)
            # Get the version up to the first decimal
            # e.g. for 3.4.1 we only keep 3.4
            vnum = version[0:version.find('.')+2]
            if vnum >= '3.5':
                return 'gcc'
        return 'clang'
    elif 'gcc' in lines[0]:
        return 'gcc'
    elif 'GCC' in lines[0]:
        return 'gcc'
    elif 'clang' in cc:
        return 'clang'
    elif 'gcc' in cc or 'g++' in cc:
        return 'gcc'
    elif 'icc' in cc or 'icpc' in cc:
        return 'icc'
    else:
        return 'unknown'

# This was supposed to remove the -Wstrict-prototypes flag
# But it doesn't work....
# Hopefully they'll fix this bug soon:
#  http://bugs.python.org/issue9031
#  http://bugs.python.org/issue1222585
#(opt,) = get_config_vars('OPT')
#os.environ['OPT'] = " ".join( flag for flag in opt.split() if flag != '-Wstrict-prototypes')

# Make a subclass of build_ext so we can do different things depending on which compiler we have.
# In particular, we want to use different compiler options for OpenMP in each case.
# cf. http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class my_builder( build_ext ):
    def build_extensions(self):
        cc = self.compiler.executables['compiler_cxx'][0]
        comp_type = get_compiler(cc)
        print 'Using compiler %s, which is %s'%(cc,comp_type)
        if copt.has_key(comp_type):
            for e in self.extensions:
                e.extra_compile_args = copt[ comp_type ]
                e.include_dirs = ['include']
        if lopt.has_key(comp_type):
            for e in self.extensions:
                e.extra_link_args = lopt[ comp_type ]
                e.include_dirs = ['include']
        build_ext.build_extensions(self)

ext=Extension("treecorr._treecorr",
              sources,
              undef_macros = undef_macros)

dependencies = ['numpy']
if py_version < '2.7':
    dependencies += ['argparse']
else:
    dependencies += ['pandas']  # These seem to have conflicting numpy requirements, so don't
                                # include pandas with argparse.

# Make sure at least some fits package is present.
try:
    import astropy
except ImportError:
    try:
        import pyfits
    except ImportError:
        dependencies += ['fitsio']

setup(name="TreeCorr", 
      version="3.0.0",
      author="Mike Jarvis",
      author_email="michael@jarvis.net",
      description="Python module for computing 2-point correlation functions",
      license = "BSD License",
      url="https://github.com/rmjarvis/TreeCorr",
      download_url="https://github.com/rmjarvis/TreeCorr/releases/tag/v3.0.0.zip",
      packages=['treecorr'],
      ext_modules=[ext],
      install_requires=dependencies,
      cmdclass = {'build_ext': my_builder },
      scripts=scripts)

