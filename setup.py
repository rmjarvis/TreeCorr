from __future__ import print_function
import sys
import os
import glob

try:
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
    import setuptools
    print("Using setuptools version",setuptools.__version__)
except ImportError:
    print('Unable to import setuptools.  Using distutils instead.')
    from distutils.core import setup, Extension
    from distutils.command.build_ext import build_ext
    import distutils
    print("Using distutils version",distutils.__version__)
try:
    from sysconfig import get_config_vars
except:
    from distutils.sysconfig import get_config_vars

print('Python version = ',sys.version)
py_version = "%d.%d"%sys.version_info[0:2]  # we check things based on the major.minor version.

scripts = ['corr2', 'corr3']
scripts = [ os.path.join('scripts',f) for f in scripts ]

sources = glob.glob(os.path.join('src','*.cpp'))

undef_macros = []

# If we build with debug, also undefine NDEBUG flag
if "--debug" in sys.argv:
    undef_macros+=['NDEBUG']

copt =  {
    'gcc' : ['-fopenmp','-O3','-ffast-math'],
    'icc' : ['-openmp','-O3'],
    'clang' : ['-O3','-ffast-math'],
    'clang w/ OpenMP' : ['-fopenmp=libomp','-O3','-ffast-math'],
    'unknown' : [],
}
lopt =  {
    'gcc' : ['-fopenmp'],
    'icc' : ['-openmp'],
    'clang' : [],
    'clang w/ OpenMP' : ['-fopenmp=libomp'],
    'unknown' : [],
}

if "--debug" in sys.argv:
    copt['gcc'].append('-g')
    copt['icc'].append('-g')
    copt['clang'].append('-g')
    copt['clang w/ OpenMP'].append('-g')

def get_compiler(cc):
    """Try to figure out which kind of compiler this really is.
    In particular, try to distinguish between clang and gcc, either of which may
    be called cc or gcc.
    """
    cmd = [cc,'--version']
    import subprocess
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = p.stdout.readlines()
    print('compiler version information: ')
    for line in lines:
        print(line.strip())
    try:
        # Python3 needs this decode bit.
        # Python2.7 doesn't need it, but it works fine.
        line = lines[0].decode(encoding='UTF-8')
        if line.startswith('Configured'):
            line = lines[1].decode(encoding='UTF-8')
    except TypeError:
        # Python2.6 throws a TypeError, so just use the lines as they are.
        line = lines[0]
        if line.startswith('Configured'):
            line = lines[1]

    if "clang" in line:
        # clang 3.7 is the first with openmp support. So check the version number.
        # It can show up in one of two places depending on whether this is Apple clang or
        # regular clang.
        import re
        if 'LLVM' in line:
            match = re.search(r'LLVM [0-9]+(\.[0-9]+)+', line)
            match_num = 1
        else:
            match = re.search(r'[0-9]+(\.[0-9]+)+', line)
            match_num = 0
        if match:
            version = match.group(0).split()[match_num]
            print('clang version = ',version)
            # Get the version up to the first decimal
            # e.g. for 3.4.1 we only keep 3.4
            vnum = version[0:version.find('.')+2]
            if vnum >= '3.7':
                return 'clang w/ OpenMP'
        return 'clang'
    elif 'gcc' in line:
        return 'gcc'
    elif 'GCC' in line:
        return 'gcc'
    elif 'clang' in cc:
        return 'clang'
    elif 'gcc' in cc or 'g++' in cc:
        return 'gcc'
    elif 'icc' in cc or 'icpc' in cc:
        return 'icc'
    else:
        # OK, the main thing we need to know is what openmp flag we need for this compiler,
        # so let's just try the various options and see what works.  Don't try icc, since 
        # the -openmp flag there gets treated as '-o penmp' by gcc and clang, which is bad.
        # Plus, icc should be detected correctly by the above procedure anyway.
        for cc_type in ['gcc', 'clang']:
            if try_cc(cc, cc_type):
                return cc_type
        # I guess none of them worked.  Now we really do have to bail.
        return 'unknown'

def try_cc(cc, cc_type):
    """
    If cc --version is not helpful, the last resort is to try each compiler type and see
    if it works.
    """
    cpp_code = """
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include "omp.h"
#endif

int get_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int main() {
    int n = 500;
    std::vector<double> x(n,0.);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i=0; i<n; ++i) x[i] = 2*i+1;

    double sum = 0.;
    for (int i=0; i<n; ++i) sum += x[i];
    // Sum should be n^2 = 250000

    std::cout<<get_max_threads()<<"  "<<sum<<std::endl;
    return 0;
}
"""
    import tempfile
    cpp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.cpp')
    cpp_file.write(cpp_code)
    cpp_file.close()

    # Just get a named temporary file to write to:
    o_file = tempfile.NamedTemporaryFile(delete=False, suffix='.os')
    o_file.close()

    # Try compiling with the given flags
    import subprocess
    cmd = [cc] + copt[cc_type] + ['-c',cpp_file.name,'-o',o_file.name]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = p.stdout.readlines()
    p.communicate()
    if p.returncode != 0:
        os.remove(cpp_file.name)
        if os.path.exists(o_file.name): os.remove(o_file.name)
        return False

    # Another named temporary file for the executable
    exe_file = tempfile.NamedTemporaryFile(delete=False, suffix='.exe')
    exe_file.close()

    # Try linking
    cmd = [cc] + lopt[cc_type] + [o_file.name,'-o',exe_file.name]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = p.stdout.readlines()
    p.communicate()

    if p.returncode and cc == 'cc':
        # The linker needs to be a c++ linker, which isn't 'cc'.  However, I couldn't figure
        # out how to get setup.py to tell me the actual command to use for linking.  All the
        # executables available from build_ext.compiler.executables are 'cc', not 'c++'.
        # I think this must be related to the bugs about not handling c++ correctly.
        #    http://bugs.python.org/issue9031
        #    http://bugs.python.org/issue1222585
        # So just switch it manually and see if that works.
        cmd = ['c++'] + lopt[cc_type] + [o_file.name,'-o',exe_file.name]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = p.stdout.readlines()
        p.communicate()

    # Remove the temp files
    os.remove(cpp_file.name)
    os.remove(o_file.name)
    if os.path.exists(exe_file.name): os.remove(exe_file.name)
    return p.returncode == 0


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
        if cc == comp_type:
            print('Using compiler %s'%(cc))
        else:
            print('Using compiler %s, which is %s'%(cc,comp_type))
        for e in self.extensions:
            e.extra_compile_args = copt[ comp_type ]
            e.extra_link_args = lopt[ comp_type ]
            e.include_dirs = ['include']
        build_ext.build_extensions(self)

ext=Extension("treecorr._treecorr",
              sources,
              undef_macros = undef_macros)

dependencies = ['numpy', 'six']
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
        if py_version < '3':
            dependencies += ['fitsio']
        else:
            # I'm getting errors with fitsio in python 3, so use astropy instead for that.
            # Revisit at some point to see if this gets fixed, since fitsio is the better package.
            dependencies += ['astropy']

with open('README.rst') as file:
    long_description = file.read()

setup(name="TreeCorr", 
      version="3.2.0",
      author="Mike Jarvis",
      author_email="michael@jarvis.net",
      description="Python module for computing 2-point correlation functions",
      long_description=long_description,
      license = "BSD License",
      url="https://github.com/rmjarvis/TreeCorr",
      download_url="https://github.com/rmjarvis/TreeCorr/releases/tag/v3.2.0.zip",
      packages=['treecorr'],
      ext_modules=[ext],
      install_requires=dependencies,
      cmdclass = {'build_ext': my_builder },
      scripts=scripts)

