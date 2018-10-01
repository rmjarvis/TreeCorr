from __future__ import print_function
import sys,os,glob,re
import select


try:
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
    from setuptools.command.install_scripts import install_scripts
    from setuptools.command.easy_install import easy_install
    import setuptools
    print("Using setuptools version",setuptools.__version__)
except ImportError:
    print('Unable to import setuptools.  Using distutils instead.')
    from distutils.core import setup, Extension
    from distutils.command.build_ext import build_ext
    from distutils.command.install_scripts import install_scripts
    easy_install = object  # Prevent error when using as base class
    import distutils
    # cf. http://stackoverflow.com/questions/1612733/including-non-python-files-with-setup-py
    from distutils.command.install import INSTALL_SCHEMES
    for scheme in INSTALL_SCHEMES.values():
        scheme['data'] = scheme['purelib']
    print("Using distutils version",distutils.__version__)


#from distutils.command.install_headers import install_headers 

try:
    from sysconfig import get_config_vars
except:
    from distutils.sysconfig import get_config_vars

print('Python version = ',sys.version)
py_version = "%d.%d"%sys.version_info[0:2]  # we check things based on the major.minor version.

scripts = ['corr2', 'corr3']
scripts = [ os.path.join('scripts',f) for f in scripts ]

sources = glob.glob(os.path.join('src','*.cpp'))

headers = glob.glob(os.path.join('include','*.h'))

undef_macros = []

# If we build with debug, also undefine NDEBUG flag
if "--debug" in sys.argv:
    undef_macros+=['NDEBUG']

copt =  {
    'gcc' : ['-fopenmp','-O3','-ffast-math'],
    'icc' : ['-openmp','-O3'],
    'clang' : ['-O3','-ffast-math'],
    #'clang w/ OpenMP' : ['-fopenmp=libomp','-O3','-ffast-math'],
    'clang w/ OpenMP' : ['-fopenmp','-O3','-ffast-math'],
    'unknown' : [],
}
lopt =  {
    'gcc' : ['-fopenmp'],
    'icc' : ['-openmp'],
    'clang' : [],
    'clang w/ OpenMP' : ['-fopenmp'],
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
        print(line.decode().strip())
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

    if 'clang' in line:
        # clang 3.7 is the first with openmp support.  But Apple lies about the version
        # number of clang, so the most reliable thing to do is to just try the compilation
        # with the openmp flag and see if it works.
        if try_cc(cc, 'clang w/ OpenMP'):
            return 'clang w/ OpenMP'
        else:
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
    cpp_file.write(cpp_code.encode())
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
        if os.path.exists(o_file.name):
            os.remove(o_file.name)
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
    if os.path.exists(exe_file.name):
        os.remove(exe_file.name)
    return p.returncode == 0

def check_ffi_compile(cc, cc_type):
    ffi_code = """
#include "ffi.h"
int main() {
    return 0;
}
"""
    import tempfile
    ffi_file = tempfile.NamedTemporaryFile(delete=False, suffix='.c')
    ffi_file.write(ffi_code.encode())
    ffi_file.close()

    # Just get a named temporary file to write to:
    o_file = tempfile.NamedTemporaryFile(delete=False, suffix='.os')
    o_file.close()

    # Try compiling with the given flags
    import subprocess
    cmd = [cc] + copt[cc_type] + ['-c',ffi_file.name,'-o',o_file.name]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = p.stdout.readlines()
    p.communicate()
    if p.returncode != 0:
        print('Unable to compile file with #include "ffi.h"')
        print("Failed command is: ",' '.join(cmd))
        # Try ffi/ffi.h
        ffi_code = """
#include "ffi/ffi.h"
int main() {
    return 0;
}
"""
        ffi_file = open(ffi_file.name, ffi_file.mode)
        ffi_file.write(ffi_code.encode())
        ffi_file.close()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = p.stdout.readlines()
        p.communicate()
        if p.returncode != 0:
            print('Unable to compile file with #include "ffi/ffi.h"')
            print("Failed command is: ",' '.join(cmd))
            print("Could not find ffi.h")
            return False
        else:
            print("Found ffi/ffi.h")
    else:
        print("Found ffi.h")

    # Another named temporary file for the executable
    exe_file = tempfile.NamedTemporaryFile(delete=False, suffix='.exe')
    exe_file.close()

    # Try linking
    cmd = [cc] + lopt[cc_type] + ['-lffi'] + [o_file.name,'-o',exe_file.name]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = p.stdout.readlines()
    p.communicate()

    if p.returncode != 0:
        print("Could not link with -lffi")
        print("Failed command is: ",' '.join(cmd))
        return False
    else:
        print("Successfully linked with -lffi")

    # Remove the temp files only if all succeeded.
    os.remove(ffi_file.name)
    os.remove(o_file.name)
    if os.path.exists(exe_file.name): 
        os.remove(exe_file.name)

    return True

# Based on recipe 577058: http://code.activestate.com/recipes/577058/
def query_yes_no(question, default="yes", timeout=30):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes":"yes",   "y":"yes",  "ye":"yes",
             "no":"no",     "n":"no"}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while 1:
        sys.stdout.write(question + prompt)
        sys.stdout.flush()
        i, _, _ = select.select( [sys.stdin], [], [], timeout )

        if i:
            choice = sys.stdin.readline().strip()
        else:
            sys.stdout.write("\nPrompt timed out after %s seconds.\n"%timeout)
            return default

        if default is not None and choice == '':
            return default
        elif choice in valid.keys():
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def check_ffi(cc, cc_type):
    try:
        raise ImportError
        import cffi
    except ImportError:
        # Then cffi will need to be installed.
        # It requires libffi, so check if it is available.
        if check_ffi_compile(cc, cc_type):
            return
        # libffi needs to be installed.  Give a helpful message about how to do so.
        prefix = '/SOME/APPROPRIATE/PREFIX'
        prefix_param = [param for param in sys.argv if param.startswith('--prefix=')]
        if len(prefix_param) == 1:
            prefix = prefix_param[0].split('=')[1]
            prefix = os.path.expanduser(prefix)
        msg = """
WARNING: TreeCorr uses cffi, which in turn requires libffi to be installed.
         As the latter is not a python package, pip cannot download and
         install it.  However, it is fairly straightforward to install.

On Linux, you can use one of the following:

    apt-get install libffi-dev
    yum install libffi-devel

On a Mac, it should be available after you do:

    xcode-select --install

If neither of those work for you, you can install it yourself with the
following commands:

    wget ftp://sourceware.org:/pub/libffi/libffi-3.2.1.tar.gz
    tar xfz libffi-3.2.1.tar.gz
    cd libffi-3.2.1
    ./configure --prefix={0}
    make
    make install
    cp */include/ffi*.h {0}/include
    cd ..

If you have already done this, then check the command (given above) that failed.  You may
need to add a directory to either C_INCLUDE_PATH, LIBRARY_PATH, or LD_LIBRARY_PATH to
make it succeed.
""".format(prefix)
        print(msg)
        q = "Stop the installation here to take care of this?"
        yn = query_yes_no(q, default='yes')
        if yn == 'yes':
            sys.exit(1)

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
        # Figure out what compiler it will use
        cc = self.compiler.compiler_so[0]
        comp_type = get_compiler(cc)
        if cc == comp_type:
            print('Using compiler %s'%(cc))
        else:
            print('Using compiler %s, which is %s'%(cc,comp_type))
        # Add the appropriate extra flags for that compiler.
        for e in self.extensions:
            e.extra_compile_args = copt[ comp_type ]
            e.extra_link_args = lopt[ comp_type ]
            e.include_dirs = ['include']
        check_ffi(cc,comp_type)
        # Now run the normal build function.
        build_ext.build_extensions(self)

# AFAICT, setuptools doesn't provide any easy access to the final installation location of the
# executable scripts.  This bit is just to save the value of script_dir so I can use it later.
# cf. http://stackoverflow.com/questions/12975540/correct-way-to-find-scripts-directory-from-setup-py-in-python-distutils/
class my_easy_install( easy_install ):  # For setuptools

    # Match the call signature of the easy_install version.
    def write_script(self, script_name, contents, mode="t", *ignored):
        # Run the normal version
        easy_install.write_script(self, script_name, contents, mode, *ignored)
        # Save the script install directory in the distribution object.
        # This is the same thing that is returned by the setup function.
        self.distribution.script_install_dir = self.script_dir

# For distutils, the appropriate thing is the install_scripts command class, not easy_install.
# So here is the appropriate thing in that case.
class my_install_scripts( install_scripts ):  # For distutils
    def run(self):
        install_scripts.run(self)
        self.distribution.script_install_dir = self.install_dir

ext=Extension("treecorr._treecorr",
              sources,
              depends=headers,
              undef_macros = undef_macros)

dependencies = ['numpy', 'future', 'cffi', 'fitsio', 'pyyaml']
if py_version <= '2.6':
    dependencies += ['argparse'] # These seem to have conflicting numpy requirements, so don't
                                 # include pandas with argparse.
else:
    dependencies += ['pandas'] 

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

dist = setup(name="TreeCorr", 
      version=treecorr_version,
      author="Mike Jarvis",
      author_email="michael@jarvis.net",
      description="Python module for computing 2-point correlation functions",
      long_description=long_description,
      license = "BSD License",
      url="https://github.com/rmjarvis/TreeCorr",
      download_url="https://github.com/rmjarvis/TreeCorr/releases/tag/v%s.zip"%treecorr_version,
      packages=['treecorr'],
      package_data={'treecorr' : headers },
      ext_modules=[ext],
      install_requires=dependencies,
      cmdclass = {'build_ext': my_builder,
                  'install_scripts': my_install_scripts,
                  'easy_install': my_easy_install,
                  },
      scripts=scripts)

# I don't actually need these installed for TreeCorr, but I wanted to figure out how to do
# it, so I played with it here.  distutils installs these automatically when the headers argument
# is given to setup.  But setuptools doesn't.  cf. http://bugs.python.org/setuptools/issue142
#cmd = install_headers(dist)
#cmd.finalize_options()
#print('Installing headers to ',cmd.install_dir)
#cmd.run()

# Check that the path includes the directory where the scripts are installed.
if (hasattr(dist,'script_install_dir') and
    dist.script_install_dir not in os.environ['PATH'].split(':')):
    print('\nWARNING: The TreeCorr executables were installed in a directory not in your PATH')
    print('         If you want to use the executables, you should add the directory')
    print('\n             ',dist.script_install_dir,'\n')
    print('         to your path.  The current path is')
    print('\n             ',os.environ['PATH'],'\n')
    print('         Alternatively, you can specify a different prefix with --prefix=PREFIX,')
    print('         in which case the scripts will be installed in PREFIX/bin.')
    print('         If you are installing via pip use --install-option="--prefix=PREFIX"')
