.. image:: https://travis-ci.org/rmjarvis/TreeCorr.svg?branch=master
        :target: https://travis-ci.org/rmjarvis/TreeCorr
.. image:: https://codecov.io/gh/rmjarvis/TreeCorr/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/rmjarvis/TreeCorr


TreeCorr is a package for efficiently computing 2-point and 3-point correlation
functions.

- The code is hosted at https://github.com/rmjarvis/TreeCorr
- It can compute correlations of regular number counts, weak lensing shears, or
  scalar quantities such as convergence or CMB temperature fluctutations.
- 2-point correlations may be auto-correlations or cross-correlations.  This
  includes shear-shear, count-shear, count-count, kappa-kappa, etc.  (Any
  combination of shear, kappa, and counts.)
- 3-point correlations currently can only be auto-correlations.  This includes
  shear-shear-shear, count-count-count, and kappa-kappa-kappa.  The cross
  varieties are planned to be added in the near future.
- Both 2- and 3-point functions can be done with the correct curved-sky 
  calculation using RA, Dec coordinates, on a Euclidean tangent plane, or in
  3D using either (RA,Dec,r) or (x,y,z) positions.
- The front end is in Python, which can be used as a Python module or as a 
  standalone executable using configuration files. (The executable is corr2
  for 2-point and corr3 for 3-point.)
- The actual computation of the correlation functions is done in C++ using ball
  trees (similar to kd trees), which make the calculation extremely efficient.
- When available, OpenMP is used to run in parallel on multi-core machines.
- Approximate running time for 2-point shear-shear is ~30 sec * (N/10^6) / core
  for a bin size b=0.1 in log(r).  It scales as b^(-2).  This is the slowest
  of the various kinds of 2-point correlations, so others will be a bit faster,
  but with the same scaling with N and b.
- The running time for 3-point functions are highly variable depending on the 
  range of triangle geometries you are calculating.  They are significantly
  slower than the 2-point functions, but many orders of magnitude faster than
  brute force algorithms.
- **If you use TreeCorr in published research, please reference:
  Jarvis, Bernstein, & Jain, 2004, MNRAS, 352, 338**
  (I'm working on new paper about TreeCorr, including some of the improvements
  I've made since then, but this will suffice as a reference for now.)
- Record on the Astrophyics Source Code Library: http://ascl.net/1508.007
- Developed by Mike Jarvis.  Fee free to contact me with questions or comments
  at mikejarvis17 at gmail.  Or post an issue (see below) if you have any
  problems with the code.

The code is licensed under a FreeBSD license.  Essentially, you can use the 
code in any way you want, but if you distribute it, you need to include the 
file ``TreeCorr_LICENSE`` with the distribution.  See that file for details.


Installation
------------

The easiest way to install TreeCorr is with pip::

    sudo pip install TreeCorr

If you have previously installed TreeCorr, and want to upgrade to a new
released version, you should do::

    sudo pip install TreeCorr --upgrade

To install TreeCorr on a system where you do not have sudo privileges,
you can do::

    pip install TreeCorr --user

This installs the Python module into ``~/.local/lib/python2.7/site-packages``,
which is normally already in your PYTHONPATH, but it puts the executables
``corr2`` and ``corr3`` into ``~/.local/bin`` which is probably not in your PATH.
To use these scripts, you should add this directory to your PATH.  If you would
rather install into a different prefix rather than ~/.local, you can use::

    pip install TreeCorr --install-option="--prefix=PREFIX"

This would install the executables into ``PREFIX/bin`` and the Python module
into ``PREFIX/lib/python2.7/site-packages``.


If you would rather download the tarball and install TreeCorr yourself,
that is also relatively straightforward:

1. Dependencies: All dependencies should be installed automatically for you by
   setup.py, so you should not need to worry about these.  But if you are
   interested, the dependencies are:

   - numpy
   - future
   - fitsio
   - pandas
   - pyyaml
   - cffi

   The last dependency is the only one that typically could cause any problems, since it in
   turn depends on a library called libffi.  This is a common thing to have installed already
   on linux machines, so it is likely that you won't have any trouble with it, but if you get
   errors about "ffi.h" not being found, then you may need to either install it yourself or
   update your paths to include the directory where ffi.h is found.

   a) Installing libffi on linux systems:

      If you have root access on your system, then one of the following should work to install
      libffi::

            apt-get install libffi-dev
            yum install libffi-devel

   b) Installing libffi on Mac systems:

      It should be installed as part of the XCode libraries after running::

            xcode-select --install

   c) Installing libffi manually:

      If neither of the above methods works for you, you can install it yourself with the
      following commands::

            wget ftp://sourceware.org:/pub/libffi/libffi-3.2.1.tar.gz
            tar xfz libffi-3.2.1.tar.gz
            cd libffi-3.2.1
            ./configure --prefix={prefix}
            make
            make install
            cp */include/ffi*.h {prefix}/include
            cd ..

      where {prefix} is wherever you want the code to be installed.  e.g. /home/username.

   d) Updating your system paths:

      If you used option (c) above and ``{prefix}`` is not ``/usr/local`` or similar, then you
      might need to update some system paths to let the compiler know where to look for the header
      file and library.  Similarly, if libffi was already installed, but it is in a non-standard
      location where gcc doesn't look by default, then this also applies.  (Try ``locate ffi.h``
      to see if it shows up somewhere.)

      Assuming ``ffi.h`` is in ``{prefix}/include/ffi/`` and ``libffi.so`` (or ``libffi.dylib`` on
      Macs) is in ``{prefix}/lib/``, then the following commands should be put in your ``.bashrc``
      or ``.bash_profile`` file::

            export C_INCLUDE_PATH=$C_INCLUDE_PATH:{prefix}/include
            export LIBRARY_PATH=$LIBRARY_PATH:{prefix}/lib
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{prefix}/lib

      Or if you use C shell, put the following in ``.cshrc`` or equivalent file::

            setenv C_INCLUDE_PATH "$C_INCLUDE_PATH":{prefix}/include
            setenv LIBRARY_PATH "$LIBRARY_PATH":{prefix}/lib
            setenv LD_LIBRARY_PATH "$LD_LIBRARY_PATH":{prefix}/lib


2. Download the zip file or tarball for the latest release from:

   https://github.com/rmjarvis/TreeCorr/releases/

3. Unzip the archive with either of the following (depending on which kind
   of archive you downloaded)::

        unzip TreeCorr-3.3.11.zip
        tar xvzf TreeCorr-3.3.11.tar.gz

   It will unzip into the directory TreeCorr-3.3.11. Change to that directory::

        cd TreeCorr-3.3.11

4. Install with the normal setup.py options.  Typically this would be the
   command::

        python setup.py install --prefix=~

   This will install the executable ``corr2`` at::

        /your/home/directory/bin/corr2

   It will also install the Python module called ``treecorr`` which you can use
   from within Python.

   .. note::

        There is a bug with numpy that it sometimes doesn't install correctly
        when included as a setup.py dependency:

            https://github.com/numpy/numpy/issues/1458  

        The bug was marked closed in 2012, but I've gotten it with numpy
        versions since then. Installation failed with a traceback that ended
        with::

            File "/private/tmp/easy_install-xl4gri/numpy-1.8.2/numpy/core/setup.py", line 631, in configuration

            AttributeError: 'Configuration' object has no attribute 'add_define_macros'

        The workaround if this happens for you seems to be to install numpy
        separately with::

            easy_install numpy

        Then the normal TreeCorr installation should work correctly.



5. (optional) If you want to run the unit tests, you can do the following::

        cd tests
        nosetests



Two-point Correlations
----------------------

This software is able to compute several varieties of two-point correlations:

:NN:  The normal two-point correlation function of number counts (typically
      galaxy counts).

:GG:  Two-point shear-shear correlation function.

:KK:  Nominally the two-point kappa-kappa correlation function, although any
      scalar quantity can be used as "kappa".  In lensing, kappa is the 
      convergence, but this could be used for temperature, size, etc.

:NG:  Cross-correlation of counts with shear.  This is what is often called
      galaxy-galaxy lensing.

:NK:  Cross-correlation of counts with kappa.  Again, "kappa here can be any scalar
      quantity.

:KG:  Cross-correlation of convergence with shear.  Like the NG calculation, but 
      weighting the pairs by the kappa values the foreground points.


Three-point Correlations
------------------------

This software is currently only able to compute three-point auto-correlations:

:NNN: Three-point correlation function of number counts.

:GGG: Three-point shear correlation function.  We use the "natural components"
      called Gamma, described by Schneider & Lombardi [Astron.Astrophys. 397
      (2003) 809-818] using the triangle centroid as the reference point.

:KKK: Three-point kappa correlation function.  Again, "kappa" here can be any
      scalar quantity.


Running corr2 and corr3
-----------------------

The executables corr2 and corr3 each take one required command-line argument,
which is the name of a configuration file::

    corr2 config_file
    corr3 config_file

A sample configuration file for corr2 is provided, called sample.params.  
See the TreeCorr wiki page

https://github.com/rmjarvis/TreeCorr/wiki/Configuration-Parameters

for the complete documentation about the allowed parameters.

You can also specify parameters on the command line after the name of 
the configuration file. e.g.::

    corr2 config_file file_name=file1.dat gg_file_name=file1.out
    corr2 config_file file_name=file2.dat gg_file_name=file2.out
    ...

This can be useful when running the program from a script for lots of input 
files.


Using the Python module
-----------------------

Here we only give a quick overview.  A Jupyter notebook tutorial here:

https://github.com/rmjarvis/TreeCorr/blob/master/tests/Tutorial.ipynb

goes into more detail.  And full Sphinx-generated documentation can be found at:

http://rmjarvis.github.io/TreeCorr/html/index.html

The TreeCorr module is called ``treecorr`` in Python.  Typical usage for
computing the shear-shear correlation function looks something like the
following::

    >>> import treecorr
    >>> cat = treecorr.Catalog('cat.fits', ra_col='RA', dec_col='DEC',
    ...                        ra_units='degrees', dec_units='degrees',
    ...                        g1_col='GAMMA1', g2_col='GAMMA2')
    >>> gg = treecorr.GGCorrelation(min_sep=1., max_sep=100., bin_size=0.1,
    ...                             sep_units='arcmin')
    >>> gg.process(cat)
    >>> xip = gg.xip  # The xi_plus correlation function
    >>> xim = gg.xim  # The xi_minus correlation function

The different correlation functions each have their own class.  You can 
access the Python documentation by calling help on the appropriate class
to get more details about the different kwarg options, attributes, and 
methods for each::

    >>> help(NNCorrelation)
    >>> help(GGCorrelation)
    >>> help(KKCorrelation)
    >>> help(NGCorrelation)
    >>> help(NKCorrelation)
    >>> help(KGCorrelation)
    >>> help(NNNCorrelation)
    >>> help(GGGCorrelation)
    >>> help(KKKCorrelation)

You can also leverage the configuration file apparatus from within Python
using a Python dict for the configuration parameters::

    >>> import treecorr
    >>> config = treecorr.read_config(config_file)
    >>> config['file_name'] = 'file1.dat'
    >>> config['gg_file_name'] = 'file1.out'
    >>> treecorr.corr2(config)
    >>> config['file_name'] = 'file2.dat'
    >>> config['gg_file_name'] = 'file2.out'
    >>> treecorr.corr2(config)

However, the Python module gives you much more flexibility in how to specify
the input and output, including going directly from and to numpy arrays within
Python.  For a slightly longer "Getting Started" guide see the wiki page:

https://github.com/rmjarvis/TreeCorr/wiki/Guide-to-using-TreeCorr-in-Python


Reporting bugs
--------------

If you find a bug running the code, please report it at:

https://github.com/rmjarvis/TreeCorr/issues

Click "New Issue", which will open up a form for you to fill in with the
details of the problem you are having.


Requesting features
-------------------

If you would like to request a new feature, do the same thing.  Open a new
issue and fill in the details of the feature you would like added to TreeCorr.
Or if there is already an issue for your desired feature, please add to the 
discussion, describing your use case.  The more people who say they want a
feature, the more likely I am to get around to it sooner than later.


