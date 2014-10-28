
TreeCorr is a package for efficiently computing 2-point correlation functions.

- The code is hosted at https://github.com/rmjarvis/TreeCorr
- It can compute correlations of regular number counts, weak lensing shears, or
  scalar quantities such as convergence or CMB temperature fluctutations.
- 2-point correlations may be auto-correlations or cross-correlations.  This
  includes shear-shear, count-shear, count-count, kappa-kappa, etc.  (Any
  combination of shear, kappa, and counts.)
- 2-point functions can be done with correct curved-sky calculation using
  RA, Dec coordinates, on a Euclidean tangent plane, or in 3D using RA, Dec
  and a distance.
- The front end is in Python, which can be used as a Python module or as a 
  standalone executable using configuration files.
- The actual computation of the correlation functions is done in C++ using ball
  trees (similar to kd trees), which make the calculation extremely
  efficient.
- When available, OpenMP is used to run in parallel on multi-core machines.
- Approximate running time for 2-point shear-shear is ~30 sec * (N/10^6) / core
  for a bin size of 0.1 in log(r).  It scales as b^(-2).  This is the slowest
  of the various kinds of correlations, so others will be a bit faster, but
  with the same scaling with N and b.
- 3-point functions have not yet been migrated to the new API, but they should
  be available soon.
- Reference: Jarvis, Bernstein, & Jain, 2004, MNRAS, 352, 338

The code is licensed under a FreeBSD license.  Essentially, you can use the 
code in any way you want, but if you distribute it, you need to include the 
file `TreeCorr_LICENSE` with the distribution.  See that file for details.


Installation
------------

The easiest way to install TreeCorr is with pip::

    sudo pip install TreeCorr

If you have previously installed TreeCorr, and want to upgrade to a new
released version, you should do::

    sudo pip install TreeCorr --upgrade

To install TreeCorr on a system where you do not have sudo privileges,
you can do::

    pip install TreeCorr --install-option="--prefix=~"

NB: There is also a `--user` option with `pip install`, which installs into
~/.local.  This is fine for the python module, but it puts the `corr2`
executable into ~/.local/bin, which is probably not in your path.  The above
command will instead install `corr2` into ~/bin.


If you would rather download the tarball and install TreeCorr yourself,
that is also relatively straightforward:

1. Dependencies: All dependencies should be installed automatically for you by
   setup.py, so you should not need to worry about these.  But if you are
   interested, the dependencies are:

    - numpy
    - fitsio: for reading/writing FITS files
    - pandas: for reading/writing ASCII files. This package significantly
      speeds up the reading of ASCII input catalogs over the numpy functions
      loadtxt or genfromtxt.

2. Download the zip file or tarball of the current code from:

   https://github.com/rmjarvis/TreeCorr/releases/tag/v3.1.0

3. Unzip the archive with either of the following (depending on which kind
   of archive you downloaded)::

        unzip TreeCorr-3.1.0.zip
        tar xvzf TreeCorr-3.1.0.tar.gz

   It will unzip into the directory TreeCorr-3.1.0. Change to that directory::

        cd TreeCorr-3.1.0

4. Install with the normal setup.py options.  Typically this would be the
   command::

        python setup.py install --prefix=/your/home/directory

   This will install the executable `corr2` at::

        /your/home/directory/bin/corr2

   It will also install the Python module called treecorr which you can use
   from within Python.

   .. note::

        There is a bug with numpy that it sometimes doesn't install correctly
        when included as a setup.py dependency:
            https://github.com/numpy/numpy/issues/1458  
        The bug was marked closed in 2012, but I've gotten it with the latest
        numpy version 1.8.2.  Installation failed with a traceback that ended
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

   They do take a bit of time to run, since I use around 1 million galaxies
   for many of the tests.  On the order of 5-10 minutes when using a single
   core, or less when OpenMP is enabled.


Two-point Correlations
----------------------

This software is able to compute several varieties of two-point correlations:

:NN:  the normal two point correlation function of things like 2dF that
      correlate the galaxy counts at each position.

:NG:  correlation of counts with shear.  This is what is often called
      galaxy-galaxy lensing.

:GG:  two-point shear correlation function.

:NK:  correlation of counts with kappa.  While kappa is nominally the lensing
      convergence, it could really be any scalar quantity, like temperature,
      size, etc.

:KG:  correlation of convergence with shear.  Like the NG calculation, but 
      weighting the pairs by the convergence values the foreground points.

:KK:  two-point kappa correlation function.


Running corr2
-------------

The executable corr2 takes one required command-line argument, which is the 
name of a configuration file::

    corr2 config_file

A sample configuration file is provided, called sample.params.  See the
TreeCorr wiki page

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

The same functionality can be achieved from within Python using a Python dict
for the configuration parameters::

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
Python.  For more information, see the wiki page:

https://github.com/rmjarvis/TreeCorr/wiki/Guide-to-using-TreeCorr-in-Python

Sphinx documentation based on the doc strings can be found at:

http://rmjarvis.github.io/TreeCorr/html/index.html


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


