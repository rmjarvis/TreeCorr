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

    pip install treecorr

If you have previously installed TreeCorr, and want to upgrade to a new
released version, you should do::

    pip install treecorr --upgrade

Depending on the write permissions of the python distribution for your specific
system, you might need to use one of the following variants::

    sudo pip install treecorr
    pip install treecorr --user

The latter installs the Python module into ``~/.local/lib/python3.7/site-packages``,
which is normally already in your PYTHONPATH, but it puts the executables
``corr2`` and ``corr3`` into ``~/.local/bin`` which is probably not in your PATH.
To use these scripts, you should add this directory to your PATH.  If you would
rather install into a different prefix rather than ~/.local, you can use::

    pip install treecorr --install-option="--prefix=PREFIX"

This would install the executables into ``PREFIX/bin`` and the Python module
into ``PREFIX/lib/python3.7/site-packages``.


If you would rather download the tarball and install TreeCorr yourself,
that is also relatively straightforward:

1. Install dependencies
^^^^^^^^^^^^^^^^^^^^^^^

   All required dependencies should be installed automatically for you by
   setup.py, so you should not need to worry about these.  But if you are
   interested, the dependencies are:

    - numpy
    - pyyaml
    - LSSTDESC.Coord
    - cffi

   They can all be installed at once by running::

        pip install -r requirements.txt

   The last dependency is the only one that typically could cause any problems, since it in
   turn depends on a library called libffi.  This is a common thing to have installed already
   on linux machines, so it is likely that you won't have any trouble with it, but if you get
   errors about "ffi.h" not being found, then you may need to either install it yourself or
   update your paths to include the directory where ffi.h is found.

   See https://cffi.readthedocs.io/en/latest/installation.html for more information about
   installing cffi, including its libffi dependency.


   .. note::

        Two additional modules are not required for basic TreeCorr operations, but are
        potentially useful.

        a) fitsio is required for reading FITS catalogs or writing to FITS output files.

        b) pandas will signficantly speed up reading from ASCII catalogs.

        These are both pip installable::

            pip install fitsio
            pip install pandas

        But they are not installed with TreeCorr automatically.


2. Download TreeCorr
^^^^^^^^^^^^^^^^^^^^

   You can download the latest tarball from::

        https://github.com/rmjarvis/TreeCorr/releases/

   Or you can clone the repository using either of the following::

        git clone git@github.com:GalSim-developers/GalSim.git
        git clone https://github.com/GalSim-developers/GalSim.git

   which will start out in the current stable release branch.

   Either way, cd into the TreeCorr directory.


3. Install
^^^^^^^^^^

   You can then install TreeCorr in the normal way with setup.py.  Typically this would be the
   command::

        python setup.py install

   If you don't have write permission in your python distribution, you might need
   to use::

        python setup.py install --user

   In addition to installing the Python module ``treecorr``, this will install
   the executables ``corr2`` and ``corr3`` in a ``bin`` folder somewhere on your
   system.  Look for a line like::

        Installing corr2 script to /anaconda3/bin

   or similar in the output to see where the scripts are installed.  If the
   directory is not in your path, you will also get a warning message at the
   end letting you know which directory you should add to your path if you want
   to run these scripts.


4. Run Tests (optional)
^^^^^^^^^^^^^^^^^^^^^^^

   If you want to run the unit tests, you can do the following::

        pip install -r test_requirements.txt
        cd tests
        nosetests



Two-point Correlations
----------------------

This software is able to compute a variety of two-point correlations:

:NN:  The normal two-point correlation function of number counts (typically
      galaxy counts).

:GG:  Two-point shear-shear correlation function.

:KK:  Nominally the two-point kappa-kappa correlation function, although any
      scalar quantity can be used as "kappa".  In lensing, kappa is the
      convergence, but this could be used for temperature, size, etc.

:NG:  Cross-correlation of counts with shear.  This is what is often called
      galaxy-galaxy lensing.

:NK:  Cross-correlation of counts with kappa.  Again, "kappa" here can be any scalar
      quantity.

:KG:  Cross-correlation of convergence with shear.  Like the NG calculation, but
      weighting the pairs by the kappa values the foreground points.

See `Two-point Correlation Functions
<https://rmjarvis.github.io/TreeCorr/_build/html/correlation2.html>`_ for more details.

Three-point Correlations
------------------------

This software is not yet able to compute three-point cross-correlations, so the
only avaiable three-point correlations are:

:NNN: Three-point correlation function of number counts.

:GGG: Three-point shear correlation function.  We use the "natural components"
      called Gamma, described by Schneider & Lombardi (2003) (Astron.Astrophys.
      397, 809) using the triangle centroid as the reference point.

:KKK: Three-point kappa correlation function.  Again, "kappa" here can be any
      scalar quantity.

See `Three-point Correlation Functions
<https://rmjarvis.github.io/TreeCorr/_build/html/correlation3.html>`_ for more details.

Running corr2 and corr3
-----------------------

The executables corr2 and corr3 each take one required command-line argument,
which is the name of a configuration file::

    corr2 config_file
    corr3 config_file

A sample configuration file for corr2 is provided, called sample.params.
See `Configuration Parameters <https://rmjarvis.github.io/TreeCorr/_build/html/params.html>`_
for the complete documentation about the allowed parameters.

You can also specify parameters on the command line after the name of
the configuration file. e.g.::

    corr2 config_file file_name=file1.dat gg_file_name=file1.out
    corr2 config_file file_name=file2.dat gg_file_name=file2.out
    ...

This can be useful when running the program from a script for lots of input
files.

See `Using configuration files <https://rmjarvis.github.io/TreeCorr/_build/html/scripts.html>`_
for more details.

Using the Python module
-----------------------

The typical usage in python is in three stages:

1. Define one or more Catalogs with the input data to be correlated.
2. Define the correlation function that you want to perform on those data.
3. Run the correlation by calling ``process``.
4. Maybe write the results to a file or use them in some way.

For instance, computing a shear-shear correlation from an input file stored
in a fits file would look something like the following::

    >>> import treecorr
    >>> cat = treecorr.Catalog('cat.fits', ra_col='RA', dec_col='DEC',
    ...                        ra_units='degrees', dec_units='degrees',
    ...                        g1_col='GAMMA1', g2_col='GAMMA2')
    >>> gg = treecorr.GGCorrelation(min_sep=1., max_sep=100., bin_size=0.1,
    ...                             sep_units='arcmin')
    >>> gg.process(cat)
    >>> xip = gg.xip  # The xi_plus correlation function
    >>> xim = gg.xim  # The xi_minus correlation function

For more involved worked examples, see our `Jupyter notebook tutorial
<https://github.com/rmjarvis/TreeCorr/blob/master/tests/Tutorial.ipynb>`_.

And for the complete details about all aspects of the code, see the `Sphinx-generated
documentation <http://rmjarvis.github.io/TreeCorr>`_.

If you are used to using ``corr2`` with a configuration file,
and want to learn how to do the same thing in pythonn, there is also a
`guide <https://rmjarvis.github.io/TreeCorr/_build/html/guide.html>`_
to migrating over.


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
