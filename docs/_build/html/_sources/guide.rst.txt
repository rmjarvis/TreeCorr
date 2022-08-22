Getting Started Guide
---------------------

Jupyter Tutorial
^^^^^^^^^^^^^^^^

The below page covers many of the same points as the
`Jupyter notebook tutorial <https://github.com/rmjarvis/TreeCorr/blob/main/tests/Tutorial.ipynb>`_
available in the TreeCorr repo.
You may find it useful to work through that as well as, or instead of, reading this guide.

Shear-shear auto-correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's start with how to calculate a shear-shear two-point auto-correlation.
It's not necessarily the simplest choice of correlation, but this specific
calculation was the original reason I wrote TreeCorr, so it's close to my heart.
The basic pattern is as follows::

    cat = treecorr.Catalog(file_name, config)
    gg = treecorr.GGCorrelation(config)
    gg.process(cat)
    gg.write(out_file_name)

Here ``file_name`` is the name of some input file, which has the shear and position
data of your galaxies.  ``config`` is a dictionary with all the configuration
parameters about how to load the data and define the binning.  We'll expand that
out shortly.  Finally, ``out_file_name`` is some output file to write the results.

You can do a cross-correlation between two sets of galaxies very similarly::

    cat1 = treecorr.Catalog(file_name1, config1)
    cat2 = treecorr.Catalog(file_name2, config2)
    gg.process(cat1, cat2)

If you would rather not write the results to an output file, but maybe plot them up or do some
further calculation with them, you can access the resulting fields directly as numpy arrays::

    xip = gg.xip            # The real part of xi+
    xim = gg.xim            # The real part of xi-
    logr = gg.logr          # The nominal center of each bin
    meanlogr = gg.meanlogr  # The mean <log(r)> within the bins
    varxi = gg.varxi        # The variance of each xi+ or xi- value
                            # taking into account shape noise only

See the doc string for `GGCorrelation` for other available attributes.

Other Two-point Correlation Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The other kinds of correlations each have their own class:

    - `NNCorrelation` = count-count  (normal LSS correlation)
    - `GGCorrelation` = shear-shear  (e.g. cosmic shear)
    - `KKCorrelation` = kappa-kappa  (or any other scalar field)
    - `NGCorrelation` = count-shear  (i.e. <gamma_t>(R))
    - `NKCorrelation` = count-kappa  (i.e. <kappa>(R))
    - `KGCorrelation` = kappa-shear

You should see their doc strings for details, but they all work similarly.
For the last three, there is no auto-correlation option, of course, just the cross-correlation.

The other main difference between these other correlation classes from GG is that there is only a
single correlation function, so it is called ``xi`` rather than ``xip`` and ``xim``.

Also, NN does not have any kind of ``xi`` attribute.  You need to perform an additional
calculation involving random catalogs for that.
See `Using random catalogs` below for more details.


Loading a Catalog
^^^^^^^^^^^^^^^^^

OK, now let's get into some of the details about how to load data into a `Catalog`.

To specify the names of the columns in the input file, as well as other details about
how to interpret the columns, you can either use a ``config`` dict, as we did above,
or specify keyword arguments.  Either way is fine, although to be honest, the keywords
are probably more typical, so we'll use that from here on.

For a shear catalog, you need to specify the position of each galaxy and the
shear values, g1 and g2.  You do this by stating which column in the input catalog
corresponds to each value you need.  For example::

    cat = treecorr.Catalog(file_name='input_cat.fits',
                           x_col='X_IMAGE', y_col='Y_IMAGE', g1_col='E1', g2_col='E2')

For FITS files, you specify the columns by name, which correspond to the column name
in the FITS table.  For ASCII input files, you specify the column number instead::

    cat = treecorr.Catalog(file_name='input_cat.dat',
                           x_col=2, y_col=3, g1_col=5, g2_col=6)

where the first column in numbered 1, not 0.

When the positions are given as right ascension and declination on the celestial
sphere, rather than x and y on a flat projection (like an image), you also need
to specify what units the angles use::

    cat = treecorr.Catalog(file_name='input_cat.fits',
                           ra_col='RA', dec_col='DEC', g1_col='E1', g2_col='E2',
                           ra_units='hours', dec_units='degrees')

For the catalog of the N part of a calculation, you can skip the ``g1_col`` and ``g2_col``.
Those only need positions. For a K correlation, you should specify ``k_col`` instead::

    cat = treecorr.Catalog(file_name='input_cat.fits',
                           ra_col='RA', dec_col='DEC', k_col='KAPPA',
                           ra_units='hours', dec_units='degrees')

See the documentation for `Catalog` for more options, such as how to flip the sign of
g1 or g2 (unfortunately not everyone follows the same conventions), use weights,
skip objects with specific flags, and more.


Building a Catalog from numpy arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the provided tools for reading in the data from an input file are insufficient, or if
the data you want to use are being generated in Python natively, so there is no file
to read, then you can instead build the `Catalog` directly from numpy arrays::

    x = numpy.array(x_values)    # These might be the output of
    y = numpy.array(y_values)    # some calculation...
    g1 = numpy.array(g1_values)
    g2 = numpy.array(g2_values)

    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)

You always need to include either ``x`` and ``y`` or ``ra`` and ``dec``.
Which other columns you need depends on what kind of correlation function you want to calculate
from the data.  For GG, you need ``g1`` and ``g2``, but for K correlations, you would use
``k`` instead.

You can optionally provide a weight column as well with ``w`` if desired.
This will then perform a weighted correlation using those weights.

Again, see the doc string for `Catalog` for more information.


Defining the binning
^^^^^^^^^^^^^^^^^^^^

For the default `bin_type <Binning>`, ``"Log"``, the correlation function is binned
in equally spaced bins in :math:`\log(r)`.  where :math:`r` represents the separation
between two points being correlated.

Typically you would specify the minimum and
maximum separation you want accumulated as ``min_sep`` and ``max_sep`` respectively,
along with ``nbins`` to specify how many bins to use::

    gg = treecorr.GGCorrelation(min_sep=1., max_sep=100., nbins=10)

When the positions are given as (ra, dec), then the separations are also angles,
so you need to specify what units to use. These do not have to be the same units
as you used for either ra or dec::

    gg = treecorr.GGCorrelation(min_sep=1., max_sep=100., nbins=10, sep_units='arcmin')

Most correlation functions of interest in astronomy are roughly power laws, so log
binning puts similar signal-to-noise in each bin, making it often a good choice.
However, for some use cases, linear binning is more appropriate.  This is possible
using the ``bin_type`` parameter::

    gg = treecorr.GGCorrelation(min_sep=10., max_sep=15., nbins=5, bin_type='Linear')

See `Binning` for more details about this option and the ``"TwoD"`` binning,
as well as some other options related to binning.

Finally, the default way of calculating separations is a normal Euclidean metric.
However, TreeCorr implements a number of other metrics as well, which are useful
in various situations.  See `Metrics` for details.

Three-point Correlation Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TreeCorr can also do three-point correlations, to measure how the product of three fields
depends on the size and shape of the triangle connecting three points.
So far, we have only implemented the auto-correlation three-point functions:

    - `NNNCorrelation`  # count-count-count
    - `GGGCorrelation`  # shear-shear-shear
    - `KKKCorrelation`  # kappa-kappa-kappa

These classes are significantly more complicated than the two-point ones,
since they have to deal with the geometry of the triangles being binned.
See their doc strings for more details.


Using random catalogs
^^^^^^^^^^^^^^^^^^^^^

For the NN and NNN correlations, the raw calculation is not sufficient to produce the real
correlation function.  You also need to account for the survey geometry (edges, mask, etc.)
by running the same calculation with a random catalog (or several) that have a uniform density,
but the same geometry::

    data = treecorr.Catalog(data_file, config)
    rand = treecorr.Catalog(rand_file, config)
    dd = treecorr.NNCorrelation(config)
    dr = treecorr.NNCorrelation(config)
    rr = treecorr.NNCorrelation(config)
    dd.process(data)
    dr.process(data,rand)
    rr.process(rand)
    xi, varxi = dd.calculateXi(rr,dr)


This calculates xi = (DD-2DR+RR)/RR for each bin.  This is the Landy-Szalay estimator,
which is the most widely used estimator for count-count correlation functions.  However,
if you want to use a simpler estimator xi = (DD-RR)/RR, then you can omit the dr parameter.
The simpler estimator is slightly biased though, so this is not recommended.

After calling `calculateXi <NNCorrelation.calculateXi>`, the ``dd`` object above will have ``xi``
and ``varxi`` attributes, which store the results of this calculation.

The NG and NK classes also have a `calculateXi <NGCorrelation.calculateXi>` method to allow
for the use of compensated estimators in those cases as well.
Calling this function updates the ``xi`` attribute from the uncompensated value to the
compensated value.
These correlations do not suffer as much from masking effects,
so the compensation is not as necessary.  However, it does produce a slightly better estimate
of the correlation function if you are able to use a random catalog.

Furthermore, the `process <GGCorrelation.process>` functions can take lists of Catalogs if desired,
in which case it will
do all the possible combinations.  This is especially relevant for doing randoms,
since the statistics get better if you generate several randoms and do all the correlations to beat down the noise::

    rand_list = [ treecorr.Catalog(f,config) for f in rand_files ]
    dr.process(data, rand_list)
    rr.process(rand_list)

The corresponding three-point NNN calculation is even more complicated, since there are 8 total
combinations that need to be computed: zeta = (DDD-DDR-DRD-RDD+DRR+RDR+RRD-RRR)/RRR.
Because of the triangle geometry, we don't have DRR = DRD = RDD, so all 8 need to be computed.
See the docstring for `calculateZeta` for more details.

Manually accumulating the correlation function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For even more control over the calculation, you can break up the steps in the
`process <GGCorrelation.process>` functions.  There are typically three steps:

1. Calculate the shear variance or kappa variance as needed (i.e. for anything but NN correlations).
2. Accumulate the correlations into the bins for each auto-correlation and cross-correlation desired.
3. Finalize the calculation.

If you have several pairs of catalogs that you want to accumulate into a single correlation
function, you could write the following::

    lens_cats = [ treecorr.Catalog(f,config) for f in lens_files ]
    source_cats = [ treecorr.Catalog(f,config) for f in source_files ]
    ng = treecorr.NGCorrelation(config)
    varg = treecorr.calculateVarG(source_cats)
    for c1, c2 in zip(lens_cats, source_cats):
        ng.process_cross(c1,c2)
    ng.finalize(varg)

In addition to `process_cross <NGCorrelation.process_cross>`,
classes that allow auto-correlations have a
`process_auto <GGCorrelation.process_auto>` method for manually processing
auto-correlations.  See the doc strings for these methods for more information.

Breaking up the calculation manually like this is probably not often necessary anymore.
It used to be useful for dividing a calculation among several machines, which would
each save their results to disk.  These results could then be reassembled and
finalized after all the results were finished.

However, this work mode is now incorporated directly into TreeCorr via the use of
"patches".  See `Patches` for details about how to automatically
divide up your input catalog into patches and to farm the calculation out to
multiple machines using MPI.
