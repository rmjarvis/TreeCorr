Getting Started Guide
---------------------

The below guide has largely been superseded by a
`Jupyter notebook tutorial <https://github.com/rmjarvis/TreeCorr/blob/master/tests/Tutorial.ipynb>`_.
That probably provides a better starting point for most users.  The below guide was mostly
written for people migrating from using the executable `corr2` to more advanced usage
within Python.


Mimicking the corr2 executable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same functionality that you have from the ``corr2`` executable is available in python via the
`corr2` function::

    import treecorr
    config = treecorr.read_config(config_file)
    config['file_name'] = 'catalog.dat'
    config['gg_file_name'] = 'gg.out'
    treecorr.corr2(config)

However, this isn't exactly a huge improvement over using the executable itself.
Basically, it allows for an alternative way to set the configuration parameters, but not much
beyond that.  The utility of the python module becomes more apparent if we drill down one
level and see what the `corr2` function is doing.


Shear-shear auto-correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The full general version of the function is pretty involved, since it checks for all the
different possible inputs and outputs and does the appropriate thing.  But lets distill out
just a simple shear-shear auto-correlation of a single file and see how that would work::

    cat = treecorr.Catalog(file_name, config)
    gg = treecorr.GGCorrelation(config)
    gg.process(cat)
    gg.write(out_file_name)

You can do a cross-correlation between two sets of galaxies very similarly::

    cat1 = treecorr.Catalog(file_name1, config)
    cat2 = treecorr.Catalog(file_name2, config)
    gg.process(cat1, cat2)

If you would rather not write the results to an output file, but maybe plot them up or do some further calculation with them, you can access the resulting fields directly as numpy arrays::

    xip = gg.xip            # The real part of xi+
    xim = gg.xim            # The real part of xi-
    logr = gg.logr          # The nominal center of each bin
    meanlogr = gg.meanlogr  # The mean <log(r)> within the bins
    varxi = gg.varxi        # The variance of each xi+ or xi- value
                            # taking into account shape noise only

See the doc string for `GGCorrelation` for other available attributes.

Also, anywhere that you can pass it a config dict, you can also pass the relevant parameters as
regular python kwargs.  For example::

    cat = treecorr.Catalog(file_name, ra_col='RA', dec_col='DEC',
                           ra_units='hours', dec_units='deg',
                           g1_col='E1', g2_col='E2')
    gg = treecorr.GGCorrelation(bin_size=0.1, min_sep=1, max_sep=100,
                                sep_units='arcmin', bin_slop=0.5)


Building a Catalog from numpy arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also bypass the I/O on the input side as well::

    x = numpy.array(x_values)    # These might be the output of
    y = numpy.array(y_values)    # some calculation...
    g1 = numpy.array(g1_values)
    g2 = numpy.array(g2_values)

    cat = treecorr.Catalog(x=x, y=y, g1=g1, g2=g2)

You always need to include either ``x`` and ``y`` or ``ra`` and ``dec``.
Which other columns you need depends on what kind of correlation function you want to calculate
from the data.  For GG, you need ``g1`` and ``g2``, but for kappa correlations, you would use
``k`` instead.

You can optionally provide a weight column as well with ``w`` if desired.  To have the calculation
skip some objects (e.g. objects with some kind of flag set), simply provide ``w`` where those
objects have ``w[i] = 0``.

See the doc string for `Catalog` for more information.


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
calculation involving random catalogs for that.  See `Using random catalogs` below for more details.


.. _dummy_3pt:

Three-point Correlation Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

The NG and NK classes also have a :meth:`~treecorr.NGCorrelation.calculateXi` method to allow for the use of compensated
estimators in those cases as well.  They already have a ``xi`` attribute though, which is the
uncompensated estimator.  These correlations do not suffer as much from masking effects,
so the compensation is not as required.  However, it does produce a slightly better estimate
of the correlation function if you are able to use a random catalog.

Furthermore, the :meth:`~treecorr.GGCorrelation.process` functions can take lists of Catalogs if desired, in which case it will
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

For even more control over the calculation, you can drill down one more level into what the
:meth:`~treecorr.GGCorrelation.process` functions are doing.  There are typically three steps:

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

In addition to :meth:`~treecorr.NGCorrelation.process_cross`, classes that allow auto-correlations
have a :meth:`~treecorr.GGCorrelation.process_auto` method for manually processing
auto-correlations.  See the doc strings for these methods for more information.
