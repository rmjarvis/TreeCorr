Binning
=======

Bin Types
---------

To be useful, the measured correlations need to be binned in some way to
find the average correlation among many pairs of nearly the same separation.
The different ways to bin the results may be specified using the ``bin_type``
parameter in :class:`~treecorr.BinnedCorr2` or :class:`~treecorr.BinnedCorr3`.

"Log"
^^^^^

The default way to bin the results in TreeCorr is uniformly in log(d),
where d is defined according to the specified metric
(cf. :ref:`Metrics`).  This corresponds to ``bin_type = 'Log'``, although
one normally omits this, as it is the default.

For most correlation functions, which tend to be approximately power laws, this
binning is the most appropriate, since it naturally handles a large dynamic range
in the separation.

The exact binning is specified using any 3 of the following 4 parameters:

    - ``nbins``       How many bins to use.
    - ``bin_size``    The width of the bins in log(d).
    - ``min_sep``     The minimum separation d to include.
    - ``max_sep``     The maximum separation d to include.

For a pair with a metric distance d, the index of the corresponding bin in the
output array is ``int(log(d) - log(min_sep))/bin_size)``.

Note that if ``nbins`` is the omitted value, then ``max_sep`` might need to be increased a little
to accommodate an integer number of bins with the given ``bin_size``.

"TwoD"
^^^^^^

To bin the correlation in two dimensions, (x,y), you can use ``bin_type = 'TwoD'``.
This will keep track of not only the distance between two points, but also the
direction.  The results are then binned linearly in both the delta x and delta y values.

The exact binning is specified using any 2 of the following 3 parameters:

    - ``nbins``       How many bins to use in each direction.
    - ``bin_size``    The width of the bins in dx and dy.
    - ``max_sep``     The maximum absolute value of dx or dy to include.

For a pair with a directed separation (dx,dy), the indices of the corresponding bin in the
2-d output array are ``int((dx + max_sep)/bin_size)``, ``int((dy + max_sep)/bin_size)``.

The binning is symmetric around (0,0), so the minimum separation in either direction is
``-max_sep``, and the maximum is ``+max_sep``.
If is also permissible to specify ``min_sep`` to exclude small separations from being 
accumulated, but the binning will still include a bin that crosses over (dx,dy) = (0,0)
if ``nbins`` is odd, or four bins that touch (0,0) if ``nbins`` is even.

Note that this metric is only valid when the input positions are given as x,y (not ra, dec),
and the metric is "Euclidean".  If you have a use case for other combinations, please
open an issue with your specific case, and we can try to figure out how it should be implemented.

Other options for binning
-------------------------

There are a few other options which impact how the binning is done.

sep_units
^^^^^^^^^

The optional parameter ``sep_units`` lets you specify what units you want for
the binned separations if the separations are angles.

Valid options are 'arcsec', 'arcmin', 'degrees', 'hours', or 'radians'.  The default if
not specified is 'radians'.

Note that this is only valid when the distance metric is an angle.
E.g. if RA and Dec values are given for the positions,
and no distance values are specified, then the default metric, "Euclidean",
is the angular separation on the sky.  "Arc" similarly is always an angle.

If the distance metric is a physical distance, then this parameter is invalid,
and the output separation will match the physical distance units in the input catalog.
E.g. if the distance from Earth is given as r, then the output units will match the
units of the r values.  Or if positions are given as x, y (and maybe z), then the
units will be whatever the units are for these values.

bin_slop
^^^^^^^^

This parameter is really the reason TreeCorr is fast.  The fundamental approximation
of TreeCorr is that if you have decided to bin with some bin size, then you apparently
don't care about the distances being more precise than this.  The approximation
TreeCorr makes is to allow some *additional* imprecision that is a fraction of this
level.  Namely ``bin_slop``.

Specifically, ``bin_slop`` specifies what fraction of the bin size can a given pair
of points be placed into the wrong bin.  You can think of it as turning all of your
rectangular bins into overlapping trapezoids, where ``bin_slop`` defines the ratio
of the angled portion to the flat mean width.  Larger ``bin_slop`` allows for more
overlap (and is thus faster), while smaller ``bin_slop`` gets closer to putting each
pair perfectly into the bin it belongs in.

The default ``bin_slop`` for the "Log" bin type is such that ``bin_slop * bin_size``
is 0.1.  Or if ``bin_size < 0.1``, then we use ``bin_slop = 1``.  This has been
found to give fairly good accuracy across a variety of applications.  However,
for high precision measurements, it may be appropriate to use a smaller value than
this.  Especially if your bins are fairly large.  A typical test to perform on your
data is to cut ``bin_slop`` in half and see if your results change significantly.
If not, you are probably fine, but if they change by an appreciable amount, then your
original ``bin_slop`` was too large.

To understand the impact of the ``bin_slop`` parameter, it helps to start by thinking
about when it is set to 0.
If ``bin_slop=0``, then TreeCorr does essentially a brute-force calculation, 
where each pair of points is always placed into the correct bin.
But if ``bin_slop > 0``, then any given pair is allowed to be placed in the wrong bin
so long as the true separation is within this fraction of a bin from the edge.
For example, if a bin nominally goes from 10 to 20 arcmin, then with bin_slop = 0.05,
TreeCorr will accumulate pairs with separations ranging from 9.5 to 20.5 arcmin into this
bin.  (I.e. the slop is 0.05 of the bin width on each side.)
Note that some of the pairs with separations from 9.5 to ~10.5 would possibly fall into the
lower bin instead.  Likewise some from 19.5 to 20.5 would fall in the higher bin.
So both edges are a little fuzzy.

For large number of objects, the shifts up and down tend to cancel out, so there is typically
very little bias in the results.  Statistically, about as many pairs scatter up as scatter
down, so the resulting counts come out pretty close to correct.  Furthermore, the total
number of pairs within the specified range is always correct, since each pair is placed
in some bin.

Finally, there is a distinction between ``bin_slop=0`` and ``brute=True``.
Internally, the latter will *always* traverse the tree all the way to the leaves.  So
every pair will be calculated individually.  This is the brute force calculation.
However, ``bin_slop=0`` will allow for the traversal to stop early if all possible pairs in a
given pair of cells fall into the same bin.  This can be quite a large speedup in some cases.
And especially for NN correlations, there is no disadvantage to doing so.

For shear correlations, there can be a slight difference between using ``bin_slop=0`` and
``brute=True`` because the shear projections won't be precisely equal in the two cases.
If the difference is seen to matter for you, this is probably a sign that you should decrease
your bin size.
