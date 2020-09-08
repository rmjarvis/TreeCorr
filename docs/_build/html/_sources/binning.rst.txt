Binning
=======

To be useful, the measured correlations need to be binned in some way to
find the average correlation among many pairs of nearly the same separation.
The different ways to bin the results may be specified using the ``bin_type``
parameter in `BinnedCorr2`.

"Log"
-----

The default way to bin the results in TreeCorr is uniformly in log(r),
where r is defined according to the specified metric
(cf. `Metrics`).  This corresponds to ``bin_type`` = "Log", although
one normally omits this, as it is the default.

For most correlation functions, which tend to be approximately power laws, this
binning is the most appropriate, since it naturally handles a large dynamic range
in the separation.

The exact binning is specified using any 3 of the following 4 parameters:

    - ``nbins``       How many bins to use.
    - ``bin_size``    The width of the bins in log(r).
    - ``min_sep``     The minimum separation r to include.
    - ``max_sep``     The maximum separation r to include.

For a pair with a metric distance r, the index of the corresponding bin in the
output array is ``int(log(r) - log(min_sep))/bin_size)``.

.. note::

    If ``nbins`` is the omitted value, then ``bin_size`` might need to be decreased
    slightly to accommodate an integer number of bins with the given ``min_sep`` and ``max_sep``.

"Linear"
--------

For use cases where the scales of interest span only a relatively small range of distances,
it may be more convenient to use linear binning rather than logarithmic.  A notable
example of this is BAO investigations, where the interesting region is near the BAO peak.
In these cases, using ``bin_type`` = "Linear" may be preferred.

As with "Log", the binning may be specified using any 3 of the following 4 parameters:

    - ``nbins``       How many bins to use.
    - ``bin_size``    The width of the bins in r.
    - ``min_sep``     The minimum separation r to include.
    - ``max_sep``     The maximum separation r to include.

For a pair with a metric distance r, the index of the corresponding bin in the
output array is ``int((r - min_sep)/bin_size)``.

.. note::

    If ``nbins`` is the omitted value, then ``bin_size`` might need to be decreased
    slightly to accommodate an integer number of bins with the given ``min_sep`` and ``max_sep``.

"TwoD"
------

To bin the correlation in two dimensions, (x,y), you can use ``bin_type`` = "TwoD".
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

Output quantities
-----------------

For all of the different binning options, the Correlation object will have the following attributes
related to the locations of the bins:

    - ``rnom`` The separation at the nominal centers of the bins.  For "Linear" binning,
      these will be spaced uniformly.
    - ``logr`` The log of the separation at the nominal centers of the bins.  For "Log"
      binning, these will be spaced uniformly.  This is always the (natural)
      log of ``rnom``.
    - ``left_edges`` The separation at the left edges of the bins.  For "Linear" binning, these
      are half-way between the ``rnom`` values of successive bins.  For "Log" binning, these are
      the geometric mean of successive ``rnom`` values, rather than the arithmetic mean.
      For "TwoD" binning, these are like "Linear" but for the x separations only.
    - ``right_edges`` Analogously, the separation at the right edges of the bins.
    - ``meanr`` The mean separation of all the pairs of points that actually ended up
      falling in each bin.
    - ``meanlogr`` The mean log(separation) of all the pairs of points that actually ended up
      falling in each bin.

The last two quantities are only available after finishing a calculation (e.g. with ``process``).

In addition to the above, "TwoD" binning also includes the following:

    - ``bottom_edges`` The y separation at the bottom edges of the 2-D bins. Like
      ``left_edges``, but for the y values rather than the x values.
    - ``top_edges`` The y separation at the top edges of the 2-D bins. Like
      ``right_edges``, but for the y values rather than the x values.

There is some subtlety about which separation to use when comparing measured correlation functions
to theoretical predictions.  See Appendix D of
`Singh et al, 2020 <https://ui.adsabs.harvard.edu/abs/2020MNRAS.491...51S/abstract>`_,
who show that one can find percent level differences among the different options.
(See their Figure D2 in particular.)
The difference is smaller as the bin size decreases, although they point out that it is not always
feasible to make the bin size very small, e.g. because of issues calculating the covariance matrix.

In most cases, if the true signal is expected to be locally well approximated by a power law, then
using ``meanlogr`` is probably the most appropriate choice.  This most closely approximates the
signal-based weighting that they recommend, but if you are concerned about the percent level
effects of this choice, you would be well-advised to investigate the different options with
simulations to see exactly what impact the choice has on your science.


Other options for binning
-------------------------

There are a few other options that affect the binning, which can be set when constructing
any of the `BinnedCorr2` or `BinnedCorr3` classes.

sep_units
^^^^^^^^^

The optional parameter ``sep_units`` lets you specify what units you want for
the binned separations if the separations are angles.

Valid options are "arcsec", "arcmin", "degrees", "hours", or "radians".  The default if
not specified is "radians".

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

One of the main reasons that TreeCorr is able to compute correlation functions
so quickly is that it allows the bin edges to be a little bit fuzzy. A pairs whose
separation is very close to a dividing line between two bins might be placed
in the next bin over from where an exact calculation would put it.

This is normally completely fine for any real-world application.
Indeed, by deciding to bin your correlation function with some non-zero bin size, you have
implicitly defined a resolution below which you don't care about the exact separation
values.  

The approximation TreeCorr makes is to allow some *additional* imprecision that is a
fraction of this level.  Namely ``bin_slop``.  Specifically, ``bin_slop`` specifies the
maximum possible error any pair can have, given as a fraction of the bin size.

You can think of it as turning all of your rectangular bins into overlapping trapezoids,
where ``bin_slop`` defines the ratio of the angled portion to the flat mean width.
Larger ``bin_slop`` allows for more overlap (and is thus faster), while smaller ``bin_slop``
gets closer to putting each pair perfectly into the bin it belongs in.

The default ``bin_slop`` for the "Log" bin type is such that ``bin_slop * bin_size``
is 0.1.  Or if ``bin_size < 0.1``, then we use ``bin_slop`` = 1.  This has been
found to give fairly good accuracy across a variety of applications.  However,
for high precision measurements, it may be appropriate to use a smaller value than
this.  Especially if your bins are fairly large.

A typical test to perform on your data is to cut ``bin_slop`` in half and see if your
results change significantly.  If not, you are probably fine, but if they change by an
appreciable amount (according to whatever you think that means for your science),
then your original ``bin_slop`` was too large.

To understand the impact of the ``bin_slop`` parameter, it helps to start by thinking
about when it is set to 0.
If ``bin_slop`` = 0, then TreeCorr does essentially a brute-force calculation,
where each pair of points is always placed into the correct bin.

But if ``bin_slop`` > 0, then any given pair is allowed to be placed in the wrong bin
so long as the true separation is within this fraction of a bin from the edge.
For example, if a bin nominally goes from 10 to 20 arcmin, then with bin_slop = 0.05,
TreeCorr will accumulate pairs with separations ranging from 9.5 to 20.5 arcmin into this
bin.  (I.e. the slop is 0.05 of the bin width on each side.)
Note that some of the pairs with separations from 9.5 to 10.5 would possibly fall into the
lower bin instead.  Likewise some from 19.5 to 20.5 would fall in the higher bin.
So both edges are a little fuzzy.

For large number of objects, the shifts up and down tend to cancel out, so there is typically
very little bias in the results.  Statistically, about as many pairs scatter up as scatter
down, so the resulting counts come out pretty close to correct.  Furthermore, the total
number of pairs within the specified range is always correct, since each pair is placed
in some bin.

brute
^^^^^

Sometimes, it can be useful to force the code to do the full brute force calculation,
skipping all of the approximations that are inherent to the tree traversal algorithm.
This of course is much slower, but this option can be useful for testing purposes especially.
For instance, comparisons to brute force results have been invaluable in TreeCorr
development of the faster algorithms.  Some science cases also use comparison to brute
force results to confirm that they are not significantly impacted by using non-zero
``bin_slop``.

Setting ``brute`` = True is roughly equivalent to setting ``bin_slop`` = 0.  However,
there is a distinction between these two cases.
Internally, the former will *always* traverse the tree all the way to the leaves.  So
every pair will be calculated individually.  This really is the brute force calculation.

However, ``bin_slop`` = 0 will allow for the traversal to stop early if all possible pairs in a
given pair of cells fall into the same bin.  This can be quite a large speedup in some cases.
And especially for NN correlations, there is no disadvantage to doing so.

For shear correlations, there can be a slight difference between using ``bin_slop`` = 0 and
``brute`` = True because the shear projections won't be precisely equal in the two cases.
Shear correlations require parallel transporting the shear values to the centers of
the cells, and then when accumulating pairs, the shears are projected onto the line joining
the two points.  Both of these lead to slight differences in the results of a ``bin_slop`` = 0
calculation compared to the true brute force calculation.
If the difference is seen to matter for you, this is probably a sign that you should decrease
your bin size.

Additionally, there is one other way to use the ``brute`` parameter.  If you set
``brute`` to 1 or 2, rather than True or False, then the forced traversal to the
leaf cells will only apply to ``cat1`` or ``cat2`` respectively.  The cells for the other
catalog will use the normal criterion based on the ``bin_slop`` parameter to decide whether
it is acceptable to use a non-leaf cell or to continue traversing the tree.
