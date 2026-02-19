Binning for three-point correlations
====================================

The binning in the three-point case is somewhat more complicated than for
two-point functions, since we need to characterize the geometry of triangles.
There are currently three different binnings available, which
may be specified using the ``bin_type`` parameter in `Corr3`.

.. note::

    The different binning options each have their own way of defining the sides,
    which we number :math:`d_1`, :math:`d_2`, and :math:`d_3`.
    In all cases, vertices 1, 2 and 3 of the triangle are defined to be the vertex opposite
    the corresponding sides (:math:`d_1`, :math:`d_2`, :math:`d_3` respectively).
    For mixed-type correlations (e.g. NNG, KNK, etc.) we only keep the triangle if
    this definition of vertices has the right field in the corresponding vertex.
    E.g. NNG only keeps triangles that have the G field in vertex 3.  For triangles
    with the G field in vertex 1 or 2, you would need to use GNN and NGN respectively.
    To fully characterize the full set of 3-point correlation information of the
    three fields with mixed type, you need all three of these.

See also `Other options for binning` for additional parameters that are relevant to
the binning. These all work the same way for three-point functions as for
two-point function.

"LogRUV"
--------

This binning option uses a Side-Side-Side (SSS) characterization of the triangle.
The three side lengths of the triangle are measured (using whatever `Metric <Metrics>`
is being used).  Then we sort their lengths so that :math:`d_1 \ge d_2 \ge d_3`.

If we just binned directly in these three side lengths, then the range of valid
values for each of these will depend on the values of the other two.  This would
make the binning extremely complicated.  Therefore, we compute three derived
quantities which have better-behaved ranges:

.. math::

    r &\equiv d_2 \\
    u &\equiv \frac{d_3}{d_2} \\
    v &\equiv \frac{d_1 - d_2}{d_3}

With this reparametrization, :math:`u` and :math:`v` are each limited to the range
:math:`[0,1]`, independent of the values of the other parameters.  The :math:`r`
parameter defines the overall size of the triangle, and that can range of whatever
set of values the user wants.

This provides a unique definition for any triangle, except for a mirror reflection.
Two congruent triangles (that are not isosceles or equilateral) are not necessarily
equivalent for 3-point correlations.  The orientation of the sides matters, at least
in many use cases.  So we need to keep track of that.  We choose to do so in the 
sign of :math:`v`, where positive values mean that the sides :math:`d_1`,
:math:`d_2` and :math:`d_3` are oriented in counter-clockwise order.
Negative values of :math:`v` mean they are oriented in clockwise order.

.. warning::

    This binning can only use the 'triangle' algorithm, which is generally much
    slower than the 'multipole' algorithm.  For most purposes, we recommend using
    `"LogSAS"` instead, which can use the 'multipole' algorithm to calculate the
    correlation function.  See `Three-point Algorithm` below for more discussion
    about this.

The binning of :math:`r` works the same way as `"Log"` for two-point correlations.
That is, the binning is specified using any 3 of the following 4 parameters:

    - ``nbins``       How many bins to use.
    - ``bin_size``    The width of the bins in log(r).
    - ``min_sep``     The minimum separation r to include.
    - ``max_sep``     The maximum separation r to include.

The :math:`u` and :math:`v` parameters are binned linearly between limits given
by the user.  If unspecified, the full range of :math:`[0,1]` is used.  We always
bin :math:`v` symmetrically for positive and negative values.  So if you give it
a range of :math:`[0.2,0.6]` say, then it will also bin clockwise triangles
with these values into negative :math:`v` bins.
The :math:`u` and :math:`v` binning is specified using the following parameters:

    - ``nubins``      How many bins to use for u.
    - ``ubin_size``   The width of the bins in u.
    - ``min_u``       The minimum u to include.
    - ``max_u``       The maximum u to include.
    - ``nvbins``      How many bins to use for v.
    - ``vbin_size``   The width of the bins in v.
    - ``min_v``       The minimum v to include.
    - ``max_v``       The maximum v to include.


"LogSAS"
--------

This binning option uses a Side-Angle-Side (SAS) characterization of the triangles.
The two sides extending from vertex 1 of a triangle are measured using whatever
`Metric <Metrics>` is being used.  In addition, we measure the angle between
these two sides.  Since vertex 1 is where the angle is, the two side lengths
being used for the binning are called :math:`d_2` and :math:`d_3`.  The angle
between these two sides is called :math:`\phi`, and the side opposite it
(not used for binning) is :math:`d_1`.

The two sides, :math:`d_2` and :math:`d_3` are each binned the same way as
`"Log"` binning for two-point correlations.
That is, the binning is specified using any 3 of the following 4 parameters:

    - ``nbins``         How many bins to use for d2 and d3.
    - ``bin_size``      The width of the bins in log(d2) or log(d3).
    - ``min_sep``       The minimum side length to include for d2 or d3.
    - ``max_sep``       The maximum side length to include for d2 or d3.

The angle :math:`\phi` is binned linearly according to the parameters:

    - ``nphi_bins``     How many bins to use for phi.
    - ``phi_bin_size``  The width of the bins in phi.
    - ``min_phi``       The minimum angle phi to include.
    - ``max_phi``       The maximum angle phi to include.
    - ``phi_units``     The angle units to use for ``min_phi`` and ``max_phi``.


"LogMultipole"
--------------

This binning option uses a multipole expansion of the `"LogSAS"` characterization.
This idea was initially developed by
`Chen & Szapudi (2005, ApJ, 635, 743)
<https://ui.adsabs.harvard.edu/abs/2005ApJ...635..743C/abstract>`_
and then further refined by
`Slepian & Eisenstein (2015, MNRAS, 454, 4142)
<https://ui.adsabs.harvard.edu/abs/2015MNRAS.448....9S/abstract>`_,
`Philcox et al (2022, MNRAS, 509, 2457)
<https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.2457P/abstract>`_, and
`Porth et al (2024, A&A, 689, 224)
<https://ui.adsabs.harvard.edu/abs/2024A%26A...689A.227P/abstract>`_.
The latter in particular showed how to use this
method for non-spin-0 correlations (GGG in particular).

The basic idea is to do a Fourier transform of the phi binning to convert the phi
bins into n bins.

.. math::

    \zeta(d_2, d_3, \phi) = \frac{1}{2\pi} \sum_n \mathcal{Z}_n(d_2,d_3) e^{i n \phi}

Formally, this is exact if the sum goes from :math:`-\infty .. \infty`.  Truncating this
sum at :math:`\pm n_\mathrm{max}` is similar to binning in theta with this many bins
for :math:`\phi` within the range :math:`0 \le \phi \le \pi`.

The above papers show that this multipole expansion allows for a much more efficient
calculation, since it can be done with a kind of 2-point calculation.
We provide methods to convert the multipole output into the SAS binning if desired, since
that is often more convenient in practice.

As for "LogSAS", the sides :math:`d_2` and :math:`d_3` are binned logarithmically
according to the parameters

    - ``nbins``         How many bins to use for d2 and d3.
    - ``bin_size``      The width of the bins in log(d2) or log(d3).
    - ``min_sep``       The minimum side length to include for d2 or d3.
    - ``max_sep``       The maximum side length to include for d2 or d3.

The binning of the multipoles for each pair of :math:`d_2`, :math:`d_3` is given by
a single parameter:

    - ``max_n``         The maximum multipole index n being stored.

The multipole values range from :math:`-n_{\rm max}` to :math:`+n_{\rm max}` inclusive.

Three-point Algorithm
---------------------

An important consideration related to the choice of binning for three-point correlations is
the algorithm used to compute the correlations.  The original algorithm used by TreeCorr
prior to version 5.0 is now called the 'triangle' algorithm.  This was described in
`Jarvis, Bernstein & Jain (2004, MNRAS, 352, 338)
<https://ui.adsabs.harvard.edu/abs/2004MNRAS.352..338J/abstract>`_,
section 4.2. (We no longer implement the algorithm described in section 4.3 due to memory
considerations.)  This algorithm is much faster than a brute-force calculation, but it is
still quite slow compared to the new multipole algorithm.

Starting in version 5.0, we now also implement the algorithm developed by
`Porth et al (2024, A&A, 689, 224)
<https://ui.adsabs.harvard.edu/abs/2024A%26A...689A.227P/abstract>`_,
called 'multipole' in TreeCorr, which is much faster for typical data sets.
This algorithm is directly used for
`"LogMultipole"` binning, but it is also available for `"LogSAS"`.  In the latter case, TreeCorr
first computes the correlation using the "LogMultipole" binning. Then it essentially does
a Fourier transform to convert the results to "LogSAS" binning.  This is the default
algorithm for "LogSAS" binning, but if desired, you may also use ``algo='triangle'`` to
use the 'triangle' algorithm.  (We use comparisons between the two algorithms extensively in
the unit tests.)

There is not currently any way to use the 'multipole' algorithm with `"LogRUV"` binning,
which means that calculations using that binning choice tend to be a lot slower than calculations
using "LogSAS" binning. For most use cases, we strongly recommend using "LogSAS" instead.
