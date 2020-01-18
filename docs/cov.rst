Covariance Estimates
====================

In addition to calculating the correlation function, TreeCorr can also
estimate the variance of the measured values or even the covariance matrix.

This simplest estimate of the variance involves propagating the shot noise
of the individual measurements into the final results.  For shear (G) mesurements,
this includes the so-called "shape noise".  For scalar (K) measurements, this
includes the point variance of the k values.  This variance estimate is the
default if you don't specify something different, and it will be recorded as
**varxi** for most types of correlations.  For GG, there are two quantities,
**varxip** and **varxim**, which give the variance of **xip** and **xim**
respectively.

However, this kind of variance estimate does not capture the sample variance.
This is the fact that the signal has real variation across the field, which
tends to dominate the total variance at large scales.  To estimate this
component of the total variance from the data, one typically needs to split
the field into patches and use the variation in the measurement among the
patches to estimate the overall sample variance.

.. note::

    So far, these patch-based covariance estimates are only available for
    2-point correlation functions.  Implementing this for 3-point functions
    is still an open issue.

Patches
-------

The first step to get an improved variance or covariance estimate is to
divide the input `Catalog` into patches.  There are several ways to do this.

1. Set the **patch** parameter in the input to give an explicit patch number to
   each object in the catalog.
2. Read a "patch" column from an input file using **patch_col**.
3. Set the **npatch** parameter to have TreeCorr split the `Catalog` into the
   given number of patches for you, using a K-Means algorithm (cf. `Field.run_kmeans`)
4. Set the **patch_centers** parameter to use an existing set of patch centers
   (typically from a previous TreeCorr K-Means calculation, written out using
   `Catalog.write_patch_centers`), so TreeCorr can assign each galaxy to the patch
   with the closest center.

Variance Methods
----------------

To get one of the patch-based variance estimates for the **varxi** or similar
attribute, you can set the **var_method** parameter in the constructor.  e.g.::

    >>> ng = treecorre.NGCorrelation(nbins=10, min_sep=1, max_sep=100, var_method='jackknife')

This tells TreeCorr to use the jackknife algorithm for computing the covariance matrix.
Then **varxi** is taken as the diagonal of this covariance matrix.
The full covariance matrix is also recorded at the **cov** attribute.

The following variance methods are implemented:

"shot"
^^^^^^

This is the default shot-noise estimate of the covariance. It includes the Poisson
counts of points for N statistics, shape noise for G statistics, and the observed
scatter in the values for K statistics.  In this case, the covariance matrix will
be diagonal, since there is no way to estimate the off-diagonal terms.

"jackknife"
^^^^^^^^^^^

This is the classic jackknife estimate of the covariance matrix.  It computes the
correlation function that would have been measured if one patch at a time is excluded
from the sample.  Then the covariance matrix is estimated as

.. math::

    C = \left(1 - \frac{1}{N_\mathrm{patch}} \right) \sum_i (\xi_i - \bar\xi)^T (\xi_i-\bar_xi)

"sample"
^^^^^^^^

This is the simplest patch-based covariance estimate estimate.  It computes the
correlation function for each patch, where at least one of the two points falls in
that patch.  Then the estimated covariance matrix is simply the sample covariance
of these vectors, scaled by the relative total weight in each patch.

.. math::

    C = \frac{1}{N_\mathrm{patch}-1} \sum_i w_i (\xi_i - \bar\xi)^T (\xi_i-\bar_xi)

For :math:`w_i`, we use the total weight in the correlation measurement for each patch
divided by the total weight in all patches.  This is roughly equal to
:math:`1/N_\mathrm{patch}` but captures somewhat any patch-to-patch variation in area
that might be present.

"bootstrap"
^^^^^^^^^^^

This estimate s based on a bootstrap resampling of the patches.  Specifically, we follow
the method described in *A valid and Fast Spatial Bootstrap for Correlation Functions*
by Ji Meng Loh, 2008.  cf. https://ui.adsabs.harvard.edu/abs/2008ApJ...681..726L/.

This method starts out the same as the "sample" method.  It computes the correlation
function for each patch where at least one of the two points falls in that patch.
However, it keeps track of the numerator and denominator separately.
These are the "marks" in Loh, 2008.

Then these marks are resampled in the normal bootstrap manner (random with replacement)
to produce mock results.  The correlation function for each bootstrap resampling is
the sum of the numerator marks divided by the sum of the denominator marks.

Then the covariance estimate is the sample variance of these resampled results:

.. math::

    C = \frac{1}{N_\mathrm{boot}-1} \sum_i (\xi_i - \bar\xi)^T (\xi_i-\bar_xi)

The default number of bootstrap resamplings is 500, but you can change this in the
Correlation constructor using the parameter **num_bootstrap**.

"bootstrap2"
^^^^^^^^^^^^

This implements a different manner of bootstrap resampling.  Rather than precomputing
the results for each patch individually as recommended by Loh, 2008, this method
calculates the final statistic using only the auto-correlations and cross-correlations
of the chosen resampled patches.

The auto-correlations are included at the selected repetition for the bootstrap
samples.  Cross-correlations between patches are included only if the two patches
aren't actually the same patch (i.e. it's not actually an auto-correlation).

This method is quite a bit slower to compute than "bootstrap", but it seems to
produce somewhat more accurate estimates of the covariance for the simple tests
that we have done comparing the two.

The default number of bootstrap resamplings is 500, but you can change this in the
Correlation constructor using the parameter **num_bootstrap**.


Covariance Matrix
-----------------

As mentioned above, the covariance matrix corresponding to the specified **var_method**
will be saved as the **cov** attribute of the correlation instance after processing
is complete.

However, if the processing was done using patches, then you can also compute the
covariance matrix for any of the above methods without redoing the processing
using `BinnedCorr2.estimate_cov`.  E.g.::

    >>> ng = treecorre.NGCorrelation(nbins=10, min_sep=1, max_sep=100)
    >>> ng.process(lens_cat, source_cat)  # At least one of these needs to have patches set.
    >>> cov_jk = ng.estimate_cov('jackknife')
    >>> cov_boot = ng.estimate_cov('bootstrap')

Additionally, you can compute the joint covariance matrix for a number of statistics
that were processed using the same patches with `treecorr.estimate_multi_cov`.  E.g.::

    >>> ng = treecorre.NGCorrelation(nbins=10, min_sep=1, max_sep=100)
    >>> ng.process(lens_cat, source_cat)
    >>> gg = treecorre.GGCorrelation(nbins=10, min_sep=1, max_sep=100)
    >>> gg.process(source_cat)
    >>> cov = treecorr.estimate_multi_cov([ng,gg], 'jackknife')

This will calculate an estimate of the covariance matrix for the full data vector
with ``ng.xi`` followed by ``gg.xip`` and then ``gg.xim``.
