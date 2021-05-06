Covariance Estimates
====================

In addition to calculating the correlation function, TreeCorr can also
estimate the variance of the resulting array of values, or even the
covariance matrix.

This simplest estimate of the variance involves propagating the shot noise
of the individual measurements into the final results.  For shear (G) mesurements,
this includes the so-called "shape noise". For scalar (K) measurements, this
includes the point variance of the k values. For count (N) measurements,
it comes from the Poisson statistics of counting. This variance estimate is the
default if you don't specify something different, and it will be recorded as
``varxi`` for most types of correlations.  For GG, there are two quantities,
``varxip`` and ``varxim``, which give the variance of ``xip`` and ``xim``
respectively.

However, this kind of variance estimate does not capture the sample variance.
This is the fact that the signal has real variation across the field, which
tends to dominate the total variance at large scales.  To estimate this
component of the total variance from the data, one typically needs to split
the field into patches and use the variation in the measurement among the
patches to estimate the overall sample variance.

See `Patches` for information on defining the patches to use for your input `Catalog`.

Variance Methods
----------------

To get one of the patch-based variance estimates for the ``varxi`` or similar
attribute, you can set the ``var_method`` parameter in the constructor.  e.g.::

    >>> ng = treecorr.NGCorrelation(nbins=10, min_sep=1, max_sep=100, var_method='jackknife')

This tells TreeCorr to use the jackknife algorithm for computing the covariance matrix.
Then ``varxi`` is taken as the diagonal of this covariance matrix.
The full covariance matrix is also recorded at the ``cov`` attribute.

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

    C = \frac{N_\mathrm{patch} - 1}{N_\mathrm{patch}} \sum_i (\xi_i - \bar\xi)^T (\xi_i-\bar\xi)

"sample"
^^^^^^^^

This is the simplest patch-based covariance estimate estimate.  It computes the
correlation function for each patch, where at least one of the two points falls in
that patch.  Then the estimated covariance matrix is simply the sample covariance
of these vectors, scaled by the relative total weight in each patch.

.. math::

    C = \frac{1}{N_\mathrm{patch} - 1} \sum_i w_i (\xi_i - \bar\xi)^T (\xi_i-\bar\xi)

For :math:`w_i`, we use the total weight in the correlation measurement for each patch
divided by the total weight in all patches.  This is roughly equal to
:math:`1/N_\mathrm{patch}` but captures somewhat any patch-to-patch variation in area
that might be present.

"bootstrap"
^^^^^^^^^^^

This estimate implements a bootstrap resampling of the patches as follows:

1. Select :math:`N_\mathrm{patch}` patch numbers at random from the full list
   :math:`[0 \dots N_\mathrm{patch}{-}1]` with replacement, so some patch numbers
   will appear more than once, and some will be missing.

2. Calculate the total correlation function that would have been computed
   from these patches rather than the original patches.

3. The auto-correlations are included at the selected repetition for the bootstrap
   samples.  So if a patch number is repeated, its auto-correlation is included that
   many times.

4. Cross-correlations between patches are included only if the two patches
   aren't actually the same patch (i.e. it's not actually an auto-correlation).
   This prevents extra auto-correlations (where most of the signal typically occurs)
   from being included in the sum.

5. Repeat steps 1-4 a total of :math:`N_\mathrm{bootstrap}` times to build up a large
   set of resampled correlation functions, :math:`\{\xi_i\}`.

6. Then the covariance estimate is the sample variance of these resampled results:

    .. math::

        C = \frac{1}{N_\mathrm{bootstrap}-1} \sum_i (\xi_i - \bar\xi)^T (\xi_i-\bar\xi)

The default number of bootstrap resamplings is 500, but you can change this in the
Correlation constructor using the parameter ``num_bootstrap``.

"marked_bootstrap"
^^^^^^^^^^^^^^^^^^

This estimate is based on a "marked-point" bootstrap resampling of the patches.
Specifically, we follow the method described in
*A valid and Fast Spatial Bootstrap for Correlation Functions*
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

    C = \frac{1}{N_\mathrm{bootstrap}-1} \sum_i (\xi_i - \bar\xi)^T (\xi_i-\bar\xi)

The default number of bootstrap resamplings is 500, but you can change this in the
Correlation constructor using the parameter ``num_bootstrap``.

Covariance Matrix
-----------------

As mentioned above, the covariance matrix corresponding to the specified ``var_method``
will be saved as the ``cov`` attribute of the correlation instance after processing
is complete.

However, if the processing was done using patches, then you can also compute the
covariance matrix for any of the above methods without redoing the processing
using `BinnedCorr2.estimate_cov` or `BinnedCorr3.estimate_cov`.  E.g.::

    >>> ng = treecorr.NGCorrelation(nbins=10, min_sep=1, max_sep=100)
    >>> ng.process(lens_cat, source_cat)  # At least one of these needs to have patches set.
    >>> cov_jk = ng.estimate_cov('jackknife')
    >>> cov_boot = ng.estimate_cov('bootstrap')

Additionally, you can compute the joint covariance matrix for a number of statistics
that were processed using the same patches with `treecorr.estimate_multi_cov`.  E.g.::

    >>> ng = treecorr.NGCorrelation(nbins=10, min_sep=1, max_sep=100)
    >>> ng.process(lens_cat, source_cat)
    >>> gg = treecorr.GGCorrelation(nbins=10, min_sep=1, max_sep=100)
    >>> gg.process(source_cat)
    >>> cov = treecorr.estimate_multi_cov([ng,gg], 'jackknife')

This will calculate an estimate of the covariance matrix for the full data vector
with ``ng.xi`` followed by ``gg.xip`` and then ``gg.xim``.

Covariance of Derived Quantities
--------------------------------

Sometimes your data vector of interest might not be just the raw correlation function,
or even a list of several correlation functions.  Rather, it might be some derived
quantity. E.g.

* The ratio or difference of two correlation functions such as ``nk1.xi / nk2.xi``.
* The aperture mass variance computed by `GGCorrelation.calculateMapSq`.
* One of the other ancillary products such as ``ng.xi_im``.
* A reordering of the data vector, such as putting several ``gg.xip`` first for multiple
  tomographic bins and then the ``gg.xim`` for each after that.

These are just examples of what kind of thing you might want. In fact, we enable
any kind of post-processing you want to do on either a single correlation object
(using `BinnedCorr2.estimate_cov` or `BinnedCorr3.estimate_cov`) or a list of
correlation objects (using `treecorr.estimate_multi_cov`).

These functions take an optional ``func`` parameter, which can be any user-defined
function that calculates the desired data vector from the given correlation(s).
For instance, in the first case, where the desired data vector is the ratio of
two NK correlations, you could find the corresponding covariance matrix as follows::

    >>> func = lambda corrs: corrs[0].xi / corrs[1].xi
    >>> nk1 = treecorr.NKCorrelation(nbins=10, min_sep=1, max_sep=100)
    >>> nk2 = treecorr.NKCorrelation(nbins=10, min_sep=1, max_sep=100)
    >>> nk1.process(cat1a, cat1b)  # Ideally, all of these use the same patches.
    >>> nk2.process(cat2a, cat2b)
    >>> corrs = [nk1, nk2]
    >>> ratio = func(corrs)  # = nk1.xi / nk2.xi
    >>> cov = treecorr.estimate_multi_cov(corrs, 'jackknife', func)

The resulting covariance matrix, ``cov``, will be the jackknife estimate for the derived
data vector, ``ratio``.

Random Catalogs
---------------

There are a few adjustements to the above prescription when using random
catalogs, which of course are required when doing an NN correlation.

1. It is not necessarily required to use patches for the random catalog.
   The random is supposed to be dense enough that it doesn't materially contribute
   to the noise in the correlation measurement.  In particular, it doesn't have
   any sample variance itself, and the shot noise component should be small
   compared to the shot noise in the data.
2. If you do use patches for the random catalog, then you need to make sure
   that you use the same patch definitions for both the data and the randoms.
   Using patches for the randoms probably leads to slightly better covariance
   estimates in most cases, but the difference in the two results is usually small.
   (Note: This seems to be less true for 3pt NNN correlations than 2pt NN.
   Using patches for the randoms gives significantly better covariance estimates
   in that case than not doing so.)
3. The covariance calculation cannot happen until you call
   `calculateXi <NNCorrelation.calculateXi>`
   to let TreeCorr know what the RR and DR (if using that) results are.
4. After calling `dd.calculateXi <NNCorrelation.calculateXi>`, ``dd``
   will have ``varxi`` and ``cov`` attributes calculated according
   to whatever ``var_method`` you specified.
5. It also allows you to call `dd.estimate_cov <BinnedCorr2.estimate_cov>`
   with any different method you want.
   And you can include ``dd`` in a list of correlation
   objects passed to `treecorr.estimate_multi_cov`.

Here is a worked example::

    >>> data = treecorr.Catalog(config, npatch=N)
    >>> rand = treecorr.Catalog(rand_config, patch_centers=data.patch_centers)
    >>> dd = treecorr.NNCorrelation(nn_config, var_method='jackknife')
    >>> dr = treecorr.NNCorrelation(nn_config)
    >>> rr = treecorr.NNCorrelation(nn_config)
    >>> dd.process(data)
    >>> dr.process(data, rand)
    >>> rr.process(rand)
    >>> dd.calculateXi(rr=rr, dr=dr)
    >>> dd_cov = dd.cov  # Can access covariance now.
    >>> dd_cov_bs = dd.estimate_cov(method='bootstrap') # Or calculate a different one.
    >>> txcov = treecorr.estimate_multi_cov([ng,gg,dd], 'bootstrap') # Or include in multi_cov

As mentioned above, using ``patch_centers`` is optional for ``rand``, but probably recommended.
In the last line, it would be required that ``ng`` and ``gg`` were also made using catalogs
with the same patch centers that ``dd`` used.

The use pattern for `NNNCorrelation` is analogous, where `NNNCorrelation.calculateZeta`
needs to be run to get the covariance estimate, after which it may be used in a list
past to `treecorr.estimate_multi_cov`.
