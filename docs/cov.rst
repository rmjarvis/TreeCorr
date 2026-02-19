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

.. admonition:: Recommendation

    For most use cases, the most accurate covariance estimate seems to be
    jackknife covariances with matched cross-patch weights, so we recommend
    this option for most users who want empirical covariance estimates from
    the data.  I.e. use ``method='jackknife', cross_patch_weight='match'``,
    as described below.

    For the 5.x version series, the matched cross-patch weights are not the default
    in order to maintain backwards-compatibility, but this may change at the
    next major version (6.0).


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
correlation function for each patch, where at least one point falls in
that patch.  Then the estimated covariance matrix is simply the sample covariance
of these vectors, scaled by the relative total weight in each patch.

.. math::

    C = \frac{1}{N_\mathrm{patch} - 1} \sum_i w_i (\xi_i - \bar\xi)^T (\xi_i-\bar\xi)

For :math:`w_i`, we use the total weight in the correlation measurement for each patch
divided by the total weight in all patches.  This is roughly equal to
:math:`1/N_\mathrm{patch}` but accounts somewhat for any patch-to-patch variation in area
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
`*A valid and Fast Spatial Bootstrap for Correlation Functions*
by Ji Meng Loh, 2008 <https://ui.adsabs.harvard.edu/abs/2008ApJ...681..726L/>`_.

This method starts out the same as the "sample" method.  It computes the correlation
function for each patch where at least one point falls in that patch.
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

Cross-patch Weights
-------------------

There is some ambiguity as to the exact calculation of :math:`\xi` in each of the above
formulae, specifically with respect to the treatment of pairs (or triples for 3 point
statistics) that cross between a selected patch and an unselected patch.
`Mohammad and Percival (2022; MP22 hereafter)
<https://ui.adsabs.harvard.edu/abs/2022MNRAS.514.1289M/>`_ explored several different options
for how much weight to give these pairs for jackknife and bootstrap.
We allow the user to choose among them using the parameter ``cross_patch_weight``,
which can be provided in the `Corr2` or `Corr3` constructor or in the call to
`estimate_cov <Corr2.estimate_cov>` or `estimate_multi_cov`.  The valid options are:

* 'simple' is the prescription that TreeCorr implicitly used prior to version 5.1,
  and it is generally the simplest treatment in each case.
  For jackknife and bootstrap, it corresponds to what MP22 calls :math:`v_{\rm mult}`,
  which means the weight is the product of the two patch weights.
  For jackknife, the weights are all 1 or 0, so this means the pair is used only if
  both points are not in the excluded patch.  For bootstrap, the weights are some
  integer corresponding the multiplicity of that patch in the bootstrap selection.
  Cross patch pairs are included at the product of the multiplicity of the two patches.
  For sample and marked_bootstrap, a pair is included if the first point is the selected
  sample.
* 'mean' involves weighting pairs by the mean of the patch weights.  For jackknife, this
  means that pairs between the unselected patch and a selected one are included, but only with
  half the weight of other pairs.  For bootstrap, the cross pairs between selected and
  unselected patches have half the weight of the selected patch, and those between two
  selected patches use the average weight of the two patches.  For sample and marked_bootstrap,
  the weight of pairs between the selected sample and another one is 0.5, but it includes
  pairs with the selected patch in either position.
* 'geom' is the same as 'mean', but using the geometric mean rather than the arithmetic mean.
  This option is only valid for 'bootstrap', since for other methods, it is equivalent to
  'simple'.
* 'match' is an innovation of MP22.  They derived an optimal weight for jackknife covariance
  that matches the effective weight of the cross-patch pairs to that of the intra-patch
  pairs.  They find that this weight is significantly more accurate than either 'simple'
  (what they call mult) or 'mean'.

The default value of ``cross_patch_weight`` is 'simple' for all variance methods.
MP22 recommends to instead use 'match' for jackknife covariances and 'geom' for
bootstrap covariances.  In order to maintain API consistency, we haven't made this
the default yet, but we may in a future version of TreeCorr.

For now, we recommend
explicitly setting ``cross_patch_weight`` to either 'match' or 'geom' as appropriate,
especially if your field has significant sample variance, but not much super-sample variance,
where these options seem to be more optimal than the default weighting.
In many practical analyses, we find jackknife with ``cross_patch_weight='match'`` to be
the most accurate default choice, with bootstrap/``'geom'`` as a useful comparison.
For 'sample' and 'marked_bootstrap', we don't see much difference between 'simple' and 'mean',
although we welcome feedback from users whether 'mean' might be a better
choice for these methods.

Recommended Baseline Workflow
-----------------------------

For most analyses, a good baseline is jackknife covariance with ``'match'`` weighting:

.. code-block:: python

    cat1 = treecorr.Catalog(cat1_file, npatch=50)
    cat2 = treecorr.Catalog(cat2_file, patch_centers=cat1.patch_centers)
    ng = treecorr.NGCorrelation(config)
    ng.process(cat1, cat2)
    cov = ng.estimate_cov(method='jackknife', cross_patch_weight='match')

Patch consistency checklist for cross-correlations:

1. Use a single shared patch definition across all relevant catalogs.
2. In particular, do not use ``npatch`` on multiple catalogs, since that will create different
   patch definitions for the different data sets.
3. Typically use ``patch_centers`` to copy the patch definitions from one catalog to any other
   catalogs being used in the correlation calculation.
4. This applies to random catalogs as well.  If you are using random catalogs, use the same
   patch definitions as the data.
5. Ensure any correlations combined with ``estimate_multi_cov`` were processed with compatible
   patches.


Covariance Matrix
-----------------

As mentioned above, the covariance matrix corresponding to the specified ``var_method``
will be saved as the ``cov`` attribute of the correlation instance after processing
is complete.

However, if the processing was done using patches, then you can also compute the
covariance matrix for any of the above methods without redoing the processing
using `Corr2.estimate_cov` or `Corr3.estimate_cov`.  E.g.::

    >>> ng = treecorr.NGCorrelation(nbins=10, min_sep=1, max_sep=100)
    >>> ng.process(lens_cat, source_cat)  # At least one of these needs to have patches set.
    >>> cov_jk = ng.estimate_cov('jackknife')
    >>> cov_boot = ng.estimate_cov('bootstrap')

Additionally, you can compute the joint covariance matrix for a number of statistics
that were processed using the same patches with `estimate_multi_cov`.  E.g.::

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
(using `Corr2.estimate_cov` or `Corr3.estimate_cov`) or a list of
correlation objects (using `estimate_multi_cov`).

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
    >>> cov = treecorr.estimate_multi_cov(corrs, method='jackknife', func=func)

The resulting covariance matrix, ``cov``, will be the jackknife estimate for the derived
data vector, ``ratio``.

Random Catalogs
---------------

There are a few adjustments to the above prescription when using random
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
5. It also allows you to call `dd.estimate_cov <Corr2.estimate_cov>`
   with any different method you want.
   And you can include ``dd`` in a list of correlation
   objects passed to `estimate_multi_cov`.

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
    >>> tx_cov = treecorr.estimate_multi_cov([ng,gg,dd], 'bootstrap') # Or include in multi_cov

As mentioned above, using ``patch_centers`` is optional for ``rand``, but probably recommended.
In the last line, it would be required that ``ng`` and ``gg`` were also made using catalogs
with the same patch centers that ``dd`` used.

The use pattern for `NNNCorrelation` is analogous, where `calculateZeta <NNNCorrelation.calculateZeta>`
needs to be run to get the covariance estimate, after which it may be used in a list
passed to `estimate_multi_cov`.

Design Matrix
-------------

Occasionally, it can be useful to access the design matrix that would be used to compute the
covariance matrix.  This is the matrix where each row is the :math:`\xi_i` vector as
described `above <Variance Methods>`.

This matrix is available using a parallel pair of functions to `estimate_cov <Corr2.estimate_cov>`
and `estimate_multi_cov`.  Namely `build_cov_design_matrix <Corr2.build_cov_design_matrix>`
and `build_multi_cov_design_matrix`.  E.g.::

    >>> A_ng_jk, w = ng.build_cov_design_matrix(method='jackknife')
    >>> A_tx_bs, w = treecorr.build_multi_cov_design_matrix([ng,gg,dd], method='bootstrap')

The second value returned here (``w``) is a vector of the total weight for each row.  Most methods
ignore this quantity, but the 'sample' method uses this to weight the rows when building the
covariance matrix.
