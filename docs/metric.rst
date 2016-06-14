
Metrics
=======

The correlation functions need to know how to calculate distances between the points,
that is, the metric defining the space.

In most cases, you will probably want to use the default Metric, called "Euclidean",
which just uses the normal Euclidean distance between two points.  However, there are a few
other options, which are useful for various applications.

Both :class:`~treecorr.BinnedCorr2` and :class:`~treecorr.BinnedCorr3` take an optional
``metric`` parameter, which should be one of the following string values:


"Euclidean"
-----------

This is the default metric, and is the only current option for 2-dimensional flat correlations,
i.e. when the coordinates are given by (x,y), rather than either (x,y,z), (ra,dec), or (ra,dec,r).

For 2-dimensional coordinate systems, the distance is defined as

:math:`d_{\rm Euclidean} = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}`

For 3-dimensional coordinate systems, the distance is defined as

:math:`d_{\rm Euclidean} = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2 + (z_2-z_1)^2}`

For spherical coordinates with distances, (ra,dec,r), the coordinates are first
converted to Cartesian coordinates and the above formula is used.

For spherical coordinates without distances, (ra, dec), the coordinates are placed on the
unit sphere and the above formula is used.  This means that all distances are really chord
distances across the sphere, not great circle distances.  For small angles, this is a small
correction, but as the angles get large, the difference between the great circle distance and
the chord distance becomes significant.  The conversion formula is

:math:`d_{\rm GC} = 2 \arcsin(d_{\rm Euclidean} / 2)`

TreeCorr applies this formula at the end as part of the ``finalize`` function
(e.g. :meth:`~treecorr.GGCorrelation.finalize`), so the ``meanr`` and ``meanlogr`` attributes
will be in terms of great circle distances.  However, they will not necessarily be spaced
precisely uniformly in log(r), since the original bin spacing will have been set up in terms
of the chord distances.

"Arc"
-----

This metric is only valid for spherical coordinates (ra,dec).

The distance is defined as

:math:`d_{\rm Arc} = 2 \arcsin(d_{\rm Euclidean} / 2)`

where :math:`d_{\rm Euclidean}` is the above "Euclidean" chord distance.

This metric is significantly slower than the "Euclidean" metric, since it requires trigonometric
functions for every pair calculation along the way, rather than just at the end.
In most cases, this extra care is unnecessary, but it provides a means to check if the
chord calculations are in any way problematic for your particular use case.

Also, unlike the "Euclidean" version, the bin spacing will be uniform in log(r) using the
actual great circle distances, rather than being based on the chord distances.

"Rperp"
-------

This metric is only valid for 3-dimensional coordinates (ra,dec,r) or (x,y,z).

The distance is defined as

:math:`d_{\rm Rperp} = \sqrt{d_{\rm Euclidean}^2 - (r_2-r_1)^2}`

That is, it breaks up the full 3-d distance into perpendicular and parallel components:
:math:`d_{\rm 3d}^2 = r_\bot^2 + r_\parallel^2`, where :math:`r_\parallel \equiv r_2-r_1`,
and it identifies the metric separation as just the perpendicular component, :math:`r_\bot`.

Note that this decomposition is really only valid for objects with a relatively small angular
separation, :math:`\theta`, on the sky, so the two radial vectors are nearly parallel.
In this limit, the formula for :math:`d` reduces to

:math:`d_{\rm Rperp} \approx \left(\sqrt{r_1 r_2}\right) \theta`

This metric also permits the use of two other parameters, ``min_rpar`` and ``max_rpar``,
which set the minimum and maximum values of :math:`r_\parallel` for pairs to be included in the
correlations. 

The sign of :math:`r_\parallel` is defined such that positive values mean
the object from the second catalog is farther away.  Thus, if the first catalog represents
lenses and the second catalog represents lensed source galaxies, then setting
``min_rpar = 0`` will restrict the sources to being in the background of each lens.
Similarly, setting ``min_rpar = -50``, ``max_rpar = 50`` will restrict the sources to be
within 50 Mpc (say, assuming the catalog distances are given in Mpc) of the lenses.

"Rlens"
-------

This metric is only valid when the first catalog uses 3-dimensional coordinates
(ra,dec,r) or (x,y,z).  The second catalog may take either 3-d coordinates or spherical
coordinates (ra,dec).

The distance is defined as

:math:`d_{\rm Rlens} = 2 r_1 \sin(\theta / 2)`

where :math:`\theta` is the opening angle between the two objects and :math:`r_1` is the
radial distance to the object in the first catalog (nominally the "lens" catalog).
In other words, this is the chord distance between the two lines of sight at the radius
of the lens galaxy.

This metric also permits the use of two other parameters, ``min_rpar`` and ``max_rpar``,
which set the minimum and maximum values of :math:`r_\parallel = r_2 - r_1` for pairs to be
included in the correlations. 

The sign of :math:`r_\parallel` is defined such that positive values mean
the object from the second catalog is farther away. Thus, setting
``min_rpar = 0`` will restrict the sources to being in the background of each lens.
Similarly, setting ``min_rpar = -50``, ``max_rpar = 50`` will restrict the sources to be
within 50 Mpc (say, assuming the catalog distances are given in Mpc) of the lenses.

Since the basic metric does not use the radial distance to the source galaxies (:math:`r_2`),
they are not required.  You may just provide (ra,dec) coordinates for the sources.
However, if you want to use the ``min_rpar`` or ``max_rpar`` options, then
the source coordinates need to include r.

