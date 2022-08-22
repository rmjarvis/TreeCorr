
Metrics
=======

The correlation functions need to know how to calculate distances between the points,
that is, the metric defining the space.

In most cases, you will probably want to use the default Metric, called "Euclidean",
which just uses the normal Euclidean distance between two points.  However, there are a few
other options, which are useful for various applications.

Both `BinnedCorr2` and `BinnedCorr3` take an optional
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

TreeCorr applies this formula at the end as part of the `finalize <GGCorrelation.finalize>`
function, so the ``meanr`` and ``meanlogr`` attributes
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


.. _Rperp:

"Rperp" or "FisherRperp"
------------------------

This metric is only valid for 3-dimensional coordinates (ra,dec,r) or (x,y,z).

The distance in this metric is defined as

:math:`d_{\rm Rperp} = \sqrt{d_{\rm Euclidean}^2 - r_\parallel^2}`

where :math:`r_\parallel` follows the defintion in Fisher et al, 1994 (MNRAS, 267, 927).
Namely, if :math:`p_1` and :math:`p_2` are the vector positions from Earth for the
two points, and

:math:`L \equiv \frac{p1 + p2}{2}`

then

:math:`r_\parallel = \frac{(p_2 - p_1) \cdot L}{|L|}`

That is, it breaks up the full 3-d distance into perpendicular and parallel components:
:math:`d_{\rm 3d}^2 = r_\bot^2 + r_\parallel^2`,
and it identifies the metric separation as just the perpendicular component, :math:`r_\bot`.

Note that this decomposition is really only valid for objects with a relatively small angular
separation, :math:`\theta`, on the sky, so the two radial vectors are nearly parallel.
In this limit, the formula for :math:`d` reduces to

:math:`d_{\rm Rperp} \approx \left(\frac{2 r_1 r_2}{r_1+r_2}\right) \theta`

.. warning::

    Prior to version 4.0, the "Rperp" name meant what is now called "OldRperp".
    The difference can be significant for some use cases, so if consistency across
    versions is importatnt to you, you should either switch to using "OldRperp"
    or investigate whether the change to "FisherRperp" is important for your
    particular science case.


"OldRperp"
----------

This metric is only valid for 3-dimensional coordinates (ra,dec,r) or (x,y,z).

This is the version of the Rperp metric that TreeCorr used in versions 3.x.
In version 4.0, we switched the definition of :math:`r_\parallel` to the one
used by Fisher et al, 1994 (MNRAS, 267, 927).  The difference turns out to be
non-trivial in some realistic use cases, so we preserve the ability to use the
old version with this metric.

Specifically, if :math:`r_1` and :math:`r_2` are the two distance from Earth,
then this metric uses :math:`r_\parallel \equiv r_2-r_1`.

The distance is then defined as

:math:`d_{\rm OldRperp} = \sqrt{d_{\rm Euclidean}^2 - r_\parallel^2}`

That is, it breaks up the full 3-d distance into perpendicular and parallel components:
:math:`d_{\rm 3d}^2 = r_\bot^2 + r_\parallel^2`,
and it identifies the metric separation as just the perpendicular component, :math:`r_\bot`.

Note that this decomposition is really only valid for objects with a relatively small angular
separation, :math:`\theta`, on the sky, so the two radial vectors are nearly parallel.
In this limit, the formula for :math:`d` reduces to

:math:`d_{\rm OldRperp} \approx \left(\sqrt{r_1 r_2}\right) \theta`


"Rlens"
-------

This metric is only valid when the first catalog uses 3-dimensional coordinates
(ra,dec,r) or (x,y,z).  The second catalog may take either 3-d coordinates or spherical
coordinates (ra,dec).

The distance is defined as

:math:`d_{\rm Rlens} = r_1 \sin(\theta)`

where :math:`\theta` is the opening angle between the two objects and :math:`r_1` is the
radial distance to the object in the first catalog.
In other words, this is the distance from the first object (nominally the "lens") to the
line of sight to the second object (nominally the "source").  This is commonly referred to
as the impact parameter of the light path from the source as it passes the lens.

Since the basic metric does not use the radial distance to the source galaxies (:math:`r_2`),
they are not required.  You may just provide (ra,dec) coordinates for the sources.
However, if you want to use the ``min_rpar`` or ``max_rpar`` options
(see `Restrictions on the Line of Sight Separation` below),
then the source coordinates need to include r.

"Periodic"
----------

This metric is equivalent to the Euclidean metric for either 2-d or 3-d coordinate systems,
except that the space is given periodic boundaries, and the distance between two
points is taken to be the *smallest* distance in the periodically repeating space.
It is invalid for Spherical coordinates.

When constructing the correlation object, you need to set ``period`` if the period is the
same in each direction.  Or if you want different periods in each direction, you can
set ``xperiod``, ``yperiod``, and (if 3-d) ``zperiod`` individually.
We call these periods :math:`L_x`, :math:`L_y`, and :math:`L_z` below.

The distance is defined as

.. math::

    dx &= \min \left(|x_2 - x_1|, L_x - |x_2-x_1| \right) \\
    dy &= \min \left(|y_2 - y_1|, L_y - |y_2-y_1| \right) \\
    dz &= \min \left(|z_2 - z_1|, L_z - |z_2-z_1| \right)

.. math::
    d_{\rm Periodic} = \sqrt{dx^2 + dy^2 + dz^2}

Of course, for 2-dimensional coordinate systems, :math:`dz = 0`.

This metric is particularly relevant for data generated from N-body simuluations, which
often use periodic boundary conditions.


Restrictions on the Line of Sight Separation
--------------------------------------------

There are two additional parameters that are tightly connected to the metric space:
``min_rpar`` and ``max_rpar``.
These set the minimum and maximum values of :math:`r_\parallel` for pairs to be included in the
correlations.

This is most typically relevant for the Rperp or Rlens metrics, but we now (as of version 4.2)
allow these parameters for any metric.

The two different Rperp conventions (FisherRperp and OldRperp) have different definitions of
:math:`r_\parallel` as described above, which are used in the definition of the metric distances.
These are the same :math:`r_\parallel` definitions that are used for the min and max values
if ``min_rpar`` and/or ``max_rpar`` are given.
For all other metrics, we use the FisherRperp definition for :math:`r_\parallel` if needed
for this purpose.

The sign of :math:`r_\parallel` is defined such that positive values mean
the object from the second catalog is farther away.  Thus, if the first catalog represents
lenses and the second catalog represents lensed source galaxies, then setting
``min_rpar`` = 0 will restrict the sources to being in the background of each lens.
Contrariwise, setting ``max_rpar`` = 0 will restrict to pairs where the object in the first
catalog is behind the object in the second catalog.

Another common use case is to restrict to pairs that are near each other in line of sight distance.
Setting ``min_rpar`` = -50, ``max_rpar`` = 50 will restrict the pairs to only those that are
separated by no more than 50 Mpc (say, assuming the catalog distances are given in Mpc) along
the radial direction.
