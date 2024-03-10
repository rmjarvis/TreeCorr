
Two-point Correlation Functions
===============================

TreeCorr can compute two- and three-point correlations for several different kinds of fields.
For notational brevity in the various classes involved, we use a single letter
to represent each kind of field as follows:

* N represents simple counting statistics.  The underlying field is a density of objects,
  which is manifested by objects appearing at specific locations.  The assumption here is
  that the probability of an object occurring at a specific location is proportional to the
  underlying density field at that spot.
* K represents a real, scalar field.  Nominally, K is short for "kappa", since TreeCorr was
  originally written for weak lensing applications, and kappa is the name of the weak lensing
  convergence, a measure of the projected matter density along the line of sight.
* Z represents a complex, spin-0 scalar field.  This is mostly for API consistency,
  since we have several other complex fields with different spin properties.  Spin-0 fields
  don't change their complex value when the orientation changes.
* V represents a complex, spin-1 vector field.  Spin-1 means that the complex value changes
  by :math:`\exp(i \phi)` when the orientation is rotated by an angle :math:`\phi`.  This kind
  of field is appropriate for normal vectors with direction, like velocity fields.
* G represents a complex, spin-2 shear field.  Spin-2 means that the complex value changes
  by :math:`\exp(2i \phi)` when the orientation is rotated by an angle :math:`\phi`.  The letter
  g is commonly used for reduced shear in the weak lensing context (and :math:`\gamma` is the
  unreduced shear), which is a spin-2 field, hence our use of G for spin-2 fields in Treecorr.
* T represents a complex, spin-3 field.  Spin-3 means that the complex value changes
  by :math:`\exp(3i \phi)` when the orientation is rotated by an angle :math:`\phi`.  The letter
  T is short for trefoil, a shape with spin-3 rotational properties.
* Q represents a complex, spin-4 field.  Spin-4 means that the complex value changes
  by :math:`\exp(4i \phi)` when the orientation is rotated by an angle :math:`\phi`.  The letter
  Q is short for quatrefoil, a shape with spin-4 rotational properties.

Not all possible pairings of two fields are currently implemented.  The following lists
all the currently implemented classes for computing two-point correlations.  If you have need
of a pairing not listed here, please file in issue asking for it.  It's not hard to add more,
but I didn't want to implement a bunch of classes that no one will use.

.. toctree::
    :maxdepth: 1

    nn
    nk
    kk
    nz
    kz
    zz
    nv
    kv
    vv
    ng
    kg
    gg
    nt
    kt
    tt
    nq
    kq
    qq

Each of the above classes is a sub-class of the base class Corr2, so they have a number of
features in common about how they are constructed.  The common features are documented here.

.. autoclass:: treecorr.Corr2
    :members:


.. autofunction:: treecorr.estimate_multi_cov

.. autofunction:: treecorr.build_multi_cov_design_matrix

.. autofunction:: treecorr.set_max_omp_threads

.. autofunction:: treecorr.set_omp_threads

.. autofunction:: treecorr.get_omp_threads
