
Three-point Correlation Functions
=================================

TreeCorr can compute three-point correlations for several different kinds of fields.
For notational brevity in the various classes involved, we use a single letter
to represent each kind of field as follows:

* N represents simple counting statistics.  The underlying field is a density of objects,
  which is manifested by objects appearing at specific locations.  The assumption here is
  that the probability of an object occurring at a specific location is proportional to the
  underlying density field at that spot.
* K represents a real, scalar field.  Nominally, K is short for "kappa", since TreeCorr was
  originally written for weak lensing applications, and kappa is the name of the weak lensing
  convergence, a measure of the projected matter density along the line of sight.
* G represents a complex, spin-2 shear field.  Spin-2 means that the complex value changes
  by :math:`\exp(2i \phi)` when the orientation is rotated by an angle :math:`\phi`.  The letter
  g is commonly used for reduced shear in the weak lensing context (and :math:`\gamma` is the
  unreduced shear), which is a spin-2 field, hence our use of G for spin-2 fields in TreeCorr.

We have not yet implemented complex fields with spin 0, 1, 3 or 4 (called Z, V, T, and Q
respectively) as we have for two-point functions.  If you have a use case that requires any
of these, please open an issue requesting this feature.

The following classes are used for computing the three-point functions according to
which field is on each vertex of the triangle.

.. toctree::
    :maxdepth: 1

    nnn
    kkk
    ggg
    nnk
    nng
    nkk
    ngg
    kkg
    kgg

Each of the above classes is a subclass of the base class Corr3, so they have a number of
features in common about how they are constructed.  The common features are documented here.

.. autoclass:: treecorr.Corr3
    :members:
