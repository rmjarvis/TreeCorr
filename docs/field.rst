Fields
======

The `Field` class and its subclasses repackage the information from a `Catalog`
into a ball tree data structure, allowing for fast calcaulation of the correlation
functions.

There are several kinds of `Field` classes.

    - `Field` itself is an abstract base class of the other kinds of fields, and has a
      few methods that are available for all `Field` types.

    - `NField` holds counts of objects and is used for correlations with an N in the name,
      including `NNCorrelation`, `NGCorrelation`, `NKCorrelation`, and `NNNCorrelation`.
    - `KField` holds both counts of objects and the mean "kappa" of those objects.
      It is used for correlations with a K in the name, including
      `KKCorrelation`, `NKCorrelation`, `KGCorrelation`, and `KKKCorrelation`.
    - `ZField` holds both counts of objects and the mean value of the complex field of those
      objects.  It is used for correlations with a Z in the name, including
      `ZZCorrelation`, `NZCorrelation`, and `KZCorrelation`.
    - `VField` holds both counts of objects and the mean vector field of those objects.
      It is used for correlations with a V in the name, including
      `VVCorrelation`, `NVCorrelation`, and `KVCorrelation`.
    - `GField` holds both counts of objects and the mean shear of those objects.
      It is used for correlations with a G in the name, including
      `GGCorrelation`, `NGCorrelation`, `KGCorrelation`, and `GGGCorrelation`.
    - `TField` holds both counts of objects and the mean trefoil field of those objects.
      It is used for correlations with a T in the name, including
      `TTCorrelation`, `NTCorrelation`, and `KTCorrelation`.
    - `QField` holds both counts of objects and the mean quatrefoil field of those objects.
      It is used for correlations with a Q in the name, including
      `QQCorrelation`, `NQCorrelation`, and `KQCorrelation`.

Typically, one would not create any of these objects directly, but would instead
use Catalog methods `getNField`, `getKField`, `getZField`, `getVField`, `getGField`.
`getTField`, or `getQField`.
Or indeed, usually, one does not even do that, and just lets the relevant ``process``
command do so for you.

.. autoclass:: treecorr.Field
    :members:

.. autoclass:: treecorr.NField
    :members:

.. autoclass:: treecorr.KField
    :members:

.. autoclass:: treecorr.ZField
    :members:

.. autoclass:: treecorr.VField
    :members:

.. autoclass:: treecorr.GField
    :members:

.. autoclass:: treecorr.TField
    :members:

.. autoclass:: treecorr.QField
    :members:
