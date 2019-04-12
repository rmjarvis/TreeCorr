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
    - `GField` holds both counts of objects and the mean shear of those objects.
      It is used for correlations with a G in the name, including
      `GGCorrelation`, `NGCorrelation`, `KGCorrelation`, and `GGGCorrelation`.
    - `KField` holds both counts of objects and the mean "kappa" of those objects.
      It is used for correlations with a K in the name, including
      `KKCorrelation`, `NKCorrelation`, `KGCorrelation`, and `KKKCorrelation`.
    - `SimpleField` is a different base class, which packages the information in a list
      rather than a tree.  Its subclasses, `NSimpleField`, `GSimpleField`, and
      `KSimpleField`, are used instead of the regular `Field` types when doing
      ``pairwise`` correlations.

Typically, one would not create any of these objects directly, but would instead
use Catalog methods `getNField`, `getGField`, `getKField`.  Or indeed, usually, one
does not even do that, and just lets the relevant ``process`` command do so for you.

.. autoclass:: treecorr.Field
    :members:

.. autoclass:: treecorr.NField
    :members:

.. autoclass:: treecorr.GField
    :members:

.. autoclass:: treecorr.KField
    :members:

.. autoclass:: treecorr.SimpleField
    :members:

.. autoclass:: treecorr.NSimpleField
    :members:

.. autoclass:: treecorr.GSimpleField
    :members:

.. autoclass:: treecorr.KSimpleField
    :members:

