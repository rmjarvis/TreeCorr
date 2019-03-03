# Taking this from GalSim v2.1.  License is in GalSim_LICENSE in this directory.
# Modified slightly to remove explicit dependence on galsim, leaving only the
# functions that we actually use.

# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

from __future__ import print_function
import numpy as np
import os
import sys
import logging
import coord
import copy


# This file has some helper functions that are used by tests from multiple files to help
# avoid code duplication.


def do_pickle(obj1, func = lambda x : x, irreprable=False):
    """Check that the object is picklable.  Also that it has basic == and != functionality.
    """
    from numbers import Integral, Real, Complex
    try:
        import cPickle as pickle
    except:
        import pickle
    import copy
    # In case the repr uses these:
    from numpy import array, uint16, uint32, int16, int32, float32, float64, complex64, complex128, ndarray
    from astropy.units import Unit

    try:
        import astropy.io.fits
        from distutils.version import LooseVersion
        if LooseVersion(astropy.__version__) < LooseVersion('1.0.6'):
            irreprable = True
    except:
        import pyfits
    print('Try pickling ',str(obj1))

    #print('pickled obj1 = ',pickle.dumps(obj1))
    obj2 = pickle.loads(pickle.dumps(obj1))
    assert obj2 is not obj1
    #print('obj1 = ',repr(obj1))
    #print('obj2 = ',repr(obj2))
    f1 = func(obj1)
    f2 = func(obj2)
    #print('func(obj1) = ',repr(f1))
    #print('func(obj2) = ',repr(f2))
    assert f1 == f2

    # Check that == works properly if the other thing isn't the same type.
    assert f1 != object()
    assert object() != f1

    # Test the hash values are equal for two equivalent objects.
    from collections import Hashable
    if isinstance(obj1, Hashable):
        #print('hash = ',hash(obj1),hash(obj2))
        assert hash(obj1) == hash(obj2)

    obj3 = copy.copy(obj1)
    assert obj3 is not obj1
    random = hasattr(obj1, 'rng') or 'rng' in repr(obj1)
    if not random:  # Things with an rng attribute won't be identical on copy.
        f3 = func(obj3)
        assert f3 == f1

    obj4 = copy.deepcopy(obj1)
    assert obj4 is not obj1
    f4 = func(obj4)
    if random: f1 = func(obj1)
    #print('func(obj1) = ',repr(f1))
    #print('func(obj4) = ',repr(f4))
    assert f4 == f1  # But everything should be identical with deepcopy.

    # Also test that the repr is an accurate representation of the object.
    # The gold standard is that eval(repr(obj)) == obj.  So check that here as well.
    # A few objects we don't expect to work this way in GalSim; when testing these, we set the
    # `irreprable` kwarg to true.  Also, we skip anything with random deviates since these don't
    # respect the eval/repr roundtrip.

    if not random and not irreprable:
        obj5 = eval(repr(obj1))
        #print('obj1 = ',repr(obj1))
        #print('obj5 = ',repr(obj5))
        f5 = func(obj5)
        #print('f1 = ',f1)
        #print('f5 = ',f5)
        assert f5 == f1, "func(obj1) = %r\nfunc(obj5) = %r"%(f1, f5)
    else:
        # Even if we're not actually doing the test, still make the repr to check for syntax errors.
        repr(obj1)

def all_obj_diff(objs, check_hash=True):
    """ Helper function that verifies that each element in `objs` is unique and, if hashable,
    produces a unique hash."""

    from collections import Hashable
    # Check that all objects are unique.
    # Would like to use `assert len(objs) == len(set(objs))` here, but this requires that the
    # elements of objs are hashable (and that they have unique hashes!, which is what we're trying
    # to test!.  So instead, we just loop over all combinations.
    for i, obji in enumerate(objs):
        assert obji == obji
        assert not (obji != obji)
        # Could probably start the next loop at `i+1`, but we start at 0 for completeness
        # (and to verify a != b implies b != a)
        for j, objj in enumerate(objs):
            if i == j:
                continue
            assert obji != objj, ("Found equivalent objects {0} == {1} at indices {2} and {3}"
                                  .format(obji, objj, i, j))

    if not check_hash:
        return
    # Now check that all hashes are unique (if the items are hashable).
    if not isinstance(objs[0], Hashable):
        return
    hashes = [hash(obj) for obj in objs]
    try:
        assert len(hashes) == len(set(hashes))
    except AssertionError as e:
        try:
            # Only valid in 2.7, but only needed if we have an error to provide more information.
            from collections import Counter
        except ImportError:
            raise e
        for k, v in Counter(hashes).items():
            if v <= 1:
                continue
            print("Found multiple equivalent object hashes:")
            for i, obj in enumerate(objs):
                if hash(obj) == k:
                    print(i, repr(obj))
        raise e


def funcname():
    import inspect
    return inspect.stack()[1][3]


def timer(f):
    import functools

    @functools.wraps(f)
    def f2(*args, **kwargs):
        import time
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        fname = repr(f).split()[1]
        print('time for %s = %.2f' % (fname, t1-t0))
        return result
    return f2


class CaptureLog(object):
    """A context manager that saves logging output into a string that is accessible for
    checking in unit tests.

    After exiting the context, the attribute `output` will have the logging output.

    Sample usage:

            >>> with CaptureLog() as cl:
            ...     cl.logger.info('Do some stuff')
            >>> assert cl.output == 'Do some stuff'

    """
    def __init__(self, level=3):
        logging_levels = { 0: logging.CRITICAL,
                           1: logging.WARNING,
                           2: logging.INFO,
                           3: logging.DEBUG }
        self.logger = logging.getLogger('CaptureLog')
        self.logger.setLevel(logging_levels[level])
        try:
            from StringIO import StringIO
        except ImportError:
            from io import StringIO
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.logger.addHandler(self.handler)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.handler.flush()
        self.output = self.stream.getvalue().strip()
        self.handler.close()


# Replicate a small part of the nose package to get the `assert_raises` function/context-manager
# without relying on nose as a dependency.
import unittest
class Dummy(unittest.TestCase):
    def nop():
        pass
_t = Dummy('nop')
assert_raises = getattr(_t, 'assertRaises')
#if sys.version_info > (3,2):
if False:
    # Note: this should work, but at least sometimes it fails with:
    #    RuntimeError: dictionary changed size during iteration
    # cf. https://bugs.python.org/issue29620
    # So just use our own (working) implementation for all Python versions.
    assert_warns = getattr(_t, 'assertWarns')
else:
    from contextlib import contextmanager
    import warnings
    @contextmanager
    def assert_warns_context(wtype):
        # When used as a context manager
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            yield w
        assert len(w) >= 1, "Expected warning %s was not raised."%(wtype)
        assert issubclass(w[0].category, wtype), \
                "Warning raised was the wrong type (got %s, expected %s)"%(
                w[0].category, wtype)

    def assert_warns(wtype, *args, **kwargs):
        if len(args) == 0:
            return assert_warns_context(wtype)
        else:
            # When used as a regular function
            func = args[0]
            args = args[1:]
            with assert_warns(wtype):
                res = func(*args, **kwargs)
            return res

del Dummy
del _t

# Context to make it easier to profile bits of the code
class profile(object):
    def __init__(self, sortby='tottime', nlines=30):
        self.sortby = sortby
        self.nlines = nlines

    def __enter__(self):
        import cProfile, pstats
        self.pr = cProfile.Profile()
        self.pr.enable()
        return self

    def __exit__(self, type, value, traceback):
        import pstats
        self.pr.disable()
        ps = pstats.Stats(self.pr).sort_stats(self.sortby)
        ps.print_stats(self.nlines)
