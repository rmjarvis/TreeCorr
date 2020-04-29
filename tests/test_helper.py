# Copyright (c) 2003-2019 by Mike Jarvis
#
# TreeCorr is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

from __future__ import print_function
import logging
import sys
import os
import time

def get_from_wiki(file_name, host=None):
    """We host some larger files used for the test suite separately on the TreeCorr wiki repo
    so people don't need to download them with the code when checking out the repo.
    Most people don't run the tests after all.

    The default host is the wiki page, but you can also download from a different host url.
    """
    local_file_name = os.path.join('data',file_name)
    if host is None:
        host = 'https://github.com/rmjarvis/TreeCorr/wiki/'
    url = host + file_name
    if not os.path.isfile(local_file_name):
        try:
            from urllib.request import urlopen
        except ImportError:
            from urllib import urlopen
        import shutil

        print('downloading %s from %s...'%(local_file_name,url))
        # urllib.request.urlretrieve(url,local_file_name)
        # The above line doesn't work very well with the SSL certificate that github puts on it.
        # It works fine in a web browser, but on my laptop I get:
        # urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:600)>
        # The solution is to open a context that doesn't do ssl verification.
        # But that can only be done with urlopen, not urlretrieve.  So, here is the solution.
        # cf. http://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
        #     http://stackoverflow.com/questions/27835619/ssl-certificate-verify-failed-error
        try:
            import ssl
            context = ssl._create_unverified_context()
            u = urlopen(url, context=context)
        except (AttributeError, TypeError):
            # Note: prior to 2.7.9, there is no such function or even the context keyword.
            u = urlopen(url)
        with open(local_file_name, 'wb') as out:
            shutil.copyfileobj(u, out)
        u.close()
        print('done.')

def which(program):
    """
    Mimic functionality of unix which command
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    if sys.platform == "win32" and not program.endswith(".exe"):
        program += ".exe"

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None

def get_script_name(file_name):
    """
    Check if the file_name is in the path.  If not, prepend appropriate path to it.
    """
    if which(file_name) is not None:
        return file_name
    else:
        test_dir = os.path.split(os.path.realpath(__file__))[0]
        root_dir = os.path.split(test_dir)[0]
        script_dir = os.path.join(root_dir, 'scripts')
        exe_file_name = os.path.join(script_dir, file_name)
        print('Warning: The script %s is not in the path.'%file_name)
        print('         Using explcit path for the test:',exe_file_name)
        return exe_file_name

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
# Usage:
#   with profile():
#       do_something

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


def do_pickle(obj1, func = lambda x : x):
    """Check that the object is picklable.  Also that it has basic == and != functionality.
    """
    try:
        import cPickle as pickle
    except:
        import pickle
    import copy
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

    obj3 = copy.copy(obj1)
    assert obj3 is not obj1
    f3 = func(obj3)
    assert f3 == f1

    obj4 = copy.deepcopy(obj1)
    assert obj4 is not obj1
    f4 = func(obj4)
    assert f4 == f1

try:
    from mpi4py import MPI
except ImportError:
    pass
else:
    # cf. https://groups.google.com/forum/#!topic/mpi4py/nArVuMXyyZI
    # This wrapper stops comm.Barrier() from taking 100% cpu on the processes that are
    # just sitting around waiting for rank 0 to be done.
    class NiceComm(MPI.Intracomm):
        """Wrapper to mpi4py's MPI.Intracomm class to avoid busy-waiting.

        As suggested by Lisandro Dalcin at:
        * http://code.google.com/p/mpi4py/issues/detail?id=4 and
        * https://groups.google.com/forum/?fromgroups=#!topic/mpi4py/nArVuMXyyZI
        """
        def __new__(cls, comm=MPI.COMM_WORLD, recvSleep=0.01, barrierSleep=0.1):
            """Construct an MPI.Comm wrapper

            @param comm            MPI.Intracomm to wrap a duplicate of
            @param recvSleep       Sleep time (seconds) for recv()
            @param barrierSleep    Sleep time (seconds) for Barrier()
            """
            self = super(NiceComm, cls).__new__(cls, comm.Dup())

            # Duplicate communicator used for Barrier point-to-point checking
            self._barrierComm = None

            self._recvSleep = recvSleep
            self._barrierSleep = barrierSleep
            return self

        def recv(self, obj=None, source=0, tag=0, status=None):
            """Version of comm.recv() that doesn't busy-wait"""
            sts = MPI.Status()
            while not self.Iprobe(source=source, tag=tag, status=sts):
                time.sleep(self._recvSleep)
            return super(NiceComm, self).recv(obj, source=sts.source, tag=sts.tag, status=status)

        def _checkBarrierComm(self):
            """Ensure the duplicate communicator is available"""
            if self._barrierComm is None:
                self._barrierComm = self.Dup()

        def Free(self):
            if self._barrierComm is not None:
                self._barrierComm.Free()
            super(NiceComm, self).Free()

        def Barrier(self, tag=0):
            """Version of comm.Barrier() that doesn't busy-wait

            A duplicate communicator is used so as not to interfere with the user's own
            communications.
            """
            self._checkBarrierComm()
            size = self._barrierComm.Get_size()

            if size == 1:
                return

            rank = self._barrierComm.Get_rank()

            mask = 1
            while mask < size:
                dst = (rank + mask) % size
                src = (rank - mask + size) % size
                req = self._barrierComm.isend(None, dst, tag)
                while not self._barrierComm.Iprobe(src, tag):
                    time.sleep(self._barrierSleep)
                self._barrierComm.recv(None, src, tag)
                req.Wait()
                mask <<= 1
