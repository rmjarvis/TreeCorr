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

import time

from mpi4py import MPI

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
