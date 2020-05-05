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
import numpy as np

# This file builds a framework for testing code that is designed to work in an
# mpi4py MPI session, but without requiring MPI.  It uses multiprocessing to
# start n processes and makes a mocked Comm object that takes the place of 
# mpi4py.COMM_WORLD.
#
# Usage:
#
# Define a function that holds the code that should work in an MPI session.
#
#     def mpi_session(comm):
#         ...
#
# If this were a real MPI session, it should work as
#
#     import mpi4py
#     comm = MPI.COMM_WORLD
#     mpi_session(comm)
#
# which you would run with mpiexec -n nproc ...
#
# To test the code without MPI, import mock_mpiexec from this file and run
#
#     mock_mpiexec(nproc, mpi_session)
#
# Caveats:
#
#   1. Absolutely no attention was paid to making this efficient.  This code
#      is designed to be used for unit testing, not production.
#   2. I only implemented the lowercase methods of Comm.  It would probably
#      be pretty simple to define Send, Recv, etc. to be equivalent to send,
#      recv, etc., but I haven't bothered.
#   3. I also didn't implement isend and irecv.  Sorry.  I don't use them,
#      so I didn't bother to understand them well enough to implement here.
#   4. There may be other functions in the mpi4py API that I didn't implement.
#      The documentation of mpi4py is pretty terrible, so while I tried to
#      identify the main functionality, it's very likely I missed some things.
#   5. It doesn't work on python 2.


class MockComm(object):
    """A class to mock up the MPI Comm API for a multiprocessing Pipe.

    """
    def __init__(self, rank, size, pipes, barrier):
        self.rank = rank
        self.size = size
        self.pipes = pipes
        self.barrier = barrier

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def send(self, msg, dest):
        if dest != self.rank:
            self.pipes[dest].send(msg)
        else:
            self.msg = msg

    def recv(self, source):
        if source != self.rank:
            msg = self.pipes[source].recv()
        else:
            msg = self.msg
        return msg

    def Barrier(self):
        self.barrier.wait()

    def bcast(self, msg, root):
        if root == self.rank:
            for p in range(self.size):
                self.send(msg,p)
        msg = self.recv(root)
        return msg

    def scatter(self, data, root):
        if root == self.rank:
            for p in range(self.size):
                self.send(data[p],p)
        data = self.recv(root)
        return data

    def gather(self, data, root):
        self.send(data,root)
        if root == self.rank:
            new_data = []
            for p in range(self.size):
                new_data.append(self.recv(p))
            return new_data
        else:
            return None

    def alltoall(self, data):
        for p in range(self.size):
            self.send(data[p],p)
        new_data = []
        for p in range(self.size):
            new_data.append(self.recv(p))
        return new_data


def mock_mpiexec(nproc, target):
    """Run a function, given as target, as though it were an MPI session using mpiexec -n nproc
    but using multiprocessing instead of mpi.
    """
    from multiprocessing import Pipe, Process, Barrier, set_start_method
    set_start_method('spawn', force=True)

    # Make the message passing pipes
    all_pipes = [ {} for p in range(nproc) ]
    for i in range(nproc):
        for j in range(i+1,nproc):
            p1, p2 = Pipe()
            all_pipes[i][j] = p1
            all_pipes[j][i] = p2

    # Make a barrier
    barrier = Barrier(nproc)

    # Make fake MPI-like comm object
    comms = [ MockComm(rank, nproc, pipes, barrier) for rank,pipes in enumerate(all_pipes) ]

    # Make processes
    procs = [ Process(target=target, args=(comm,)) for comm in comms ]

    for p in procs:
        p.start()
    
    for p in procs:
        p.join()

def test_mpi_session(comm):
    """A simple MPI session we want to run in mock MPI mode.

    This serves as a test of the above code.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank,'rank, size = ',rank,size,flush=True)

    comm.send('Hello! My rank is %d.'%rank, dest=0)
    print(rank,'sent hello message ',flush=True)

    if rank == 0:
        for p in range(size):
            print(rank,'try to read from ',p,flush=True)
            msg = comm.recv(source=p)
            print(rank,'received message: ',repr(msg),flush=True)

    print(rank,'Before barrier',flush=True)
    comm.Barrier()
    print(rank,'After barrier',flush=True)

    if rank == 0:
        data = np.arange(size) + 10
    else:
        data = None

    print(rank,'Before bcast: data = ',data,flush=True)
    data = comm.bcast(data, root=0)
    print(rank,'After bcast: data = ',data,flush=True)
    np.testing.assert_array_equal(data, np.arange(size) + 10)
    comm.Barrier()

    if rank != 0:
        data = None

    print(rank,'Before scatter: data = ',data,flush=True)
    data = comm.scatter(data, root=0)
    print(rank,'After scatter: data = ',data,flush=True)
    assert data == rank + 10
    comm.Barrier()

    print(rank,'Before gather: data = ',data,flush=True)
    data = comm.gather(data, root=0)
    print(rank,'After gather: data = ',data,flush=True)
    if rank == 0:
        np.testing.assert_array_equal(data, np.arange(size) + 10)
    else:
        assert data is None
    comm.Barrier()

    data = np.arange(size) + rank**2 + 5
    print(rank,'Before alltoall: data = ',data,flush=True)
    data = comm.alltoall(data)
    print(rank,'After alltoall: data = ',data,flush=True)
    np.testing.assert_array_equal(data, np.arange(size)**2 + rank + 5)


if __name__ == '__main__':
    # Test this code.
    mock_mpiexec(4, test_mpi_session)
