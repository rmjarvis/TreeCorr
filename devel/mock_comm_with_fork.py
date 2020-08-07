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
import os
import time

# This is an attempt to use fork to mock up an MPI session.
# It doesn't work.  And I've given up trying to figure out how to make it work.

class MockMPI(object):
    """A context manager that mocks up an MPI session using fork, so it can be run
    in normal unit testing.

    It makes no attempt to be efficient, so it is really only useful for unit testing
    functions that are intended to use an mpi4py Comm object.

    It can also only communicate between rank=0 and other ranks.  So it won't work for
    use cases that need communication among all the different ranks.

    TODO: So far it only implements send and recv communication, not the more complicated
          bcast, scatter, etc.

    Sample usage:

            >>> with MockMPI(size=4) as comm:
            ...     rank = comm.Get_rank()
            ...     size = comm.Get_size()
            ...     print('rank, size = ',rank,size,flush=True)
    """
    def __init__(self, size=2):
        self.size = size
        self.rank = 0
        self.write_pipes = {}
        self.read_pipes = {}

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def send(self, msg, dest):
        print(self.rank,'sending to ',dest,msg,flush=True)
        if dest == self.rank:
            self.self_msg = msg
        else:
            print(self.rank,'writing on ',self.write_pipes[dest],flush=True)
            self.write_pipes[dest].write(msg)
            self.write_pipes[dest].flush()
            #print('fno = ',self.write_pipes[dest].name, self.write_pipes[dest].fileno(),flush=True)
            #fno = self.write_pipes[dest].fileno()
            #self.write_pipes[dest].close()
            #self.write_pipes[dest] = os.fdopen(fno,'w')
            #print(self.rank,'reopened ',self.write_pipes[dest],flush=True)
        print(self.rank,'sent to ',dest,flush=True)

    def recv(self, source):
        print(self.rank,'receiving from ',source,flush=True)
        if source == self.rank:
            msg = self.self_msg
        else:
            print(self.rank,'reading from ',self.read_pipes[source],flush=True)
            msg = self.read_pipes[source].read()
        print(self.rank,'received from ',source,msg,flush=True)
        return msg

    def Barrier(self):
        print(self.rank,'staring Barrier',flush=True)
        # Sync up by checking in with everyone
        # 0 -> all, then reply back to 0.
        if self.rank == 0:
            for p in range(1,self.size):
                self.send('check',p)
            for p in range(1,self.size):
                self.recv(p)
        else:
            self.recv(0)
            self.send('ready',0)

    def __enter__(self):
        size = self.size
        next_rank = 1
        while size > 1:
            r1,w1 = os.pipe()  # communication from 0 to rank
            r2,w2 = os.pipe()  # communication from rank to 0
            print('pipes for',next_rank,'are',r1,w1,r2,w2)
            pid = os.fork()
            if pid:
                # Parent
                os.close(r1)
                os.close(w2)
                self.read_pipes[next_rank] = r2
                self.write_pipes[next_rank] = w1
                next_rank += 1
                size -= 1
            else:
                # Child
                self.rank = next_rank
                size = 0  # Don't do further forks from non-parent.
                os.close(r2)
                os.close(w1)
                # Clear these, since it gets copies of the rank 0 ones, which we don't want
                self.read_pipes.clear()
                self.write_pipes.clear()
                self.read_pipes[0] = r1
                self.write_pipes[0] = w2
        if self.rank == 0:
            # Let rank 0 read/write to itself.
            r,w = os.pipe()
            self.read_pipes[0] = r
            self.write_pipes[0] = w
        for p in self.read_pipes:
            self.read_pipes[p] = os.fdopen(self.read_pipes[p])
        for p in self.write_pipes:
            os.set_blocking(self.write_pipes[p],False)
            self.write_pipes[p] = os.fdopen(self.write_pipes[p],'w')
        return self

    def __exit__(self, type, value, traceback):
        print(rank, 'is exiting.')
        #self.Barrier()
        # I can't figure out how to make a Barrier work right.
        for p in self.write_pipes:
            self.write_pipes[p].close()
        print(rank, 'closed all writes')
        # Without this it fails
        time.sleep(3)
        for p in self.read_pipes:
            self.read_pipes[p].close()
        print(rank, 'closed all reads')
        if self.rank > 0:
            os._exit(0)
        print(rank, 'exited')

msg = 'default_msg'

with MockMPI(2) as comm:
    rank = comm.Get_rank()
    size = comm.Get_size()
    print('rank, size = ',rank,size,flush=True)
    print(rank, 'can read from ',list(comm.read_pipes.keys()),flush=True)
    print(rank, 'can write to ',list(comm.write_pipes.keys()),flush=True)

    comm.send('my rank is %d'%rank, dest=0)

    if rank == 0:
        print('Final section',flush=True)
        for p in range(size):
            print('Try to read from ',p,flush=True)
            msg = comm.recv(source=p)
            print('rank 0 received message: ',msg,flush=True)
            comm.send('done', dest=p)

    print(rank,'done',flush=True)
    # This next line causes it to freeze.
    #final_msg = comm.recv(0)
    #print(rank,'final message = ',final_msg,flush=True)
