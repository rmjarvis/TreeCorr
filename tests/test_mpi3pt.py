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
import unittest
import sys
if sys.version_info > (3,0):
    from mockmpi import mock_mpiexec

from test_helper import timer
from mpi_test3pt import *

@unittest.skipIf(sys.version_info < (3, 0), "mock_mpiexec doesn't support python 2")
@timer
def test_mpi_ggg():
    output = __name__ == '__main__'
    mock_mpiexec(1, do_mpi_ggg, output)
    mock_mpiexec(4, do_mpi_ggg, output)

@unittest.skipIf(sys.version_info < (3, 0), "mock_mpiexec doesn't support python 2")
@timer
def test_mpi_kkk():
    output = __name__ == '__main__'
    mock_mpiexec(1, do_mpi_kkk, output)
    mock_mpiexec(4, do_mpi_kkk, output)

@unittest.skipIf(sys.version_info < (3, 0), "mock_mpiexec doesn't support python 2")
@timer
def test_mpi_kkk2():
    output = __name__ == '__main__'
    mock_mpiexec(4, do_mpi_kkk2, output)

if __name__ == '__main__':
    if sys.version_info > (3,0):
        print("mockmpi does not support python 2")
        exit()
    setup()
    test_mpi_ggg()
    test_mpi_kkk()
    test_mpi_kkk2()
