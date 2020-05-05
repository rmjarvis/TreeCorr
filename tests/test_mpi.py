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
import treecorr
import unittest
import sys

from test_helper import timer
from mock_mpi import mock_mpiexec
from mpi_test import *

@unittest.skipIf(sys.version_info < (3, 0), "mock_mpiexec doesn't support python 2")
@timer
def test_mpi_gg():
    mock_mpiexec(4, do_mpi_gg)
    mock_mpiexec(1, do_mpi_gg)

@unittest.skipIf(sys.version_info < (3, 0), "mock_mpiexec doesn't support python 2")
@timer
def test_mpi_ng():
    mock_mpiexec(4, do_mpi_ng)
    mock_mpiexec(1, do_mpi_ng)

@unittest.skipIf(sys.version_info < (3, 0), "mock_mpiexec doesn't support python 2")
@timer
def test_mpi_nk():
    mock_mpiexec(4, do_mpi_nk)
    mock_mpiexec(1, do_mpi_nk)

@unittest.skipIf(sys.version_info < (3, 0), "mock_mpiexec doesn't support python 2")
@timer
def test_mpi_nn():
    mock_mpiexec(4, do_mpi_nn)
    mock_mpiexec(1, do_mpi_nn)

@unittest.skipIf(sys.version_info < (3, 0), "mock_mpiexec doesn't support python 2")
@timer
def test_mpi_kg():
    mock_mpiexec(4, do_mpi_kg)
    mock_mpiexec(1, do_mpi_kg)

@unittest.skipIf(sys.version_info < (3, 0), "mock_mpiexec doesn't support python 2")
@timer
def test_mpi_kk():
    mock_mpiexec(4, do_mpi_kk)
    mock_mpiexec(1, do_mpi_kk)

if __name__ == '__main__':
    setup()
    test_mpi_gg()
    test_mpi_ng()
    test_mpi_nk()
    test_mpi_nn()
    test_mpi_kg()
    test_mpi_kk()
