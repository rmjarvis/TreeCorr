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
from mockmpi import mock_mpiexec

from test_helper import timer
from mpi_test import setup, do_mpi_gg, do_mpi_ng, do_mpi_nk, do_mpi_nn, do_mpi_kk, do_mpi_kg, do_mpi_cov

@timer
def test_mpi_gg():
    output = __name__ == '__main__'
    mock_mpiexec(4, do_mpi_gg, output)
    mock_mpiexec(1, do_mpi_gg, output)

@timer
def test_mpi_ng():
    output = __name__ == '__main__'
    mock_mpiexec(4, do_mpi_ng, output)
    mock_mpiexec(1, do_mpi_ng, output)

@timer
def test_mpi_nk():
    output = __name__ == '__main__'
    mock_mpiexec(4, do_mpi_nk, output)
    mock_mpiexec(1, do_mpi_nk, output)

@timer
def test_mpi_nn():
    output = __name__ == '__main__'
    mock_mpiexec(4, do_mpi_nn, output)
    mock_mpiexec(1, do_mpi_nn, output)

@timer
def test_mpi_kg():
    output = __name__ == '__main__'
    mock_mpiexec(4, do_mpi_kg, output)
    mock_mpiexec(1, do_mpi_kg, output)

@timer
def test_mpi_kk():
    output = __name__ == '__main__'
    mock_mpiexec(4, do_mpi_kk, output)
    mock_mpiexec(1, do_mpi_kk, output)


@unittest.skipIf(sys.version_info < (3, 0), "mock_mpiexec doesn't support python 2")
@timer
def test_mpi_cov_jackknife():
    mock_mpiexec(1, do_mpi_cov, "jackknife")
    mock_mpiexec(2, do_mpi_cov, "jackknife")
    mock_mpiexec(4, do_mpi_cov, "jackknife")

@unittest.skipIf(sys.version_info < (3, 0), "mock_mpiexec doesn't support python 2")
@timer
def test_mpi_cov_bootstrap():
    mock_mpiexec(1, do_mpi_cov, "bootstrap")
    mock_mpiexec(2, do_mpi_cov, "bootstrap")
    mock_mpiexec(4, do_mpi_cov, "bootstrap")

@unittest.skipIf(sys.version_info < (3, 0), "mock_mpiexec doesn't support python 2")
@timer
def test_mpi_cov_marked_bootstrap():
    mock_mpiexec(1, do_mpi_cov, "marked_bootstrap")
    mock_mpiexec(2, do_mpi_cov, "marked_bootstrap")
    mock_mpiexec(4, do_mpi_cov, "marked_bootstrap")

@unittest.skipIf(sys.version_info < (3, 0), "mock_mpiexec doesn't support python 2")
@timer
def test_mpi_cov_sample():
    mock_mpiexec(1, do_mpi_cov, "sample")
    mock_mpiexec(2, do_mpi_cov, "sample")
    mock_mpiexec(4, do_mpi_cov, "sample")


if __name__ == '__main__':
    setup()
    test_mpi_gg()
    test_mpi_ng()
    test_mpi_nk()
    test_mpi_nn()
    test_mpi_kg()
    test_mpi_kk()
