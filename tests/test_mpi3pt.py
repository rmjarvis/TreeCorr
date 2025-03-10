# Copyright (c) 2003-2024 by Mike Jarvis
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

import sys

from test_helper import timer
from mpi_test3pt import *

try:
    import fitsio
    from mockmpi import mock_mpiexec
except ImportError:
    # All the mpi tests use Aardvark.fit, so skip them when fitsio isn't installed.
    # Also skip if mockmpi is not installed.
    skip=True
else:
    skip=False

@timer
def test_mpi_ggg():
    if skip: return
    output = __name__ == '__main__'
    mock_mpiexec(1, do_mpi_ggg, output)
    mock_mpiexec(4, do_mpi_ggg, output)

@timer
def test_mpi_kkk():
    if skip: return
    output = __name__ == '__main__'
    mock_mpiexec(1, do_mpi_kkk, output)
    mock_mpiexec(4, do_mpi_kkk, output)

@timer
def test_mpi_kkk2():
    if skip: return
    output = __name__ == '__main__'
    mock_mpiexec(4, do_mpi_kkk2, output)

@timer
def test_mpi_nnk():
    if skip: return
    output = __name__ == '__main__'
    mock_mpiexec(1, do_mpi_nnk, output)
    mock_mpiexec(4, do_mpi_nnk, output)

@timer
def test_mpi_nng():
    if skip: return
    output = __name__ == '__main__'
    mock_mpiexec(1, do_mpi_nng, output)
    mock_mpiexec(4, do_mpi_nng, output)

@timer
def test_mpi_nkk():
    if skip: return
    output = __name__ == '__main__'
    mock_mpiexec(1, do_mpi_nkk, output)
    mock_mpiexec(4, do_mpi_nkk, output)

@timer
def test_mpi_ngg():
    if skip: return
    output = __name__ == '__main__'
    mock_mpiexec(1, do_mpi_ngg, output)
    mock_mpiexec(4, do_mpi_ngg, output)

@timer
def test_mpi_kkg():
    if skip: return
    output = __name__ == '__main__'
    mock_mpiexec(1, do_mpi_kkg, output)
    mock_mpiexec(4, do_mpi_kkg, output)

@timer
def test_mpi_kgg():
    if skip: return
    output = __name__ == '__main__'
    mock_mpiexec(1, do_mpi_kgg, output)
    mock_mpiexec(4, do_mpi_kgg, output)


if __name__ == '__main__':
    setup()
    test_mpi_ggg()
    test_mpi_kkk()
    test_mpi_kkk2()
    test_mpi_nnk()
    test_mpi_nng()
    test_mpi_nkk()
    test_mpi_ngg()
    test_mpi_kkg()
    test_mpi_kgg()
