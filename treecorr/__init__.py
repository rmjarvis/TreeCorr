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

import os

# The version is stored in _version.py as recommended here:
# http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
from ._version import __version__, __version_info__

# Also let treecorr.version show the version.
version = __version__

# Set module level attributes for the include directory and the library file name.
treecorr_dir = os.path.dirname(__file__)
include_dir = os.path.join(treecorr_dir,'include')
from . import _treecorr
lib_file = os.path.abspath(_treecorr.__file__)

Rperp_alias = 'FisherRperp'

from .config import read_config
from .util import set_omp_threads, get_omp_threads, set_max_omp_threads

from .catalog import Catalog, read_catalogs
from .catalog import calculateVarK, calculateVarG, calculateVarZ
from .catalog import calculateVarV, calculateVarT, calculateVarQ

from .corr2base import Corr2, estimate_multi_cov, build_multi_cov_design_matrix
from .corr3base import Corr3
from .field import Field, NField, KField, ZField, VField, GField, TField, QField

from .nncorrelation import NNCorrelation
from .nkcorrelation import NKCorrelation
from .kkcorrelation import KKCorrelation

from .nzcorrelation import NZCorrelation, BaseNZCorrelation
from .kzcorrelation import KZCorrelation, BaseKZCorrelation
from .zzcorrelation import ZZCorrelation, BaseZZCorrelation

from .nvcorrelation import NVCorrelation
from .kvcorrelation import KVCorrelation
from .vvcorrelation import VVCorrelation

from .ngcorrelation import NGCorrelation
from .kgcorrelation import KGCorrelation
from .ggcorrelation import GGCorrelation

from .ntcorrelation import NTCorrelation
from .ktcorrelation import KTCorrelation
from .ttcorrelation import TTCorrelation

from .nqcorrelation import NQCorrelation
from .kqcorrelation import KQCorrelation
from .qqcorrelation import QQCorrelation

from .nnncorrelation import NNNCorrelation
from .kkkcorrelation import KKKCorrelation
from .gggcorrelation import GGGCorrelation

from .exec_corr2 import corr2, print_corr2_params, corr2_valid_params, corr2_aliases
from .exec_corr3 import corr3, print_corr3_params, corr3_valid_params, corr3_aliases
