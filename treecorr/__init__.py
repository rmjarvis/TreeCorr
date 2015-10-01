# Copyright (c) 2003-2015 by Mike Jarvis
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


version = '3.2.0'

from . import util
from .celestial import CelestialCoord, angle_units, arcsec, arcmin, degrees, hours, radians
from .config import read_config, set_omp_threads
from .catalog import Catalog, read_catalogs, calculateVarG, calculateVarK
from .binnedcorr2 import BinnedCorr2
from .ggcorrelation import GGCorrelation
from .nncorrelation import NNCorrelation
from .kkcorrelation import KKCorrelation
from .ngcorrelation import NGCorrelation
from .nkcorrelation import NKCorrelation
from .kgcorrelation import KGCorrelation
from .field import NField, KField, GField, NSimpleField, KSimpleField, GSimpleField
from .binnedcorr3 import BinnedCorr3
from .nnncorrelation import NNNCorrelation
from .kkkcorrelation import KKKCorrelation
from .gggcorrelation import GGGCorrelation
from .corr2 import corr2, print_corr2_params
from .corr3 import corr3, print_corr3_params
