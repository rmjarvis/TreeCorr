/* Copyright (c) 2003-2024 by Mike Jarvis
 *
 * TreeCorr is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

#include "PyBind11Helper.h"

void pyExportField(py::module&);
void pyExportCorr2(py::module&);
void pyExportCorr3(py::module&);

PYBIND11_MODULE(_treecorr, _treecorr)
{
    pyExportField(_treecorr);
    pyExportCorr2(_treecorr);
    pyExportCorr3(_treecorr);
}
