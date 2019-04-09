/* Copyright (c) 2003-2019 by Mike Jarvis
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

#ifndef TreeCorr_Field_H
#define TreeCorr_Field_H

#include "Cell.h"

// Most of the functionality for building Cells and doing the correlation functions is the
// same regardless of which kind of Cell we have (N, K, G) or which kind of positions we
// are using (Flat, ThreeD, Sphere), or what metric we use for the distances between points
// (Euclidean, Rperp, Rlens, Arc).  So most of the C++ code uses templates.
//
// D = NData for counts
//     KData for kappa
//     GData for shear
//
// C = Flat for (x,y) coordinates
//     ThreeD for (x,y,z) coordinates
//     Sphere for spherical coordinates
//
// M = Euclidean for Euclidean distances in either (x,y) or (x,y,z)
//     Rperp for the perpendicular component of the 3D distance
//     Rlens for the perpendicular component at the location of the "lens" (c1)
//     Arc for great circle distances on the sphere

template <int D, int C>
class Field
{
public:
    Field(double* x, double* y, double* z, double* g1, double* g2, double* k,
          double* w, double* wpos, long nobj,
          double minsize, double maxsize,
          int sm_int, bool brute, int mintop, int maxtop);
    ~Field();

    long getNObj() const { return _nobj; }
    long getNTopLevel() const { return long(_cells.size()); }
    const std::vector<Cell<D,C>*>& getCells() const { return _cells; }
    long countNear(double x, double y, double z, double sep) const;
    void getNear(double x, double y, double z, double sep, long* indices, int n) const;

private:

    long _nobj;
    double _minsize;
    double _maxsize;
    SplitMethod _sm;
    std::vector<Cell<D,C>*> _cells;
};

// A SimpleField just stores the celldata.  It doesn't go on to build up the Cells.
// It is used by processPairwise.
template <int D, int C>
class SimpleField
{
public:
    SimpleField(double* x, double* y, double* z, double* g1, double* g2, double* k,
                double* w, double* wpos, long nobj);
    ~SimpleField();

    long getNObj() const { return long(_cells.size()); }
    const std::vector<Cell<D,C>*>& getCells() const { return _cells; }

private:
    std::vector<Cell<D,C>*> _cells;
};

#endif
