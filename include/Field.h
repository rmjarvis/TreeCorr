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

#ifndef TreeCorr_Field_H
#define TreeCorr_Field_H

#if defined(__GNUC__) && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "Cell.h"

// Most of the functionality for building Cells and doing the correlation functions is the
// same regardless of which kind of Cell we have (N, K, G) or which kind of positions we
// are using (Flat, ThreeD, Sphere), or what metric we use for the distances between points
// (Euclidean, Rperp, Rlens, Arc).  So most of the C++ code uses templates.
//
// D = NData for counts
//     KData for kappa
//     ZData for spin-0
//     VData for vector
//     GData for shear
//     TData for spin-3
//     QData for spin-4
//
// C = Flat for (x,y) coordinates
//     ThreeD for (x,y,z) coordinates
//     Sphere for spherical coordinates
//
// M = Euclidean for Euclidean distances in either (x,y) or (x,y,z)
//     Rperp for the perpendicular component of the 3D distance
//     Rlens for the perpendicular component at the location of the "lens" (c1)
//     Arc for great circle distances on the sphere

template <int C>
class BaseField
{
public:
    BaseField(long nobj, double minsize, double maxsize,
              SplitMethod sm, long long seed, bool brute, int mintop, int maxtop);
    virtual ~BaseField();

    long getNObj() const { return _nobj; }
    double getSizeSq() const { return _sizesq; }
    Position<C> getCenter() const { return _center; }
    double getSize() const { return std::sqrt(_sizesq); }
    long getNTopLevel() const { BuildCells(); return long(_cells.size()); }
    const std::vector<const BaseCell<C>*>& getCells() const
    {
        BuildCells();
        // const_cast is insufficient to turn this into a vector of const BaseCell*.
        // cf. https://stackoverflow.com/questions/19122858/why-is-a-vector-of-pointers-not-castable-to-a-const-vector-of-const-pointers
        // But reinterpret_cast here is safe.
        return reinterpret_cast<std::vector<const BaseCell<C>*>&>(_cells);
    }
    long countNear(double x, double y, double z, double sep) const;
    void getNear(double x, double y, double z, double sep, long* indices, long n) const;

    long _nobj;
    double _minsize;
    double _maxsize;
    SplitMethod _sm;
    bool _brute;
    int _mintop;
    int _maxtop;
    Position<C> _center;
    double _sizesq;
    mutable std::vector<BaseCell<C>*> _cells;

    // This is set at the start, but once we finish making all the cells, we don't need it anymore.
    mutable std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> > _celldata;

protected:
    virtual void BuildCells() const = 0;
};

template <int D, int C>
class Field : public BaseField<C>
{
public:
    Field(const double* x, const double* y, const double* z,
          const double* d1, const double* d2,
          const double* w, const double* wpos, long nobj,
          double minsize, double maxsize,
          SplitMethod sm, long long seed, bool brute, int mintop, int maxtop);

protected:
    // This finishes the work of the Field constructor.
    void BuildCells() const;

private:

    template <int SM>
    void DoBuildCells() const;
};

#endif
