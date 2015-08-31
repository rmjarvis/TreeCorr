/* Copyright (c) 2003-2014 by Mike Jarvis
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

#include "Cell.h"

// The C++ class
template <int DC, int M>
class Field
{
public:
    Field(double* x, double* y, double* z, double* g1, double* g2, double* k, double* w,
          long nobj, double minsep, double maxsep, double b, int sm_int, int maxtop,
          bool spher=false);
    ~Field();

    long getNObj() const { return _nobj; }
    long getNTopLevel() const { return long(_cells.size()); }
    const std::vector<Cell<DC,M>*>& getCells() const { return _cells; }

private:

    long _nobj;
    double _minsep;
    double _maxsep;
    double _b;
    SplitMethod _sm;
    std::vector<Cell<DC,M>*> _cells;
};

// A SimpleField just stores the celldata.  It doesn't go on to build up the Cells.
// It is used by processPairwise.
template <int DC, int M>
class SimpleField
{
public:
    SimpleField(double* x, double* y, double* z, double* g1, double* g2, double* k, double* w,
                long nobj, bool spher=false);
    ~SimpleField();

    long getNObj() const { return long(_cells.size()); }
    const std::vector<Cell<DC,M>*>& getCells() const { return _cells; }

private:
    std::vector<Cell<DC,M>*> _cells;
};

// The C interface for python
extern "C" {

    extern void* BuildGFieldFlat(double* x, double* y, double* g1, double* g2, double* w,
                                 long nobj, double minsep, double maxsep, double b, int sm_int,
                                 int maxtop);
    extern void* BuildGField3D(double* x, double* y, double* z,
                               double* g1, double* g2, double* w,
                               long nobj, double minsep, double maxsep, double b, int sm_int,
                               int maxtop, int spher);
    extern void* BuildGFieldPerp(double* x, double* y, double* z,
                                 double* g1, double* g2, double* w,
                                 long nobj, double minsep, double maxsep, double b, int sm_int,
                                 int maxtop);

    extern void* BuildKFieldFlat(double* x, double* y, double* k, double* w,
                                 long nobj, double minsep, double maxsep, double b, int sm_int,
                                 int maxtop);
    extern void* BuildKField3D(double* x, double* y, double* z, double* k, double* w,
                               long nobj, double minsep, double maxsep, double b, int sm_int,
                               int maxtop, int spher);
    extern void* BuildKFieldPerp(double* x, double* y, double* z, double* k, double* w,
                                 long nobj, double minsep, double maxsep, double b, int sm_int,
                                 int maxtop);

    extern void* BuildNFieldFlat(double* x, double* y, double* w,
                                 long nobj, double minsep, double maxsep, double b, int sm_int,
                                 int maxtop);
    extern void* BuildNField3D(double* x, double* y, double* z, double* w,
                               long nobj, double minsep, double maxsep, double b, int sm_int,
                               int maxtop, int spher);
    extern void* BuildNFieldPerp(double* x, double* y, double* z, double* w,
                                 long nobj, double minsep, double maxsep, double b, int sm_int,
                                 int maxtop);

    extern void DestroyGFieldFlat(void* field);
    extern void DestroyGField3D(void* field);
    extern void DestroyGFieldPerp(void* field);

    extern void DestroyKFieldFlat(void* field);
    extern void DestroyKField3D(void* field);
    extern void DestroyKFieldPerp(void* field);

    extern void DestroyNFieldFlat(void* field);
    extern void DestroyNField3D(void* field);
    extern void DestroyNFieldPerp(void* field);


    extern void* BuildGSimpleFieldFlat(double* x, double* y,
                                       double* g1, double* g2, double* w, long nobj);
    extern void* BuildGSimpleField3D(double* x, double* y, double* z,
                                     double* g1, double* g2, double* w, long nobj, int spher);
    extern void* BuildGSimpleFieldPerp(double* x, double* y, double* z,
                                       double* g1, double* g2, double* w, long nobj);

    extern void* BuildKSimpleFieldFlat(double* x, double* y,
                                       double* k, double* w, long nobj);
    extern void* BuildKSimpleField3D(double* x, double* y, double* z,
                                     double* k, double* w, long nobj, int spher);
    extern void* BuildKSimpleFieldPerp(double* x, double* y, double* z,
                                       double* k, double* w, long nobj);

    extern void* BuildNSimpleFieldFlat(double* x, double* y, double* w, long nobj);
    extern void* BuildNSimpleField3D(double* x, double* y, double* z,
                                     double* w, long nobj, int spher);
    extern void* BuildNSimpleFieldPerp(double* x, double* y, double* z, double* w, long nobj);

    extern void DestroyGSimpleFieldFlat(void* field);
    extern void DestroyGSimpleField3D(void* field);
    extern void DestroyGSimpleFieldPerp(void* field);

    extern void DestroyKSimpleFieldFlat(void* field);
    extern void DestroyKSimpleField3D(void* field);
    extern void DestroyKSimpleFieldPerp(void* field);

    extern void DestroyNSimpleFieldFlat(void* field);
    extern void DestroyNSimpleField3D(void* field);
    extern void DestroyNSimpleFieldPerp(void* field);

    extern long GFieldFlatGetNTopLevel(void* field);
    extern long GField3DGetNTopLevel(void* field);
    extern long GFieldPerpGetNTopLevel(void* field);
    extern long KFieldFlatGetNTopLevel(void* field);
    extern long KField3DGetNTopLevel(void* field);
    extern long KFieldPerpGetNTopLevel(void* field);
    extern long NFieldFlatGetNTopLevel(void* field);
    extern long NField3DGetNTopLevel(void* field);
    extern long NFieldPerpGetNTopLevel(void* field);

}

