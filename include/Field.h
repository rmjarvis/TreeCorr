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
    // Note: for M=Sphere, x,y here are really ra,dec.
    Field(double* x, double* y, double* r, double* g1, double* g2, double* k, double* w,
          long nobj, double minsep, double maxsep, double b, int sm_int, int maxtop);
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
    SimpleField(double* x, double* y, double* r, double* g1, double* g2, double* k, double* w,
                long nobj);
    ~SimpleField();

    long getNObj() const { return long(_cells.size()); }
    const std::vector<Cell<DC,M>*>& getCells() const { return _cells; }

private:
    std::vector<Cell<DC,M>*> _cells;
};

// The C interface for python
extern "C" {

    extern void* BuildGFieldFlat(double* x, double* y, double* g1, double* g2, double* w,
<<<<<<< HEAD
                                 long nobj, double minsep, double maxsep, double b, int sm_int,
                                 int maxtop);

    extern void* BuildGFieldSphere(double* ra, double* dec, double* r,
                                   double* g1, double* g2, double* w,
                                   long nobj, double minsep, double maxsep, double b, int sm_int,
                                   int maxtop);

    extern void* BuildKFieldFlat(double* x, double* y, double* k, double* w,
                                 long nobj, double minsep, double maxsep, double b, int sm_int,
                                 int maxtop);

    extern void* BuildKFieldSphere(double* ra, double* dec, double* r, double* k, double* w,
                                   long nobj, double minsep, double maxsep, double b, int sm_int,
                                   int maxtop);

    extern void* BuildNFieldFlat(double* x, double* y, double* w,
                                 long nobj, double minsep, double maxsep, double b, int sm_int,
                                 int maxtop);

    extern void* BuildNFieldSphere(double* ra, double* dec, double* r, double* w,
                                   long nobj, double minsep, double maxsep, double b, int sm_int,
                                   int maxtop);
=======
                                 long nobj, double minsep, double maxsep, double b, int sm_int);
    extern void* BuildGFieldSphere(double* ra, double* dec, double* r,
                                   double* g1, double* g2, double* w,
                                   long nobj, double minsep, double maxsep, double b, int sm_int);
    extern void* BuildGFieldPerp(double* ra, double* dec, double* r,
                                 double* g1, double* g2, double* w,
                                 long nobj, double minsep, double maxsep, double b, int sm_int);

    extern void* BuildKFieldFlat(double* x, double* y, double* k, double* w,
                                 long nobj, double minsep, double maxsep, double b, int sm_int);
    extern void* BuildKFieldSphere(double* ra, double* dec, double* r, double* k, double* w,
                                   long nobj, double minsep, double maxsep, double b, int sm_int);
    extern void* BuildKFieldPerp(double* ra, double* dec, double* r, double* k, double* w,
                                 long nobj, double minsep, double maxsep, double b, int sm_int);

    extern void* BuildNFieldFlat(double* x, double* y, double* w,
                                 long nobj, double minsep, double maxsep, double b, int sm_int);
    extern void* BuildNFieldSphere(double* ra, double* dec, double* r, double* w,
                                   long nobj, double minsep, double maxsep, double b, int sm_int);
    extern void* BuildNFieldPerp(double* ra, double* dec, double* r, double* w,
                                 long nobj, double minsep, double maxsep, double b, int sm_int);
>>>>>>> #17

    extern void DestroyGFieldFlat(void* field);
    extern void DestroyGFieldSphere(void* field);
    extern void DestroyGFieldPerp(void* field);

    extern void DestroyKFieldFlat(void* field);
    extern void DestroyKFieldSphere(void* field);
    extern void DestroyKFieldPerp(void* field);

    extern void DestroyNFieldFlat(void* field);
    extern void DestroyNFieldSphere(void* field);
    extern void DestroyNFieldPerp(void* field);


    extern void* BuildGSimpleFieldFlat(double* x, double* y,
                                       double* g1, double* g2, double* w, long nobj);
    extern void* BuildGSimpleFieldSphere(double* ra, double* dec, double* r,
                                         double* g1, double* g2, double* w, long nobj);
    extern void* BuildGSimpleFieldPerp(double* ra, double* dec, double* r,
                                       double* g1, double* g2, double* w, long nobj);

    extern void* BuildGSimpleFieldFlat(double* x, double* y,
                                       double* g1, double* g2, double* w, long nobj);
    extern void* BuildGSimpleFieldSphere(double* ra, double* dec, double* r,
                                         double* g1, double* g2, double* w, long nobj);
    extern void* BuildGSimpleFieldPerp(double* ra, double* dec, double* r,
                                       double* g1, double* g2, double* w, long nobj);

    extern void* BuildKSimpleFieldFlat(double* x, double* y,
                                       double* k, double* w, long nobj);
    extern void* BuildKSimpleFieldSphere(double* ra, double* dec, double* r,
                                         double* k, double* w, long nobj);
    extern void* BuildKSimpleFieldPerp(double* ra, double* dec, double* r,
                                       double* k, double* w, long nobj);

    extern void* BuildNSimpleFieldFlat(double* x, double* y, double* w, long nobj);
    extern void* BuildNSimpleFieldSphere(double* ra, double* dec, double* r, double* w, long nobj);
    extern void* BuildNSimpleFieldPerp(double* ra, double* dec, double* r, double* w, long nobj);

    extern void DestroyGSimpleFieldFlat(void* field);
    extern void DestroyGSimpleFieldSphere(void* field);
    extern void DestroyGSimpleFieldPerp(void* field);

    extern void DestroyKSimpleFieldFlat(void* field);
    extern void DestroyKSimpleFieldSphere(void* field);
    extern void DestroyKSimpleFieldPerp(void* field);

    extern void DestroyNSimpleFieldFlat(void* field);
    extern void DestroyNSimpleFieldSphere(void* field);
    extern void DestroyNSimpleFieldPerp(void* field);


}

