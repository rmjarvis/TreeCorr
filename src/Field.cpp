/* Copyright (c) 2003-2015 by Mike Jarvis
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

#include <cstddef>  // for ptrdiff_t
#include "Field.h"
#include "Cell.h"
#include "dbg.h"

// To turn on debugging statements, set dbgout to &std::cerr or some other stream.
std::ostream* dbgout=0;
//std::ostream* dbgout=&std::cerr;
// For even more debugging, set this to true.
bool XDEBUG = false;
// Note: You will also need to compile with
//     python setup.py build --debug`
//     python setup.py install

// This function just works on the top level data to figure out which data goes into
// each top-level Cell.  It is building up the top_* vectors, which can then be used
// to build the actual Cells.
template <int D, int C>
void SetupTopLevelCells(
    std::vector<CellData<D,C>*>& celldata, double maxsizesq,
    SplitMethod sm, size_t start, size_t end, int maxtop,
    std::vector<CellData<D,C>*>& top_data,
    std::vector<double>& top_sizesq,
    std::vector<size_t>& top_start, std::vector<size_t>& top_end)
{
    xdbg<<"Start SetupTopLevelCells: start,end = "<<start<<','<<end<<std::endl;
    xdbg<<"maxsizesq = "<<maxsizesq<<std::endl;
    xdbg<<"celldata has "<<celldata.size()<<" entries\n";
    // The structure of this is very similar to the Cell constructor.
    // The difference is that here we only construct a new Cell (and do the corresponding
    // calculation of the averages) if the size is small enough.  At that point, the
    // rest of the construction is passed onto the Cell class.
    CellData<D,C>* ave;
    double sizesq;
    if (end-start == 1) {
        xdbg<<"Only 1 CellData entry: size = 0\n";
        ave = celldata[start];
        celldata[start] = 0; // Make sure the calling function doesn't delete this!
        sizesq = 0.;
    } else {
        ave = new CellData<D,C>(celldata,start,end);
        xdbg<<"ave pos = "<<ave->getPos()<<std::endl;
        xdbg<<"n = "<<ave->getN()<<std::endl;
        xdbg<<"w = "<<ave->getW()<<std::endl;
        sizesq = CalculateSizeSq(ave->getPos(),celldata,start,end);
        xdbg<<"size = "<<sqrt(sizesq)<<std::endl;
    }

    if (sizesq <= maxsizesq) {
        xdbg<<"Small enough.  Make a cell.\n";
        if (end-start > 1) ave->finishAverages(celldata,start,end);
        top_data.push_back(ave);
        top_sizesq.push_back(sizesq);
        top_start.push_back(start);
        top_end.push_back(end);
    } else if (maxtop <= 0) {
        xdbg<<"At specified end of top layer recusion\n";
        if (end-start > 1) ave->finishAverages(celldata,start,end);
        top_data.push_back(ave);
        top_sizesq.push_back(sizesq);
        top_start.push_back(start);
        top_end.push_back(end);
    } else {
        size_t mid = SplitData(celldata,sm,start,end,ave->getPos());
        xdbg<<"Too big.  Recurse with mid = "<<mid<<std::endl;
        SetupTopLevelCells(celldata, maxsizesq, sm, start, mid, maxtop-1,
                           top_data, top_sizesq, top_start, top_end);
        SetupTopLevelCells(celldata, maxsizesq, sm, mid, end, maxtop-1,
                           top_data, top_sizesq, top_start, top_end);
    }
}

// A helper struct to build the right kind of CellData object
template <int D, int C>
struct CellDataHelper;

// Specialize for each D,C
template <>
struct CellDataHelper<NData,Flat>
{
    static CellData<NData,Flat>* build(double x, double y, double,
                                       double , double , double, double w, double wpos)
    { return new CellData<NData,Flat>(Position<Flat>(x,y), w, wpos); }
};
template <>
struct CellDataHelper<KData,Flat>
{
    static CellData<KData,Flat>* build(double x, double y, double,
                                       double , double , double k, double w, double wpos)
    { return new CellData<KData,Flat>(Position<Flat>(x,y), k, w, wpos); }
};
template <>
struct CellDataHelper<GData,Flat>
{
    static CellData<GData,Flat>* build(double x, double y,  double,
                                       double g1, double g2, double, double w, double wpos)
    { return new CellData<GData,Flat>(Position<Flat>(x,y), std::complex<double>(g1,g2), w, wpos); }
};


template <>
struct CellDataHelper<NData,ThreeD>
{
    static CellData<NData,ThreeD>* build(double x, double y, double z,
                                         double , double , double, double w, double wpos)
    { return new CellData<NData,ThreeD>(Position<ThreeD>(x,y,z), w, wpos); }
};
template <>
struct CellDataHelper<KData,ThreeD>
{
    static CellData<KData,ThreeD>* build(double x, double y, double z,
                                         double , double , double k, double w, double wpos)
    { return new CellData<KData,ThreeD>(Position<ThreeD>(x,y,z), k, w, wpos); }
};
template <>
struct CellDataHelper<GData,ThreeD>
{
    static CellData<GData,ThreeD>* build(double x, double y, double z,
                                         double g1, double g2, double, double w, double wpos)
    {
        return new CellData<GData,ThreeD>(Position<ThreeD>(x,y,z), std::complex<double>(g1,g2),
                                          w, wpos);
    }
};


// Sphere
template <>
struct CellDataHelper<NData,Sphere>
{
    static CellData<NData,Sphere>* build(double x, double y, double z,
                                         double , double , double, double w, double wpos)
    { return new CellData<NData,Sphere>(Position<Sphere>(x,y,z), w, wpos); }
};
template <>
struct CellDataHelper<KData,Sphere>
{
    static CellData<KData,Sphere>* build(double x, double y, double z,
                                         double , double , double k, double w, double wpos)
    { return new CellData<KData,Sphere>(Position<Sphere>(x,y,z), k, w, wpos); }
};
template <>
struct CellDataHelper<GData,Sphere>
{
    static CellData<GData,Sphere>* build(double x, double y, double z,
                                         double g1, double g2, double, double w, double wpos)
    {
        return new CellData<GData,Sphere>(Position<Sphere>(x,y,z), std::complex<double>(g1,g2),
                                          w, wpos);
    }
};

template <int D, int C>
Field<D,C>::Field(
    double* x, double* y, double* z, double* g1, double* g2, double* k,
    double* w, double* wpos, long nobj,
    double minsize, double maxsize,
    int sm_int, int maxtop)
{
    dbg<<"Starting to Build Field with "<<nobj<<" objects\n";
    std::vector<CellData<D,C>*> celldata;
    celldata.reserve(nobj);
    if (z) {
        for(int i=0;i<nobj;++i)
            if (wpos[i] != 0.)
                celldata.push_back(CellDataHelper<D,C>::build(x[i],y[i],z[i],g1[i],g2[i],k[i],
                                                              w[i],wpos[i]));
    } else {
        Assert(C == Flat);
        for(int i=0;i<nobj;++i)
            if (wpos[i] != 0.)
                celldata.push_back(CellDataHelper<D,C>::build(x[i],y[i],0.,g1[i],g2[i],k[i],
                                                              w[i],wpos[i]));
    }
    dbg<<"Built celldata with "<<celldata.size()<<" entries\n";

    // We don't build Cells that are too big or too small based on the min/max separation:

    double minsizesq = minsize * minsize;
    xdbg<<"minsizesq = "<<minsizesq<<std::endl;

    double maxsizesq = maxsize * maxsize;
    xdbg<<"maxsizesq = "<<maxsizesq<<std::endl;

    // Convert from the int to our enum.
    SplitMethod sm = static_cast<SplitMethod>(sm_int);

    // This is done in two parts so that we can do the (time-consuming) second part in
    // parallel.
    // First we setup what all the top-level cells are going to be.
    // Then we build them and their sub-nodes.

    std::vector<CellData<D,C>*> top_data;
    std::vector<double> top_sizesq;
    std::vector<size_t> top_start;
    std::vector<size_t> top_end;

    // Setup the top level cells:
    SetupTopLevelCells(celldata,maxsizesq,sm,0,celldata.size(),maxtop,
                       top_data,top_sizesq,top_start,top_end);
    const ptrdiff_t n = top_data.size();
    dbg<<"Field has "<<n<<" top-level nodes.  Building lower nodes...\n";
    _cells.resize(n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(ptrdiff_t i=0;i<n;++i)
        _cells[i] = new Cell<D,C>(top_data[i],top_sizesq[i],celldata,minsizesq,sm,
                                  top_start[i],top_end[i]);

    // delete any CellData elements that didn't get kept in the _cells object.
    for (size_t i=0;i<celldata.size();++i) if (celldata[i]) delete celldata[i];
}

template <int D, int C>
Field<D,C>::~Field()
{
    for(size_t i=0; i<_cells.size(); ++i) delete _cells[i];
}

template <int D, int C>
SimpleField<D,C>::SimpleField(
    double* x, double* y, double* z, double* g1, double* g2, double* k,
    double* w, double* wpos, long nobj)
{
    // This bit is the same as the start of the Field constructor.
    dbg<<"Starting to Build SimpleField with "<<nobj<<" objects\n";
    std::vector<CellData<D,C>*> celldata;
    celldata.reserve(nobj);
    if (z) {
        for(long i=0;i<nobj;++i)
            if (wpos[i] != 0.)
                celldata.push_back(CellDataHelper<D,C>::build(x[i],y[i],z[i],g1[i],g2[i],k[i],
                                                              w[i],wpos[i]));
    } else {
        Assert(C == Flat);
        for(long i=0;i<nobj;++i)
            if (wpos[i] != 0.)
                celldata.push_back(CellDataHelper<D,C>::build(x[i],y[i],0.,g1[i],g2[i],k[i],
                                                              w[i],wpos[i]));
    }
    dbg<<"Built celldata with "<<celldata.size()<<" entries\n";

    // However, now we just turn each item into a leaf Cell and keep them all in a single vector.
    ptrdiff_t n = celldata.size();
    _cells.resize(n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(ptrdiff_t i=0;i<n;++i)
        _cells[i] = new Cell<D,C>(celldata[i]);
}

template <int D, int C>
SimpleField<D,C>::~SimpleField()
{
    for(size_t i=0; i<_cells.size(); ++i) delete _cells[i];
}


//
//
// Now the C-C++ interface functions that get used in python:
//
//

extern "C" {
#include "Field_C.h"
}

void* BuildGFieldFlat(double* x, double* y, double* g1, double* g2,
                      double* w, double* wpos, long nobj,
                      double minsize, double maxsize,
                      int sm_int, int maxtop)
{
    dbg<<"Start BuildGFieldFlat\n";
    // Use w for k in this call to make sure k[i] is valid and won't seg fault.
    // The actual value of k[i] is ignored.
    Field<GData,Flat>* field = new Field<GData,Flat>(x, y, 0, g1, g2, w,
                                                     w, wpos, nobj,
                                                     minsize, maxsize,
                                                     sm_int, maxtop);
    xdbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}
void* BuildGField3D(double* x, double* y, double* z, double* g1, double* g2,
                    double* w, double* wpos, long nobj,
                    double minsize, double maxsize,
                    int sm_int, int maxtop)
{
    dbg<<"Start BuildGField3D\n";
    Field<GData,ThreeD>* field = new Field<GData,ThreeD>(x, y, z, g1, g2, w,
                                                         w, wpos, nobj,
                                                         minsize, maxsize,
                                                         sm_int, maxtop);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}
void* BuildGFieldSphere(double* x, double* y, double* z, double* g1, double* g2,
                        double* w, double* wpos, long nobj,
                        double minsize, double maxsize,
                        int sm_int, int maxtop)
{
    dbg<<"Start BuildGField3D\n";
    Field<GData,Sphere>* field = new Field<GData,Sphere>(x, y, z, g1, g2, w,
                                                         w, wpos, nobj,
                                                         minsize, maxsize,
                                                         sm_int, maxtop);
    xdbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void* BuildKFieldFlat(double* x, double* y, double* k,
                      double* w, double* wpos, long nobj,
                      double minsize, double maxsize,
                      int sm_int, int maxtop)
{
    dbg<<"Start BuildKFieldFlat\n";
    Field<KData,Flat>* field = new Field<KData,Flat>(x, y, 0, w, w, k,
                                                     w, wpos, nobj,
                                                     minsize, maxsize,
                                                     sm_int, maxtop);
    xdbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}
void* BuildKField3D(double* x, double* y, double* z, double* k,
                    double* w, double* wpos, long nobj,
                    double minsize, double maxsize,
                    int sm_int, int maxtop)
{
    dbg<<"Start BuildKField3D\n";
    Field<KData,ThreeD>* field = new Field<KData,ThreeD>(x, y, z, w, w, k,
                                                         w, wpos, nobj,
                                                         minsize, maxsize,
                                                         sm_int, maxtop);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}
void* BuildKFieldSphere(double* x, double* y, double* z, double* k,
                        double* w, double* wpos, long nobj,
                        double minsize, double maxsize,
                        int sm_int, int maxtop)
{
    dbg<<"Start BuildKField3D\n";
    Field<KData,Sphere>* field = new Field<KData,Sphere>(x, y, z, w, w, k,
                                                         w, wpos, nobj,
                                                         minsize, maxsize,
                                                         sm_int, maxtop);
    xdbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void* BuildNFieldFlat(double* x, double* y,
                      double* w, double* wpos, long nobj,
                      double minsize, double maxsize,
                      int sm_int, int maxtop)
{
    dbg<<"Start BuildNFieldFlat\n";
    Field<NData,Flat>* field = new Field<NData,Flat>(x, y, 0, w, w, w,
                                                     w, wpos, nobj,
                                                     minsize, maxsize,
                                                     sm_int, maxtop);
    xdbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}
void* BuildNField3D(double* x, double* y, double* z,
                    double* w, double* wpos, long nobj,
                    double minsize, double maxsize,
                    int sm_int, int maxtop)
{
    dbg<<"Start BuildNField3D\n";
    Field<NData,ThreeD>* field = new Field<NData,ThreeD>(x, y, z, w, w, w,
                                                         w, wpos, nobj,
                                                         minsize, maxsize,
                                                         sm_int, maxtop);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}
void* BuildNFieldSphere(double* x, double* y, double* z,
                        double* w, double* wpos, long nobj,
                        double minsize, double maxsize,
                        int sm_int, int maxtop)
{
    dbg<<"Start BuildNField3D\n";
    Field<NData,Sphere>* field = new Field<NData,Sphere>(x, y, z, w, w, w,
                                                         w, wpos, nobj,
                                                         minsize, maxsize,
                                                         sm_int, maxtop);
    xdbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void DestroyGFieldFlat(void* field)
{
    dbg<<"Start DestroyGFieldFlat\n";
    xdbg<<"field = "<<field<<std::endl;
    delete static_cast<Field<GData,Flat>*>(field);
}
void DestroyGField3D(void* field)
{
    dbg<<"Start DestroyGField3D\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<Field<GData,ThreeD>*>(field);
}
void DestroyGFieldSphere(void* field)
{
    dbg<<"Start DestroyGField3D\n";
    xdbg<<"field = "<<field<<std::endl;
    delete static_cast<Field<GData,Sphere>*>(field);
}

void DestroyKFieldFlat(void* field)
{
    dbg<<"Start DestroyKFieldFlat\n";
    xdbg<<"field = "<<field<<std::endl;
    delete static_cast<Field<KData,Flat>*>(field);
}
void DestroyKField3D(void* field)
{
    dbg<<"Start DestroyKField3D\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<Field<KData,ThreeD>*>(field);
}
void DestroyKFieldSphere(void* field)
{
    dbg<<"Start DestroyKField3D\n";
    xdbg<<"field = "<<field<<std::endl;
    delete static_cast<Field<KData,Sphere>*>(field);
}

void DestroyNFieldFlat(void* field)
{
    dbg<<"Start DestroyNFieldFlat\n";
    xdbg<<"field = "<<field<<std::endl;
    delete static_cast<Field<NData,Flat>*>(field);
}
void DestroyNField3D(void* field)
{
    dbg<<"Start DestroyNField3D\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<Field<NData,ThreeD>*>(field);
}
void DestroyNFieldSphere(void* field)
{
    dbg<<"Start DestroyNField3D\n";
    xdbg<<"field = "<<field<<std::endl;
    delete static_cast<Field<NData,Sphere>*>(field);
}


void* BuildGSimpleFieldFlat(double* x, double* y, double* g1, double* g2,
                            double* w, double* wpos, long nobj)
{
    dbg<<"Start BuildGSimpleFieldFlat\n";
    // Use w for k in this call to make sure k[i] is valid and won't seg fault.
    // The actual value of k[i] is ignored.
    SimpleField<GData,Flat>* field = new SimpleField<GData,Flat>(x, y, 0, g1, g2, w,
                                                                 w, wpos, nobj);
    xdbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}
void* BuildGSimpleField3D(double* x, double* y, double* z, double* g1, double* g2,
                          double* w, double* wpos, long nobj)
{
    dbg<<"Start BuildGSimpleField3D\n";
    SimpleField<GData,ThreeD>* field = new SimpleField<GData,ThreeD>(
        x, y, z, g1, g2, w,
        w, wpos, nobj);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}
void* BuildGSimpleFieldSphere(double* x, double* y, double* z, double* g1, double* g2,
                              double* w, double* wpos, long nobj)
{
    dbg<<"Start BuildGSimpleField3D\n";
    SimpleField<GData,Sphere>* field = new SimpleField<GData,Sphere>(
        x, y, z, g1, g2, w,
        w, wpos, nobj);
    xdbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void* BuildKSimpleFieldFlat(double* x, double* y, double* k,
                            double* w, double* wpos, long nobj)
{
    dbg<<"Start BuildKSimpleFieldFlat\n";
    SimpleField<KData,Flat>* field = new SimpleField<KData,Flat>(
        x, y, 0, w, w, k,
        w, wpos, nobj);
    xdbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}
void* BuildKSimpleField3D(double* x, double* y, double* z, double* k,
                          double* w, double* wpos, long nobj)
{
    dbg<<"Start BuildKSimpleField3D\n";
    SimpleField<KData,ThreeD>* field = new SimpleField<KData,ThreeD>(
        x, y, z, w, w, k,
        w, wpos, nobj);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}
void* BuildKSimpleFieldSphere(double* x, double* y, double* z, double* k,
                              double* w, double* wpos, long nobj)
{
    dbg<<"Start BuildKSimpleField3D\n";
    SimpleField<KData,Sphere>* field = new SimpleField<KData,Sphere>(
        x, y, z, w, w, k,
        w, wpos, nobj);
    xdbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void* BuildNSimpleFieldFlat(double* x, double* y,
                            double* w, double* wpos, long nobj)
{
    dbg<<"Start BuildNSimpleFieldFlat\n";
    SimpleField<NData,Flat>* field = new SimpleField<NData,Flat>(
        x, y, 0, w, w, w,
        w, wpos, nobj);
    xdbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}
void* BuildNSimpleField3D(double* x, double* y, double* z,
                          double* w, double* wpos, long nobj)
{
    dbg<<"Start BuildNSimpleField3D\n";
    SimpleField<NData,ThreeD>* field = new SimpleField<NData,ThreeD>(
        x, y, z, w, w, w,
        w, wpos, nobj);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}
void* BuildNSimpleFieldSphere(double* x, double* y, double* z,
                              double* w, double* wpos, long nobj)
{
    dbg<<"Start BuildNSimpleField3D\n";
    SimpleField<NData,Sphere>* field = new SimpleField<NData,Sphere>(
        x, y, z, w, w, w,
        w, wpos, nobj);
    xdbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void DestroyGSimpleFieldFlat(void* field)
{
    dbg<<"Start DestroyGSimpleFieldFlat\n";
    xdbg<<"field = "<<field<<std::endl;
    delete static_cast<SimpleField<GData,Flat>*>(field);
}
void DestroyGSimpleField3D(void* field)
{
    dbg<<"Start DestroyGSimpleField3D\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<SimpleField<GData,ThreeD>*>(field);
}
void DestroyGSimpleFieldSphere(void* field)
{
    dbg<<"Start DestroyGSimpleField3D\n";
    xdbg<<"field = "<<field<<std::endl;
    delete static_cast<SimpleField<GData,Sphere>*>(field);
}

void DestroyKSimpleFieldFlat(void* field)
{
    dbg<<"Start DestroyKSimpleFieldFlat\n";
    xdbg<<"field = "<<field<<std::endl;
    delete static_cast<SimpleField<KData,Flat>*>(field);
}
void DestroyKSimpleField3D(void* field)
{
    dbg<<"Start DestroyKSimpleField3D\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<SimpleField<KData,ThreeD>*>(field);
}
void DestroyKSimpleFieldSphere(void* field)
{
    dbg<<"Start DestroyKSimpleField3D\n";
    xdbg<<"field = "<<field<<std::endl;
    delete static_cast<SimpleField<KData,Sphere>*>(field);
}

void DestroyNSimpleFieldFlat(void* field)
{
    dbg<<"Start DestroyNSimpleFieldFlat\n";
    xdbg<<"field = "<<field<<std::endl;
    delete static_cast<SimpleField<NData,Flat>*>(field);
}
void DestroyNSimpleField3D(void* field)
{
    dbg<<"Start DestroyNSimpleField3D\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<SimpleField<NData,ThreeD>*>(field);
}
void DestroyNSimpleFieldSphere(void* field)
{
    dbg<<"Start DestroyNSimpleField3D\n";
    xdbg<<"field = "<<field<<std::endl;
    delete static_cast<SimpleField<NData,Sphere>*>(field);
}

long GFieldFlatGetNTopLevel(void* field)
{ return static_cast<Field<GData,Flat>*>(field)->getNTopLevel(); }
long GField3DGetNTopLevel(void* field)
{ return static_cast<Field<GData,ThreeD>*>(field)->getNTopLevel(); }
long GFieldSphereGetNTopLevel(void* field)
{ return static_cast<Field<GData,Sphere>*>(field)->getNTopLevel(); }

long KFieldFlatGetNTopLevel(void* field)
{ return static_cast<Field<KData,Flat>*>(field)->getNTopLevel(); }
long KField3DGetNTopLevel(void* field)
{ return static_cast<Field<KData,ThreeD>*>(field)->getNTopLevel(); }
long KFieldSphereGetNTopLevel(void* field)
{ return static_cast<Field<KData,Sphere>*>(field)->getNTopLevel(); }

long NFieldFlatGetNTopLevel(void* field)
{ return static_cast<Field<NData,Flat>*>(field)->getNTopLevel(); }
long NField3DGetNTopLevel(void* field)
{ return static_cast<Field<NData,ThreeD>*>(field)->getNTopLevel(); }
long NFieldSphereGetNTopLevel(void* field)
{ return static_cast<Field<NData,Sphere>*>(field)->getNTopLevel(); }

