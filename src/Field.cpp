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

#include <cstddef>  // for ptrdiff_t
#include "Field.h"
#include "Cell.h"
#include "dbg.h"

// To turn on debugging statements, set dbgout to &std::cerr or some other stream.
//std::ostream* dbgout=0;
std::ostream* dbgout=&std::cerr;
// For even more debugging, set this to true.
bool XDEBUG = true;
// Note: You will also need to compile with 
//     python setup.py build --debug`
//     python setup.py install 

// Most of the functionality for building Cells and doing the correlation functions is the
// same regardless of which kind of Cell we have (N, K, G) or which kind of positions we
// are using (Flat, Sphere).  So most of the C++ code uses templates.  
// DC = GData for shear 
//      KData for kappa
//      NData for counts
// M = Flat for flat-sky coordinates
//     Sphere for spherical coordinates

// This function just works on the top level data to figure out which data goes into
// each top-level Cell.  It is building up the top_* vectors, which can then be used
// to build the actual Cells.
template <int DC, int M>
void SetupTopLevelCells(
    std::vector<CellData<DC,M>*>& celldata, double maxsizesq, double bsq,
    SplitMethod sm, size_t start, size_t end, int maxtop,
    std::vector<CellData<DC,M>*>& top_data,
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
    CellData<DC,M>* ave;
    double sizesq;
    if (end-start == 1) {
        xdbg<<"Only 1 CellData entry: size = 0\n";
        ave = celldata[start];
        celldata[start] = 0; // Make sure the calling function doesn't delete this!
        sizesq = 0.;
    } else {
        ave = new CellData<DC,M>(celldata,start,end);
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
        // We want to stop splitting if size <= b * maxsep, but this check makes sure that
        // maxsep is the largest possible separation of interest.  It could be the histograms
        // maxsep value, which is what the initial call to SetupTopLevelCells calculates,
        // but it could also be 2*size of the first cell.  So check here to see if we need
        // to make maxsizesq larger.
        // This can only happen on the first call, but it's not too much overhead, so it's not
        // worth doing something special to avoid this check on subsequent calls.
        double temp = 4. * bsq * sizesq;
        if (temp > maxsizesq) maxsizesq = temp;

        size_t mid = SplitData(celldata,sm,start,end,ave->getPos());
        xdbg<<"Too big.  Recurse with mid = "<<mid<<std::endl;
        SetupTopLevelCells(celldata, maxsizesq, bsq, sm, start, mid, maxtop-1,
                           top_data, top_sizesq, top_start, top_end);
        SetupTopLevelCells(celldata, maxsizesq, bsq, sm, mid, end, maxtop-1,
                           top_data, top_sizesq, top_start, top_end);
    }
}

// Specialize for all the different kinds of CellData possibilities.
template <int DC, int M>
CellData<DC,M>* BuildCellData(double x, double y, double g1, double g2, double, double w);

template <>
CellData<GData,Flat>* BuildCellData(double x, double y, double g1, double g2, double, double w)
{ return new CellData<GData,Flat>(Position<Flat>(x,y), std::complex<double>(g1,g2), w); }

template <>
CellData<GData,Sphere>* BuildCellData(double ra, double dec, double g1, double g2, double, double w)
{ return new CellData<GData,Sphere>(Position<Sphere>(ra,dec), std::complex<double>(g1,g2), w); }

template <>
CellData<KData,Flat>* BuildCellData(double x, double y, double , double , double k, double w)
{ return new CellData<KData,Flat>(Position<Flat>(x,y), k, w); }

template <>
CellData<KData,Sphere>* BuildCellData(double ra, double dec, double , double , double k, double w)
{ return new CellData<KData,Sphere>(Position<Sphere>(ra,dec), k, w); }

template <>
CellData<NData,Flat>* BuildCellData(double x, double y, double , double , double, double w)
{ return new CellData<NData,Flat>(Position<Flat>(x,y), w); }

template <>
CellData<NData,Sphere>* BuildCellData(double ra, double dec, double , double , double, double w)
{ return new CellData<NData,Sphere>(Position<Sphere>(ra,dec), w); }

// For the 3D ones, we use a default implementation for when M=Flat, since we don't need those.
template <int DC, int M>
CellData<DC,M>* BuildCellData(
    double x, double y, double r, double g1, double g2, double, double w)
{ return 0; }

template <>
CellData<GData,Sphere>* BuildCellData(
    double ra, double dec, double r, double g1, double g2, double, double w)
{ return new CellData<GData,Sphere>(Position<Sphere>(ra,dec,r), std::complex<double>(g1,g2), w); }

template <>
CellData<KData,Sphere>* BuildCellData(
    double ra, double dec, double r, double , double , double k, double w)
{ return new CellData<KData,Sphere>(Position<Sphere>(ra,dec,r), k, w); }

template <>
CellData<NData,Sphere>* BuildCellData(
    double ra, double dec, double r, double , double , double, double w)
{ return new CellData<NData,Sphere>(Position<Sphere>(ra,dec,r), w); }


template <int DC, int M>
Field<DC,M>::Field(
    double* x, double* y, double* r, double* g1, double* g2, double* k, double* w,
    long nobj, double minsep, double maxsep, double b, int sm_int, int maxtop)
{
    dbg<<"Starting to Build Field with "<<nobj<<" objects\n";
    dbg<<"r = "<<r<<std::endl;
    std::vector<CellData<DC,M>*> celldata;
    celldata.reserve(nobj);
    //xdbg<<"First values are: x,y = "<<x[0]<<"  "<<y[0]<<std::endl;
    //if (r) { xdbg<<"r = "<<r[0]<<std::endl; }
    //xdbg<<"g1,g2 = "<<g1[0]<<','<<g2[0]<<std::endl;
    //xdbg<<"k = "<<k[0]<<std::endl;
    //xdbg<<"w = "<<w[0]<<std::endl;
    if (r) {
        for(int i=0;i<nobj;++i) 
            if (w[i] != 0.)
                celldata.push_back(BuildCellData<DC,M>(x[i],y[i],r[i],g1[i],g2[i],k[i],w[i]));
    } else {
        for(int i=0;i<nobj;++i) 
            if (w[i] != 0.)
                celldata.push_back(BuildCellData<DC,M>(x[i],y[i],g1[i],g2[i],k[i],w[i]));
    }
    dbg<<"Built celldata with "<<celldata.size()<<" entries\n";

    // We don't build Cells that are too big or too small based on the min/max separation:

    // The minimum size cell that will be useful is one where two cells that just barely
    // don't split have (d + s1 + s2) = minsep
    // The largest s2 we need to worry about is s2 = 2s1.
    // i.e. d = minsep - 3s1  and s1 = 0.5 * bd
    //      d = minsep - 1.5 bd
    //      d = minsep / (1+1.5 b)
    //      s = 0.5 * b * minsep / (1+1.5 b)
    //        = b * minsep / (2+3b)
    double minsize = minsep * b / (2.+3.*b);
    xdbg<<"minsize = "<<minsize<<std::endl;
    double minsizesq = minsize * minsize;
    xdbg<<"minsizesq = "<<minsizesq<<std::endl;

    // The maximum size cell that will be useful is one where a cell of size s will
    // be split at the maximum separation even if the other size = 0.
    // i.e. s = b * maxsep
    // However, the maxsep here is not necessarily the maxsep that we will use for the bins.
    // It is the maximum separation of any two cells.  So on the first pass to SetupTopLevelCells,
    // we'll update this to make sure maxsizesq >= bsq * sizesq.
    double maxsize = maxsep * b;
    xdbg<<"maxsize = "<<maxsize<<std::endl;
    double maxsizesq = maxsize * maxsize;
    xdbg<<"maxsizesq = "<<maxsizesq<<std::endl;

    // Convert from the int to our enum.
    SplitMethod sm = static_cast<SplitMethod>(sm_int);

    // This is done in two parts so that we can do the (time-consuming) second part in 
    // parallel.
    // First we setup what all the top-level cells are going to be.
    // Then we build them and their sub-nodes.

    if (maxsizesq == 0.) {
        dbg<<"Doing brute-force calculation (all cells are leaf nodes).\n";
        // If doing a brute-force calculation, the top-level cell data are the same as celldata.
        const ptrdiff_t n = celldata.size();
        _cells.resize(n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(ptrdiff_t i=0;i<n;++i) {
            _cells[i] = new Cell<DC,M>(celldata[i]);
            celldata[i] = 0; // Make sure the calling routing doesn't delete this one.
        }
    } else {
        std::vector<CellData<DC,M>*> top_data;
        std::vector<double> top_sizesq;
        std::vector<size_t> top_start;
        std::vector<size_t> top_end;

        // Setup the top level cells:
        SetupTopLevelCells(celldata,maxsizesq,b*b,sm,0,celldata.size(),maxtop,
                           top_data,top_sizesq,top_start,top_end);
        const ptrdiff_t n = top_data.size();
        dbg<<"Field has "<<n<<" top-level nodes.  Building lower nodes...\n";
        _cells.resize(n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(ptrdiff_t i=0;i<n;++i)
            _cells[i] = new Cell<DC,M>(top_data[i],top_sizesq[i],celldata,minsizesq,sm,
                                       top_start[i],top_end[i]);
    }

    // delete any CellData elements that didn't get kept in the _cells object.
    for (size_t i=0;i<celldata.size();++i) if (celldata[i]) delete celldata[i];
}

template <int DC, int M>
Field<DC,M>::~Field()
{
    for(size_t i=0; i<_cells.size(); ++i) delete _cells[i];
}

template <int DC, int M>
SimpleField<DC,M>::SimpleField(
    double* x, double* y, double* r, double* g1, double* g2, double* k, double* w, long nobj)
{
    // This bit is the same as the start of the Field constructor.
    dbg<<"Starting to Build SimpleField with "<<nobj<<" objects\n";
    std::vector<CellData<DC,M>*> celldata;
    celldata.reserve(nobj);
    if (r) {
        for(long i=0;i<nobj;++i)
            if (w[i] != 0.) 
                celldata.push_back(BuildCellData<DC,M>(x[i],y[i],r[i],g1[i],g2[i],k[i],w[i]));
    } else {
        for(long i=0;i<nobj;++i)
            if (w[i] != 0.) 
                celldata.push_back(BuildCellData<DC,M>(x[i],y[i],g1[i],g2[i],k[i],w[i]));
    }
    dbg<<"Built celldata with "<<celldata.size()<<" entries\n";

    // However, now we just turn each item into a leaf Cell and keep them all in a single vector.
    ptrdiff_t n = celldata.size();
    _cells.resize(n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(ptrdiff_t i=0;i<n;++i) 
        _cells[i] = new Cell<DC,M>(celldata[i]);
}

template <int DC, int M>
SimpleField<DC,M>::~SimpleField()
{
    for(size_t i=0; i<_cells.size(); ++i) delete _cells[i];
}


//
//
// Now the C-C++ interface functions that get used in python:
//
//

void* BuildGFieldFlat(double* x, double* y, double* g1, double* g2, double* w,
                      long nobj, double minsep, double maxsep, double b, int sm_int,
                      int maxtop)
{
    dbg<<"Start BuildGFieldFlat\n";
    // Use w for k in this call to make sure k[i] is valid and won't seg fault.
    // The actual value of k[i] is ignored.
    Field<GData,Flat>* field = new Field<GData,Flat>(x, y, 0, g1, g2, w, w, nobj, 
                                                     minsep, maxsep, b, sm_int, maxtop);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void* BuildGFieldSphere(double* ra, double* dec, double* r, double* g1, double* g2, double* w,
                        long nobj, double minsep, double maxsep, double b, int sm_int,
                        int maxtop)
{
    dbg<<"Start BuildGFieldSphere\n";
    Field<GData,Sphere>* field = new Field<GData,Sphere>(ra, dec, r, g1, g2, w, w, nobj,
                                                         minsep, maxsep, b, sm_int, maxtop);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void* BuildKFieldFlat(double* x, double* y, double* k, double* w,
                      long nobj, double minsep, double maxsep, double b, int sm_int,
                      int maxtop)
{
    dbg<<"Start BuildKFieldFlat\n";
    Field<KData,Flat>* field = new Field<KData,Flat>(x, y, 0, w, w, k, w, nobj,
                                                     minsep, maxsep, b, sm_int, maxtop);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void* BuildKFieldSphere(double* ra, double* dec, double* r, double* k, double* w,
                        long nobj, double minsep, double maxsep, double b, int sm_int,
                        int maxtop)
{
    dbg<<"Start BuildKFieldSphere\n";
    Field<KData,Sphere>* field = new Field<KData,Sphere>(ra, dec, r, w, w, k, w, nobj,
                                                         minsep, maxsep, b, sm_int, maxtop);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void* BuildNFieldFlat(double* x, double* y, double* w,
                      long nobj, double minsep, double maxsep, double b, int sm_int,
                      int maxtop)
{
    dbg<<"Start BuildNFieldFlat\n";
    Field<NData,Flat>* field = new Field<NData,Flat>(x, y, 0, w, w, w, w, nobj, 
                                                     minsep, maxsep, b, sm_int, maxtop);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void* BuildNFieldSphere(double* ra, double* dec, double* r, double* w,
                        long nobj, double minsep, double maxsep, double b, int sm_int,
                        int maxtop)
{
    dbg<<"Start BuildNFieldSphere\n";
    Field<NData,Sphere>* field = new Field<NData,Sphere>(ra, dec, r, w, w, w, w, nobj,
                                                         minsep, maxsep, b, sm_int, maxtop);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void DestroyGFieldFlat(void* field)
{
    dbg<<"Start DestroyGFieldFlat\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<Field<GData,Flat>*>(field);
}

void DestroyGFieldSphere(void* field)
{
    dbg<<"Start DestroyGFieldSphere\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<Field<GData,Sphere>*>(field);
}

void DestroyKFieldFlat(void* field)
{ 
    dbg<<"Start DestroyKFieldFlat\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<Field<KData,Flat>*>(field);
}

void DestroyKFieldSphere(void* field)
{
    dbg<<"Start DestroyKFieldSphere\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<Field<KData,Sphere>*>(field);
}

void DestroyNFieldFlat(void* field)
{ 
    dbg<<"Start DestroyNFieldFlat\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<Field<NData,Flat>*>(field);
}

void DestroyNFieldSphere(void* field)
{
    dbg<<"Start DestroyNFieldSphere\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<Field<NData,Sphere>*>(field);
}




void* BuildGSimpleFieldFlat(double* x, double* y, double* g1, double* g2, double* w, long nobj)
{
    dbg<<"Start BuildGSimpleFieldFlat\n";
    // Use w for k in this call to make sure k[i] is valid and won't seg fault.
    // The actual value of k[i] is ignored.
    SimpleField<GData,Flat>* field = new SimpleField<GData,Flat>(x, y, 0, g1, g2, w, w, nobj);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void* BuildGSimpleFieldSphere(double* ra, double* dec, double* r, double* g1, double* g2, double* w, long nobj)
{
    dbg<<"Start BuildGSimpleFieldSphere\n";
    SimpleField<GData,Sphere>* field = new SimpleField<GData,Sphere>(ra, dec, r, g1, g2, w, w, nobj);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void* BuildKSimpleFieldFlat(double* x, double* y, double* k, double* w, long nobj)
{
    dbg<<"Start BuildKSimpleFieldFlat\n";
    SimpleField<KData,Flat>* field = new SimpleField<KData,Flat>(x, y, 0, w, w, k, w, nobj);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void* BuildKSimpleFieldSphere(double* ra, double* dec, double* r, double* k, double* w, long nobj)
{
    dbg<<"Start BuildKSimpleFieldSphere\n";
    SimpleField<KData,Sphere>* field = new SimpleField<KData,Sphere>(ra, dec, r, w, w, k, w, nobj);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void* BuildNSimpleFieldFlat(double* x, double* y, double* w, long nobj)
{
    dbg<<"Start BuildNSimpleFieldFlat\n";
    SimpleField<NData,Flat>* field = new SimpleField<NData,Flat>(x, y, 0, w, w, w, w, nobj);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void* BuildNSimpleFieldSphere(double* ra, double* dec, double* r, double* w, long nobj)
{
    dbg<<"Start BuildNSimpleFieldSphere\n";
    SimpleField<NData,Sphere>* field = new SimpleField<NData,Sphere>(ra, dec, r, w, w, w, w, nobj);
    dbg<<"field = "<<field<<std::endl;
    return static_cast<void*>(field);
}

void DestroyGSimpleFieldFlat(void* field)
{
    dbg<<"Start DestroyGSimpleFieldFlat\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<SimpleField<GData,Flat>*>(field);
}

void DestroyGSimpleFieldSphere(void* field)
{
    dbg<<"Start DestroyGSimpleFieldSphere\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<SimpleField<GData,Sphere>*>(field);
}

void DestroyKSimpleFieldFlat(void* field)
{ 
    dbg<<"Start DestroyKSimpleFieldFlat\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<SimpleField<KData,Flat>*>(field);
}

void DestroyKSimpleFieldSphere(void* field)
{
    dbg<<"Start DestroyKSimpleFieldSphere\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<SimpleField<KData,Sphere>*>(field);
}

void DestroyNSimpleFieldFlat(void* field)
{ 
    dbg<<"Start DestroyNSimpleFieldFlat\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<SimpleField<NData,Flat>*>(field);
}

void DestroyNSimpleFieldSphere(void* field)
{
    dbg<<"Start DestroyNSimpleFieldSphere\n";
    dbg<<"field = "<<field<<std::endl;
    delete static_cast<SimpleField<NData,Sphere>*>(field);
}


