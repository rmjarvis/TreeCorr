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

//#define DEBUGLOGGING

#include <cstddef>  // for ptrdiff_t
#include "Field.h"
#include "Cell.h"
#include "dbg.h"

// This function just works on the top level data to figure out which data goes into
// each top-level Cell.  It is building up the top_* vectors, which can then be used
// to build the actual Cells.
template <int D, int C>
void SetupTopLevelCells(
    std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& celldata,
    double maxsizesq, SplitMethod sm, size_t start, size_t end, int mintop, int maxtop,
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
        ave = celldata[start].first;
        celldata[start].first = 0; // Make sure the calling function doesn't delete this!
        sizesq = 0.;
    } else {
        ave = new CellData<D,C>(celldata,start,end);
        xdbg<<"ave pos = "<<ave->getPos()<<std::endl;
        xdbg<<"n = "<<ave->getN()<<std::endl;
        xdbg<<"w = "<<ave->getW()<<std::endl;
        sizesq = CalculateSizeSq(ave->getPos(),celldata,start,end);
        xdbg<<"size = "<<sqrt(sizesq)<<std::endl;
    }

    if (sizesq == 0 || (sizesq <= maxsizesq && mintop<=0)) {
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
        SetupTopLevelCells(celldata, maxsizesq, sm, start, mid, mintop-1, maxtop-1,
                           top_data, top_sizesq, top_start, top_end);
        SetupTopLevelCells(celldata, maxsizesq, sm, mid, end, mintop-1, maxtop-1,
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
                                       double , double , double, double w)
    { return new CellData<NData,Flat>(Position<Flat>(x,y), w); }
};
template <>
struct CellDataHelper<KData,Flat>
{
    static CellData<KData,Flat>* build(double x, double y, double,
                                       double , double , double k, double w)
    { return new CellData<KData,Flat>(Position<Flat>(x,y), k, w); }
};
template <>
struct CellDataHelper<GData,Flat>
{
    static CellData<GData,Flat>* build(double x, double y,  double,
                                       double g1, double g2, double, double w)
    { return new CellData<GData,Flat>(Position<Flat>(x,y), std::complex<double>(g1,g2), w); }
};


template <>
struct CellDataHelper<NData,ThreeD>
{
    static CellData<NData,ThreeD>* build(double x, double y, double z,
                                         double , double , double, double w)
    { return new CellData<NData,ThreeD>(Position<ThreeD>(x,y,z), w); }
};
template <>
struct CellDataHelper<KData,ThreeD>
{
    static CellData<KData,ThreeD>* build(double x, double y, double z,
                                         double , double , double k, double w)
    { return new CellData<KData,ThreeD>(Position<ThreeD>(x,y,z), k, w); }
};
template <>
struct CellDataHelper<GData,ThreeD>
{
    static CellData<GData,ThreeD>* build(double x, double y, double z,
                                         double g1, double g2, double, double w)
    { return new CellData<GData,ThreeD>(Position<ThreeD>(x,y,z), std::complex<double>(g1,g2), w); }
};


// Sphere
template <>
struct CellDataHelper<NData,Sphere>
{
    static CellData<NData,Sphere>* build(double x, double y, double z,
                                         double , double , double, double w)
    { return new CellData<NData,Sphere>(Position<Sphere>(x,y,z), w); }
};
template <>
struct CellDataHelper<KData,Sphere>
{
    static CellData<KData,Sphere>* build(double x, double y, double z,
                                         double , double , double k, double w)
    { return new CellData<KData,Sphere>(Position<Sphere>(x,y,z), k, w); }
};
template <>
struct CellDataHelper<GData,Sphere>
{
    static CellData<GData,Sphere>* build(double x, double y, double z,
                                         double g1, double g2, double, double w)
    { return new CellData<GData,Sphere>(Position<Sphere>(x,y,z), std::complex<double>(g1,g2), w); }
};

inline WPosLeafInfo get_wpos(double* wpos, double* w, int i)
{
    WPosLeafInfo wp;
    wp.wpos = wpos ? wpos[i] : w[i];
    wp.index = i;
    return wp;
}

template <int D, int C>
Field<D,C>::Field(
    double* x, double* y, double* z, double* g1, double* g2, double* k,
    double* w, double* wpos, long nobj,
    double minsize, double maxsize,
    int sm_int, bool brute, int mintop, int maxtop)
{
    //set_verbose(2);
    dbg<<"Starting to Build Field with "<<nobj<<" objects\n";
    xdbg<<"D,C = "<<D<<','<<C<<std::endl;
    xdbg<<"First few values are:\n";
    for(int i=0;i<5;++i) {
        xdbg<<x[i]<<"  "<<y[i]<<"  "<<(z?z[i]:0)<<"  "<<g1[i]<<"  "<<g2[i]<<"  "<<k[i]<<"  "<<w[i]<<"  "<<(wpos?wpos[i]:0)<<std::endl;
    }
    std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> > celldata;
    celldata.reserve(nobj);
    if (z) {
        for(int i=0;i<nobj;++i) {
            WPosLeafInfo wp = get_wpos(wpos,w,i);
            if (wp.wpos != 0.)
                celldata.push_back(std::make_pair(
                        CellDataHelper<D,C>::build(x[i],y[i],z[i],g1[i],g2[i],k[i],w[i]),
                        wp));
        }
    } else {
        Assert(C == Flat);
        for(int i=0;i<nobj;++i) {
            WPosLeafInfo wp = get_wpos(wpos,w,i);
            if (wp.wpos != 0.)
                celldata.push_back(std::make_pair(
                        CellDataHelper<D,C>::build(x[i],y[i],0.,g1[i],g2[i],k[i],w[i]),
                        wp));
        }
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
    SetupTopLevelCells(celldata,maxsizesq,sm,0,celldata.size(),mintop,maxtop,
                       top_data,top_sizesq,top_start,top_end);
    const ptrdiff_t n = top_data.size();
    dbg<<"Field has "<<n<<" top-level nodes.  Building lower nodes...\n";
    _cells.resize(n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(ptrdiff_t i=0;i<n;++i) {
        _cells[i] = new Cell<D,C>(top_data[i],top_sizesq[i],celldata,minsizesq,sm,brute,
                                  top_start[i],top_end[i]);
        xdbg<<i<<": "<<_cells[i]->getN()<<"  "<<_cells[i]->getW()<<"  "<<
            _cells[i]->getPos()<<"  "<<_cells[i]->getSize()<<"  "<<_cells[i]->getSizeSq()<<std::endl;
    }

    // delete any CellData elements that didn't get kept in the _cells object.
    for (size_t i=0;i<celldata.size();++i) if (celldata[i].first) delete celldata[i].first;
    //set_verbose(1);
}

template <int D, int C>
Field<D,C>::~Field()
{
    for(size_t i=0; i<_cells.size(); ++i) delete _cells[i];
}

template <int D, int C>
long CountNear(const Cell<D,C>* cell, const Position<C>& pos, double sep, double sepsq)
{
    double s = cell->getSize();
    const double dsq = (cell->getPos() - pos).normSq();
    dbg<<"CountNear: "<<cell->getPos()<<"  "<<pos<<"  "<<dsq<<"  "<<s<<"  "<<sepsq<<std::endl;

    // If s == 0, then just check dsq
    if (s==0.) {
        if (dsq <= sepsq) {
            dbg<<"s==0, d < sep   N = "<<cell->getN()<<std::endl;
            Assert(sqrt(dsq) <= sep);
            return cell->getN();
        }
        else {
            Assert(sqrt(dsq) > sep);
            dbg<<"s==0, d >= sep\n";
            return 0;
        }
    } else {
        // If d - s > sep, then no points are close enough.
        if (dsq > sepsq && dsq > SQR(sep+s)) {
            Assert(sqrt(dsq) - s > sep);
            dbg<<"d - s > sep: "<<sqrt(dsq)-s<<" > "<<sep<<std::endl;
            return 0;
        }

        // If d + s < sep, then all points are close enough.
        if (dsq <= sepsq && s < sep && dsq <= SQR(sep-s)) {
            Assert(sqrt(dsq) + s <= sep);
            dbg<<"d + s <= sep: "<<sqrt(dsq)+s<<" <= "<<sep<<"  N = "<<cell->getN()<<std::endl;
            return cell->getN();
        }

        // Otherwise check the subcells.
        Assert(cell->getLeft());
        Assert(cell->getRight());
        dbg<<"Recurse to subcells\n";
        return (CountNear(cell->getLeft(), pos, sep, sepsq) +
                CountNear(cell->getRight(), pos, sep, sepsq));
    }
}

template <int D, int C>
long Field<D,C>::countNear(double x, double y, double z, double sep) const
{
    Position<C> pos(x,y,z);
    double sepsq = sep*sep;
    long ntot = 0;
    dbg<<"Start countNear: "<<_cells.size()<<" top level cells\n";
    for(size_t i=0; i<_cells.size(); ++i) {
        dbg<<"Top level "<<i<<" with N="<<_cells[i]->getN()<<std::endl;
        ntot += CountNear(_cells[i], pos, sep, sepsq);
        dbg<<"ntot -> "<<ntot<<std::endl;
    }
    return ntot;
}

template <int D, int C>
void GetNear(const Cell<D,C>* cell, const Position<C>& pos, double sep, double sepsq,
             long* indices, int& k, int n)
{
    double s = cell->getSize();
    const double dsq = (cell->getPos() - pos).normSq();
    dbg<<"GetNear: "<<cell->getPos()<<"  "<<pos<<"  "<<dsq<<"  "<<s<<"  "<<sepsq<<std::endl;

    // If s == 0, then just check dsq
    if (s==0.) {
        if (dsq <= sepsq) {
            dbg<<"s==0, d < sep   N = "<<cell->getN()<<std::endl;
            Assert(sqrt(dsq) <= sep);
            Assert(k < n);
            long n1 = cell->getN();
            Assert(k + n1 <= n);
            // This shouldn't happen, but if it does, we can get a seg fault, so check.
            if (k + n1 > n) return;
            if (n1 == 1) {
                dbg<<"N == 1 case\n";
                indices[k++] = cell->getInfo().index;
            } else {
                dbg<<"N > 1 case: "<<n1<<std::endl;
                std::vector<long>* leaf_indices = cell->getListInfo().indices;
                Assert(int(leaf_indices->size()) == n1);
                for (int m=0; m<n1; ++m)
                    indices[k++] = (*leaf_indices)[m];
            }
            Assert(k <= n);
        } else {
            Assert(sqrt(dsq) > sep);
            dbg<<"s==0, d >= sep\n";
        }
    } else {
        // If d - s > sep, then no points are close enough.
        if (dsq > sepsq && dsq > SQR(sep+s)) {
            Assert(sqrt(dsq) - s > sep);
            dbg<<"d - s > sep: "<<sqrt(dsq)-s<<" > "<<sep<<std::endl;
        } else {
            // Otherwise check the subcells.
            Assert(cell->getLeft());
            Assert(cell->getRight());
            dbg<<"Recurse to subcells\n";
            GetNear(cell->getLeft(), pos, sep, sepsq, indices, k, n);
            GetNear(cell->getRight(), pos, sep, sepsq, indices, k, n);
        }
    }
}

template <int D, int C>
void Field<D,C>::getNear(double x, double y, double z, double sep, long* indices, int n) const
{
    Position<C> pos(x,y,z);
    double sepsq = sep*sep;
    dbg<<"Start getNear: "<<_cells.size()<<" top level cells\n";
    int k = 0;
    for(size_t i=0; i<_cells.size(); ++i) {
        dbg<<"Top level "<<i<<" with N="<<_cells[i]->getN()<<std::endl;
        GetNear(_cells[i], pos, sep, sepsq, indices, k, n);
        dbg<<"k -> "<<k<<std::endl;
    }
}

template <int D, int C>
SimpleField<D,C>::SimpleField(
    double* x, double* y, double* z, double* g1, double* g2, double* k,
    double* w, double* wpos, long nobj)
{
    // This bit is the same as the start of the Field constructor.
    dbg<<"Starting to Build SimpleField with "<<nobj<<" objects\n";
    std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> > celldata;
    celldata.reserve(nobj);
    if (z) {
        for(long i=0;i<nobj;++i) {
            WPosLeafInfo wp = get_wpos(wpos,w,i);
            if (wp.wpos != 0.)
                celldata.push_back(std::make_pair(
                        CellDataHelper<D,C>::build(x[i],y[i],z[i],g1[i],g2[i],k[i],w[i]),
                        wp));
        }
    } else {
        Assert(C == Flat);
        for(long i=0;i<nobj;++i) {
            WPosLeafInfo wp = get_wpos(wpos,w,i);
            if (wp.wpos != 0.)
                celldata.push_back(std::make_pair(
                        CellDataHelper<D,C>::build(x[i],y[i],0.,g1[i],g2[i],k[i],w[i]),
                        wp));
        }
    }
    dbg<<"Built celldata with "<<celldata.size()<<" entries\n";

    // However, now we just turn each item into a leaf Cell and keep them all in a single vector.
    ptrdiff_t n = celldata.size();
    _cells.resize(n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(ptrdiff_t i=0;i<n;++i)
        _cells[i] = new Cell<D,C>(celldata[i].first, celldata[i].second);
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

template <int D>
void* BuildField(double* x, double* y, double* z, double* g1, double* g2, double* k,
                 double* w, double* wpos, long nobj,
                 double minsize, double maxsize,
                 int sm_int, int brute, int mintop, int maxtop, int coords)
{
    dbg<<"Start BuildField "<<D<<"  "<<coords<<std::endl;
    void* field=0;
    switch(coords) {
      case Flat:
           // Note: Use w for k, since we access k[i], even though value will be ignored.
           field = static_cast<void*>(new Field<D,Flat>(x, y, 0, g1, g2, k,
                                                        w, wpos, nobj,
                                                        minsize, maxsize,
                                                        sm_int, bool(brute), mintop, maxtop));
           break;
      case Sphere:
           field = static_cast<void*>(new Field<D,Sphere>(x, y, z, g1, g2, k,
                                                          w, wpos, nobj,
                                                          minsize, maxsize,
                                                          sm_int, bool(brute), mintop, maxtop));
           break;
      case ThreeD:
           field = static_cast<void*>(new Field<D,ThreeD>(x, y, z, g1, g2, k,
                                                          w, wpos, nobj,
                                                          minsize, maxsize,
                                                          sm_int, bool(brute), mintop, maxtop));
           break;
    }
    xdbg<<"field = "<<field<<std::endl;
    return field;
}

void* BuildGField(double* x, double* y, double* z, double* g1, double* g2,
                  double* w, double* wpos, long nobj,
                  double minsize, double maxsize,
                  int sm_int, int brute, int mintop, int maxtop, int coords)
{
    // Note: Use w for k, since we access k[i], even though value will be ignored.
    return BuildField<GData>(x,y,z, g1,g2,w, w,wpos,nobj, minsize,maxsize, sm_int,
                             brute,mintop,maxtop,coords);
}


void* BuildKField(double* x, double* y, double* z, double* k,
                  double* w, double* wpos, long nobj,
                  double minsize, double maxsize,
                  int sm_int, int brute, int mintop, int maxtop, int coords)
{
    // Note: Use w for g1,g2, since we access g1[i],g2[i] even though values are ignored.
    return BuildField<KData>(x,y,z, w,w,k, w,wpos,nobj, minsize,maxsize, sm_int,
                             brute,mintop,maxtop,coords);
}

void* BuildNField(double* x, double* y, double* z,
                  double* w, double* wpos, long nobj,
                  double minsize, double maxsize,
                  int sm_int, int brute, int mintop, int maxtop, int coords)
{
    // Note: Use w for g1,g2,k for same reasons as above.
    return BuildField<NData>(x,y,z, w,w,w, w,wpos,nobj, minsize,maxsize, sm_int,
                             brute,mintop,maxtop,coords);
}

template <int D>
void DestroyField(void* field, int coords)
{
    dbg<<"Start DestroyField "<<D<<"  "<<coords<<std::endl;
    xdbg<<"field = "<<field<<std::endl;
    switch(coords) {
      case Flat:
           delete static_cast<Field<D,Flat>*>(field);
           break;
      case Sphere:
           delete static_cast<Field<D,Sphere>*>(field);
           break;
      case ThreeD:
           delete static_cast<Field<D,ThreeD>*>(field);
           break;
    }
}

void DestroyGField(void* field, int coords)
{ DestroyField<GData>(field, coords); }

void DestroyKField(void* field, int coords)
{ DestroyField<KData>(field, coords); }

void DestroyNField(void* field, int coords)
{ DestroyField<NData>(field, coords); }

template <int D>
long FieldGetNTopLevel1(void* field, int coords)
{
    switch(coords) {
      case Flat:
           return static_cast<Field<D,Flat>*>(field)->getNTopLevel();
           break;
      case Sphere:
           return static_cast<Field<D,Sphere>*>(field)->getNTopLevel();
           break;
      case ThreeD:
           return static_cast<Field<D,ThreeD>*>(field)->getNTopLevel();
           break;
    }
    return 0;  // Can't get here, but saves a compiler warning
}

long FieldGetNTopLevel(void* field, int d, int coords)
{
    switch(d) {
      case NData:
        return FieldGetNTopLevel1<NData>(field, coords);
        break;
      case KData:
        return FieldGetNTopLevel1<KData>(field, coords);
        break;
      case GData:
        return FieldGetNTopLevel1<GData>(field, coords);
        break;
    }
    return 0;  // Can't get here, but saves a compiler warning
}

template <int D>
long FieldCountNear1(void* field, double x, double y, double z, double sep, int coords)
{
    switch(coords) {
      case Flat:
           return static_cast<Field<D,Flat>*>(field)->countNear(x,y,z,sep);
           break;
      case Sphere:
           return static_cast<Field<D,Sphere>*>(field)->countNear(x,y,z,sep);
           break;
      case ThreeD:
           return static_cast<Field<D,ThreeD>*>(field)->countNear(x,y,z,sep);
           break;
    }
    return 0;  // Can't get here, but saves a compiler warning
}

long FieldCountNear(void* field, double x, double y, double z, double sep, int d, int coords)
{
    switch(d) {
      case NData:
           return FieldCountNear1<NData>(field, x, y, z, sep, coords);
           break;
      case KData:
           return FieldCountNear1<KData>(field, x, y, z, sep, coords);
           break;
      case GData:
           return FieldCountNear1<GData>(field, x, y, z, sep, coords);
           break;
    }
    return 0;  // Can't get here, but saves a compiler warning
}

template <int D>
void FieldGetNear1(void* field, double x, double y, double z, double sep, int coords,
                   long* indices, int n)
{
    switch(coords) {
      case Flat:
           static_cast<Field<D,Flat>*>(field)->getNear(x,y,z,sep,indices,n);
           break;
      case Sphere:
           static_cast<Field<D,Sphere>*>(field)->getNear(x,y,z,sep,indices,n);
           break;
      case ThreeD:
           static_cast<Field<D,ThreeD>*>(field)->getNear(x,y,z,sep,indices,n);
           break;
    }
}

void FieldGetNear(void* field, double x, double y, double z, double sep, int d, int coords,
                  long* indices, int n)
{
    switch(d) {
      case NData:
           FieldGetNear1<NData>(field, x, y, z, sep, coords, indices, n);
           break;
      case KData:
           FieldGetNear1<KData>(field, x, y, z, sep, coords, indices, n);
           break;
      case GData:
           FieldGetNear1<GData>(field, x, y, z, sep, coords, indices, n);
           break;
    }
}

template <int D>
void* BuildSimpleField(double* x, double* y, double* z, double* g1, double* g2, double* k,
                       double* w, double* wpos, long nobj, int coords)
{
    dbg<<"Start BuildSimpleField "<<D<<"  "<<coords<<std::endl;
    void* field=0;
    switch (coords) {
      case Flat:
           field = static_cast<void*>(new SimpleField<D,Flat>(x, y, 0, g1, g2, k,
                                                              w, wpos, nobj));
           break;
      case Sphere:
           field = static_cast<void*>(new SimpleField<D,Sphere>(x, y, z, g1, g2, k,
                                                                w, wpos, nobj));
           break;
      case ThreeD:
           field = static_cast<void*>(new SimpleField<D,ThreeD>(x, y, z, g1, g2, k,
                                                                w, wpos, nobj));
           break;
    }
    xdbg<<"field = "<<field<<std::endl;
    return field;
}

void* BuildGSimpleField(double* x, double* y, double* z, double* g1, double* g2,
                        double* w, double* wpos, long nobj, int coords)
{ return BuildSimpleField<GData>(x,y,z,g1,g2,w,w,wpos,nobj,coords); }

void* BuildKSimpleField(double* x, double* y, double* z, double* k,
                        double* w, double* wpos, long nobj, int coords)
{ return BuildSimpleField<KData>(x,y,z,w,w,k,w,wpos,nobj,coords); }

void* BuildNSimpleField(double* x, double* y, double* z,
                        double* w, double* wpos, long nobj, int coords)
{ return BuildSimpleField<NData>(x,y,z,w,w,w,w,wpos,nobj,coords); }

template <int D>
void DestroySimpleField(void* field, int coords)
{
    dbg<<"Start DestroySimpleField "<<D<<"  "<<coords<<std::endl;
    xdbg<<"field = "<<field<<std::endl;
    switch(coords) {
      case Flat:
           delete static_cast<SimpleField<D,Flat>*>(field);
           break;
      case Sphere:
           delete static_cast<SimpleField<D,Sphere>*>(field);
           break;
      case ThreeD:
           delete static_cast<SimpleField<D,ThreeD>*>(field);
           break;
    }
}

void DestroyGSimpleField(void* field, int coords)
{ DestroySimpleField<GData>(field, coords); }

void DestroyKSimpleField(void* field, int coords)
{ DestroySimpleField<KData>(field, coords); }

void DestroyNSimpleField(void* field, int coords)
{ DestroySimpleField<NData>(field, coords); }


