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

#include "PyBind11Helper.h"

#include <cstddef>  // for ptrdiff_t
#include "Field.h"
#include "Cell.h"
#include "dbg.h"

// This function just works on the top level data to figure out which data goes into
// each top-level Cell.  It is building up the top_* vectors, which can then be used
// to build the actual Cells.
template <int D, int C, int SM>
double SetupTopLevelCells(
    std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& celldata,
    double maxsizesq, size_t start, size_t end, int mintop, int maxtop,
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
        size_t mid = SplitData<D,C,SM>(celldata,start,end,ave->getPos());
        xdbg<<"Too big.  Recurse with mid = "<<mid<<std::endl;
        SetupTopLevelCells<D,C,SM>(celldata, maxsizesq, start, mid, mintop-1, maxtop-1,
                                   top_data, top_sizesq, top_start, top_end);
        SetupTopLevelCells<D,C,SM>(celldata, maxsizesq, mid, end, mintop-1, maxtop-1,
                                   top_data, top_sizesq, top_start, top_end);
    }
    return sizesq;
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

inline WPosLeafInfo get_wpos(double* wpos, double* w, long i)
{
    WPosLeafInfo wp;
    wp.wpos = wpos ? wpos[i] : w[i];
    wp.index = i;
    return wp;
}

template <bool B>
double at(double* x, int i) { return x[i]; }
template <>
double at<false>(double* x, int i) { return 0.; }

template <int D, int C>
Field<D,C>::Field(double* x, double* y, double* z, double* g1, double* g2, double* k,
                  double* w, double* wpos, long nobj,
                  double minsize, double maxsize,
                  SplitMethod sm, long long seed, bool brute, int mintop, int maxtop) :
    _nobj(nobj), _minsize(minsize), _maxsize(maxsize), _sm(sm),
    _brute(brute), _mintop(mintop), _maxtop(maxtop)
{
    //set_verbose(2);
    dbg<<"Starting to Build Field with "<<nobj<<" objects\n";
    xdbg<<"D,C = "<<D<<','<<C<<std::endl;
    xdbg<<"First few values are:\n";
    for(int i=0;i<5;++i) {
        xdbg<<x[i]<<"  "<<y[i]<<"  "<<(z?z[i]:0)<<"  "<<at<D==GData>(g1,i)<<"  "<<
            at<D==GData>(g2,i)<<"  "<<at<D==KData>(k,i)<<"  "<<w[i]<<"  "<<
            (wpos?wpos[i]:0)<<std::endl;
    }

    if (seed != 0) { urand(seed); }
    _celldata.reserve(nobj);
    if (z) {
        for(long i=0;i<nobj;++i) {
            WPosLeafInfo wp = get_wpos(wpos,w,i);
            _celldata.push_back(std::make_pair(
                    CellDataHelper<D,C>::build(
                        x[i], y[i], z[i], at<D==GData>(g1,i), at<D==GData>(g2,i),
                        at<D==KData>(k,i), w[i]), wp));
        }
    } else {
        Assert(C == Flat);
        for(long i=0;i<nobj;++i) {
            WPosLeafInfo wp = get_wpos(wpos,w,i);
            _celldata.push_back(std::make_pair(
                    CellDataHelper<D,C>::build(
                        x[i], y[i], 0., at<D==GData>(g1,i), at<D==GData>(g2,i),
                        at<D==KData>(k,i), w[i]), wp));
        }
    }
    dbg<<"Built celldata with "<<_celldata.size()<<" entries\n";

    // Calculate the overall center and size
    CellData<D,C> ave(_celldata, 0, _celldata.size());
    ave.finishAverages(_celldata, 0, _celldata.size());
    _center = ave.getPos();
    _sizesq = CalculateSizeSq(_center, _celldata, 0, _celldata.size());
}

template <int D, int C>
void Field<D,C>::BuildCells() const
{
    // Signal that we already built the cells.
    if (_celldata.size() == 0) return;

    switch (_sm) {
      case Middle:
           DoBuildCells<Middle>();
           break;
      case Median:
           DoBuildCells<Median>();
           break;
      case Mean:
           DoBuildCells<Mean>();
           break;
      case Random:
           DoBuildCells<Random>();
           break;
      default:
           throw std::runtime_error("Invalid SplitMethod");
    };
}

template <int D, int C> template <int SM>
void Field<D,C>::DoBuildCells() const
{
    // We don't build Cells that are too big or too small based on the min/max separation:

    double minsizesq = _minsize * _minsize;
    xdbg<<"minsizesq = "<<minsizesq<<std::endl;

    double maxsizesq = _maxsize * _maxsize;
    xdbg<<"maxsizesq = "<<maxsizesq<<std::endl;

    // This is done in two parts so that we can do the (time-consuming) second part in
    // parallel.
    // First we setup what all the top-level cells are going to be.
    // Then we build them and their sub-nodes.

    // Setup the top level cells:
    std::vector<CellData<D,C>*> top_data;
    std::vector<double> top_sizesq;
    std::vector<size_t> top_start;
    std::vector<size_t> top_end;

    SetupTopLevelCells<D,C,SM>(_celldata, maxsizesq, 0, _celldata.size(), _mintop, _maxtop,
                               top_data, top_sizesq, top_start, top_end);
    const ptrdiff_t n = top_data.size();

    // Now build the lower cells in parallel
    dbg<<"Field has "<<n<<" top-level nodes.  Building lower nodes...\n";
    _cells.resize(n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(ptrdiff_t i=0;i<n;++i) {
        _cells[i] = BuildCell<D,C,SM>(_celldata, minsizesq, _brute,
                                      top_start[i], top_end[i],
                                      top_data[i], top_sizesq[i]);
        xdbg<<i<<": "<<_cells[i]->getN()<<"  "<<_cells[i]->getW()<<"  "<<
            _cells[i]->getPos()<<"  "<<_cells[i]->getSize()<<std::endl;
    }

    // delete any CellData elements that didn't get kept in the _cells object.
    for (size_t i=0;i<_celldata.size();++i) if (_celldata[i].first) delete _celldata[i].first;
    //set_verbose(1);
    _celldata.clear();
}

template <int D, int C>
Field<D,C>::~Field()
{
    for (size_t i=0; i<_cells.size(); ++i) delete _cells[i];
    // If this is still around, need to delete those too.
    for (size_t i=0; i<_celldata.size(); ++i) if (_celldata[i].first) delete _celldata[i].first;
}

template <int D, int C>
long CountNear(const Cell<D,C>* cell, const Position<C>& pos, double sep, double sepsq)
{
    double s = cell->getSize();
    const double dsq = (cell->getPos() - pos).normSq();
    xdbg<<"CountNear: "<<cell->getPos()<<"  "<<pos<<"  "<<dsq<<"  "<<s<<"  "<<sepsq<<std::endl;

    // If s == 0, then just check dsq
    if (s==0.) {
        if (dsq <= sepsq) {
            dbg<<"s==0, d < sep   N = "<<cell->getN()<<std::endl;
            Assert(sqrt(dsq) <= sep);
            return cell->getN();
        }
        else {
            Assert(sqrt(dsq) > sep);
            xdbg<<"s==0, d >= sep\n";
            return 0;
        }
    } else {
        // If d - s > sep, then no points are close enough.
        if (dsq > sepsq && dsq > SQR(sep+s)) {
            Assert(sqrt(dsq) - s > sep);
            xdbg<<"d - s > sep: "<<sqrt(dsq)-s<<" > "<<sep<<std::endl;
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
        xdbg<<"Recurse to subcells\n";
        return (CountNear(cell->getLeft(), pos, sep, sepsq) +
                CountNear(cell->getRight(), pos, sep, sepsq));
    }
}

template <int D, int C>
long Field<D,C>::countNear(double x, double y, double z, double sep) const
{
    BuildCells();  // Make sure this is done.
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
             long* indices, long& k, long n)
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
                Assert(long(leaf_indices->size()) == n1);
                for (long m=0; m<n1; ++m)
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
void Field<D,C>::getNear(double x, double y, double z, double sep, long* indices, long n) const
{
    BuildCells();  // Make sure this is done.
    Position<C> pos(x,y,z);
    double sepsq = sep*sep;
    dbg<<"Start getNear: "<<_cells.size()<<" top level cells\n";
    long k = 0;
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
            celldata.push_back(std::make_pair(
                    CellDataHelper<D,C>::build(
                        x[i], y[i], z[i], at<D==GData>(g1,i), at<D==GData>(g2,i),
                        at<D==KData>(k,i), w[i]), wp));
        }
    } else {
        Assert(C == Flat);
        for(long i=0;i<nobj;++i) {
            WPosLeafInfo wp = get_wpos(wpos,w,i);
            celldata.push_back(std::make_pair(
                    CellDataHelper<D,C>::build(
                        x[i], y[i], 0., at<D==GData>(g1,i), at<D==GData>(g2,i),
                        at<D==KData>(k,i), w[i]), wp));
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
// The functions we call from Python.
//
//

template <int D>
BaseField<D>* BuildField(double* x, double* y, double* z, double* g1, double* g2, double* k,
                 double* w, double* wpos, long nobj,
                 double minsize, double maxsize,
                 SplitMethod sm, long long seed, bool brute, int mintop, int maxtop, Coord coords)
{
    dbg<<"Start BuildField "<<D<<"  "<<coords<<std::endl;
    switch(coords) {
      case Flat:
           return new Field<D,Flat>(x, y, 0, g1, g2, k,
                                    w, wpos, nobj,
                                    minsize, maxsize,
                                    sm, seed,
                                    bool(brute), mintop, maxtop);
      case Sphere:
           return new Field<D,Sphere>(x, y, z, g1, g2, k,
                                      w, wpos, nobj,
                                      minsize, maxsize,
                                      sm, seed,
                                      bool(brute), mintop, maxtop);
      case ThreeD:
           return new Field<D,ThreeD>(x, y, z, g1, g2, k,
                                      w, wpos, nobj,
                                      minsize, maxsize,
                                      sm, seed,
                                      bool(brute), mintop, maxtop);
      default:
           Assert(false);
    }
    return 0;
}

BaseField<GData>* BuildGField(
    py::array_t<double>& xp, py::array_t<double>& yp, py::array_t<double>& zp,
    py::array_t<double>& g1p, py::array_t<double>& g2p,
    py::array_t<double>& wp, py::array_t<double>& wposp,
    double minsize, double maxsize,
    SplitMethod sm, long long seed, bool brute, int mintop, int maxtop, Coord coords)
{
    long nobj = xp.request().size;
    Assert(yp.request().size == nobj);
    Assert(zp.request().size == nobj || zp.request().size == 0);
    Assert(g1p.request().size == nobj);
    Assert(g2p.request().size == nobj);
    Assert(wp.request().size == nobj);
    Assert(wposp.request().size == nobj || wposp.request().size == 0);

    double* x = static_cast<double*>(xp.request().ptr);
    double* y = static_cast<double*>(yp.request().ptr);
    double* z = zp.request().size == 0 ? 0 : static_cast<double*>(zp.request().ptr);
    double* g1 = static_cast<double*>(g1p.request().ptr);
    double* g2 = static_cast<double*>(g2p.request().ptr);
    double* w = static_cast<double*>(wp.request().ptr);
    double* wpos = wposp.request().size == 0 ? 0 : static_cast<double*>(wposp.request().ptr);

    return BuildField<GData>(x, y, z, g1, g2, 0, w, wpos,
                             nobj, minsize, maxsize, sm, seed,
                             brute, mintop, maxtop, coords);
}

BaseField<KData>* BuildKField(
    py::array_t<double>& xp, py::array_t<double>& yp, py::array_t<double>& zp,
    py::array_t<double>& kp, py::array_t<double>& wp, py::array_t<double>& wposp,
    double minsize, double maxsize,
    SplitMethod sm, long long seed, bool brute, int mintop, int maxtop, Coord coords)
{
    long nobj = xp.request().size;
    Assert(yp.request().size == nobj);
    Assert(zp.request().size == nobj || zp.request().size == 0);
    Assert(kp.request().size == nobj);
    Assert(wp.request().size == nobj);
    Assert(wposp.request().size == nobj || wposp.request().size == 0);

    double* x = static_cast<double*>(xp.request().ptr);
    double* y = static_cast<double*>(yp.request().ptr);
    double* z = zp.request().size == 0 ? 0 : static_cast<double*>(zp.request().ptr);
    double* k = static_cast<double*>(kp.request().ptr);
    double* w = static_cast<double*>(wp.request().ptr);
    double* wpos = wposp.request().size == 0 ? 0 : static_cast<double*>(wposp.request().ptr);

    return BuildField<KData>(x, y, z, 0, 0, k, w, wpos,
                             nobj, minsize, maxsize, sm, seed,
                             brute, mintop, maxtop, coords);
}

BaseField<NData>* BuildNField(
    py::array_t<double>& xp, py::array_t<double>& yp, py::array_t<double>& zp,
    py::array_t<double>& wp, py::array_t<double>& wposp,
    double minsize, double maxsize,
    SplitMethod sm, long long seed, bool brute, int mintop, int maxtop, Coord coords)
{
    long nobj = xp.request().size;
    Assert(yp.request().size == nobj);
    Assert(zp.request().size == nobj || zp.request().size == 0);
    Assert(wp.request().size == nobj);
    Assert(wposp.request().size == nobj || wposp.request().size == 0);

    double* x = static_cast<double*>(xp.request().ptr);
    double* y = static_cast<double*>(yp.request().ptr);
    double* z = zp.request().size == 0 ? 0 : static_cast<double*>(zp.request().ptr);
    double* w = static_cast<double*>(wp.request().ptr);
    double* wpos = wposp.request().size == 0 ? 0 : static_cast<double*>(wposp.request().ptr);

    return BuildField<NData>(x, y, z, 0, 0, 0, w, wpos,
                             nobj, minsize, maxsize, sm, seed,
                             brute, mintop, maxtop, coords);
}

template <int D>
void FieldGetNear(BaseField<D>* field, double x, double y, double z, double sep,
                  py::array_t<long>& inp)
{
    long n = inp.request().size;
    long* indices = static_cast<long*>(inp.request().ptr);
    field->getNear(x,y,z,sep,indices,n);
}

template <int D>
BaseSimpleField<D>* BuildSimpleField(
    double* x, double* y, double* z, double* g1, double* g2, double* k,
    double* w, double* wpos, long nobj, Coord coords)
{
    dbg<<"Start BuildSimpleField "<<D<<"  "<<coords<<std::endl;
    switch (coords) {
      case Flat:
           return new SimpleField<D,Flat>(x, y, 0, g1, g2, k,
                                          w, wpos, nobj);
      case Sphere:
           return new SimpleField<D,Sphere>(x, y, z, g1, g2, k,
                                            w, wpos, nobj);
      case ThreeD:
           return new SimpleField<D,ThreeD>(x, y, z, g1, g2, k,
                                            w, wpos, nobj);
      default:
           Assert(false);
    }
    return 0;
}

BaseSimpleField<GData>* BuildGSimpleField(
    py::array_t<double>& xp, py::array_t<double>& yp, py::array_t<double>& zp,
    py::array_t<double>& g1p, py::array_t<double>& g2p,
    py::array_t<double>& wp, py::array_t<double>& wposp,
    Coord coords)
{
    long nobj = xp.request().size;
    Assert(yp.request().size == nobj);
    Assert(zp.request().size == nobj || zp.request().size == 0);
    Assert(g1p.request().size == nobj);
    Assert(g2p.request().size == nobj);
    Assert(wp.request().size == nobj);
    Assert(wposp.request().size == nobj || wposp.request().size == 0);

    double* x = static_cast<double*>(xp.request().ptr);
    double* y = static_cast<double*>(yp.request().ptr);
    double* z = zp.request().size == 0 ? 0 : static_cast<double*>(zp.request().ptr);
    double* g1 = static_cast<double*>(g1p.request().ptr);
    double* g2 = static_cast<double*>(g2p.request().ptr);
    double* w = static_cast<double*>(wp.request().ptr);
    double* wpos = wposp.request().size == 0 ? 0 : static_cast<double*>(wposp.request().ptr);

    return BuildSimpleField<GData>(x, y, z, g1, g2, 0, w, wpos, nobj, coords);
}

BaseSimpleField<KData>* BuildKSimpleField(
    py::array_t<double>& xp, py::array_t<double>& yp, py::array_t<double>& zp,
    py::array_t<double>& kp,
    py::array_t<double>& wp, py::array_t<double> wposp,
    Coord coords)
{
    long nobj = xp.request().size;
    Assert(yp.request().size == nobj);
    Assert(zp.request().size == nobj || zp.request().size == 0);
    Assert(kp.request().size == nobj);
    Assert(wp.request().size == nobj);
    Assert(wposp.request().size == nobj || wposp.request().size == 0);

    double* x = static_cast<double*>(xp.request().ptr);
    double* y = static_cast<double*>(yp.request().ptr);
    double* z = zp.request().size == 0 ? 0 : static_cast<double*>(zp.request().ptr);
    double* k = static_cast<double*>(kp.request().ptr);
    double* w = static_cast<double*>(wp.request().ptr);
    double* wpos = wposp.request().size == 0 ? 0 : static_cast<double*>(wposp.request().ptr);

    return BuildSimpleField<KData>(x, y, z, 0, 0, k, w, wpos, nobj, coords);
}

BaseSimpleField<NData>* BuildNSimpleField(
    py::array_t<double>& xp, py::array_t<double>& yp, py::array_t<double>& zp,
    py::array_t<double>& wp, py::array_t<double>& wposp,
    Coord coords)
{
    long nobj = xp.request().size;
    Assert(yp.request().size == nobj);
    Assert(zp.request().size == nobj || zp.request().size == 0);
    Assert(wp.request().size == nobj);
    Assert(wposp.request().size == nobj || wposp.request().size == 0);

    double* x = static_cast<double*>(xp.request().ptr);
    double* y = static_cast<double*>(yp.request().ptr);
    double* z = zp.request().size == 0 ? 0 : static_cast<double*>(zp.request().ptr);
    double* w = static_cast<double*>(wp.request().ptr);
    double* wpos = wposp.request().size == 0 ? 0 : static_cast<double*>(wposp.request().ptr);

    return BuildSimpleField<NData>(x, y, z, 0, 0, 0, w, wpos, nobj, coords);
}

// Export the above functions using pybind11

// Also wrap some functions that live in KMeans.cpp
template <int D>
void KMeansInitTree(BaseField<D>* field, py::array_t<double>& cenp, int npatch,
                    Coord coords, long long seed);
template <int D>
void KMeansInitRand(BaseField<D>* field, py::array_t<double>& cenp, int npatch,
                    Coord coords, long long seed);
template <int D>
void KMeansInitKMPP(BaseField<D>* field, py::array_t<double>& cenp, int npatch,
                    Coord coords, long long seed);
template <int D>
void KMeansRun(BaseField<D>* field, py::array_t<double>& cenp, int npatch,
               int max_iter, double tol, bool alt, Coord coords);
template <int D>
void KMeansAssign(BaseField<D>* field, py::array_t<double>& cenp, int npatch,
                  py::array_t<long>& pp, Coord coords);
void QuickAssign(py::array_t<double>& cenp, int npatch,
                 py::array_t<double>& xp, py::array_t<double>& yp,
                 py::array_t<double>& zp, py::array_t<long>& pp);
void SelectPatch(int patch, py::array_t<double>& cenp, int npatch,
                 py::array_t<double>& xp, py::array_t<double>& yp,
                 py::array_t<double>& zp, py::array_t<long>& usep);
void GenerateXYZ(
    py::array_t<double>& xp, py::array_t<double>& yp, py::array_t<double>& zp,
    py::array_t<double>& rap, py::array_t<double>& decp, py::array_t<double>& rp);


template <int D, typename F>
void WrapField(F& field)
{
    typedef void (*getNear_type)(BaseField<D>* field, double x, double y, double z,
                                 double sep, py::array_t<long>& inp);

    typedef void (*init_type)(BaseField<D>* field, py::array_t<double>& cenp, int npatch,
                              Coord coords, long long seed);
    typedef void (*run_type)(BaseField<D>* field, py::array_t<double>& cenp, int npatch,
                            int max_iter, double tol, bool alt, Coord coords);
    typedef void (*assign_type)(BaseField<D>* field, py::array_t<double>& cenp, int npatch,
                                py::array_t<long>& pp, Coord coords);

    field.def("getNObj", &BaseField<D>::getNObj);
    field.def("getSize", &BaseField<D>::getSize);
    field.def("countNear", &BaseField<D>::countNear);
    field.def("getNear", getNear_type(&FieldGetNear));
    field.def("getNTopLevel", &BaseField<D>::getNTopLevel);

    field.def("KMeansInitTree", init_type(&KMeansInitTree));
    field.def("KMeansInitRand", init_type(&KMeansInitRand));
    field.def("KMeansInitKMPP", init_type(&KMeansInitKMPP));
    field.def("KMeansRun", run_type(&KMeansRun));
    field.def("KMeansAssign", assign_type(&KMeansAssign));
}

void pyExportField(py::module& _treecorr)
{
    py::class_<BaseField<NData> > nfield(_treecorr, "NField");
    py::class_<BaseField<KData> > kfield(_treecorr, "KField");
    py::class_<BaseField<GData> > gfield(_treecorr, "GField");

    // These aren't included in the WrapField template, since the function signatures
    // aren't the same.
    nfield.def(py::init(&BuildNField));
    kfield.def(py::init(&BuildKField));
    gfield.def(py::init(&BuildGField));

    WrapField<NData>(nfield);
    WrapField<KData>(kfield);
    WrapField<GData>(gfield);

    py::class_<BaseSimpleField<NData> > nsimplefield(_treecorr, "NSimpleField");
    py::class_<BaseSimpleField<KData> > ksimplefield(_treecorr, "KSimpleField");
    py::class_<BaseSimpleField<GData> > gsimplefield(_treecorr, "GSimpleField");

    nsimplefield.def(py::init(&BuildNSimpleField));
    ksimplefield.def(py::init(&BuildKSimpleField));
    gsimplefield.def(py::init(&BuildGSimpleField));

    _treecorr.def("QuickAssign", &QuickAssign);
    _treecorr.def("SelectPatch", &SelectPatch);
    _treecorr.def("GenerateXYZ", &GenerateXYZ);
}
