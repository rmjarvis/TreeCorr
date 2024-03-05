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
    std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& celldata,
    double maxsizesq, size_t start, size_t end, int mintop, int maxtop,
    std::vector<BaseCellData<C>*>& top_data,
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
        ave = static_cast<CellData<D,C>*>(celldata[start].first);
        celldata[start].first = 0; // Make sure the calling function doesn't delete this!
        sizesq = 0.;
    } else {
        ave = new CellData<D,C>(celldata, start, end);
        xdbg<<"ave pos = "<<ave->getPos()<<std::endl;
        xdbg<<"n = "<<ave->getN()<<std::endl;
        xdbg<<"w = "<<ave->getW()<<std::endl;
        sizesq = CalculateSizeSq(ave->getPos(), celldata, start, end);
        xdbg<<"size = "<<sqrt(sizesq)<<std::endl;
    }

    if (sizesq == 0 || (sizesq <= maxsizesq && mintop<=0)) {
        xdbg<<"Small enough.  Make a cell.\n";
        if (end-start > 1) ave->finishAverages(celldata, start, end);
        top_data.push_back(ave);
        top_sizesq.push_back(sizesq);
        top_start.push_back(start);
        top_end.push_back(end);
    } else if (maxtop <= 0) {
        xdbg<<"At specified end of top layer recusion\n";
        if (end-start > 1) ave->finishAverages(celldata, start, end);
        top_data.push_back(ave);
        top_sizesq.push_back(sizesq);
        top_start.push_back(start);
        top_end.push_back(end);
    } else {
        size_t mid = SplitData<C,SM>(celldata, start, end, ave->getPos());
        xdbg<<"Too big.  Recurse with mid = "<<mid<<std::endl;
        SetupTopLevelCells<D,C,SM>(celldata, maxsizesq, start, mid, mintop-1, maxtop-1,
                                   top_data, top_sizesq, top_start, top_end);
        SetupTopLevelCells<D,C,SM>(celldata, maxsizesq, mid, end, mintop-1, maxtop-1,
                                   top_data, top_sizesq, top_start, top_end);
    }
    return sizesq;
}

// A helper struct to build the right kind of CellData object
template <int algo, int D, int C>
struct CellDataHelper2;

// Specialize for each D,C
// Flat
template <>
struct CellDataHelper2<0,NData,Flat>
{
    static CellData<NData,Flat>* build(double x, double y, double , double , double, double w)
    { return new CellData<NData,Flat>(Position<Flat>(x,y), w); }
};
template <>
struct CellDataHelper2<1,KData,Flat>
{
    static CellData<KData,Flat>* build(double x, double y, double , double k, double , double w)
    { return new CellData<KData,Flat>(Position<Flat>(x,y), k, w); }
};
template <int D>
struct CellDataHelper2<2,D,Flat>
{
    static CellData<D,Flat>* build(double x, double y,  double, double g1, double g2, double w)
    { return new CellData<D,Flat>(Position<Flat>(x,y), std::complex<double>(g1,g2), w); }
};

// ThreeD, Sphere
template <int C>
struct CellDataHelper2<3,NData,C>
{
    static CellData<NData,C>* build(double x, double y, double z, double , double, double w)
    { return new CellData<NData,C>(Position<C>(x,y,z), w); }
};
template <int C>
struct CellDataHelper2<4,KData,C>
{
    static CellData<KData,C>* build(double x, double y, double z, double d1, double , double w)
    { return new CellData<KData,C>(Position<C>(x,y,z), d1, w); }
};
template <int D, int C>
struct CellDataHelper2<5,D,C>
{
    static CellData<D,C>* build(double x, double y, double z, double d1, double d2, double w)
    { return new CellData<D,C>(Position<C>(x,y,z), std::complex<double>(d1,d2), w); }
};

template <int D, int C>
struct CellDataHelper
{
    static CellData<D,C>* build(double x, double y, double z, double d1, double d2, double w)
    {
        const int algo =
            C==Flat ? (
                D == NData ? 0 :
                D == KData ? 1 :
                D >= GData ? 2 : -1 )
            : (
                D == NData ? 3 :
                D == KData ? 4 :
                D >= GData ? 5 : -1 );
        return CellDataHelper2<algo,D,C>::build(x,y,z,d1,d2,w);
    }
};

inline WPosLeafInfo get_wpos(const double* wpos, const double* w, long i)
{
    WPosLeafInfo wp;
    wp.wpos = wpos ? wpos[i] : w[i];
    wp.index = i;
    return wp;
}

template <int C>
BaseField<C>::BaseField(long nobj, double minsize, double maxsize,
                        SplitMethod sm, long long seed, bool brute, int mintop, int maxtop) :
    _nobj(nobj), _minsize(minsize), _maxsize(maxsize), _sm(sm),
    _brute(brute), _mintop(mintop), _maxtop(maxtop)
{
    if (seed != 0) { urand(seed); }
}

template <int D, int C>
Field<D,C>::Field(const double* x, const double* y, const double* z,
                  const double* d1, const double* d2,
                  const double* w, const double* wpos, long nobj,
                  double minsize, double maxsize,
                  SplitMethod sm, long long seed, bool brute, int mintop, int maxtop) :
    BaseField<C>(nobj, minsize, maxsize, sm, seed, brute, mintop, maxtop)
{
    //set_verbose(2);
    dbg<<"Starting to Build Field with "<<nobj<<" objects\n";
    xdbg<<"D,C = "<<D<<','<<C<<std::endl;
    xdbg<<"First few values are:\n";
    for(int i=0;i<5;++i) {
        xdbg<<x[i]<<"  "<<y[i]<<"  "<<(z?z[i]:0)<<"  "<<(d1?d1[i]:0)<<"  "<<
            (d2?d2[i]:0)<<"  "<<w[i]<<"  "<<(wpos?wpos[i]:0)<<std::endl;
    }

    this->_celldata.reserve(nobj);
    if (z) {
        for(long i=0;i<nobj;++i) {
            WPosLeafInfo wp = get_wpos(wpos, w, i);
            this->_celldata.push_back(std::make_pair(
                    CellDataHelper<D,C>::build(
                        x[i], y[i], z[i], d1?d1[i]:0., d2?d2[i]:0., w[i]), wp));
        }
    } else {
        Assert(C == Flat);
        for(long i=0;i<nobj;++i) {
            WPosLeafInfo wp = get_wpos(wpos, w, i);
            this->_celldata.push_back(std::make_pair(
                    CellDataHelper<D,C>::build(
                        x[i], y[i], 0., d1?d1[i]:0., d2?d2[i]:0., w[i]), wp));
        }
    }
    dbg<<"Built celldata with "<<this->_celldata.size()<<" entries\n";

    // Calculate the overall center and size
    CellData<D,C> ave(this->_celldata, 0, this->_celldata.size());
    ave.finishAverages(this->_celldata, 0, this->_celldata.size());
    this->_center = ave.getPos();
    this->_sizesq = CalculateSizeSq(this->_center, this->_celldata, 0, this->_celldata.size());
}

template <int D, int C>
void Field<D,C>::BuildCells() const
{
    // Signal that we already built the cells.
    if (this->_celldata.size() == 0) return;

    switch (this->_sm) {
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

    double minsizesq = this->_minsize * this->_minsize;
    xdbg<<"minsizesq = "<<minsizesq<<std::endl;

    double maxsizesq = this->_maxsize * this->_maxsize;
    xdbg<<"maxsizesq = "<<maxsizesq<<std::endl;

    // This is done in two parts so that we can do the (time-consuming) second part in
    // parallel.
    // First we setup what all the top-level cells are going to be.
    // Then we build them and their sub-nodes.

    // Setup the top level cells:
    std::vector<BaseCellData<C>*> top_data;
    std::vector<double> top_sizesq;
    std::vector<size_t> top_start;
    std::vector<size_t> top_end;

    SetupTopLevelCells<D,C,SM>(this->_celldata, maxsizesq, 0, this->_celldata.size(),
                               this->_mintop, this->_maxtop,
                               top_data, top_sizesq, top_start, top_end);
    const ptrdiff_t n = top_data.size();

    // Now build the lower cells in parallel
    dbg<<"Field has "<<n<<" top-level nodes.  Building lower nodes...\n";
    this->_cells.resize(n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(ptrdiff_t i=0;i<n;++i) {
        CellData<D,C>* top_data_i = static_cast<CellData<D,C>*>(top_data[i]);
        this->_cells[i] = BuildCell<D,C,SM>(this->_celldata, minsizesq, this->_brute,
                                            top_start[i], top_end[i],
                                            top_data_i, top_sizesq[i]);
        xdbg<<i<<": "<<this->_cells[i]->getN()<<"  "<<this->_cells[i]->getW()<<"  "<<
            this->_cells[i]->getPos()<<"  "<<this->_cells[i]->getSize()<<std::endl;
    }

    // delete any CellData elements that didn't get kept in the _cells object.
    for (size_t i=0;i<this->_celldata.size();++i) if (this->_celldata[i].first)
        delete this->_celldata[i].first;
    //set_verbose(1);
    this->_celldata.clear();
}

template <int C>
BaseField<C>::~BaseField()
{
    for (size_t i=0; i<_cells.size(); ++i) delete _cells[i];
    // If this is still around, need to delete those too.
    for (size_t i=0; i<_celldata.size(); ++i) if (_celldata[i].first) delete _celldata[i].first;
}

template <int C>
long CountNear(const BaseCell<C>& cell, const Position<C>& pos, double sep, double sepsq)
{
    double s = cell.getSize();
    const double dsq = (cell.getPos() - pos).normSq();
    xdbg<<"CountNear: "<<cell.getPos()<<"  "<<pos<<"  "<<dsq<<"  "<<s<<"  "<<sepsq<<std::endl;

    // If s == 0, then just check dsq
    if (s==0.) {
        if (dsq <= sepsq) {
            dbg<<"s==0, d < sep   N = "<<cell.getN()<<std::endl;
            Assert(sqrt(dsq) <= sep);
            return cell.getN();
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
            dbg<<"d + s <= sep: "<<sqrt(dsq)+s<<" <= "<<sep<<"  N = "<<cell.getN()<<std::endl;
            return cell.getN();
        }

        // Otherwise check the subcells.
        Assert(cell.getLeft());
        Assert(cell.getRight());
        xdbg<<"Recurse to subcells\n";
        return (CountNear(*cell.getLeft(), pos, sep, sepsq) +
                CountNear(*cell.getRight(), pos, sep, sepsq));
    }
}

template <int C>
long BaseField<C>::countNear(double x, double y, double z, double sep) const
{
    BuildCells();  // Make sure this is done.
    Position<C> pos(x, y, z);
    double sepsq = sep*sep;
    long ntot = 0;
    dbg<<"Start countNear: "<<_cells.size()<<" top level cells\n";
    for(size_t i=0; i<_cells.size(); ++i) {
        dbg<<"Top level "<<i<<" with N="<<_cells[i]->getN()<<std::endl;
        ntot += CountNear(*_cells[i], pos, sep, sepsq);
        dbg<<"ntot -> "<<ntot<<std::endl;
    }
    return ntot;
}

template <int C>
void GetNear(const BaseCell<C>& cell, const Position<C>& pos, double sep, double sepsq,
             long* indices, long& k, long n)
{
    double s = cell.getSize();
    const double dsq = (cell.getPos() - pos).normSq();
    dbg<<"GetNear: "<<cell.getPos()<<"  "<<pos<<"  "<<dsq<<"  "<<s<<"  "<<sepsq<<std::endl;

    // If s == 0, then just check dsq
    if (s==0.) {
        if (dsq <= sepsq) {
            dbg<<"s==0, d < sep   N = "<<cell.getN()<<std::endl;
            Assert(sqrt(dsq) <= sep);
            Assert(k < n);
            long n1 = cell.getN();
            Assert(k + n1 <= n);
            // This shouldn't happen, but if it does, we can get a seg fault, so check.
            if (k + n1 > n) return;
            if (n1 == 1) {
                dbg<<"N == 1 case\n";
                indices[k++] = cell.getInfo().index;
            } else {
                dbg<<"N > 1 case: "<<n1<<std::endl;
                const std::vector<long>& leaf_indices = *cell.getListInfo().indices;
                Assert(long(leaf_indices.size()) == n1);
                for (long m=0; m<n1; ++m)
                    indices[k++] = leaf_indices[m];
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
            Assert(cell.getLeft());
            Assert(cell.getRight());
            dbg<<"Recurse to subcells\n";
            GetNear(*cell.getLeft(), pos, sep, sepsq, indices, k, n);
            GetNear(*cell.getRight(), pos, sep, sepsq, indices, k, n);
        }
    }
}

template <int C>
void BaseField<C>::getNear(double x, double y, double z, double sep, long* indices, long n) const
{
    BuildCells();  // Make sure this is done.
    Position<C> pos(x, y, z);
    double sepsq = sep*sep;
    dbg<<"Start getNear: "<<_cells.size()<<" top level cells\n";
    long k = 0;
    for(size_t i=0; i<_cells.size(); ++i) {
        dbg<<"Top level "<<i<<" with N="<<_cells[i]->getN()<<std::endl;
        GetNear(*_cells[i], pos, sep, sepsq, indices, k, n);
        dbg<<"k -> "<<k<<std::endl;
    }
}

//
//
// The functions we call from Python.
//
//

template <int D, int C>
Field<D,C>* BuildField(const double* x, const double* y, const double* z,
                       const double* d1, const double* d2,
                       const double* w, const double* wpos, long nobj,
                       double minsize, double maxsize,
                       SplitMethod sm, long long seed, bool brute,
                       int mintop, int maxtop)
{
    dbg<<"Start BuildField "<<D<<"  "<<C<<std::endl;
    return new Field<D,C>(x, y, z, d1, d2,
                          w, wpos, nobj, minsize, maxsize,
                          sm, seed, brute, mintop, maxtop);
}

template <int C>
Field<NData,C>* BuildNField(
    py::array_t<double>& xp, py::array_t<double>& yp, py::array_t<double>& zp,
    py::array_t<double>& wp, py::array_t<double>& wposp,
    double minsize, double maxsize,
    SplitMethod sm, long long seed, bool brute, int mintop, int maxtop)
{
    long nobj = xp.size();
    Assert(yp.size() == nobj);
    Assert(zp.size() == nobj || zp.size() == 0);
    Assert(wp.size() == nobj);
    Assert(wposp.size() == nobj || wposp.size() == 0);

    const double* x = static_cast<const double*>(xp.data());
    const double* y = static_cast<const double*>(yp.data());
    const double* z = zp.size() == 0 ? 0 : static_cast<const double*>(zp.data());
    const double* w = static_cast<const double*>(wp.data());
    const double* wpos = wposp.size() == 0 ? 0 : static_cast<const double*>(wposp.data());

    return BuildField<NData,C>(x, y, z, 0, 0, w, wpos,
                               nobj, minsize, maxsize, sm, seed,
                               brute, mintop, maxtop);
}

template <int C>
Field<KData,C>* BuildKField(
    py::array_t<double>& xp, py::array_t<double>& yp, py::array_t<double>& zp,
    py::array_t<double>& kp, py::array_t<double>& wp, py::array_t<double>& wposp,
    double minsize, double maxsize,
    SplitMethod sm, long long seed, bool brute, int mintop, int maxtop)
{
    long nobj = xp.size();
    Assert(yp.size() == nobj);
    Assert(zp.size() == nobj || zp.size() == 0);
    Assert(kp.size() == nobj);
    Assert(wp.size() == nobj);
    Assert(wposp.size() == nobj || wposp.size() == 0);

    const double* x = static_cast<const double*>(xp.data());
    const double* y = static_cast<const double*>(yp.data());
    const double* z = zp.size() == 0 ? 0 : static_cast<const double*>(zp.data());
    const double* k = static_cast<const double*>(kp.data());
    const double* w = static_cast<const double*>(wp.data());
    const double* wpos = wposp.size() == 0 ? 0 : static_cast<const double*>(wposp.data());

    return BuildField<KData,C>(x, y, z, k, 0, w, wpos,
                               nobj, minsize, maxsize, sm, seed,
                               brute, mintop, maxtop);
}

template <int D, int C>
Field<D,C>* BuildAnyZField(
    py::array_t<double>& xp, py::array_t<double>& yp, py::array_t<double>& zp,
    py::array_t<double>& d1p, py::array_t<double>& d2p,
    py::array_t<double>& wp, py::array_t<double>& wposp,
    double minsize, double maxsize,
    SplitMethod sm, long long seed, bool brute, int mintop, int maxtop)
{
    long nobj = xp.size();
    Assert(yp.size() == nobj);
    Assert(zp.size() == nobj || zp.size() == 0);
    Assert(d1p.size() == nobj);
    Assert(d2p.size() == nobj);
    Assert(wp.size() == nobj);
    Assert(wposp.size() == nobj || wposp.size() == 0);

    const double* x = static_cast<const double*>(xp.data());
    const double* y = static_cast<const double*>(yp.data());
    const double* z = zp.size() == 0 ? 0 : static_cast<const double*>(zp.data());
    const double* d1 = static_cast<const double*>(d1p.data());
    const double* d2 = static_cast<const double*>(d2p.data());
    const double* w = static_cast<const double*>(wp.data());
    const double* wpos = wposp.size() == 0 ? 0 : static_cast<const double*>(wposp.data());

    return BuildField<D,C>(x, y, z, d1, d2, w, wpos,
                           nobj, minsize, maxsize, sm, seed,
                           brute, mintop, maxtop);
}

template <int C>
Field<GData,C>* BuildGField(
    py::array_t<double>& x, py::array_t<double>& y, py::array_t<double>& z,
    py::array_t<double>& g1, py::array_t<double>& g2,
    py::array_t<double>& w, py::array_t<double>& wpos,
    double minsize, double maxsize,
    SplitMethod sm, long long seed, bool brute, int mintop, int maxtop)
{
    return BuildAnyZField<GData,C>(x, y, z, g1, g2, w, wpos,
                                   minsize, maxsize, sm, seed,
                                   brute, mintop, maxtop);
}

template <int C>
Field<ZData,C>* BuildZField(
    py::array_t<double>& x, py::array_t<double>& y, py::array_t<double>& z,
    py::array_t<double>& z1, py::array_t<double>& z2,
    py::array_t<double>& w, py::array_t<double>& wpos,
    double minsize, double maxsize,
    SplitMethod sm, long long seed, bool brute, int mintop, int maxtop)
{
    return BuildAnyZField<ZData,C>(x, y, z, z1, z2, w, wpos,
                                   minsize, maxsize, sm, seed,
                                   brute, mintop, maxtop);
}

template <int C>
Field<VData,C>* BuildVField(
    py::array_t<double>& x, py::array_t<double>& y, py::array_t<double>& z,
    py::array_t<double>& v1, py::array_t<double>& v2,
    py::array_t<double>& w, py::array_t<double>& wpos,
    double minsize, double maxsize,
    SplitMethod sm, long long seed, bool brute, int mintop, int maxtop)
{
    return BuildAnyZField<VData,C>(x, y, z, v1, v2, w, wpos,
                                   minsize, maxsize, sm, seed,
                                   brute, mintop, maxtop);
}

template <int C>
Field<TData,C>* BuildTField(
    py::array_t<double>& x, py::array_t<double>& y, py::array_t<double>& z,
    py::array_t<double>& t1, py::array_t<double>& t2,
    py::array_t<double>& w, py::array_t<double>& wpos,
    double minsize, double maxsize,
    SplitMethod sm, long long seed, bool brute, int mintop, int maxtop)
{
    return BuildAnyZField<TData,C>(x, y, z, t1, t2, w, wpos,
                                   minsize, maxsize, sm, seed,
                                   brute, mintop, maxtop);
}

template <int C>
Field<QData,C>* BuildQField(
    py::array_t<double>& x, py::array_t<double>& y, py::array_t<double>& z,
    py::array_t<double>& q1, py::array_t<double>& q2,
    py::array_t<double>& w, py::array_t<double>& wpos,
    double minsize, double maxsize,
    SplitMethod sm, long long seed, bool brute, int mintop, int maxtop)
{
    return BuildAnyZField<QData,C>(x, y, z, q1, q2, w, wpos,
                                   minsize, maxsize, sm, seed,
                                   brute, mintop, maxtop);
}

template <int C>
void FieldGetNear(BaseField<C>& field, double x, double y, double z, double sep,
                  py::array_t<long>& inp)
{
    long n = inp.size();
    long* indices = static_cast<long*>(inp.mutable_data());
    field.getNear(x, y, z, sep, indices, n);
}

// Export the above functions using pybind11

// Also wrap some functions that live in KMeans.cpp
template <int C>
void KMeansInitTree(BaseField<C>& field, py::array_t<double>& cen, int npatch, long long seed);
template <int C>
void KMeansInitRand(BaseField<C>& field, py::array_t<double>& cen, int npatch, long long seed);
template <int C>
void KMeansInitKMPP(BaseField<C>& field, py::array_t<double>& cen, int npatch, long long seed);
template <int C>
void KMeansRun(BaseField<C>& field, py::array_t<double>& cen, int npatch,
               int max_iter, double tol, bool alt);
template <int C>
void KMeansAssign(BaseField<C>& field, py::array_t<double>& cen, int npatch,
                  py::array_t<long>& p);

void QuickAssign(py::array_t<double>& cen, int npatch,
                 py::array_t<double>& x, py::array_t<double>& y,
                 py::array_t<double>& z, py::array_t<long>& p);
void SelectPatch(int patch, py::array_t<double>& cenp, int npatch,
                 py::array_t<double>& x, py::array_t<double>& y,
                 py::array_t<double>& z, py::array_t<long>& use);
void GenerateXYZ(
    py::array_t<double>& x, py::array_t<double>& y, py::array_t<double>& z,
    py::array_t<double>& ra, py::array_t<double>& dec, py::array_t<double>& r);


template <int C>
void WrapField(py::module& _treecorr, std::string Cstr)
{
    typedef void (*getnear_type)(BaseField<C>& field, double x, double y, double z,
                                 double sep, py::array_t<long>& inp);
    typedef void (*init_type)(BaseField<C>& field, py::array_t<double>& cenp, int npatch,
                              long long seed);
    typedef void (*run_type)(BaseField<C>& field, py::array_t<double>& cenp, int npatch,
                             int max_iter, double tol, bool alt);
    typedef void (*assign_type)(BaseField<C>& field, py::array_t<double>& cenp, int npatch,
                                py::array_t<long>& pp);

    py::class_<BaseField<C> > field(_treecorr, ("Field" + Cstr).c_str());

    field.def("getNObj", &BaseField<C>::getNObj);
    field.def("getSize", &BaseField<C>::getSize);
    field.def("countNear", &BaseField<C>::countNear);
    field.def("getNear", getnear_type(&FieldGetNear));
    field.def("getNTopLevel", &BaseField<C>::getNTopLevel);

    field.def("KMeansInitTree", init_type(&KMeansInitTree));
    field.def("KMeansInitRand", init_type(&KMeansInitRand));
    field.def("KMeansInitKMPP", init_type(&KMeansInitKMPP));
    field.def("KMeansRun", run_type(&KMeansRun));
    field.def("KMeansAssign", assign_type(&KMeansAssign));

    py::class_<Field<NData,C>, BaseField<C> > nfield(_treecorr, ("NField" + Cstr).c_str());
    py::class_<Field<KData,C>, BaseField<C> > kfield(_treecorr, ("KField" + Cstr).c_str());
    py::class_<Field<ZData,C>, BaseField<C> > zfield(_treecorr, ("ZField" + Cstr).c_str());
    py::class_<Field<VData,C>, BaseField<C> > vfield(_treecorr, ("VField" + Cstr).c_str());
    py::class_<Field<GData,C>, BaseField<C> > gfield(_treecorr, ("GField" + Cstr).c_str());
    py::class_<Field<TData,C>, BaseField<C> > tfield(_treecorr, ("TField" + Cstr).c_str());
    py::class_<Field<QData,C>, BaseField<C> > qfield(_treecorr, ("QField" + Cstr).c_str());

    typedef Field<NData,C>* (*nfield_type)(
        py::array_t<double>& x, py::array_t<double>& y, py::array_t<double>& z,
        py::array_t<double>& w, py::array_t<double>& wpos,
        double minsize, double maxsize,
        SplitMethod sm, long long seed, bool brute, int mintop, int maxtop);
    typedef Field<KData,C>* (*kfield_type)(
        py::array_t<double>& x, py::array_t<double>& y, py::array_t<double>& z,
        py::array_t<double>& k, py::array_t<double>& w, py::array_t<double>& wpos,
        double minsize, double maxsize,
        SplitMethod sm, long long seed, bool brute, int mintop, int maxtop);
    typedef Field<ZData,C>* (*zfield_type)(
        py::array_t<double>& x, py::array_t<double>& y, py::array_t<double>& z,
        py::array_t<double>& z1, py::array_t<double>& z2,
        py::array_t<double>& w, py::array_t<double>& wpos,
        double minsize, double maxsize,
        SplitMethod sm, long long seed, bool brute, int mintop, int maxtop);
    typedef Field<VData,C>* (*vfield_type)(
        py::array_t<double>& x, py::array_t<double>& y, py::array_t<double>& z,
        py::array_t<double>& v1, py::array_t<double>& v2,
        py::array_t<double>& w, py::array_t<double>& wpos,
        double minsize, double maxsize,
        SplitMethod sm, long long seed, bool brute, int mintop, int maxtop);
    typedef Field<GData,C>* (*gfield_type)(
        py::array_t<double>& x, py::array_t<double>& y, py::array_t<double>& z,
        py::array_t<double>& g1, py::array_t<double>& g2,
        py::array_t<double>& w, py::array_t<double>& wpos,
        double minsize, double maxsize,
        SplitMethod sm, long long seed, bool brute, int mintop, int maxtop);
    typedef Field<TData,C>* (*tfield_type)(
        py::array_t<double>& x, py::array_t<double>& y, py::array_t<double>& z,
        py::array_t<double>& t1, py::array_t<double>& t2,
        py::array_t<double>& w, py::array_t<double>& wpos,
        double minsize, double maxsize,
        SplitMethod sm, long long seed, bool brute, int mintop, int maxtop);
    typedef Field<QData,C>* (*qfield_type)(
        py::array_t<double>& x, py::array_t<double>& y, py::array_t<double>& z,
        py::array_t<double>& q1, py::array_t<double>& q2,
        py::array_t<double>& w, py::array_t<double>& wpos,
        double minsize, double maxsize,
        SplitMethod sm, long long seed, bool brute, int mintop, int maxtop);

    nfield.def(py::init(nfield_type(&BuildNField)));
    kfield.def(py::init(kfield_type(&BuildKField)));
    zfield.def(py::init(zfield_type(&BuildZField)));
    vfield.def(py::init(vfield_type(&BuildVField)));
    gfield.def(py::init(gfield_type(&BuildGField)));
    tfield.def(py::init(tfield_type(&BuildTField)));
    qfield.def(py::init(qfield_type(&BuildQField)));
}

void pyExportField(py::module& _treecorr)
{
    WrapField<Flat>(_treecorr, "Flat");
    WrapField<Sphere>(_treecorr, "Sphere");
    WrapField<ThreeD>(_treecorr, "ThreeD");

    _treecorr.def("QuickAssign", &QuickAssign);
    _treecorr.def("SelectPatch", &SelectPatch);
    _treecorr.def("GenerateXYZ", &GenerateXYZ);
}
