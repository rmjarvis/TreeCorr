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

#ifndef TreeCorr_Cell_H
#define TreeCorr_Cell_H

#include <iostream>
#include <algorithm>
#include <complex>
#include <vector>

#include "Position.h"
#include "dbg.h"

enum SplitMethod { Middle, Median, Mean, Random };

const double PI = 3.141592653589793;
const double TWOPI = 2.*PI;
const double IOTA = 1.e-10;

// We use a code (to be used as a template parameter) to indicate which kind of data we
// are using for a particular use.
// NData means just count the point.
// KData means use a scalar.  Nominally kappa, but works with any scalar (e.g. temperature).
// GData means use a shear.
enum DataType { NData, KData, GData, ZData, VData, TData, QData };

// Return a random number between 0 and 1.
double urand(long long seed=0);

// This is usually what we store in the leaf cells. It has size 4, which is always <= the
// size of a pointer on modern machines, so it never adds any space to the memory needed.
// (Since it is in a union with the _right pointer.)
struct LeafInfo
{
    long index;
};

// This is used when building.  We don't need to store the wpos values, but we need them
// while we're building up the tree.
struct WPosLeafInfo : public LeafInfo
{
    double wpos;
};

// When we decide we're at a leaf, but we have >1 index to include, we use this instead.
struct ListLeafInfo
{
    std::vector<long>* indices;
};


// This class encapsulates the differences in the different kinds of data being
// stored in a Cell.  It is used both for the input data from the file and also
// for the mean values for a given Cell.  Some extra useful information is sometimes
// also stored.
template <int D, int C>
class CellData;

template <int C>
class BaseCellData
{
public:
    BaseCellData() {}

    virtual ~BaseCellData() {}

    BaseCellData(const Position<C>& pos, double w) :
        _pos(pos), _w(w), _n(1)
    {}

    BaseCellData(const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
                 size_t start, size_t end);

    const Position<C>& getPos() const { return _pos; }
    double getW() const { return _w; }
    long getN() const { return _n; }

protected:

    Position<C> _pos;
    float _w;
    long _n;
};


template <int C>
class CellData<NData,C> : public BaseCellData<C>
{
public:
    CellData() {}

    CellData(const Position<C>& pos, double w) :
        BaseCellData<C>(pos, w) {}

    template <int C2>
    CellData(const Position<C2>& pos, double w) :
        BaseCellData<C>(pos, w) {}

    CellData(const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
             size_t start, size_t end) :
        BaseCellData<C>(vdata, start, end) {}

    // This doesn't do anything, but is provided for consistency with the other
    // kinds of CellData.
    void finishAverages(const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >&,
                        size_t , size_t ) {}
};

template <int C>
std::ostream& operator<<(std::ostream& os, const CellData<NData,C>& c)
{ return os << c.getPos() << " " << c.getN(); }

template <int C>
class CellData<KData,C> : public BaseCellData<C>
{
public:
    CellData() {}

    CellData(const Position<C>& pos, double k, double w) :
        BaseCellData<C>(pos, w), _wk(w*k)
    {}

    template <int C2>
    CellData(const Position<C2>& pos, double k, double w) :
        BaseCellData<C>(pos, w), _wk(w*k)
    {}

    CellData(const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
             size_t start, size_t end) :
        BaseCellData<C>(vdata, start, end) {}

    // The above constructor just computes the mean pos, since sometimes that's all we
    // need.  So this function will finish the rest of the construction when desired.
    void finishAverages(const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >&,
                        size_t start, size_t end);

    double getWK() const { return _wk; }
    void setWK(double wk) { _wk = wk; }

protected:

    float _wk;
};

template <int C>
std::ostream& operator<<(std::ostream& os, const CellData<KData,C>& c)
{ return os << c.getPos() << " " << c.getWK() << " " << c.getW() << " " << c.getN(); }

template <int C>
class CellData<GData,C> : public BaseCellData<C>
{
public:
    CellData() {}

    CellData(const Position<C>& pos, const std::complex<double>& g, double w) :
        BaseCellData<C>(pos, w), _wg(w*g)
    {}

    template <int C2>
    CellData(const Position<C2>& pos, const std::complex<double>& g, double w) :
        BaseCellData<C>(pos, w), _wg(w*g)
    {}

    CellData(const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
             size_t start, size_t end) :
        BaseCellData<C>(vdata, start, end) {}

    // The above constructor just computes the mean pos, since sometimes that's all we
    // need.  So this function will finish the rest of the construction when desired.
    void finishAverages(const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >&,
                        size_t start, size_t end);

    std::complex<double> getWG() const { return _wg; }
    void setWG(const std::complex<double>& wg) { _wg = wg; }

protected:

    std::complex<float> _wg;
};

template <int C>
std::ostream& operator<<(std::ostream& os, const CellData<GData,C>& c)
{ return os << c.getPos() << " " << c.getWG() << " " << c.getW() << " " << c.getN(); }

template <int C>
class CellData<ZData,C> : public CellData<GData, C>
{
public:
    CellData() {}

    CellData(const Position<C>& pos, const std::complex<double>& v, double w) :
        CellData<GData,C>(pos, v, w) {}

    template <int C2>
    CellData(const Position<C2>& pos, const std::complex<double>& v, double w) :
        CellData<GData,C>(pos, v, w) {}

    CellData(const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
             size_t start, size_t end) :
        CellData<GData,C>(vdata, start, end) {}

    // The above constructor just computes the mean pos, since sometimes that's all we
    // need.  So this function will finish the rest of the construction when desired.
    void finishAverages(const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >&,
                        size_t start, size_t end);

    std::complex<double> getWZ() const { return this->getWG(); }
    void setWZ(const std::complex<double>& wv) { this->setWG(wv); }
};

template <int C>
class CellData<VData,C> : public CellData<GData, C>
{
public:
    CellData() {}

    CellData(const Position<C>& pos, const std::complex<double>& v, double w) :
        CellData<GData,C>(pos, v, w) {}

    template <int C2>
    CellData(const Position<C2>& pos, const std::complex<double>& v, double w) :
        CellData<GData,C>(pos, v, w) {}

    CellData(const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
             size_t start, size_t end) :
        CellData<GData,C>(vdata, start, end) {}

    // The above constructor just computes the mean pos, since sometimes that's all we
    // need.  So this function will finish the rest of the construction when desired.
    void finishAverages(const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >&,
                        size_t start, size_t end);

    std::complex<double> getWV() const { return this->getWG(); }
    void setWV(const std::complex<double>& wv) { this->setWG(wv); }
};

template <int C>
class CellData<TData,C> : public CellData<GData, C>
{
public:
    CellData() {}

    CellData(const Position<C>& pos, const std::complex<double>& t, double w) :
        CellData<GData,C>(pos, t, w) {}

    template <int C2>
    CellData(const Position<C2>& pos, const std::complex<double>& t, double w) :
        CellData<GData,C>(pos, t, w) {}

    CellData(const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
             size_t start, size_t end) :
        CellData<GData,C>(vdata, start, end) {}

    // The above constructor just computes the mean pos, since sometimes that's all we
    // need.  So this function will finish the rest of the construction when desired.
    void finishAverages(const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >&,
                        size_t start, size_t end);

    std::complex<double> getWT() const { return this->getWG(); }
    void setWT(const std::complex<double>& wt) { this->setWG(wt); }
};

template <int C>
class CellData<QData,C> : public CellData<GData, C>
{
public:
    CellData() {}

    CellData(const Position<C>& pos, const std::complex<double>& q, double w) :
        CellData<GData,C>(pos, q, w) {}

    template <int C2>
    CellData(const Position<C2>& pos, const std::complex<double>& q, double w) :
        CellData<GData,C>(pos, q, w) {}

    CellData(const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
             size_t start, size_t end) :
        CellData<GData,C>(vdata, start, end) {}

    // The above constructor just computes the mean pos, since sometimes that's all we
    // need.  So this function will finish the rest of the construction when desired.
    void finishAverages(const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >&,
                        size_t start, size_t end);

    std::complex<double> getWQ() const { return this->getWG(); }
    void setWQ(const std::complex<double>& wq) { this->setWG(wq); }
};

template <int C>
class BaseCell
{
public:

    BaseCell(BaseCellData<C>* data, const LeafInfo& info) :
        _data(data), _size(0.), _left(0), _info(info) {}

    BaseCell(BaseCellData<C>* data, const ListLeafInfo& listinfo) :
        _data(data), _size(0.), _left(0), _listinfo(listinfo) {}

    BaseCell(BaseCellData<C>* data, double size, BaseCell<C>* l, BaseCell<C>* r) :
        _data(data), _size(size), _left(l), _right(r) {}

    virtual ~BaseCell()
    {
        if (_left) {
            Assert(_right);
            delete _left;
            delete _right;
        } else if (_data && _data->getN() > 1) {
            delete _listinfo.indices;
        } // if !left and N==1, then _info, which doesn't need anything to be deleted.
        if (_data) {
            delete (_data);
        }
    }

    const BaseCellData<C>& getData() const { return *_data; }
    const Position<C>& getPos() const { return _data->getPos(); }
    double getW() const { return _data->getW(); }
    long getN() const { return _data->getN(); }

    double getSize() const { return _size; }
    double calculateInertia() const;
    double calculateSumWSq() const;

    const BaseCell<C>* getLeft() const { return _left; }
    const BaseCell<C>* getRight() const { return _left ? _right : 0; }
    const LeafInfo& getInfo() const { Assert(!_left && getN()==1); return _info; }
    const ListLeafInfo& getListInfo() const { Assert(!_left && getN()!=1); return _listinfo; }

    // These are mostly used for debugging purposes.
    long countLeaves() const;
    std::vector<const BaseCell<C>*> getAllLeaves() const;
    bool includesIndex(long index) const;
    std::vector<long> getAllIndices() const;
    const BaseCell<C>* getLeafNumber(long i) const;

    void Write(std::ostream& os) const;
    void WriteTree(std::ostream& os, int indent=0) const;

protected:

    BaseCellData<C>* _data;
    float _size;

    BaseCell<C>* _left;
    union {
        BaseCell<C>* _right;    // Use this when _left != 0
        LeafInfo _info;         // Use this when _left == 0 and N == 1
        ListLeafInfo _listinfo; // Use this when _left == 0 and N > 1
    };
};

// A Cell contains the accumulated data for a bunch of galaxies.
// It is characterized primarily by a centroid and a size.
// The centroid is simply the weighted centroid of all the galaxy positions.
// The size is the maximum deviation of any one of these galaxies
// from the centroid.  That is, all galaxies fall within a radius
// size from the centroid.
// The structure also keeps track of some averages and sums about
// the galaxies which are used in the correlation function calculations.
template <int D, int C>
class Cell : public BaseCell<C>
{
public:
    Cell(CellData<D,C>* data, const LeafInfo& info) :
        BaseCell<C>(data, info) {}

    Cell(CellData<D,C>* data, const ListLeafInfo& listinfo) :
        BaseCell<C>(data, listinfo) {}

    Cell(CellData<D,C>* data, double size, Cell<D,C>* l, Cell<D,C>* r) :
        BaseCell<C>(data, size, l, r) {}

    const CellData<D,C>& getData() const
    { return static_cast<const CellData<D,C>&>(BaseCell<C>::getData()); }

    const Cell<D,C>* getLeft() const
    { return static_cast<const Cell<D,C>*>(BaseCell<C>::getLeft()); }

    const Cell<D,C>* getRight() const
    { return static_cast<const Cell<D,C>*>(BaseCell<C>::getRight()); }
};

// The above is fine for NData, but K and G need a couple more methods.
// (When we eventually do 3pt for Z,V,T,Q, they will also need specializations.)
template <int C>
class Cell<KData,C> : public BaseCell<C>
{
public:
    Cell(CellData<KData,C>* data, const LeafInfo& info) :
        BaseCell<C>(data, info) {}

    Cell(CellData<KData,C>* data, const ListLeafInfo& listinfo) :
        BaseCell<C>(data, listinfo) {}

    Cell(CellData<KData,C>* data, double size, Cell<KData,C>* l, Cell<KData,C>* r) :
        BaseCell<C>(data, size, l, r) {}

    const CellData<KData,C>& getData() const
    { return static_cast<const CellData<KData,C>&>(BaseCell<C>::getData()); }

    const Cell<KData,C>* getLeft() const
    { return static_cast<const Cell<KData,C>*>(BaseCell<C>::getLeft()); }

    const Cell<KData,C>* getRight() const
    { return static_cast<const Cell<KData,C>*>(BaseCell<C>::getRight()); }

    double getWK() const { return getData().getWK(); }
    double calculateSumWKSq() const;
};

template <int C>
class Cell<GData, C> : public BaseCell<C>
{
public:
    Cell(CellData<GData,C>* data, const LeafInfo& info) :
        BaseCell<C>(data, info) {}

    Cell(CellData<GData,C>* data, const ListLeafInfo& listinfo) :
        BaseCell<C>(data, listinfo) {}

    Cell(CellData<GData,C>* data, double size, Cell<GData,C>* l, Cell<GData,C>* r) :
        BaseCell<C>(data, size, l, r) {}

    const CellData<GData,C>& getData() const
    { return static_cast<const CellData<GData,C>&>(BaseCell<C>::getData()); }

    const Cell<GData,C>* getLeft() const
    { return static_cast<const Cell<GData,C>*>(BaseCell<C>::getLeft()); }

    const Cell<GData,C>* getRight() const
    { return static_cast<const Cell<GData,C>*>(BaseCell<C>::getRight()); }

    std::complex<double> getWG() const { return getData().getWG(); }
    std::complex<double> calculateSumWGSq() const;
    double calculateSumAbsWGSq() const;
};

template <int C>
double CalculateSizeSq(
    const Position<C>& cen, const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end);

template <int C, int SM>
size_t SplitData(
    std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end, const Position<C>& meanpos);

template <int D, int C, int SM>
Cell<D,C>* BuildCell(std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
                     double minsizesq, bool brute, size_t start, size_t end,
                     CellData<D,C>* ave=0, double sizesq=0.);

template <int C>
inline std::ostream& operator<<(std::ostream& os, const BaseCell<C>& c)
{ c.Write(os); return os; }

#endif
