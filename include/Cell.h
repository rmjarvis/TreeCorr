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
#ifndef Cell_H
#define Cell_H

enum SplitMethod { MIDDLE, MEDIAN, MEAN };

#include <iostream>
#include <algorithm>
#include <complex>
#include <vector>

#include "Bounds.h"
#include "dbg.h"

const double PI = 3.141592653589793;
const double TWOPI = 2.*PI;
const double IOTA = 1.e-10;

// We use a code (to be used as a template parameter) to indicate which kind of data we
// are using for a particular use. 
// NData means just count the point.
// KData means use a scalar.  Nominally kappa, but works with any scalar (e.g. temperature).
// GData means use a shear. 
enum DataCode { NData=1 , KData=2 , GData=3 };


// This class encapsulates the differences in the different kinds of data being
// stored in a Cell.  It is used both for the input data from the file and also
// for the mean values for a given Cell.  Some extra useful information is sometimes
// also stored.
template <int DC, int M>
class CellData;

template <int M>
class CellData<NData,M>
{
public:
    CellData() {}

    CellData(const Position<M>& pos, double w) : _pos(pos), _w(w), _n(1) {}

    template <int M2>
    CellData(const Position<M2>& pos, double w) : _pos(pos), _w(w), _n(1) {}

    CellData(const std::vector<CellData<NData,M>*>& vdata,
             size_t start, size_t end);

    // This doesn't do anything, but is provided for consistency with the other
    // kinds of CellData.
    void finishAverages(const std::vector<CellData<NData,M>*>&, size_t , size_t ) {}

    const Position<M>& getPos() const { return _pos; }
    double getW() const { return _w; }
    long getN() const { return _n; }

private:

    Position<M> _pos;
    float _w;
    long _n;
};

template <int M>
std::ostream& operator<<(std::ostream& os, const CellData<NData,M>& c)
{ return os << c.getPos() << " " << c.getN(); }

template <int M>
class CellData<KData,M> 
{
public:
    CellData() {}

    CellData(const Position<M>& pos, double k, double w) : 
        _pos(pos), _wk(w*k), _w(w), _n(1)
    {}

    template <int M2>
    CellData(const Position<M2>& pos, double k, double w) : 
        _pos(pos), _wk(w*k), _w(w), _n(1)
    {}

    CellData(const std::vector<CellData<KData,M>*>& vdata, size_t start, size_t end);

    // The above constructor just computes the mean pos, since sometimes that's all we
    // need.  So this function will finish the rest of the construction when desired.
    void finishAverages(const std::vector<CellData<KData,M>*>& vdata, size_t start, size_t end);

    const Position<M>& getPos() const { return _pos; }
    double getWK() const { return _wk; }
    double getW() const { return _w; }
    long getN() const { return _n; }

private:

    Position<M> _pos;
    float _wk;
    float _w;
    long _n;
};

template <int M>
std::ostream& operator<<(std::ostream& os, const CellData<KData,M>& c)
{ return os << c.getPos() << " " << c.getWK() << " " << c.getW() << " " << c.getN(); }

template <int M>
class CellData<GData,M> 
{
public:
    CellData() {}

    CellData(const Position<M>& pos, const std::complex<double>& g, double w) : 
        _pos(pos), _wg(w*g), _w(w), _n(1)
    {}

    template <int M2>
    CellData(const Position<M2>& pos, const std::complex<double>& g, double w) : 
        _pos(pos), _wg(w*g), _w(w), _n(1)
    {}

    CellData(const std::vector<CellData<GData,M>*>& vdata, size_t start, size_t end);

    // The above constructor just computes the mean pos, since sometimes that's all we
    // need.  So this function will finish the rest of the construction when desired.
    void finishAverages(const std::vector<CellData<GData,M>*>& vdata, size_t start, size_t end);

    const Position<M>& getPos() const { return _pos; }
    std::complex<double> getWG() const { return _wg; }
    double getW() const { return _w; }
    long getN() const { return _n; }

private:

    Position<M> _pos;
    std::complex<float> _wg;
    float _w;
    long _n;
};

template <int M>
std::ostream& operator<<(std::ostream& os, const CellData<GData,M>& c)
{ return os << c.getPos() << " " << c.getWG() << " " << c.getW() << " " << c.getN(); }

template <int DC, int M>
class Cell
{
public:

    // A Cell contains the accumulated data for a bunch of galaxies.
    // It is characterized primarily by a centroid and a size.
    // The centroid is simply the weighted centroid of all the galaxy positions.
    // The size is the maximum deviation of any one of these galaxies 
    // from the centroid.  That is, all galaxies fall within a radius
    // size from the centroid.
    // The structure also keeps track of some averages and sums about
    // the galaxies which are used in the correlation function calculations.

    Cell(CellData<DC,M>* data) : _size(0.), _sizesq(0.), _data(data), _left(0), _right(0) {}

    Cell(std::vector<CellData<DC,M>*>& vdata, 
         double minsizesq, SplitMethod sm, size_t start, size_t end);

    Cell(CellData<DC,M>* ave, double sizesq, std::vector<CellData<DC,M>*>& vdata, 
         double minsizesq, SplitMethod sm, size_t start, size_t end);

    ~Cell() 
    {
        Assert(_data);
        delete (_data);
        if (_left) {
            Assert(_right);
            delete _left;
            delete _right;
        }
    }


    const CellData<DC,M>& getData() const { return *_data; }
    double getSize() const { return _size; }
    double getSizeSq() const { return _sizesq; }
    // For PairCells, getAllSize is different from getSize.
    double getAllSize() const { return _size; }

    const Cell<DC,M>* getLeft() const { return _left; }
    const Cell<DC,M>* getRight() const { return _right; }

    long countLeaves() const;
    std::vector<const Cell<DC,M>*> getAllLeaves() const;

    void Write(std::ostream& os) const;
    void WriteTree(std::ostream& os, int indent=0) const;

protected:

    float _size;
    float _sizesq;

    CellData<DC,M>* _data;
    Cell<DC,M>* _left;
    Cell<DC,M>* _right;
};

template <int DC, int M>
double CalculateSizeSq(
    const Position<M>& cen, const std::vector<CellData<DC,M>*>& vdata,
    size_t start, size_t end);

template <int DC, int M>
size_t SplitData(
    std::vector<CellData<DC,M>*>& vdata, SplitMethod sm, 
    size_t start, size_t end, const Position<M>& meanpos);

template <int DC, int M>
inline std::ostream& operator<<(std::ostream& os, const Cell<DC,M>& c)
{ c.Write(os); return os; }

#endif
