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

#include <sys/time.h>
#include <fstream>
#include <limits>

#include "dbg.h"
#include "Cell.h"
#include "Bounds.h"


// Helper functions to setup random numbers properly
inline void seed_urandom()
{
    // This implementation shamelessly taken from:
    // http://stackoverflow.com/questions/2572366/how-to-use-dev-random-or-urandom-in-c
    std::ifstream rand("/dev/urandom");
    long myRandomInteger;
    rand.read((char*)&myRandomInteger, sizeof myRandomInteger);
    rand.close();
    srand(myRandomInteger);
}

inline void seed_time()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    srand(tp.tv_usec);
}

// Return a random number between 0 and 1.
double urand(long seed)
{
    static bool first = true;

    if (seed != 0) {
        srand(seed);
        first = false;
    } else if (first) {
        // This is a copy of the way GalSim seeds its random number generator using urandom
        // first, and then if that fails, using the time.
        // Except we just use this to seed the std rand function, not a boost rng.
        // Should be fine for this purpose.
        try {
            seed_urandom();
        } catch(...) {
            seed_time();
        }
        first = false;
    }
    // Get a random number between 0 and 1
    double r = rand();
    r /= RAND_MAX;
    return r;
}


//
// CellData
//

template <int D, int C>
double CalculateSizeSq(
    const Position<C>& cen, const std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    double sizesq = 0.;
    for(size_t i=start;i<end;++i) {
        double devsq = (cen-vdata[i].first->getPos()).normSq();
        if (devsq > sizesq) sizesq = devsq;
    }
    return sizesq;
}

template <int D, int C>
double Cell<D,C>::calculateInertia() const
{
    if (getSize() == 0.) return 0.;
    else if (getN() == 1) return 0.;
    else {
        const Position<C> p1 = getLeft()->getPos();
        double i1 = getLeft()->calculateInertia();
        double w1 = getLeft()->getW();
        const Position<C> p2 = getRight()->getPos();
        double i2 = getRight()->calculateInertia();
        double w2 = getRight()->getW();
        const Position<C> cen = getPos();
        double inertia = i1 + i2 + (p1-cen).normSq() * w1 + (p2-cen).normSq() * w2;
#ifdef DEBUGLOGGING
        std::vector<const Cell<D,C>*> leaves = getAllLeaves();
        double inertia2 = 0.;
        for (size_t k=0; k<leaves.size(); ++k) {
            const Position<C>& p = leaves[k]->getPos();
            double w = leaves[k]->getW();
            inertia2 += w * (p-cen).normSq();
        }
        if (std::abs(inertia2 - inertia) > 1.e-8 * inertia) {
            dbg<<"Cell = "<<*this<<std::endl;
            dbg<<"cen = "<<cen<<std::endl;
            dbg<<"w = "<<getW()<<std::endl;
            dbg<<"nleaves = "<<leaves.size()<<std::endl;
            dbg<<"direct inertia from leaves = "<<inertia2<<std::endl;
            dbg<<"recursive inertia = "<<inertia<<std::endl;
            Assert(false);
        }
#endif
        return inertia;
    }
}


template <int D, int C>
void BuildCellData(
    const std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata, size_t start, size_t end,
    Position<C>& pos, float& w)
{
    Assert(start < end);
    double wp = vdata[start].second.wpos;
    pos = vdata[start].first->getPos();
    pos *= wp;
    w = vdata[start].first->getW();
    double sumwp = wp;
    for(size_t i=start+1; i!=end; ++i) {
        const CellData<D,C>& data = *vdata[i].first;
        wp = vdata[i].second.wpos;
        pos += data.getPos() * wp;
        sumwp += wp;
        w += data.getW();
    }
    if (sumwp != 0.) {
        pos /= sumwp;
        // If C == Sphere, the average position is no longer on the surface of the unit sphere.
        // Divide by the new r.  (This is a noop if C == Flat or ThreeD.)
        pos.normalize();
    } else {
        // Make sure we don't have an invalid position, even if all wpos == 0.
        pos = vdata[start].first->getPos();
        // But in this case, we should have w == 0 too!
        Assert(w == 0.);
    }
}

template <int C>
CellData<NData,C>::CellData(
    const std::vector<std::pair<CellData<NData,C>*,WPosLeafInfo> >& vdata, size_t start, size_t end) :
    _w(0.), _n(end-start)
{ BuildCellData(vdata,start,end,_pos,_w); }

template <int C>
CellData<KData,C>::CellData(
    const std::vector<std::pair<CellData<KData,C>*,WPosLeafInfo> >& vdata, size_t start, size_t end) :
    _wk(0.), _w(0.), _n(end-start)
{ BuildCellData(vdata,start,end,_pos,_w); }

template <int C>
CellData<GData,C>::CellData(
    const std::vector<std::pair<CellData<GData,C>*,WPosLeafInfo> >& vdata, size_t start, size_t end) :
    _wg(0.), _w(0.), _n(end-start)
{ BuildCellData(vdata,start,end,_pos,_w); }

template <int C>
void CellData<KData,C>::finishAverages(
    const std::vector<std::pair<CellData<KData,C>*,WPosLeafInfo> >& vdata, size_t start, size_t end)
{
    // Accumulate in double precision for better accuracy.
    double dwk = 0.;
    for(size_t i=start;i<end;++i) dwk += vdata[i].first->getWK();
    _wk = dwk;
}

template <>
void CellData<GData,Flat>::finishAverages(
    const std::vector<std::pair<CellData<GData,Flat>*,WPosLeafInfo> >& vdata, size_t start, size_t end)
{
    // Accumulate in double precision for better accuracy.
    std::complex<double> dwg(0.);
    for(size_t i=start;i<end;++i) dwg += vdata[i].first->getWG();
    _wg = dwg;
}

template <int C>
std::complex<double> ParallelTransportShift(
    const std::vector<std::pair<CellData<GData,C>*,WPosLeafInfo> >& vdata,
    const Position<C>& center, size_t start, size_t end)
{
    // For the average shear, we need to parallel transport each one to the center
    // to account for the different coordinate systems for each measurement.
    xdbg<<"Finish Averages for Center = "<<center<<std::endl;
    std::complex<double> dwg=0.;
    for(size_t i=start;i<end;++i) {
        xxdbg<<"Project shear "<<(vdata[i].first->getWG()/vdata[i].first->getW())<<
            " at point "<<vdata[i].first->getPos()<<std::endl;
        // This is a lot like the ProjectShear function in BinCorr2.cpp
        // The difference is that here, we just rotate the single shear by
        // (Pi-A-B).  See the comments in ProjectShear2 for understanding
        // the initial bit where we calculate A,B.
        double x1 = center.getX();
        double y1 = center.getY();
        double z1 = center.getZ();
        double x2 = vdata[i].first->getPos().getX();
        double y2 = vdata[i].first->getPos().getY();
        double z2 = vdata[i].first->getPos().getZ();
        double temp = x1*x2+y1*y2;
        double cosA = z1*(1.-z2*z2) - z2*temp;
        double sinA = y1*x2 - x1*y2;
        double normAsq = sinA*sinA + cosA*cosA;
        double cosB = z2*(1.-z1*z1) - z1*temp;
        double sinB = sinA;
        double normBsq = sinB*sinB + cosB*cosB;
        xxdbg<<"A = atan("<<sinA<<"/"<<cosA<<") = "<<atan2(sinA,cosA)*180./M_PI<<std::endl;
        xxdbg<<"B = atan("<<sinB<<"/"<<cosB<<") = "<<atan2(sinB,cosB)*180./M_PI<<std::endl;
        if (normAsq < 1.e-12 && normBsq < 1.e-12) {
            // Then this point is at the center, no need to project.
            dwg += vdata[i].first->getWG();
        } else {
            // The angle we need to rotate the shear by is (Pi-A-B)
            // cos(beta) = -cos(A+B)
            // sin(beta) = sin(A+B)
            double cosbeta = -cosA * cosB + sinA * sinB;
            double sinbeta = sinA * cosB + cosA * sinB;
            xxdbg<<"beta = "<<atan2(sinbeta,cosbeta)*180/M_PI<<std::endl;
            std::complex<double> expibeta(cosbeta,-sinbeta);
            xxdbg<<"expibeta = "<<expibeta/sqrt(normAsq*normBsq)<<std::endl;
            std::complex<double> exp2ibeta = (expibeta * expibeta) / (normAsq*normBsq);
            xxdbg<<"exp2ibeta = "<<exp2ibeta<<std::endl;
            dwg += vdata[i].first->getWG() * exp2ibeta;
        }
    }
    return dwg;
}

// These two need to do the same thing, so pull it out into the above function.
template <>
void CellData<GData,ThreeD>::finishAverages(
    const std::vector<std::pair<CellData<GData,ThreeD>*,WPosLeafInfo> >& vdata, size_t start, size_t end)
{
    _wg = ParallelTransportShift(vdata,_pos,start,end);
}

template <>
void CellData<GData,Sphere>::finishAverages(
    const std::vector<std::pair<CellData<GData,Sphere>*,WPosLeafInfo> >& vdata, size_t start, size_t end)
{
    _wg = ParallelTransportShift(vdata,_pos,start,end);
}


//
// Cell
//

template <int D, int C>
struct DataCompare
{
    int split;
    DataCompare(int s) : split(s) {}
    bool operator()(const std::pair<CellData<D,C>*,WPosLeafInfo> cd1,
                    const std::pair<CellData<D,C>*,WPosLeafInfo> cd2) const
    { return cd1.first->getPos().get(split) < cd2.first->getPos().get(split); }
};

template <int D, int C>
struct DataCompareToValue
{
    int split;
    double splitvalue;

    DataCompareToValue(int s, double v) : split(s), splitvalue(v) {}
    bool operator()(const std::pair<CellData<D,C>*,WPosLeafInfo> cd) const
    { return cd.first->getPos().get(split) < splitvalue; }
};

size_t select_random(size_t lo, size_t hi)
{
    if (lo == hi) {
        return lo;
    } else {
        // Get a random number between 0 and 1
        double r = urand();
        size_t mid = size_t(r * (hi-lo+1)) + lo;
        if (mid > hi) mid = hi;  // Just in case
        return mid;
    }
}

// This is the core calculation of the below SplitData function.
// Specialize it for each SM value
template <int D, int C, int SM>
struct SplitDataCore;

template <int D, int C>
struct SplitDataCore<D, C, MIDDLE>
{
    // Middle is the average of the min and max value of x or y
    static size_t run(std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata,
                      size_t start, size_t end, const Position<C>& meanpos,
                      const Bounds<C>& b, int split)
    {
        double splitvalue = b.getMiddle(split);
        DataCompareToValue<D,C> comp(split,splitvalue);
        typename std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >::iterator middle =
            std::partition(vdata.begin()+start,vdata.begin()+end,comp);
        return middle - vdata.begin();
    }
};

template <int D, int C>
struct SplitDataCore<D, C, MEDIAN>
{
    // Median is the point which divides the group into equal numbers
    static size_t run(std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata,
                      size_t start, size_t end, const Position<C>& meanpos,
                      const Bounds<C>& b, int split)
    {
        DataCompare<D,C> comp(split);
        size_t mid = (start+end)/2;
        typename std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >::iterator middle =
            vdata.begin()+mid;
        std::nth_element(vdata.begin()+start,middle,vdata.begin()+end,comp);
        return mid;
    }
};

template <int D, int C>
struct SplitDataCore<D, C, MEAN>
{
    // Mean is the weighted average value of x or y
    static size_t run(std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata,
                      size_t start, size_t end, const Position<C>& meanpos,
                      const Bounds<C>& b, int split)
    {
        double splitvalue = meanpos.get(split);
        DataCompareToValue<D,C> comp(split,splitvalue);
        typename std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >::iterator middle =
            std::partition(vdata.begin()+start,vdata.begin()+end,comp);
        return middle - vdata.begin();
    }
};

template <int D, int C>
struct SplitDataCore<D, C, RANDOM>
{
    // Random is a random point from the first quartile to the third quartile
    static size_t run(std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata,
                      size_t start, size_t end, const Position<C>& meanpos,
                      const Bounds<C>& b, int split)
    {
        DataCompare<D,C> comp(split);

        // The code for RANDOM is same as MEDIAN except for the next line.
        // Note: The lo and hi values are slightly subtle.  We want to make sure if there
        // are only two values, we actually split.  So if start=1, end=3, the only possible
        // result should be mid=2.  Otherwise, we want roughly 2/5 and 3/5 of the span.
        size_t mid = select_random(end-3*(end-start)/5,start+3*(end-start)/5);

        typename std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >::iterator middle =
            vdata.begin()+mid;
        std::nth_element(vdata.begin()+start,middle,vdata.begin()+end,comp);
        return mid;
    }
};

template <int D, int C, int SM>
size_t SplitData(
    std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end, const Position<C>& meanpos)
{
    Assert(end-start > 1);

    Bounds<C> b;
    for(size_t i=start;i<end;++i) b += vdata[i].first->getPos();
    int split = b.getSplit();

    size_t mid = SplitDataCore<D,C,SM>::run(vdata, start, end, meanpos, b, split);

    if (mid == start || mid == end) {
        xdbg<<"Found mid not in middle.  Probably duplicate entries.\n";
        xdbg<<"start = "<<start<<std::endl;
        xdbg<<"end = "<<end<<std::endl;
        xdbg<<"mid = "<<mid<<std::endl;
        xdbg<<"b = "<<b<<std::endl;
        xdbg<<"split = "<<split<<std::endl;
        for(size_t i=start; i!=end; ++i) {
            xdbg<<"v["<<i<<"] = "<<vdata[i].first<<std::endl;
        }
        // With duplicate entries, can get mid == start or mid == end.
        // This should only happen if all entries in this set are equal.
        // So it should be safe to just take the mid = (start + end)/2.
        // But just to be safe, re-call this function with sm = MEDIAN to
        // make sure.
        Assert(SM != MEDIAN);
        return SplitData<D,C,MEDIAN>(vdata,start,end,meanpos);
    }
    Assert(mid > start);
    Assert(mid < end);
    return mid;
}

template <int D, int C, int SM>
Cell<D,C>* BuildCell(std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata,
                     double minsizesq, bool brute, size_t start, size_t end,
                     CellData<D,C>* data, double sizesq)
{
    xdbg<<"Build "<<minsizesq<<" "<<brute<<" "<<start<<" "<<end<<" "<<data<<" "<<sizesq<<std::endl;
    Assert(sizesq >= 0.);
    Assert(vdata.size()>0);
    Assert(end <= vdata.size());
    Assert(end > start);

    if (end - start == 1) {
        if (!data) {
            data = vdata[start].first;
            vdata[start].first = 0; // Make sure calling routine doesn't delete this one!
        }
        xdbg<<"Make leaf cell from "<<*data<<std::endl;
        LeafInfo info = vdata[start].second; // Only copies as a LeafInfo, so throws away wpos.
        xdbg<<"info.index = "<<info.index<<"  "<<vdata[start].second.index<<std::endl;
        return new Cell<D,C>(data, info);
    }

    // Not a leaf.  Calculate size and data for this Cell.
    if (data) {
        xdbg<<"Make cell starting with ave = "<<*data<<std::endl;
        xdbg<<"sizesq = "<<sizesq<<", brute = "<<brute<<std::endl;
    } else {
        data = new CellData<D,C>(vdata,start,end);
        data->finishAverages(vdata,start,end);
        xdbg<<"Make cell from "<<start<<".."<<end<<" = "<<*data<<std::endl;
        sizesq = CalculateSizeSq(data->getPos(),vdata,start,end);
        Assert(sizesq >= 0.);
    }

    xdbg<<"sizesq = "<<sizesq<<" cf. "<<minsizesq<<", brute="<<brute<<std::endl;
    if (sizesq > minsizesq) {
        // If size is large enough, recurse to leaves.
        double size = brute ? std::numeric_limits<double>::infinity() : sqrt(sizesq);
        if (brute) sizesq = std::numeric_limits<double>::infinity();
        xdbg<<"size,sizesq = "<<size<<","<<sizesq<<std::endl;
        size_t mid = SplitData<D,C,SM>(vdata,start,end,data->getPos());
        Cell<D,C>* l = BuildCell<D,C,SM>(vdata,minsizesq,brute,start,mid);
        xdbg<<"Made left"<<std::endl;
        Cell<D,C>* r = BuildCell<D,C,SM>(vdata,minsizesq,brute,mid,end);
        xdbg<<"Made right"<<std::endl;
        xdbg<<data<<"  "<<size<<"  "<<sizesq<<"  "<<l<<"  "<<r<<std::endl;
        return new Cell<D,C>(data, size, sizesq, l, r);
    } else {
        // Too small, so stop here anyway.
        ListLeafInfo info;
        info.indices = new std::vector<long>(end-start);
        for (size_t i=start; i<end; ++i) {
            xdbg<<"Set indices["<<i-start<<"] = "<<vdata[i].second.index<<std::endl;
            (*info.indices)[i-start] = vdata[i].second.index;
        }
        xdbg<<"Made indices"<<std::endl;
        return new Cell<D,C>(data, info);
    }
}

template <int D, int C>
long Cell<D,C>::countLeaves() const
{
    if (_left) {
        Assert(_right);
        return _left->countLeaves() + _right->countLeaves();
    } else return 1;
}

template <int D, int C>
bool Cell<D,C>::includesIndex(long index) const
{
    if (_left) {
        return _left->includesIndex(index) || _right->includesIndex(index);
    } else if (getN() == 1) {
        return _info.index == index;
    } else {
        const std::vector<long>& indices = *_listinfo.indices;
        return std::find(indices.begin(), indices.end(), index) != indices.end();
    }
}

template <int D, int C>
std::vector<const Cell<D,C>*> Cell<D,C>::getAllLeaves() const
{
    std::vector<const Cell<D,C>*> ret;
    if (_left) {
        std::vector<const Cell<D,C>*> temp = _left->getAllLeaves();
        ret.insert(ret.end(),temp.begin(),temp.end());
        Assert(_right);
        temp = _right->getAllLeaves();
        ret.insert(ret.end(),temp.begin(),temp.end());
    } else {
        ret.push_back(this);
    }
    return ret;
}

template <int D, int C>
std::vector<long> Cell<D,C>::getAllIndices() const
{
    std::vector<long> ret;
    if (_left) {
        std::vector<long> temp = _left->getAllIndices();
        ret.insert(ret.end(),temp.begin(),temp.end());
        Assert(_right);
        temp = _right->getAllIndices();
        ret.insert(ret.end(),temp.begin(),temp.end());
    } else if (getN() == 1) {
        ret.push_back(_info.index);
    } else {
        const std::vector<long>& indices = *_listinfo.indices;
        ret.insert(ret.end(),indices.begin(),indices.end());
    }
    return ret;
}

template <int D, int C>
const Cell<D,C>* Cell<D,C>::getLeafNumber(long i) const
{
    if (_left) {
        if (i < _left->getN())
            return _left->getLeafNumber(i);
        else
            return _right->getLeafNumber(i-_left->getN());
    } else {
        return this;
    }
}

template <int D, int C>
void Cell<D,C>::Write(std::ostream& os) const
{
    os<<getData().getPos()<<"  "<<getSize()<<"  "<<getData().getN();
}

template <int D, int C>
void Cell<D,C>::WriteTree(std::ostream& os, int indent) const
{
    os<<std::string(indent*2,'.')<<*this<<std::endl;
    if (getLeft()) {
        getLeft()->WriteTree(os, indent+1);
        getRight()->WriteTree(os, indent+1);
    }
}

#define Inst(D,C)\
    template class CellData<D,C>; \
    template class Cell<D,C>; \
    template double CalculateSizeSq( \
        const Position<C>& cen, \
        const std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata, \
        size_t start, size_t end); \
    template Cell<D,C>* BuildCell<D,C,MIDDLE>( \
        std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata, \
        double minsizesq, bool brute, size_t start, size_t end, \
        CellData<D,C>* data, double sizesq); \
    template Cell<D,C>* BuildCell<D,C,MEDIAN>( \
        std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata, \
        double minsizesq, bool brute, size_t start, size_t end, \
        CellData<D,C>* data, double sizesq); \
    template Cell<D,C>* BuildCell<D,C,MEAN>( \
        std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata, \
        double minsizesq, bool brute, size_t start, size_t end, \
        CellData<D,C>* data, double sizesq); \
    template Cell<D,C>* BuildCell<D,C,RANDOM>( \
        std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata, \
        double minsizesq, bool brute, size_t start, size_t end, \
        CellData<D,C>* data, double sizesq); \
    template size_t SplitData<D,C,MIDDLE>( \
        std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata, \
        size_t start, size_t end, const Position<C>& meanpos); \
    template size_t SplitData<D,C,MEDIAN>( \
        std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata, \
        size_t start, size_t end, const Position<C>& meanpos); \
    template size_t SplitData<D,C,MEAN>( \
        std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata, \
        size_t start, size_t end, const Position<C>& meanpos); \
    template size_t SplitData<D,C,RANDOM>( \
        std::vector<std::pair<CellData<D,C>*,WPosLeafInfo> >& vdata, \
        size_t start, size_t end, const Position<C>& meanpos); \

Inst(NData,Flat);
Inst(NData,ThreeD);
Inst(NData,Sphere);
Inst(KData,Flat);
Inst(KData,ThreeD);
Inst(KData,Sphere);
Inst(GData,Flat);
Inst(GData,ThreeD);
Inst(GData,Sphere);
