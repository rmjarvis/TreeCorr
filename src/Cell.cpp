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

#ifdef _WIN32
#define _USE_MATH_DEFINES  // To get M_PI
#endif

#ifndef _WIN32
#include <sys/time.h>   // Unix-only
#endif

#include <fstream>
#include <limits>

#include "dbg.h"
#include "Cell.h"
#include "Bounds.h"
#include "ProjectHelper.h"

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
#ifndef _WIN32
    // Windows deosn't have this.  sys/time.h is Unix-only.
    // And /dev/urandom I think also doesn't exist, so seed_urandom will always throw.
    // But honestly, this isn't that important in TreeCorr, so just skip it.
    // I.e. use the default random seed using whatever the OS does for us.
    struct timeval tp;
    gettimeofday(&tp,NULL);
    srand(tp.tv_usec);
#endif
}

// Return a random number between 0 and 1.
double urand(long long seed)
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

template <int C>
double CalculateSizeSq(
    const Position<C>& cen, const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    double sizesq = 0.;
    for(size_t i=start;i<end;++i) {
        double devsq = (cen-vdata[i].first->getPos()).normSq();
        if (devsq > sizesq) sizesq = devsq;
    }
    return sizesq;
}

template <int C>
double BaseCell<C>::calculateInertia() const
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
        std::vector<const BaseCell<C>*> leaves = getAllLeaves();
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

template <int C>
double BaseCell<C>::calculateSumWSq() const
{
    if (getSize() == 0.) return SQR(getW());
    else if (getN() == 1) return SQR(getW());
    else return getLeft()->calculateSumWSq() + getRight()->calculateSumWSq();
}

template <int C>
double Cell<KData,C>::calculateSumWKSq() const
{
    if (this->getSize() == 0.) return SQR(getWK());
    else if (this->getN() == 1) return SQR(getWK());
    else return this->getLeft()->calculateSumWKSq() + this->getRight()->calculateSumWKSq();
}

template <int C>
std::complex<double> Cell<GData,C>::calculateSumWGSq() const
{
    if (this->getSize() == 0.) return SQR(getWG());
    else if (this->getN() == 1) return SQR(getWG());
    else return this->getLeft()->calculateSumWGSq() + this->getRight()->calculateSumWGSq();
}

template <int C>
double Cell<GData,C>::calculateSumAbsWGSq() const
{
    if (this->getSize() == 0.) return std::norm(getWG());
    else if (this->getN() == 1) return std::norm(getWG());
    else return this->getLeft()->calculateSumAbsWGSq() + this->getRight()->calculateSumAbsWGSq();
}

template <int C>
void BuildCellData(
    const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata, size_t start, size_t end,
    Position<C>& pos, float& w)
{
    Assert(start < end);
    double wp = vdata[start].second.wpos;
    pos = vdata[start].first->getPos();
    pos *= wp;
    double ww = vdata[start].first->getW();
    double sumwp = wp;
    for(size_t i=start+1; i!=end; ++i) {
        const BaseCellData<C>& data = *vdata[i].first;
        wp = vdata[i].second.wpos;
        pos += data.getPos() * wp;
        sumwp += wp;
        ww += data.getW();
    }
    w = float(ww);
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
BaseCellData<C>::BaseCellData(
    const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata, size_t start, size_t end) :
    _w(0.), _n(end-start)
{ BuildCellData(vdata,start,end,_pos,_w); }

template <int C>
void CellData<KData,C>::finishAverages(
    const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata, size_t start, size_t end)
{
    // Accumulate in double precision for better accuracy.
    double sum_wk = 0.;
    for(size_t i=start;i<end;++i) {
        const CellData<KData,C>* vdata_k = static_cast<const CellData<KData,C>*>(vdata[i].first);
        sum_wk += vdata_k->getWK();
    }
    setWK(sum_wk);
}

template <int D, int C>
std::complex<double> SimpleSum(
    const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    // Accumulate in double precision for better accuracy.
    std::complex<double> sum_wg(0.);
    for(size_t i=start;i<end;++i) {
        const CellData<D,Flat>* vdata_g =
            static_cast<const CellData<D,Flat>*>(vdata[i].first);
        sum_wg += vdata_g->getWG();
    }
    return sum_wg;
}

template <>
void CellData<GData,Flat>::finishAverages(
    const std::vector<std::pair<BaseCellData<Flat>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    setWG(SimpleSum<GData>(vdata, start, end));
}

template <>
void CellData<ZData,Flat>::finishAverages(
    const std::vector<std::pair<BaseCellData<Flat>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    setWZ(SimpleSum<ZData>(vdata, start, end));
}

template <>
void CellData<VData,Flat>::finishAverages(
    const std::vector<std::pair<BaseCellData<Flat>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    setWV(SimpleSum<VData>(vdata, start, end));
}

template <>
void CellData<TData,Flat>::finishAverages(
    const std::vector<std::pair<BaseCellData<Flat>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    setWT(SimpleSum<TData>(vdata, start, end));
}

template <>
void CellData<QData,Flat>::finishAverages(
    const std::vector<std::pair<BaseCellData<Flat>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    setWQ(SimpleSum<QData>(vdata, start, end));
}

// C here is either ThreeD or Sphere
template <int D, int C>
std::complex<double> ParallelTransportSum(
    const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
    const Position<C>& center, size_t start, size_t end)
{
    // For the average shear, we need to parallel transport each one to the center
    // to account for the different coordinate systems for each measurement.
    xdbg<<"Finish Averages for Center = "<<center<<std::endl;
    std::complex<double> sum_wg=0.;
    Position<Sphere> cen(center);
    for(size_t i=start;i<end;++i) {
        const CellData<GData,C>* vdata_g = static_cast<const CellData<GData,C>*>(vdata[i].first);
        xxdbg<<"Project shear "<<(vdata_g->getWG()/vdata_g->getW())<<
            " at point "<<vdata_g->getPos()<<std::endl;
        // This is a lot like the ProjectHelper<Sphere>::calculate_direction function.
        // The difference is that here, we just rotate the single shear by
        // (Pi-A-B).  See the comments in ProjectShear2 for understanding
        // the initial bit where we calculate A,B.  (We don't call that function directly,
        // because there is a slight efficiency gain that sinA = sinB, which we use here.)
        Position<Sphere> pi(vdata_g->getPos());
        double z1 = center.getZ();
        double z2 = pi.getZ();
        double dsq = (cen - pi).normSq();
        if (dsq < 1.e-12) {
            // i.e. d < 1.e-6 radians = 0.2 arcsec
            // Then this point is at the center, no need to project.
            sum_wg += vdata_g->getWG();
        } else {
            double cosA = (z1 - z2) + 0.5 * z2 * dsq;
            double sinA = cen.getY()*pi.getX() - cen.getX()*pi.getY();
            double cosB = (z2 - z1) + 0.5 * z1 * dsq;
            double sinB = sinA;

            // The angle we need to rotate the shear by is (Pi-A-B)
            // cos(beta) = -cos(A+B)
            // sin(beta) = sin(A+B)
            double cosbeta = -cosA * cosB + sinA * sinB;
            double sinbeta = sinA * cosB + cosA * sinB;
            std::complex<double> expibeta(cosbeta, sinbeta);
            std::complex<double> expmsibeta = _expmsialpha<D>(expibeta);
            sum_wg += vdata_g->getWG() * expmsibeta;
        }
    }
    return sum_wg;
}

// These two need to do the same thing, so pull it out into the above function.
template <>
void CellData<GData,ThreeD>::finishAverages(
    const std::vector<std::pair<BaseCellData<ThreeD>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    setWG(ParallelTransportSum<GData>(vdata,_pos,start,end));
}

template <>
void CellData<GData,Sphere>::finishAverages(
    const std::vector<std::pair<BaseCellData<Sphere>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    setWG(ParallelTransportSum<GData>(vdata,_pos,start,end));
}

template <>
void CellData<ZData,ThreeD>::finishAverages(
    const std::vector<std::pair<BaseCellData<ThreeD>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    setWZ(ParallelTransportSum<ZData>(vdata,_pos,start,end));
}

template <>
void CellData<ZData,Sphere>::finishAverages(
    const std::vector<std::pair<BaseCellData<Sphere>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    setWZ(ParallelTransportSum<ZData>(vdata,_pos,start,end));
}

template <>
void CellData<VData,ThreeD>::finishAverages(
    const std::vector<std::pair<BaseCellData<ThreeD>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    setWV(ParallelTransportSum<VData>(vdata,_pos,start,end));
}

template <>
void CellData<VData,Sphere>::finishAverages(
    const std::vector<std::pair<BaseCellData<Sphere>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    setWV(ParallelTransportSum<VData>(vdata,_pos,start,end));
}

template <>
void CellData<TData,ThreeD>::finishAverages(
    const std::vector<std::pair<BaseCellData<ThreeD>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    setWT(ParallelTransportSum<TData>(vdata,_pos,start,end));
}

template <>
void CellData<TData,Sphere>::finishAverages(
    const std::vector<std::pair<BaseCellData<Sphere>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    setWT(ParallelTransportSum<TData>(vdata,_pos,start,end));
}

template <>
void CellData<QData,ThreeD>::finishAverages(
    const std::vector<std::pair<BaseCellData<ThreeD>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    setWQ(ParallelTransportSum<QData>(vdata,_pos,start,end));
}

template <>
void CellData<QData,Sphere>::finishAverages(
    const std::vector<std::pair<BaseCellData<Sphere>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end)
{
    setWQ(ParallelTransportSum<QData>(vdata,_pos,start,end));
}


//
// Cell
//

template <int C>
struct DataCompare
{
    int split;
    DataCompare(int s) : split(s) {}
    bool operator()(const std::pair<BaseCellData<C>*,WPosLeafInfo> cd1,
                    const std::pair<BaseCellData<C>*,WPosLeafInfo> cd2) const
    { return cd1.first->getPos().get(split) < cd2.first->getPos().get(split); }
};

template <int C>
struct DataCompareToValue
{
    int split;
    double splitvalue;

    DataCompareToValue(int s, double v) : split(s), splitvalue(v) {}
    bool operator()(const std::pair<BaseCellData<C>*,WPosLeafInfo> cd) const
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
template <int C, int SM>
struct SplitDataCore;

template <int C>
struct SplitDataCore<C,Middle>
{
    // Middle is the average of the min and max value of x or y
    static size_t run(std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
                      size_t start, size_t end, const Position<C>& meanpos,
                      const Bounds<C>& b, int split)
    {
        double splitvalue = b.getMiddle(split);
        DataCompareToValue<C> comp(split,splitvalue);
        typename std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >::iterator middle =
            std::partition(vdata.begin()+start,vdata.begin()+end,comp);
        return middle - vdata.begin();
    }
};

template <int C>
struct SplitDataCore<C,Median>
{
    // Median is the point which divides the group into equal numbers
    static size_t run(std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
                      size_t start, size_t end, const Position<C>& meanpos,
                      const Bounds<C>& b, int split)
    {
        DataCompare<C> comp(split);
        size_t mid = (start+end)/2;
        typename std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >::iterator middle =
            vdata.begin()+mid;
        std::nth_element(vdata.begin()+start,middle,vdata.begin()+end,comp);
        return mid;
    }
};

template <int C>
struct SplitDataCore<C,Mean>
{
    // Mean is the weighted average value of x or y
    static size_t run(std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
                      size_t start, size_t end, const Position<C>& meanpos,
                      const Bounds<C>& b, int split)
    {
        double splitvalue = meanpos.get(split);
        DataCompareToValue<C> comp(split,splitvalue);
        typename std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >::iterator middle =
            std::partition(vdata.begin()+start,vdata.begin()+end,comp);
        return middle - vdata.begin();
    }
};

template <int C>
struct SplitDataCore<C,Random>
{
    // Random is a random point from the first quartile to the third quartile
    static size_t run(std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
                      size_t start, size_t end, const Position<C>& meanpos,
                      const Bounds<C>& b, int split)
    {
        DataCompare<C> comp(split);

        // The code for Random is same as Median except for the next line.
        // Note: The lo and hi values are slightly subtle.  We want to make sure if there
        // are only two values, we actually split.  So if start=1, end=3, the only possible
        // result should be mid=2.  Otherwise, we want roughly 2/5 and 3/5 of the span.
        size_t mid = select_random(end-3*(end-start)/5,start+3*(end-start)/5);

        typename std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >::iterator middle =
            vdata.begin()+mid;
        std::nth_element(vdata.begin()+start,middle,vdata.begin()+end,comp);
        return mid;
    }
};

template <int C, int SM>
size_t SplitData(
    std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
    size_t start, size_t end, const Position<C>& meanpos)
{
    Assert(end-start > 1);

    Bounds<C> b;
    for(size_t i=start;i<end;++i) b += vdata[i].first->getPos();
    int split = b.getSplit();

    size_t mid = SplitDataCore<C,SM>::run(vdata, start, end, meanpos, b, split);

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
        // But just to be safe, re-call this function with sm = Median to
        // make sure.
        Assert(SM != Median);
        return SplitData<C,Median>(vdata,start,end,meanpos);
    }
    Assert(mid > start);
    Assert(mid < end);
    return mid;
}

template <int D, int C, int SM>
Cell<D,C>* BuildCell(std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata,
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
            data = static_cast<CellData<D,C>*>(vdata[start].first);
            vdata[start].first = 0; // Make sure calling routine doesn't delete this one!
        }
        xdbg<<"Make leaf cell from "<<*data<<std::endl;
        Assert(data->getN() == 1);
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
        size_t mid = SplitData<C,SM>(vdata,start,end,data->getPos());
        Cell<D,C>* l = BuildCell<D,C,SM>(vdata,minsizesq,brute,start,mid);
        xdbg<<"Made left"<<std::endl;
        Cell<D,C>* r = BuildCell<D,C,SM>(vdata,minsizesq,brute,mid,end);
        xdbg<<"Made right"<<std::endl;
        xdbg<<data<<"  "<<size<<"  "<<sizesq<<"  "<<l<<"  "<<r<<std::endl;
        return new Cell<D,C>(data, size, l, r);
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

template <int C>
long BaseCell<C>::countLeaves() const
{
    if (_left) {
        Assert(_right);
        return _left->countLeaves() + _right->countLeaves();
    } else return 1;
}

template <int C>
bool BaseCell<C>::includesIndex(long index) const
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

template <int C>
std::vector<const BaseCell<C>*> BaseCell<C>::getAllLeaves() const
{
    std::vector<const BaseCell<C>*> ret;
    if (_left) {
        std::vector<const BaseCell<C>*> temp = _left->getAllLeaves();
        ret.insert(ret.end(),temp.begin(),temp.end());
        Assert(_right);
        temp = _right->getAllLeaves();
        ret.insert(ret.end(),temp.begin(),temp.end());
    } else {
        ret.push_back(this);
    }
    return ret;
}

template <int C>
std::vector<long> BaseCell<C>::getAllIndices() const
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

template <int C>
const BaseCell<C>* BaseCell<C>::getLeafNumber(long i) const
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

template <int C>
void BaseCell<C>::Write(std::ostream& os) const
{
    os<<getData().getPos()<<"  "<<getSize()<<"  "<<getData().getN();
}

template <int C>
void BaseCell<C>::WriteTree(std::ostream& os, int indent) const
{
    os<<std::string(indent*2,'.')<<*this<<std::endl;
    if (getLeft()) {
        getLeft()->WriteTree(os, indent+1);
        getRight()->WriteTree(os, indent+1);
    }
}

#define InstD(D,C)\
    template class CellData<D,C>; \
    template class Cell<D,C>; \
    template Cell<D,C>* BuildCell<D,C,Middle>( \
        std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata, \
        double minsizesq, bool brute, size_t start, size_t end, \
        CellData<D,C>* data, double sizesq); \
    template Cell<D,C>* BuildCell<D,C,Median>( \
        std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata, \
        double minsizesq, bool brute, size_t start, size_t end, \
        CellData<D,C>* data, double sizesq); \
    template Cell<D,C>* BuildCell<D,C,Mean>( \
        std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata, \
        double minsizesq, bool brute, size_t start, size_t end, \
        CellData<D,C>* data, double sizesq); \
    template Cell<D,C>* BuildCell<D,C,Random>( \
        std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata, \
        double minsizesq, bool brute, size_t start, size_t end, \
        CellData<D,C>* data, double sizesq); \

#define Inst(C)\
    template class BaseCellData<C>; \
    template class BaseCell<C>; \
    template double CalculateSizeSq( \
        const Position<C>& cen, \
        const std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata, \
        size_t start, size_t end); \
    template size_t SplitData<C,Middle>( \
        std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata, \
        size_t start, size_t end, const Position<C>& meanpos); \
    template size_t SplitData<C,Median>( \
        std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata, \
        size_t start, size_t end, const Position<C>& meanpos); \
    template size_t SplitData<C,Mean>( \
        std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata, \
        size_t start, size_t end, const Position<C>& meanpos); \
    template size_t SplitData<C,Random>( \
        std::vector<std::pair<BaseCellData<C>*,WPosLeafInfo> >& vdata, \
        size_t start, size_t end, const Position<C>& meanpos); \
    InstD(NData,C); \
    InstD(KData,C); \
    InstD(GData,C); \
    InstD(ZData,C); \
    InstD(VData,C); \
    InstD(TData,C); \
    InstD(QData,C); \

Inst(Flat);
Inst(ThreeD);
Inst(Sphere);
