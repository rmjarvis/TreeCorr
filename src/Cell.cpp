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

#include <sys/time.h>
#include <fstream>

#include "dbg.h"
#include "Cell.h"
#include "Bounds.h"
#include "Metric.h"

//
// CellData
//

template <int D, int C>
double CalculateSizeSq(
    const Position<C>& cen, const std::vector<CellData<D,C>*>& vdata,
    size_t start, size_t end)
{
    double sizesq = 0.;
    for(size_t i=start;i<end;++i) {
        double s=0.;
        double devsq = MetricHelper<Euclidean>::DistSq(cen,vdata[i]->getPos(),s,s);
        if (devsq > sizesq) sizesq = devsq;
    }
    return sizesq;
}

template <int D, int C>
void BuildCellData(
    const std::vector<CellData<D,C>*>& vdata, size_t start, size_t end,
    Position<C>& pos, float& w, long& n)
{
    Assert(start < end);
    double wp = vdata[start]->getWPos();
    pos = vdata[start]->getPos() * wp;
    w = vdata[start]->getW();
    n = (w != 0);
    double sumwp = wp;
    for(size_t i=start+1; i!=end; ++i) {
        const CellData<D,C>& data = *vdata[i];
        pos += data.getPos() * wp;
        sumwp += wp;
        w += data.getW();
        if (data.getW() != 0.) ++n;
    }
    if (sumwp > 0.) {
        pos /= sumwp;
        // If C == Sphere, the average position is no longer on the surface of the unit sphere.
        // Divide by the new r.  (This is a noop if C == Flat or ThreeD.)
        pos.normalize();
    } else {
        // Make sure we don't have an invalid position, even if all wpos == 0.
        pos = vdata[start]->getPos();
        // But in this case, we should have w == 0 too!
        Assert(w == 0.);
    }
}

template <int C>
CellData<NData,C>::CellData(
    const std::vector<CellData<NData,C>*>& vdata, size_t start, size_t end) :
    _w(0.), _n(0)
{ BuildCellData(vdata,start,end,_pos,_w,_n); }

template <int C>
CellData<KData,C>::CellData(
    const std::vector<CellData<KData,C>*>& vdata, size_t start, size_t end) :
    _wk(0.), _w(0.), _n(0)
{ BuildCellData(vdata,start,end,_pos,_w,_n); }

template <int C>
CellData<GData,C>::CellData(
    const std::vector<CellData<GData,C>*>& vdata, size_t start, size_t end) :
    _wg(0.), _w(0.), _n(0)
{ BuildCellData(vdata,start,end,_pos,_w,_n); }

template <int C>
void CellData<KData,C>::finishAverages(
    const std::vector<CellData<KData,C>*>& vdata, size_t start, size_t end)
{
    // Accumulate in double precision for better accuracy.
    double dwk = 0.;
    for(size_t i=start;i<end;++i) dwk += vdata[i]->getWK();
    _wk = dwk;
}

template <>
void CellData<GData,Flat>::finishAverages(
    const std::vector<CellData<GData,Flat>*>& vdata, size_t start, size_t end)
{
    // Accumulate in double precision for better accuracy.
    std::complex<double> dwg(0.);
    for(size_t i=start;i<end;++i) dwg += vdata[i]->getWG();
    _wg = dwg;
}

template <int C>
std::complex<double> ParallelTransportShift(const std::vector<CellData<GData,C>*>& vdata,
                                            const Position<C>& center, size_t start, size_t end)
{
    // For the average shear, we need to parallel transport each one to the center
    // to account for the different coordinate systems for each measurement.
    //xdbg<<"Finish Averages for Center = "<<center<<std::endl;
    std::complex<double> dwg=0.;
    for(size_t i=start;i<end;++i) {
        //xxdbg<<"Project shear "<<(vdata[i]->wg/vdata[i]->w)<<" at point "<<vdata[i]->getPos()<<std::endl;
        // This is a lot like the ProjectShear function in BinCorr2.cpp
        // The difference is that here, we just rotate the single shear by
        // (Pi-A-B).  See the comments in ProjectShear2 for understanding
        // the initial bit where we calculate A,B.
        double x1 = center.getX();
        double y1 = center.getY();
        double z1 = center.getZ();
        double x2 = vdata[i]->getPos().getX();
        double y2 = vdata[i]->getPos().getY();
        double z2 = vdata[i]->getPos().getZ();
        double temp = x1*x2+y1*y2;
        double cosA = z1*(1.-z2*z2) - z2*temp;
        double sinA = y1*x2 - x1*y2;
        double normAsq = sinA*sinA + cosA*cosA;
        double cosB = z2*(1.-z1*z1) - z1*temp;
        double sinB = sinA;
        double normBsq = sinB*sinB + cosB*cosB;
        //xxdbg<<"A = atan("<<sinA<<"/"<<cosA<<") = "<<atan2(sinA,cosA)*180./M_PI<<std::endl;
        //xxdbg<<"B = atan("<<sinB<<"/"<<cosB<<") = "<<atan2(sinB,cosB)*180./M_PI<<std::endl;
        if (normAsq == 0. || normBsq == 0.) {
            // Then this point is at the center, no need to project.
            dwg += vdata[i]->getWG();
        } else {
            // The angle we need to rotate the shear by is (Pi-A-B)
            // cos(beta) = -cos(A+B)
            // sin(beta) = sin(A+B)
            double cosbeta = -cosA * cosB + sinA * sinB;
            double sinbeta = sinA * cosB + cosA * sinB;
            //xxdbg<<"beta = "<<atan2(sinbeta,cosbeta)*180/M_PI<<std::endl;
            std::complex<double> expibeta(cosbeta,-sinbeta);
            //xxdbg<<"expibeta = "<<expibeta/sqrt(normAsq*normBsq)<<std::endl;
            std::complex<double> exp2ibeta = (expibeta * expibeta) / (normAsq*normBsq);
            //xxdbg<<"exp2ibeta = "<<exp2ibeta<<std::endl;
            dwg += vdata[i]->getWG() * exp2ibeta;
        }
    }
    return dwg;
}

// These two need to do the same thing, so pull it out into the above function.
template <>
void CellData<GData,ThreeD>::finishAverages(
    const std::vector<CellData<GData,ThreeD>*>& vdata, size_t start, size_t end)
{
    _wg = ParallelTransportShift(vdata,_pos,start,end);
}

template <>
void CellData<GData,Sphere>::finishAverages(
    const std::vector<CellData<GData,Sphere>*>& vdata, size_t start, size_t end)
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
    bool operator()(const CellData<D,C>* cd1, const CellData<D,C>* cd2) const
    { return cd1->getPos().get(split) < cd2->getPos().get(split); }
};

template <int D, int C>
struct DataCompareToValue
{
    int split;
    double splitvalue;

    DataCompareToValue(int s, double v) : split(s), splitvalue(v) {}
    bool operator()(const CellData<D,C>* cd) const
    { return cd->getPos().get(split) < splitvalue; }
};

void seed_urandom()
{
    // This implementation shamelessly taken from:
    // http://stackoverflow.com/questions/2572366/how-to-use-dev-random-or-urandom-in-c
    std::ifstream rand("/dev/urandom");
    int myRandomInteger;
    rand.read((char*)&myRandomInteger, sizeof myRandomInteger);
    rand.close();
    srand(myRandomInteger);
}

void seed_time()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    srand(tp.tv_usec);
}

size_t select_random(size_t lo, size_t hi)
{
    static bool first = true;

    if (first) {
        // This is a copy of the way GalSim seeds its random number generator using urandom first,
        // and then if that fails, using the time.
        // Except we just use this to seed the std rand function, not a boost rng.
        // Should be fine for this purpose.
        try {
            seed_urandom();
        } catch(...) {
            seed_time();
        }
        first = false;
    }
    if (lo == hi) {
        return lo;
    } else {
        size_t mid = int((rand() / RAND_MAX) * (hi-lo+1)) + lo;
        if (mid > hi) mid = hi;  // Just in case rand() == RAND_MAX
        return mid;
    }
}

template <int D, int C>
size_t SplitData(
    std::vector<CellData<D,C>*>& vdata, SplitMethod sm,
    size_t start, size_t end, const Position<C>& meanpos)
{
    Assert(end-start > 1);
    size_t mid=0;

    Bounds<C> b;
    for(size_t i=start;i<end;++i) b += vdata[i]->getPos();

    int split = b.getSplit();

    switch (sm) { // three different split methods
      case MIDDLE :
           { // Middle is the average of the min and max value of x or y
               double splitvalue = b.getMiddle(split);
               DataCompareToValue<D,C> comp(split,splitvalue);
               typename std::vector<CellData<D,C>*>::iterator middle =
                   std::partition(vdata.begin()+start,vdata.begin()+end,comp);
               mid = middle - vdata.begin();
           } break;
      case MEDIAN :
           { // Median is the point which divides the group into equal numbers
               DataCompare<D,C> comp(split);
               mid = (start+end)/2;
               typename std::vector<CellData<D,C>*>::iterator middle = vdata.begin()+mid;
               std::nth_element(vdata.begin()+start,middle,vdata.begin()+end,comp);
           } break;
      case MEAN :
           { // Mean is the weighted average value of x or y
               double splitvalue = meanpos.get(split);
               DataCompareToValue<D,C> comp(split,splitvalue);
               typename std::vector<CellData<D,C>*>::iterator middle =
                   std::partition(vdata.begin()+start,vdata.begin()+end,comp);
               mid = middle - vdata.begin();
           } break;
      case RANDOM :
           { // Random is a random point from the first quartile to the third quartile
               DataCompare<D,C> comp(split);

               // The code for RANDOM is same as MEDIAN except for the next line.
               // Note: The lo and hi values are slightly subtle.  We want to make sure if there
               // are only two values, we actually split.  So if start=1, end=3, the only possible
               // result should be mid=2.  Otherwise, we want roughly 1/4 and 3/4 of the span.
               mid = select_random(end-3*(end-start)/4,start+3*(end-start)/4);

               typename std::vector<CellData<D,C>*>::iterator middle = vdata.begin()+mid;
               std::nth_element(vdata.begin()+start,middle,vdata.begin()+end,comp);
           } break;
      default :
           myerror("Invalid SplitMethod");
    }

    if (mid == start || mid == end) {
        xdbg<<"Found mid not in middle.  Probably duplicate entries.\n";
        xdbg<<"start = "<<start<<std::endl;
        xdbg<<"end = "<<end<<std::endl;
        xdbg<<"mid = "<<mid<<std::endl;
        xdbg<<"sm = "<<sm<<std::endl;
        xdbg<<"b = "<<b<<std::endl;
        xdbg<<"split = "<<split<<std::endl;
        for(size_t i=start; i!=end; ++i) {
            xdbg<<"v["<<i<<"] = "<<vdata[i]<<std::endl;
        }
        // With duplicate entries, can get mid == start or mid == end.
        // This should only happen if all entries in this set are equal.
        // So it should be safe to just take the mid = (start + end)/2.
        // But just to be safe, re-call this function with sm = MEDIAN to
        // make sure.
        Assert(sm != MEDIAN);
        return SplitData(vdata,MEDIAN,start,end,meanpos);
    }
    Assert(mid > start);
    Assert(mid < end);
    return mid;
}

template <int D, int C>
Cell<D,C>::Cell(std::vector<CellData<D,C>*>& vdata,
                 double minsizesq, SplitMethod sm, size_t start, size_t end) :
    _size(0.), _sizesq(0.), _left(0), _right(0)
{
    Assert(vdata.size()>0);
    Assert(end <= vdata.size());
    Assert(end > start);

    if (end - start == 1) {
        //xdbg<<"Make leaf cell from "<<*vdata[start]<<std::endl;
        //xdbg<<"size = "<<_size<<std::endl;
        _data = vdata[start];
        vdata[start] = 0; // Make sure calling routine doesn't delete this one!
    } else {
        _data = new CellData<D,C>(vdata,start,end);
        _data->finishAverages(vdata,start,end);
        //xdbg<<"Make cell from "<<start<<".."<<end<<" = "<<*_data<<std::endl;

        _sizesq = CalculateSizeSq(_data->getPos(),vdata,start,end);
        Assert(_sizesq >= 0.);

        if (_sizesq > minsizesq) {
            _size = sqrt(_sizesq);
            //xdbg<<"size = "<<_size<<std::endl;
            size_t mid = SplitData(vdata,sm,start,end,_data->getPos());
            try {
                _left = new Cell<D,C>(vdata,minsizesq,sm,start,mid);
                _right = new Cell<D,C>(vdata,minsizesq,sm,mid,end);
            } catch (std::bad_alloc) {
                myerror("out of memory - cannot create new Cell");
            }
        } else {
            // This shouldn't be necessary for 2-point, but 3-point calculations sometimes
            // have triangles that have two sides that are almost the same, so splits can
            // go arbitrarily small to switch which one is d1,d2 or d2,d3.  This isn't
            // actually an important distinction, so just abort that by calling the size
            // exactly zero.
            _size = _sizesq = 0.;
        }
    }
}

template <int D, int C>
Cell<D,C>::Cell(CellData<D,C>* ave, double sizesq,
                 std::vector<CellData<D,C>*>& vdata,
                 double minsizesq, SplitMethod sm, size_t start, size_t end) :
    _sizesq(sizesq), _data(ave), _left(0), _right(0)
{
    Assert(sizesq >= 0.);
    //xdbg<<"Make cell starting with ave = "<<*ave<<std::endl;
    //xdbg<<"size = "<<_size<<std::endl;
    Assert(vdata.size()>0);
    Assert(end <= vdata.size());
    Assert(end > start);

    if (_sizesq > minsizesq) {
        _size = sqrt(_sizesq);
        size_t mid = SplitData(vdata,sm,start,end,_data->getPos());
        try {
            _left = new Cell<D,C>(vdata,minsizesq,sm,start,mid);
            _right = new Cell<D,C>(vdata,minsizesq,sm,mid,end);
        } catch (std::bad_alloc) {
            myerror("out of memory - cannot create new Cell");
        }
    } else {
        _size = _sizesq = 0.;
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
        Assert(!_right);
        ret.push_back(this);
    }
    return ret;
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

template class CellData<NData,Flat>;
template class CellData<NData,ThreeD>;
template class CellData<NData,Sphere>;
template class CellData<KData,Flat>;
template class CellData<KData,ThreeD>;
template class CellData<KData,Sphere>;
template class CellData<GData,Flat>;
template class CellData<GData,ThreeD>;
template class CellData<GData,Sphere>;

template class Cell<NData,Flat>;
template class Cell<NData,ThreeD>;
template class Cell<NData,Sphere>;
template class Cell<KData,Flat>;
template class Cell<KData,ThreeD>;
template class Cell<KData,Sphere>;
template class Cell<GData,Flat>;
template class Cell<GData,ThreeD>;
template class Cell<GData,Sphere>;

template double CalculateSizeSq(
    const Position<Flat>& cen, const std::vector<CellData<NData,Flat>*>& vdata,
    size_t start, size_t end);
template double CalculateSizeSq(
    const Position<ThreeD>& cen, const std::vector<CellData<NData,ThreeD>*>& vdata,
    size_t start, size_t end);
template double CalculateSizeSq(
    const Position<Sphere>& cen, const std::vector<CellData<NData,Sphere>*>& vdata,
    size_t start, size_t end);
template double CalculateSizeSq(
    const Position<Flat>& cen, const std::vector<CellData<KData,Flat>*>& vdata,
    size_t start, size_t end);
template double CalculateSizeSq(
    const Position<ThreeD>& cen, const std::vector<CellData<KData,ThreeD>*>& vdata,
    size_t start, size_t end);
template double CalculateSizeSq(
    const Position<Sphere>& cen, const std::vector<CellData<KData,Sphere>*>& vdata,
    size_t start, size_t end);
template double CalculateSizeSq(
    const Position<Flat>& cen, const std::vector<CellData<GData,Flat>*>& vdata,
    size_t start, size_t end);
template double CalculateSizeSq(
    const Position<ThreeD>& cen, const std::vector<CellData<GData,ThreeD>*>& vdata,
    size_t start, size_t end);
template double CalculateSizeSq(
    const Position<Sphere>& cen, const std::vector<CellData<GData,Sphere>*>& vdata,
    size_t start, size_t end);
