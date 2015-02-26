#ifndef CELL_H
#define CELL_H

enum SplitMethod { MIDDLE, MEDIAN, MEAN };

#define INF 1.e200

#include <iostream>
#include <algorithm>
#include <complex>
#include <vector>
#include "dbg.h"
#include "OldBounds.h"

const double PI = 3.141592653589793;
const double TWOPI = 2.*PI;
const double IOTA = 1.e-10;

struct NCellData 
{
    Position2D pos;

    NCellData() {}
    NCellData(const Position2D& _pos) : pos(_pos) {}
};

struct TCellData : 
    public NCellData 
{
    double t;
    double w;

    TCellData() {}
    TCellData(
        const Position2D& _pos, double _t, double _w) : 
        NCellData(_pos), t(_t), w(_w) 
    {}
};

struct CellData : 
    public NCellData 
{
    std::complex<double> e;
    double w;

    CellData() {}
    CellData(
        const Position2D& _pos, const std::complex<double>& _e, double _w) : 
        NCellData(_pos), e(_e), w(_w) 
    {}
};

template <class CellType>
inline int DoCountLeaves(const CellType* c)
{
    if (c->getLeft()) {
        Assert(c->getRight());
        return c->getLeft()->countLeaves() + c->getRight()->countLeaves();
    } else return 1;
}

template <class CellType>
inline std::vector<const CellType*> DoGetAllLeaves(const CellType* c)
{
    std::vector<const CellType*> ret;
    if (c->getLeft()) {
        std::vector<const CellType*> temp = c->getLeft()->getAllLeaves();
        ret.insert(ret.end(),temp.begin(),temp.end());
        Assert(c->getRight());
        temp = c->getRight()->getAllLeaves();
        ret.insert(ret.end(),temp.begin(),temp.end()); 
    } else {
        Assert(!c->getRight());
        ret.push_back(static_cast<const CellType*>(c));
    }
    return ret;
}

template <class DataType>
struct DataCompare 
{

    bool splitonx;

    DataCompare(bool s) : splitonx(s) {}
    bool operator()(const DataType& cd1, const DataType& cd2) const 
    {
        return (splitonx ?
                cd1.pos.getX() < cd2.pos.getX() :
                cd1.pos.getY() < cd2.pos.getY());
    }
};

template <class DataType>
struct DataCompareToValue 
{

    bool splitonx;
    double splitvalue;

    DataCompareToValue(bool s, double v) : splitonx(s), splitvalue(v) {}
    bool operator()(const DataType& cd) const 
    {
        return (splitonx ? cd.pos.getX() : cd.pos.getY()) < splitvalue;
    }
};

template <class DataType>
inline size_t SplitCell(
    std::vector<DataType>& vdata, SplitMethod sm, 
    size_t start, size_t end, const Position2D& meanpos)
{
    size_t mid=0;

    Bounds2D b;
    for(size_t i=start;i<end;++i) b += vdata[i].pos;

    bool splitonx = ((b.getXMax()-b.getXMin()) > (b.getYMax() - b.getYMin()));

    switch (sm) { // three different split methods
      case MIDDLE :
           { // Middle is the average of the min and max value of x or y
               double splitvalue = 
                   ( splitonx ?
                     (b.getXMax()+b.getXMin())/2. :
                     (b.getYMax()+b.getYMin())/2.);
               DataCompareToValue<DataType> comp(splitonx,splitvalue);
               typename std::vector<DataType>::iterator middle = 
                   std::partition(vdata.begin()+start,vdata.begin()+end,comp);
               mid = middle - vdata.begin();
           } break;
      case MEDIAN :
           { // Median is the point which divides the group into equal numbers
               DataCompare<DataType> comp(splitonx);
               mid = (start+end)/2;
               typename std::vector<DataType>::iterator middle =
                   vdata.begin()+mid;
               std::nth_element(
                   vdata.begin()+start,middle,vdata.begin()+end,comp);
           } break;
      case MEAN :
           { // Mean is the weighted average value of x or y
               double splitvalue = 
                   (splitonx ? meanpos.getX() : meanpos.getY());
               DataCompareToValue<DataType> comp(splitonx,splitvalue);
               typename std::vector<DataType>::iterator middle = 
                   std::partition(vdata.begin()+start,vdata.begin()+end,comp);
               mid = middle - vdata.begin();
           } break;
      default :
           myerror("Invalid SplitMethod");
    }

    if (mid == start || mid == end) {
        // With duplicate entries, can get mid == start or mid == end. 
        // This should only happen if all entries in this set are equal.
        // So it should be safe to just take the mid = (start + end)/2.
        // But just to be safe, re-call this function with sm = MEDIAN to 
        // make sure.
        Assert(sm != MEDIAN);
        return SplitCell(vdata,MEDIAN,start,end,meanpos);
    }
    Assert(mid > start);
    Assert(mid < end);
    return mid;
}

class NCell 
{

    // A Cell that only counts the number of galaxies.

public:

    NCell() : meanpos(),size(0.),sizesq(0.),ngals(0),left(0),right(0) {}
    NCell(std::vector<NCellData>& data, SplitMethod sm=MEAN, size_t start=0, size_t end=0);
    ~NCell() { if (left) delete left; if(right) delete right;}

    const Position2D& getMeanPos() const { return meanpos; }
    double getSize() const { return size; }
    double getSizeSq() const { return sizesq; }
    double getAllSize() const { return size; } 
    // For PairCell's getAllSize is different
    int getN() const { return ngals; }

    const NCell* getLeft() const { return left; }
    const NCell* getRight() const { return right; }

    int countLeaves() const { return DoCountLeaves(this); }
    std::vector<const NCell*> getAllLeaves() const 
    { return DoGetAllLeaves(this); }

    typedef Position2D PosType;

protected:
    Position2D meanpos;
    double size;
    double sizesq;
    int ngals;

    NCell* left;
    NCell* right;
};

inline NCell::NCell(std::vector<NCellData>& vdata, SplitMethod sm, size_t start, size_t end) :
    meanpos(),size(0.),sizesq(0.),ngals(0),left(0),right(0)
{
    if (end == 0) end = vdata.size();
    Assert(vdata.size()>0);
    Assert(end <= vdata.size());
    Assert(end > start);

    ngals = end-start;

    if (end - start == 1) {
        meanpos = vdata[start].pos;
    } else {
        for(size_t i=start;i<end;++i) {
            meanpos += vdata[i].pos;
        }
        meanpos /= double(ngals);

        for(size_t i=start;i<end;++i) {
            double devsq = DistSq(vdata[i].pos,meanpos);
            if (devsq > sizesq) sizesq = devsq;
        }
        size = sqrt(sizesq);

        if (size > 0.) {
            size_t mid = SplitCell(vdata,sm,start,end,meanpos);

            left = new NCell(vdata,sm,start,mid);
            right = new NCell(vdata,sm,mid,end);
            if (!left || !right) 
                myerror("out of memory - cannot create new NCell");
        }
    }
}

class TCell : public NCell 
{
public:

    TCell(std::vector<TCellData>& data, SplitMethod sm=MEAN, size_t start=0, size_t end=0);
    ~TCell() {}

    double getWT() const { return wt; }
    double getWeight() const { return w; }

    const TCell* getLeft() const 
    { return static_cast<const TCell*>(left); }
    const TCell* getRight() const 
    { return static_cast<const TCell*>(right); }

    std::vector<const TCell*> getAllLeaves() const 
    { return DoGetAllLeaves(this); }

protected:
    double wt;
    double w;
};

inline TCell::TCell(std::vector<TCellData>& vdata, SplitMethod sm, size_t start, size_t end) :
    NCell(),wt(0.),w(0.)
{
    if (end == 0) end = vdata.size();
    Assert(vdata.size()>0);
    Assert(end <= vdata.size());
    Assert(end > start);

    ngals = end-start;

    if (end - start == 1) {
        const TCellData& data = vdata[start];
        meanpos = data.pos;
        wt = data.t*data.w;
        w = data.w;
    } else {
        for(size_t i=start;i<end;++i) {
            const TCellData& data = vdata[i];
            meanpos += data.pos * data.w;
            wt += data.t * data.w;
            w += data.w;
        }
        meanpos /= w;

        sizesq = 0.;
        for(size_t i=start;i<end;++i) {
            double devsq = DistSq(vdata[i].pos,meanpos);
            if (devsq > sizesq) sizesq = devsq;
        }
        size = sqrt(sizesq);

        if (size > 0.) {
            size_t mid = SplitCell(vdata,sm,start,end,meanpos);

            left = new TCell(vdata,sm,start,mid);
            right = new TCell(vdata,sm,mid,end);
            if (!left || !right) 
                myerror("out of memory - cannot create new TCell");
        }
    }
}

class Cell : public NCell 
{

    // A Cell contains the accumulated data for a bunch of galaxies.
    // It is characterized primarily by a centroid and a size.
    // The centroid is simply the weighted centroid of all the galaxy positions.
    // The size is the maximum deviation of any one of these galaxies 
    // from the centroid.  That is, all galaxies fall within a radius
    // size from the centroid.
    // The structure also keeps track of some averages and sums about
    // the galaxies which are used in the correlation function calculations.

public:

    Cell(std::vector<CellData>& data, SplitMethod sm=MEAN,
         size_t start=0, size_t end=0);
    ~Cell() {}

    const std::complex<double>& getWE() const { return we; }
    double getENorm() const { return enorm; }
    double getWeight() const { return w; }

    const Cell* getLeft() const 
    { return static_cast<const Cell*>(left); }
    const Cell* getRight() const 
    { return static_cast<const Cell*>(right); }

    std::vector<const Cell*> getAllLeaves() const 
    { return DoGetAllLeaves(this); }

protected:
    std::complex<double> we;
    double enorm;
    double w;
};

inline Cell::Cell(std::vector<CellData>& vdata, SplitMethod sm, size_t start, size_t end) :
    NCell(),we(0.),enorm(0.),w(0.)
{
    if (end == 0) end = vdata.size();
    Assert(vdata.size()>0);
    Assert(end <= vdata.size());
    Assert(end > start);

    ngals = end-start;

    if (end - start == 1) {
        const CellData& data = vdata[start];
        meanpos = data.pos;
        we = data.e*data.w;
        enorm = std::norm(data.e);
        w = data.w;
    } else {
        for(size_t i=start;i<end;++i) {
            const CellData& data = vdata[i];
            meanpos += data.pos * data.w;
            we += data.e * data.w;
            w += data.w;
        }
        enorm = std::norm(we/w);
        meanpos /= w;

        sizesq = 0.;
        for(size_t i=start;i<end;++i) {
            double devsq = DistSq(vdata[i].pos,meanpos);
            if (devsq > sizesq) sizesq = devsq;
        }
        size = sqrt(sizesq);

        if (size > 0.) {
            size_t mid = SplitCell(vdata,sm,start,end,meanpos);

            left = new Cell(vdata,sm,start,mid);
            right = new Cell(vdata,sm,mid,end);
            if (!left || !right) 
                myerror("out of memory - cannot create new Cell");
        }
    }
}

#endif
