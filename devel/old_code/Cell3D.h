#ifndef CELL3D_H
#define CELL3D_H

#include "OldCell.h"

struct NCell3DData 
{
    Position3D pos;

    NCell3DData() {}
    NCell3DData(const Position3D& _pos) : pos(_pos) {}
};

struct TCell3DData : 
    public NCell3DData 
{
    double t;
    double w;

    TCell3DData() {}
    TCell3DData(const Position3D& _pos, double _t, double _w) : 
        NCell3DData(_pos), t(_t), w(_w) {}
};

template <class DataType>
struct DataCompare3D 
{

    int splitonxyz; // x=0, y=1, z=2

    DataCompare3D(int s) : splitonxyz(s) {}
    bool operator()(const DataType& cd1, const DataType& cd2) const 
    {
        return (splitonxyz==0 ?  cd1.pos.getX() < cd2.pos.getX() : 
                splitonxyz==1 ? cd1.pos.getY() < cd2.pos.getY() : 
                cd1.pos.getZ() < cd2.pos.getZ());
    }
};

template <class DataType>
struct DataCompareToValue3D 
{

    int splitonxyz;
    double splitvalue;

    DataCompareToValue3D(int s, double svalue) : 
        splitonxyz(s), splitvalue(svalue) {}
    bool operator()(const DataType& cd) const 
    {
        return (splitonxyz==0 ? cd.pos.getX() : 
                splitonxyz==1 ? cd.pos.getY() : cd.pos.getZ()) < splitvalue;
    }
};

template <class DataType>
inline size_t SplitCell3D(
    std::vector<DataType>& vdata, SplitMethod sm, 
    size_t start, size_t end, const Position3D& meanpos)
{
    //xdbg<<"vdata.size = "<<vdata.size()<<", start = "<<start<<", end = "<<end<<endl;
    //for(size_t i=start;i<end;++i) { xdbg<<vdata[i].pos<<endl; }
    //xdbg<<"meanpos = "<<meanpos<<endl;
    size_t mid;

    Bounds3D b;
    for(size_t i=start;i<end;++i) b += vdata[i].pos;

    int splitonxyz = 
        (b.getXMax()-b.getXMin() > b.getYMax()-b.getYMin()) ?
        (b.getZMax()-b.getZMin() > b.getXMax()-b.getXMin() ? 2 : 0) :
        (b.getZMax()-b.getZMin() > b.getYMax()-b.getYMin() ? 2 : 1);
    //xdbg<<"b = "<<b<<endl;
    //xdbg<<"xrange = "<<b.getXMax()-b.getXMin()<<endl;
    //xdbg<<"yrange = "<<b.getYMax()-b.getYMin()<<endl;
    //xdbg<<"zrange = "<<b.getZMax()-b.getZMin()<<endl;
    //xdbg<<"split = "<<splitonxyz<<endl;

    switch (sm) { // three different split methods
      case MIDDLE :
           { // Middle is the average of the min and max value of x or y
               double splitvalue = 
                   splitonxyz==0 ? (b.getXMax()+b.getXMin())/2. :
                   splitonxyz==1 ? (b.getYMax()+b.getYMin())/2. : 
                   (b.getZMax()+b.getZMin())/2.;
               DataCompareToValue3D<DataType> comp(splitonxyz,splitvalue);
               typename std::vector<DataType>::iterator middle = 
                   std::partition(vdata.begin()+start,vdata.begin()+end,comp);
               mid = middle - vdata.begin();
           } break;
      case MEDIAN :
           { // Median is the point which divides the group into equal numbers
               DataCompare3D<DataType> comp(splitonxyz);
               mid = (start+end)/2;
               typename std::vector<DataType>::iterator middle =
                   vdata.begin()+mid;
               std::nth_element(
                   vdata.begin()+start,middle,vdata.begin()+end,comp);
           } break;
      case MEAN :
           { // Mean is the weighted average value of x or y
               double splitvalue = 
                   (splitonxyz==0 ? meanpos.getX() : 
                    splitonxyz==1 ? meanpos.getY() : meanpos.getZ());
               //xdbg<<"splitvalue = "<<splitvalue<<endl;
               DataCompareToValue3D<DataType> comp(splitonxyz,splitvalue);
               typename std::vector<DataType>::iterator middle = 
                   std::partition(vdata.begin()+start,vdata.begin()+end,comp);
               mid = middle - vdata.begin();
           } break;
      default :
           myerror("Invalid SplitMethod");
    }
    //dbg<<"mid = "<<mid<<endl;
    //dbg<<"\n\nLeft half:\n";
    //for(size_t i=start;i<mid;++i) { xdbg<<vdata[i].pos<<endl; }
    //dbg<<"\n\nRight half:\n";
    //for(size_t i=mid;i<end;++i) { xdbg<<vdata[i].pos<<endl; }

    Assert(mid > start);
    Assert(mid < end);
    return mid;
}

class NCell3D 
{

    // A Cell that only counts the number of galaxies.

public:

    NCell3D() : meanpos(),size(0.),ngals(0),left(0),right(0) {}
    NCell3D(std::vector<NCell3DData>& data, SplitMethod sm=MEAN,
            size_t start=0, size_t end=0);
    ~NCell3D() { if (left) delete left; if(right) delete right;}

    const Position3D& getMeanPos() const { return meanpos; }
    double getSize() const { return size; }
    double getSizeSq() const { return sizesq; }
    double getAllSize() const { return size; } 
    // For PairCell's getAllSize is different
    int getN() const { return ngals; }

    const NCell3D* getLeft() const { return left; }
    const NCell3D* getRight() const { return right; }

    int countLeaves() const { return DoCountLeaves(this); }
    std::vector<const NCell3D*> getAllLeaves() const 
    { return DoGetAllLeaves(this); }

    typedef Position3D PosType;

protected:
    Position3D meanpos;
    double size;
    double sizesq;
    int ngals;

    NCell3D* left;
    NCell3D* right;
};

inline NCell3D::NCell3D(
    std::vector<NCell3DData>& vdata, SplitMethod sm,
    size_t start, size_t end) :
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
            //xdbg<<"meanpos = "<<meanpos<<"  size = "<<size<<std::endl;
            size_t mid = SplitCell3D(vdata,sm,start,end,meanpos);

            left = new NCell3D(vdata,sm,start,mid);
            right = new NCell3D(vdata,sm,mid,end);
            if (!left || !right) 
                myerror("out of memory - cannot create new NCell3D");
        }
    }
}

class TCell3D : 
    public NCell3D 
{

public:

    TCell3D(std::vector<TCell3DData>& data, SplitMethod sm=MEAN,
            size_t start=0, size_t end=0);
    ~TCell3D() {}

    double getWT() const { return wt; }
    double getWeight() const { return w; }

    const TCell3D* getLeft() const 
    { return static_cast<const TCell3D*>(left); }
    const TCell3D* getRight() const 
    { return static_cast<const TCell3D*>(right); }

    std::vector<const TCell3D*> getAllLeaves() const 
    { return DoGetAllLeaves(this); }

protected:
    double wt;
    double w;
};

inline TCell3D::TCell3D(
    std::vector<TCell3DData>& vdata, SplitMethod sm,
    size_t start, size_t end) : NCell3D(),wt(0.),w(0.)
{
    if (end == 0) end = vdata.size();
    Assert(vdata.size()>0);
    Assert(end <= vdata.size());
    Assert(end > start);

    ngals = end-start;

    if (end - start == 1) {
        const TCell3DData& data = vdata[start];
        meanpos = data.pos;
        wt = data.t*data.w;
        w = data.w;
        Assert(w>0.);
    } else {
        for(size_t i=start;i<end;++i) {
            const TCell3DData& data = vdata[i];
            meanpos += data.pos * data.w;
            wt += data.t * data.w;
            w += data.w;
        }
        Assert(w>0.);
        meanpos /= w;

        sizesq = 0.;
        for(size_t i=start;i<end;++i) {
            double devsq = DistSq(vdata[i].pos,meanpos);
            if (devsq > sizesq) sizesq = devsq;
        }
        size = sqrt(sizesq);

        if (size > 0.) {
            size_t mid = SplitCell3D(vdata,sm,start,end,meanpos);

            left = new TCell3D(vdata,sm,start,mid);
            right = new TCell3D(vdata,sm,mid,end);
            if (!left || !right) 
                myerror("out of memory - cannot create new TCell3D");
        }
    }
}

#endif
