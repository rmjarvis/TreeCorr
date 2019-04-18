
#include "dbg.h"
#include "Cell.h"

//
// CellData
//

template <int DC, int M>
double CalculateSizeSq(
    const Position<M>& cen, const std::vector<CellData<DC,M>*>& vdata,
    size_t start, size_t end)
{
    double sizesq = 0.;
    for(size_t i=start;i<end;++i) {
        double devsq = DistSq(cen,vdata[i]->getPos());
        if (devsq > sizesq) sizesq = devsq;
    }
    return sizesq;
}

template <int M>
CellData<NData,M>::CellData(
    const std::vector<CellData<NData,M>*>& vdata,
    size_t start, size_t end) :
    _n(end-start)
{
    for(size_t i=start; i!=end; ++i) _pos += vdata[i]->getPos();
    Assert(_n != 0);
    _pos /= _n;
    // If M == Sphere, the average position is no longer on the surface of the unit sphere.
    // Divide by the new r.
    if (M == Sphere) {
        Assert(_pos.norm() != 0.);
        _pos /= _pos.norm();
    }
}

template <int M>
CellData<KData,M>::CellData(
    const std::vector<CellData<KData,M>*>& vdata,
    size_t start, size_t end) :
    _pos(), _wk(0.), _w(0.), _n(end-start)
{
    for(size_t i=start;i<end;++i) {
        const CellData<KData,M>& data = *vdata[i];
        _pos += data.getPos() * data.getW();
        _wk += data.getWK();
        _w += data.getW();
    }
    Assert(_w != 0.);
    _pos /= _w;
    if (M == Sphere) {
        Assert(_pos.norm() != 0.);
        _pos /= _pos.norm();
    }
}

template <int M>
void CellData<KData,M>::finishAverages(
    const std::vector<CellData<KData,M>*>& vdata,
    size_t start, size_t end) 
{
    double dwk=0.;
    for(size_t i=start;i<end;++i) dwk += vdata[i]->getWK();
    _wk = dwk;
}

template <int M>
CellData<GData,M>::CellData(
    const std::vector<CellData<GData,M>*>& vdata,
    size_t start, size_t end) :
    _pos(), _wg(0.), _w(0.), _n(end-start)
{
    for(size_t i=start;i<end;++i) {
        const CellData<GData,M>& data = *vdata[i];
        _pos += data.getPos() * data.getW();
        _w += data.getW();
    }
    Assert(_w != 0.);
    _pos /= _w;
    if (M == Sphere) {
        Assert(_pos.norm() != 0.);
        _pos /= _pos.norm();
    }
}

template <>
void CellData<GData,Flat>::finishAverages(
    const std::vector<CellData<GData,Flat>*>& vdata,
    size_t start, size_t end) 
{
    std::complex<double> dwg=0.;
    for(size_t i=start;i<end;++i) dwg += vdata[i]->getWG();
    _wg = dwg;
}

template <>
void CellData<GData,Sphere>::finishAverages(
    const std::vector<CellData<GData,Sphere>*>& vdata,
    size_t start, size_t end) 
{
    // For the average shear, we need to parallel transport each one to the center
    // to account for the different coordinate systems for each measurement.
    //xdbg<<"Finish Averages for Center = "<<pos<<std::endl;
    std::complex<double> dwg=0.;
    for(size_t i=start;i<end;++i) {
        //xxdbg<<"Project shear "<<(vdata[i]->wg/vdata[i]->w)<<" at point "<<vdata[i]->getPos()<<std::endl;
        // This is a lot like the ProjectSphere function in BinData2.h
        // The difference is that here, we just rotate the single shear by
        // (Pi-A-B).  See the comments in ProjectSphere for understanding
        // the initial bit where we calculate A,B.
        double x1 = _pos.getX(); 
        double y1 = _pos.getY(); 
        double z1 = _pos.getZ(); 
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
    _wg = dwg;
}

//
// Cell
//

template <int DC, int M>
struct DataCompare 
{
    int split;
    DataCompare(int s) : split(s) {}
    bool operator()(const CellData<DC,M>* cd1, const CellData<DC,M>* cd2) const 
    { return cd1->getPos().get(split) < cd2->getPos().get(split); }
};

template <int DC, int M>
struct DataCompareToValue 
{
    int split;
    double splitvalue;

    DataCompareToValue(int s, double v) : split(s), splitvalue(v) {}
    bool operator()(const CellData<DC,M>* cd) const 
    { return cd->getPos().get(split) < splitvalue; }
};

template <int DC, int M>
size_t SplitData(
    std::vector<CellData<DC,M>*>& vdata, SplitMethod sm, 
    size_t start, size_t end, const Position<M>& meanpos)
{
    Assert(end-start > 1);
    size_t mid=0;

    Bounds<M> b;
    for(size_t i=start;i<end;++i) b += vdata[i]->getPos();

    int split = b.getSplit();

    switch (sm) { // three different split methods
      case MIDDLE :
           { // Middle is the average of the min and max value of x or y
               double splitvalue = b.getMiddle(split);
               DataCompareToValue<DC,M> comp(split,splitvalue);
               typename std::vector<CellData<DC,M>*>::iterator middle =
                   std::partition(vdata.begin()+start,vdata.begin()+end,comp);
               mid = middle - vdata.begin();
           } break;
      case MEDIAN :
           { // Median is the point which divides the group into equal numbers
               DataCompare<DC,M> comp(split);
               mid = (start+end)/2;
               typename std::vector<CellData<DC,M>*>::iterator middle =
                   vdata.begin()+mid;
               std::nth_element(
                   vdata.begin()+start,middle,vdata.begin()+end,comp);
           } break;
      case MEAN :
           { // Mean is the weighted average value of x or y
               double splitvalue = meanpos.get(split);
               DataCompareToValue<DC,M> comp(split,splitvalue);
               typename std::vector<CellData<DC,M>*>::iterator middle =
                   std::partition(vdata.begin()+start,vdata.begin()+end,comp);
               mid = middle - vdata.begin();
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

template <int DC, int M>
Cell<DC,M>::Cell(std::vector<CellData<DC,M>*>& vdata, 
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
        _data = new CellData<DC,M>(vdata,start,end);
        _data->finishAverages(vdata,start,end);
        //xdbg<<"Make cell from "<<start<<".."<<end<<" = "<<*_data<<std::endl;

        _sizesq = CalculateSizeSq(_data->getPos(),vdata,start,end);
        Assert(_sizesq >= 0.);
        _size = sqrt(_sizesq);
        //xdbg<<"size = "<<_size<<std::endl;

        if (_sizesq > minsizesq) {
            size_t mid = SplitData(vdata,sm,start,end,_data->getPos());
            try {
                _left = new Cell<DC,M>(vdata,minsizesq,sm,start,mid);
                _right = new Cell<DC,M>(vdata,minsizesq,sm,mid,end);
            } catch (std::bad_alloc) {
                myerror("out of memory - cannot create new Cell");
            }
        }
    }
}

template <int DC, int M>
Cell<DC,M>::Cell(CellData<DC,M>* ave, double sizesq,
                 std::vector<CellData<DC,M>*>& vdata, 
                 double minsizesq, SplitMethod sm, size_t start, size_t end) :
    _size(sqrt(sizesq)), _sizesq(sizesq), _data(ave), _left(0), _right(0)
{
    Assert(sizesq >= 0.);
    xdbg<<"Make cell starting with ave = "<<*ave<<std::endl;
    xdbg<<"size = "<<_size<<std::endl;
    Assert(vdata.size()>0);
    Assert(end <= vdata.size());
    Assert(end > start);

    if (_sizesq > minsizesq) {
        size_t mid = SplitData(vdata,sm,start,end,_data->getPos());
        try {
            _left = new Cell<DC,M>(vdata,minsizesq,sm,start,mid);
            _right = new Cell<DC,M>(vdata,minsizesq,sm,mid,end);
        } catch (std::bad_alloc) {
            myerror("out of memory - cannot create new Cell");
        }
    }
}

template <int DC, int M>
long Cell<DC,M>::countLeaves() const 
{
    if (_left) {
        Assert(_right);
        return _left->countLeaves() + _right->countLeaves();
    } else return 1;
}


template <int DC, int M>
std::vector<const Cell<DC,M>*> Cell<DC,M>::getAllLeaves() const 
{
    std::vector<const Cell<DC,M>*> ret;
    if (_left) {
        std::vector<const Cell<DC,M>*> temp = _left->getAllLeaves();
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

template class CellData<NData,Flat>;
template class CellData<NData,Sphere>;
template class CellData<KData,Flat>;
template class CellData<KData,Sphere>;
template class CellData<GData,Flat>;
template class CellData<GData,Sphere>;

template class Cell<NData,Flat>;
template class Cell<NData,Sphere>;
template class Cell<KData,Flat>;
template class Cell<KData,Sphere>;
template class Cell<GData,Flat>;
template class Cell<GData,Sphere>;

template double CalculateSizeSq(
    const Position<Flat>& cen,
    const std::vector<CellData<NData,Flat>*>& vdata,
    size_t start, size_t end);
template double CalculateSizeSq(
    const Position<Sphere>& cen,
    const std::vector<CellData<NData,Sphere>*>& vdata,
    size_t start, size_t end);
template double CalculateSizeSq(
    const Position<Flat>& cen,
    const std::vector<CellData<KData,Flat>*>& vdata,
    size_t start, size_t end);
template double CalculateSizeSq(
    const Position<Sphere>& cen,
    const std::vector<CellData<KData,Sphere>*>& vdata,
    size_t start, size_t end);
template double CalculateSizeSq(
    const Position<Flat>& cen,
    const std::vector<CellData<GData,Flat>*>& vdata,
    size_t start, size_t end);
template double CalculateSizeSq(
    const Position<Sphere>& cen,
    const std::vector<CellData<GData,Sphere>*>& vdata,
    size_t start, size_t end);
