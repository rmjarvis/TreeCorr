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

#ifndef TreeCorr_PairCell_H
#define TreeCorr_PairCell_H

#include "OldCell.h"

struct NPairCellData :
    public NCellData
{
    // NPairCellData is a structure that keeps track of all the accumulated
    // data for a given set of NCell pairs.
    // pos is the average of the midpoints between the NCells.
    // sover2 is the vector (complex number) from the midpoint to c2.
    // n1n2 is the product of the counts.

    NPairCellData() : NCellData(),n1n2(0.),d3(0.),sover2(0.) {}
    template <class CellType>
    NPairCellData(const CellType& c1, const CellType& c2,
                  const double d, const std::complex<double>& r);

    void operator+=(const NPairCellData& rhs)
    {
        double w=rhs.n1n2;
        this->pos+=rhs.pos*w; n1n2+=w;
        sover2+=rhs.sover2*w;
        Assert(imag(rhs.sover2) >= 0.);
    }

    void changeSumToAve()
    {
        if(n1n2 > 0.) {
            this->pos/=n1n2; sover2/=n1n2;
            Assert(imag(sover2) >= 0.);
            double abssover2 = std::abs(sover2);
            d3 = 2.*abssover2;
        }
    }

    // pos = (x1+x2)/2 is in NCellData
    double n1n2;
    double d3;
    std::complex<double> sover2;
};

struct PairCellData :
    public NPairCellData
{
    // PairCellData is a structure that keeps track of all the accumulated
    // data for a given set of Cell pairs.
    // pos is the average of the midpoints between the Cells.
    // sover2 is the vector (complex number) from the midpoint to c2.
    // e1e2, e1e2c, w1w2, and n1n2 all store sums, not averages.

    PairCellData() : NPairCellData(),w1w2(0.),e1e2(0.),e1e2c(0.),expm2ialpha3(0.) {}
    PairCellData(const Cell& c1, const Cell& c2,
                 const double d, const std::complex<double>& r);

    void operator+=(const PairCellData& rhs)
    {
        NPairCellData::operator+=(rhs);
        w1w2+=rhs.w1w2;
        e1e2+=rhs.e1e2;
        e1e2c+=rhs.e1e2c;
    }

    void changeSumToAve()
    {
        if(w1w2 > 0.) {
            this->pos/=w1w2;
            sover2/=w1w2;
            Assert(imag(sover2) >= 0.);
            double abssover2 = std::abs(sover2);
            d3 = 2.*abssover2;
            expm2ialpha3 = conj(sover2)/abssover2;
            expm2ialpha3 *= expm2ialpha3;
        }
    }

    double w1w2;
    std::complex<double> e1e2;  // technically w1w2e1e2exp(-4ialpha3)
    std::complex<double> e1e2c; // technically w1w2e1e2*
    std::complex<double> expm2ialpha3;
};

template <class CellType>
inline NPairCellData::NPairCellData(
    const CellType& c1, const CellType& c2,
    const double d, const std::complex<double>& r) :
    NCellData( (c1.getMeanPos()+c2.getMeanPos())/2. ),
    n1n2(c1.getN()*c2.getN()), d3(d), sover2(r/2.)
{
    // make it so 2 is _above_ 1 (y2 > y1)
    if (imag(r) < 0.) {
        sover2 = -sover2;
    }
    Assert(imag(sover2) >= 0.);
}

inline PairCellData::PairCellData(
    const Cell& c1, const Cell& c2,
    const double d, const std::complex<double>& r) :
    NPairCellData(c1,c2,d,r), w1w2(c1.getWeight()*c2.getWeight())
{
    expm2ialpha3 = conj(r)/d; // only e^ia so far
    expm2ialpha3 *= expm2ialpha3; // now e^2ia
    const std::complex<double> expm4ialpha3 = expm2ialpha3*expm2ialpha3;
    const std::complex<double> e1 = c1.getWE();
    const std::complex<double> e2 = c2.getWE();
    e1e2 = e1*e2*expm4ialpha3;
    e1e2c = e1*conj(e2);

    // make it so 2 is _above_ 1 (y2 > y1)
    if (imag(r) < 0.) {
        e1e2c = conj(e1e2c);
        //sover2 = -sover2; This already done in making NPairCellData
    }
    Assert(imag(sover2) >= 0.);
}

template <class DataType>
class GenPairCell
{

    //
    // A PairCell is a set of pairs of regular Cells (or NCells).
    // It is characterized by a centroid (the centroid of the midpoints
    // between the various pairs of Cells), two sizes (see below), and
    // a vector of the data about the pairs binned according to the
    // orientation of the pairs.
    //
    // allsize is simply the maximum deviation of any component pair from
    // the overall centroid.
    // If we make the same kind of calculation for each orientation bin
    // (the max dev from _its_ centroid, not the overall centroid), then
    // size is the maximum size of any of the bins.
    //
    //

public:

    GenPairCell(int nthetabins, double minsize,
                std::vector<DataType>& pairdata, SplitMethod sm=MEAN,
                size_t start=0, size_t end=0);
    ~GenPairCell()
    { if(left) delete left; if(right) delete right; }

    const Position2D& getMeanPos() const { return meanpos; }
    double getMeanD3() const { return meand3; }
    int getNBins() const {return data.size(); }

    const DataType& getData(size_t i) const
    { Assert(i<data.size()); return data[i]; }

    double getSize() const { return size; }
    double getSizeSq() const { return sizesq; }
    double getAllSize() const { return allsize; }
    int getN() const { return ntot; }

    const GenPairCell* getLeft() const { return left; }
    const GenPairCell* getRight() const { return right; }

    void makeLeftRight(
        int nthetabins, double minsize,
        std::vector<DataType>& pairdata, SplitMethod sm,
        size_t start=0, size_t end=0) const;

    int countLeaves() const { return DoCountLeaves(this); }
    std::vector<const GenPairCell*> getAllLeaves() const
    { return DoGetAllLeaves(this); }

    typedef Position2D PosType;

protected:

    Position2D meanpos;
    double meand3;

    std::vector<DataType> data; // binned by alpha3

    double size;
    double sizesq;
    double allsize;
    int ntot;

    const GenPairCell* left;
    const GenPairCell* right;
};

template <class DataType>
inline GenPairCell<DataType>::GenPairCell(
    int nthetabins, double minsize, std::vector<DataType>& vdata,
    SplitMethod sm, size_t start, size_t end) :
    meanpos(0.,0.),meand3(0.),data(nthetabins), size(0.), sizesq(0.),
    allsize(0.), ntot(0), left(0),right(0)
{
    if (end == 0) end = vdata.size();
    Assert(vdata.size()>0);
    Assert(end <= vdata.size());
    Assert(end > start);

    ntot = end-start;

    if (end - start == 1) {
        const DataType& vdatai = vdata[start];

        double alpha3 = arg(vdatai.sover2);
        Assert(alpha3 >= 0.);
        Assert(alpha3 <= PI);
        int k = int(floor(alpha3/PI*nthetabins));
        if (k==nthetabins) { Assert(std::abs(alpha3-PI)<0.0001); --k; }

        meanpos = vdatai.pos;
        meand3 = vdatai.d3;
        data[k] = vdatai;
    } else {
        std::vector<int> vk(end-start);
        for(size_t i=start;i<end;++i) {
            const DataType& vdatai = vdata[i];

            double alpha3 = arg(vdatai.sover2);
            Assert(alpha3 >= 0.);
            Assert(alpha3 <= PI);
            int k = vk[i-start] = int(floor(alpha3/PI*nthetabins));
            if (k==nthetabins) { Assert(std::abs(alpha3-PI)<0.0001); --k; }

            DataType& datak = data[k];
            datak += vdatai;
        }
        double totw=0.;
        for (int k=0;k<nthetabins;++k) {
            DataType& datak = data[k];
            meanpos += datak.pos; // Here pos is really pos*w1w2
            datak.changeSumToAve();
            meand3 += datak.d3*datak.w1w2;
            totw += datak.w1w2;
            Assert(imag(datak.sover2) >= 0.);
        }
        meanpos /= totw;
        meand3 /= totw;

        std::vector<double> binsizesq(nthetabins,0.);
        double allsizesq=0.;
        for(size_t i=start;i<end;++i) {
            const DataType& vdatai = vdata[i];
            int k = vk[i-start];
            double devsq = DistSq(vdatai.pos,data[k].pos);
            if (devsq > binsizesq[k]) binsizesq[k] = devsq;
            devsq = DistSq(vdatai.pos,meanpos);
            if (devsq > allsizesq) allsizesq = devsq;
        }
        sizesq = 0.;
        for(int k=0;k<nthetabins;++k) {
            if (binsizesq[k] > sizesq) sizesq = binsizesq[k];
        }
        size = sqrt(sizesq);
        allsize = sqrt(allsizesq);

        if (allsize < minsize) {
            size = sizesq = allsize = 0.;
        } else {
            size_t mid = SplitCell(vdata,sm,start,end,meanpos);

            left = new GenPairCell(nthetabins,minsize,vdata,sm,start,mid);
            right = new GenPairCell(nthetabins,minsize,vdata,sm,mid,end);
            if (!left || !right)
                myerror("out of memory - cannot create new PairCell");
        }
    }
}

typedef GenPairCell<PairCellData> PairCell;
typedef GenPairCell<NPairCellData> NPairCell;

#endif
