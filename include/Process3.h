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

#ifndef TreeCorr_Process3_H
#define TreeCorr_Process3_H

template <class DataType, class CellType1, class CellType2, class CellType3>
void ProcessV(
    std::vector<DataType>& vdata,
    const CellType1& c1, const CellType2& c2, const CellType3& c3,
    const double d3, const bool swap12, const double u);

template <class DataType, class CellType1, class CellType2, class CellType3>
void ProcessV(
    std::vector<DataType>& vdata,
    const CellType1& c1, const CellType2& c2, const CellType3& c3,
    const double d1, const double d2, const double d3,
    const bool swap12, const double u)
// Preconditions: s1+s2/d3 < b
//                s3/d2 < b/u
//                d3 < d2 < d1
{
    static const double sqrttwob = sqrt(2.*b);
#define altb (b/(1.-b))

    ++recursen;
    xdbg<<std::string(recursen,'.')<<"Start ProcessV "<<d1<<" >= "<<d2<<" >= "<<d3<<std::endl;
    xdbg<<std::string(recursen,'.')<<"sizes = "<<c1.getSize()<<" -- "<<c2.getSize()<<" -- "<<c3.getSize()<<std::endl;

    XAssert(NoSplit(c1,c2,d3,b));
    XAssert(Check(c1,c2,c3,d1,d2,d3));

    Assert(d2 >= d3);
    Assert(d1 >= d2);
    XAssert(std::abs(u-d3/d2) < altb);
    XAssert(c3.getSize()/d2 < altb*d2/MAX(d3-altb*d2,1.e-10));

    // v is a bit more complicated than u:
    // Consider the angle bisector of d1,d2.  Call this b
    // Then let phi = the angle between d1 (or d2) and b
    // And let theta = the (smaller) angle between d3 and b
    // Then projecting d1,d2,d3 onto b, one finds:
    // d1 cos phi - d2 cos phi = d3 cos theta
    // v = (d1-d2)/d3 = cos theta / cos phi
    // Note that phi < 30 degrees, so cos phi won't make much
    // of a difference here.
    // So the biggest change in v from moving c3 is in theta:
    // dv = abs(dv/dtheta) dtheta = abs(sin theta)/cos phi (s3/b)
    // b cos phi > d2, so
    // dv < b ==> s < m b cos phi / sqrt(1-v^2) < m d2 / sqrt(1-v^2)
    // But - just like with u where the denominator was u+m, not just u,
    // we need to modify this.
    // If v = 1, dv = 1-cos(dtheta) = 1-cos(s3/b) ~= 1/2(s3/b)^2
    // So in this case, s3/b < sqrt(2m)
    // In general we require both to be true.

    double soverd = c3.getSize()/d2;
    double v = (d1-d2)/d3;
    const bool split3 = c3.getSize() > 0. &&
        ( soverd > sqrttwob ||
          soverd > b / sqrt(1-SQR(v)) );

    if (split3) {
        Assert(c3.getSize()>0.);
        Assert(c3.getLeft());
        Assert(c3.getRight());
        ProcessV(vdata,c1,c2,*c3.getLeft(),d3,swap12,u);
        ProcessV(vdata,c1,c2,*c3.getRight(),d3,swap12,u);
    } else {
        if (c3.getSize() == 0.) { // then v didn't get set above
            v = (d1-d2)/d3;
        }
        if (v > 0.99999) v = 0.99999;

        const typename CellType1::PosType r3 = c2.getMeanPos() - c1.getMeanPos();
        const typename CellType1::PosType r1 = c3.getMeanPos() - c2.getMeanPos();
        // This next thing only makes sense for 2D.
        // Need to think about this if I want to extend 3 point stuff into 3D.
        std::complex<double> cr3(r3.getX(),r3.getY());
        std::complex<double> cr1(r1.getX(),r1.getY());
        if (imag(cr1*conj(cr3)) < 0.) v = -v;

        int kv = int(floor((v+1.)/dv));
        //if (kv < 0) {Assert(kv==-1); kv=0;}
        //if (kv >= nvbins) {Assert(kv==nvbins); kv = nvbins-1;}
        Assert(kv >= 0); Assert(kv<int(vdata.size()));
        DirectProcessV(vdata[kv],c1,c2,c3,d1,d2,d3,swap12,u,v);

    }
    xdbg<<std::string(recursen,'.')<<"Done PV\n";
    --recursen;
}

template <class DataType, class CellType1, class CellType2, class CellType3>
void ProcessV(
    std::vector<DataType>& vdata,
    const CellType1& c1, const CellType2& c2, const CellType3& c3,
    const double d3, const bool swap12, const double u)
// Preconditions: s1+s2/d3 < b
//                s3/d2 < b/u
{
#define altb (b/(1.-b))

    XAssert(NoSplit(c1,c2,d3,b));

    const double d1 = Dist(c3.getMeanPos(),c2.getMeanPos());
    const double d2 = Dist(c1.getMeanPos(),c3.getMeanPos());

    if (d3 > d2) return;
    if (d3 > d1) return;

    ++recursen;
    xdbg<<std::string(recursen,'.')<<"Start ProcessV1 "<<d1<<" >= "<<d2<<" >= "<<d3<<std::endl;
    xdbg<<std::string(recursen,'.')<<"sizes = "<<c1.getSize()<<" -- "<<c2.getSize()<<" -- "<<c3.getSize()<<std::endl;

    XAssert(Check(c1,c2,c3,d1,d2,d3));
    XAssert(std::abs(u-d3/d2) < altb);

    if (d1 >= d2) {
        ProcessV(vdata,c1,c2,c3,d1,d2,d3,swap12,u);
    } else {
        ProcessV(vdata,c2,c1,c3,d2,d1,d3,!swap12,u);
    }
    xdbg<<std::string(recursen,'.')<<"Done PV3\n";
    --recursen;
}

template <class DataType, class CellType1, class CellType2, class CellType3>
void ProcessU(
    std::vector<std::vector<DataType> >& uvdata,
    const CellType1& c1, const CellType2& c2, const CellType3& c3,
    const double d3, const bool swap12);

template <class DataType, class CellType1, class CellType2, class CellType3>
void ProcessU(
    std::vector<std::vector<DataType> >& uvdata,
    const CellType1& c1, const CellType2& c2, const CellType3& c3,
    const double d1, const double d2, const double d3, const bool swap12)
// Preconditions: s1+s2/d3 < b
//                d1 >= d2 >= d3
{
    ++recursen;
    xdbg<<std::string(recursen,'.')<<"Start ProcessU3 "<<d1<<" >= "<<d2<<" >= "<<d3<<std::endl;
    xdbg<<std::string(recursen,'.')<<"sizes = "<<c1.getSize()<<" -- "<<c2.getSize()<<" -- "<<c3.getSize()<<std::endl;

    XAssert(NoSplit(c1,c2,d3,b));
    XAssert(Check(c1,c2,c3,d1,d2,d3));

    Assert(d1 >= d2);
    Assert(d2 >= d3);

    // u = d3/d2
    // u_max = d3/(d2-s) = d3/d2 * 1/(1-s/d2)
    // du = u_max - u = u * s/d2 / (1-s/d2)
    // du < b ==> s/d2 < m / (u+m)
    // Note: u-u_min < max ==> s/d2 < m / (u-m)
    //       which is less stringent

    const double u = d3/d2;
    const bool split3 = c3.getSize()*(u+b) > d2*b;

    if (split3) {
        Assert(c3.getSize()>0);
        Assert(c3.getLeft());
        Assert(c3.getRight());
        ProcessU(uvdata,c1,c2,*c3.getLeft(),d3,swap12);
        ProcessU(uvdata,c1,c2,*c3.getRight(),d3,swap12);
    } else {
        int ku = int(floor(u/du));
        if (ku >= nubins) { Assert(ku==nubins); --ku; }
        std::vector<DataType>& vdata = uvdata[ku];

        ProcessV(vdata,c1,c2,c3,d1,d2,d3,swap12,u);
    }
    xdbg<<std::string(recursen,'.')<<"Done PU3\n";
    --recursen;
}

template <class DataType, class CellType1, class CellType2, class CellType3>
void ProcessU(
    std::vector<std::vector<DataType> >& uvdata,
    const CellType1& c1, const CellType2& c2, const CellType3& c3,
    const double d3, const bool swap12)
// Preconditions: s1+s2/d3 < b
{
    ++recursen;
    xdbg<<std::string(recursen,'.')<<"Start ProcessU1 "<<Dist(c2.getMeanPos(),c3.getMeanPos())<<" , "<<Dist(c1.getMeanPos(),c3.getMeanPos())<<" , "<<d3<<std::endl;
    xdbg<<std::string(recursen,'.')<<"sizes = "<<c1.getSize()<<" -- "<<c2.getSize()<<" -- "<<c3.getSize()<<std::endl;
    const double d2 = Dist(c1.getMeanPos(),c3.getMeanPos());
    if (d2 < d3) {
        if (d2 + c3.getSize() < d3) {
            --recursen;
            return;
        } else {
            Assert(c3.getSize()>0);
            Assert(c3.getLeft());
            Assert(c3.getRight());
            ProcessU(uvdata,c1,c2,*c3.getLeft(),d3,swap12);
            ProcessU(uvdata,c1,c2,*c3.getRight(),d3,swap12);
        }
    } else {
        const double d1 = Dist(c3.getMeanPos(),c2.getMeanPos());
        if (d1 < d3) {
            if (d1 + c3.getSize() < d3) {
                --recursen;
                return;
            } else {
                Assert(c3.getSize()>0);
                Assert(c3.getLeft());
                Assert(c3.getRight());
                ProcessU(uvdata,c1,c2,*c3.getLeft(),d3,swap12);
                ProcessU(uvdata,c1,c2,*c3.getRight(),d3,swap12);
            }
        } else {

            XAssert(NoSplit(c1,c2,d3,b));
            XAssert(Check(c1,c2,c3,d1,d2,d3));

            if (d1 >= d2) {
                ProcessU(uvdata,c1,c2,c3,d1,d2,d3,swap12);
            } else {
                ProcessU(uvdata,c2,c1,c3,d2,d1,d3,!swap12);
            }
        }
    }
    xdbg<<std::string(recursen,'.')<<"Done PU1\n";
    --recursen;
}

template <class DataType, class CellType1, class CellType2, class CellType3>
void Process111(
    std::vector<std::vector<std::vector<DataType> > >& data,
    double minr, double maxr,
    const CellType1& c1, const CellType2& c2, const CellType3& c3)
// Does all triangles with 1 point each in c1, c2, c3
// for which d3 is the smallest side
{
    static const double logminsep = log(minsep);

    const typename CellType1::PosType r3 = c2.getMeanPos() - c1.getMeanPos();
    const double d3 = Dist(c2.getMeanPos(),c1.getMeanPos());
    const double s1ps2 = c1.getAllSize()+c2.getAllSize();
    if (d3+s1ps2 < minr) return;
    if (d3-s1ps2 > maxr) return;

    ++recursen;
    std::ostream* tempdbg = 0;
    if (dbgout) {
        tempdbg = dbgout;
        if (std::max(c1.getSize(),c2.getSize()) < outputsize) dbgout = 0;
        dbg<<std::string(recursen,'.')<<"Start Process111 d3 = "<<d3;
        dbg<<"  sizes = "<<c1.getSize()<<" -- "<<c2.getSize()<<std::endl;
    }

    bool split1 = false, split2 = false;
    CalcSplit(split1,split2,c1,c2,d3,b);

    if (split1) {
        if (split2) {
            // split 1,2
            xdbg<<std::string(recursen,'.')<<"split 1, 2\n";
            Assert(c1.getLeft());
            Assert(c1.getRight());
            Assert(c2.getLeft());
            Assert(c2.getRight());
            Process111(data,minr,maxr,*c1.getLeft(),*c2.getLeft(),c3);
            Process111(data,minr,maxr,*c1.getLeft(),*c2.getRight(),c3);
            Process111(data,minr,maxr,*c1.getRight(),*c2.getLeft(),c3);
            Process111(data,minr,maxr,*c1.getRight(),*c2.getRight(),c3);
        } else {
            // split 1 only
            xdbg<<std::string(recursen,'.')<<"split 1\n";
            Assert(c1.getLeft());
            Assert(c1.getRight());
            Process111(data,minr,maxr,*c1.getLeft(),c2,c3);
            Process111(data,minr,maxr,*c1.getRight(),c2,c3);
        }
    } else {
        if (split2) {
            // split 2 only
            xdbg<<std::string(recursen,'.')<<"split 2\n";
            Assert(c2.getLeft());
            Assert(c2.getRight());
            Process111(data,minr,maxr,c1,*c2.getLeft(),c3);
            Process111(data,minr,maxr,c1,*c2.getRight(),c3);
        } else if (d3 <= maxr) {
            // don't split any
            xdbg<<std::string(recursen,'.')<<"do uv\n";

            double logr = log(d3);
            const int kr = int(floor((logr-logminsep)/binsize));
            Assert(kr >= 0);
            Assert(kr < nrbins);

            std::vector<std::vector<DataType> >& datakr = data[kr];
            std::vector<std::vector<DataType> > tempuvdata(
                datakr.size(),
                std::vector<DataType>(datakr[0].size()));

            ProcessU(tempuvdata,c1,c2,c3,d3,false);

            Add(datakr,tempuvdata,c1,c2,r3,d3);

        } else {
            xdbg<<std::string(recursen,'.')<<"d3 > maxsep: "<<d3<<" > "<<maxsep<<std::endl;
        }
    }

    if (tempdbg) {
        dbg<<std::string(recursen,'.')<<"Done P111\n";
        dbgout = tempdbg;
    }
    --recursen;
}

template <class DataType, class CellType12, class CellType3>
void Process21(
    std::vector<std::vector<std::vector<DataType> > >& data,
    double minr, double maxr,
    const CellType12& c12, const CellType3& c3)
// Does all triangles with 2 points in c12 and 3rd point in c3
// for which d3 is the smallest side
{
    const double minsize = minr/2.;
    if (c12.getSize() < minsize) return;

    ++recursen;
    std::ostream* tempdbg = 0;
    if (dbgout) {
        tempdbg = dbgout;
        if (c12.getSize() < outputsize) dbgout = 0;
        dbg<<std::string(recursen,'.')<<"Start Process21: size = "<<c12.getSize();
        dbg<<"  MeanPos = "<<c12.getMeanPos()<<std::endl;
    }

    Assert(c12.getLeft());
    Assert(c12.getRight());

    Process21(data,minr,maxr,*c12.getLeft(),c3);
    Process21(data,minr,maxr,*c12.getRight(),c3);
    Process111(data,minr,maxr,*c12.getLeft(),*c12.getRight(),c3);

    if (tempdbg) {
        dbg<<std::string(recursen,'.')<<"Done P21: size = "<<c12.getSize()<<std::endl;
        dbgout = tempdbg;
    }
    --recursen;
}

template <class DataType, class CellType>
void Process3(
    std::vector<std::vector<std::vector<DataType> > >& data,
    double minr, double maxr, const CellType& c123)
// Does all triangles with 3 points in c123 for which d3 is the smallest side
{
    static const double sqrt3 = sqrt(3.);
    const double minsize = minr/sqrt3;
    if (c123.getSize() < minsize) return;

    ++recursen;
    std::ostream* tempdbg = 0;
    if (dbgout) {
        tempdbg = dbgout;
        if (c123.getSize() < outputsize) dbgout = 0;
        dbg<<std::string(recursen,'.')<<"Start Process3: size = "<<c123.getSize();
        dbg<<"  MeanPos = "<<c123.getMeanPos()<<std::endl;
    }

    Assert(c123.getLeft());
    Assert(c123.getRight());
    Process21(data,minr,maxr,*c123.getLeft(),*c123.getRight());
    Process21(data,minr,maxr,*c123.getRight(),*c123.getLeft());
    Process3(data,minr,maxr,*c123.getLeft());
    Process3(data,minr,maxr,*c123.getRight());

    if (tempdbg) {
        dbg<<std::string(recursen,'.')<<"Done P3: size = "<<c123.getSize()<<std::endl;
        dbgout = tempdbg;
    }
    --recursen;
}

#endif
