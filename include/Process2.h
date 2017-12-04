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

#ifndef TreeCorr_Process2_H
#define TreeCorr_Process2_H

template <class DataType, class CellType1, class CellType2>
void Process11(
    std::vector<DataType>& data, double minr, double maxr,
    double minrsq, double maxrsq,
    const CellType1& c1, const CellType2& c2)
{
    double s1 = c1.getAllSize();
    double s2 = c2.getAllSize();
    const double dsq = DistSq(c1.getMeanPos(),c2.getMeanPos(),s1,s2);
    const double s1ps2 = s1+s2;

    if (dsq < minrsq) if (sqrt(dsq)+s1ps2 < minr) return;
    if (dsq > maxrsq) if (sqrt(dsq)-s1ps2 > maxr) return;

    ++recursen;
    xdbg<<std::string(recursen,'.')<<"Start P11: "<<c1.getMeanPos()<<" -- "<<c2.getMeanPos()<<
        "   N = "<<c1.getN()<<","<<c2.getN()<<" d = "<<sqrt(dsq)<<std::endl;

    // See if need to split:
    bool split1=false, split2=false;
    CalcSplitSq(split1,split2,c1,c2,dsq,bsq);

    if (split1) {
        if (split2) {
            if (!c1.getLeft()) {
                std::cerr<<"minr = "<<minr<<", maxr = "<<maxr<<std::endl;
                std::cerr<<"minrsq = "<<minrsq<<", maxrsq = "<<maxrsq<<std::endl;
                std::cerr<<"c1.Size = "<<c1.getSize()<<", c2.Size = "<<c2.getSize()<<std::endl;
                std::cerr<<"c1.SizeSq = "<<c1.getSizeSq()<<
                    ", c2.SizeSq = "<<c2.getSizeSq()<<std::endl;
                std::cerr<<"c1.getN = "<<c1.getN()<<", c2.getN = "<<c2.getN()<<std::endl;
                std::cerr<<"c1.Pos = "<<c1.getMeanPos()<<", c2.Pos = "<<c2.getMeanPos()<<std::endl;
                std::cerr<<"dsq = "<<dsq<<", s1ps2 = "<<s1ps2<<std::endl;
            }
            Assert(c1.getLeft());
            Assert(c1.getRight());
            Assert(c2.getLeft());
            Assert(c2.getRight());
            Process11(data,minr,maxr,minrsq,maxrsq,*c1.getLeft(),*c2.getLeft());
            Process11(data,minr,maxr,minrsq,maxrsq,*c1.getLeft(),*c2.getRight());
            Process11(data,minr,maxr,minrsq,maxrsq,*c1.getRight(),*c2.getLeft());
            Process11(data,minr,maxr,minrsq,maxrsq,*c1.getRight(),*c2.getRight());
        } else {
            Assert(c1.getLeft());
            Assert(c1.getRight());
            Process11(data,minr,maxr,minrsq,maxrsq,*c1.getLeft(),c2);
            Process11(data,minr,maxr,minrsq,maxrsq,*c1.getRight(),c2);
        }
    } else {
        if (split2) {
            Assert(c2.getLeft());
            Assert(c2.getRight());
            Process11(data,minr,maxr,minrsq,maxrsq,c1,*c2.getLeft());
            Process11(data,minr,maxr,minrsq,maxrsq,c1,*c2.getRight());
        } else if (dsq > minrsq && dsq < maxrsq) {
            const typename CellType1::PosType& r = c2.getMeanPos()-c1.getMeanPos();
            XAssert(NoSplit(c1,c2,sqrt(dsq),b));
            XAssert(Dist(c2.getMeanPos() - c1.getMeanPos(),r) < 0.0001);
            DirectProcess11(data,c1,c2,dsq,r);
        }
    }

    xdbg<<std::string(recursen,'.')<<"Done P11\n";
    --recursen;
}

template <class DataType, class CellType>
void Process2(
    std::vector<DataType>& data, double minr, double maxr,
    double minrsq, double maxrsq, const CellType& c12)
{
    if (c12.getSize() < minr/2.) return;

    ++recursen;

    std::ostream* tempdbg = dbgout;
    if (dbgout) {
        if (c12.getSize() < outputsize) dbgout = 0;
        dbg<<std::string(recursen,'.')<<"Start P2: size = "<<c12.getSize()<<
            ", center = "<<c12.getMeanPos()<<"   N = "<<c12.getN()<<std::endl;
    }

    xdbg<<std::string(recursen,'.')<<"P2 ("<<c12.getSize()<<") call Left P2\n";
    Assert(c12.getLeft());
    Assert(c12.getRight());
    Process2(data,minr,maxr,minrsq,maxrsq,*c12.getLeft());

    xdbg<<std::string(recursen,'.')<<"P2 ("<<c12.getSize()<<") call Right P2\n";
    Process2(data,minr,maxr,minrsq,maxrsq,*c12.getRight());

    xdbg<<std::string(recursen,'.')<<"P2 ("<<c12.getSize()<<") call P11\n";
    Process11(data,minr,maxr,minrsq,maxrsq,*c12.getLeft(),*c12.getRight());

    if (tempdbg) {
        dbg<<std::string(recursen,'.')<<"Done P2: size = "<<c12.getSize()<<
            ", center = "<<c12.getMeanPos()<<std::endl;
        dbgout = tempdbg;
    }
    --recursen;
}

#endif
