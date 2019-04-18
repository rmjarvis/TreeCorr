#ifndef CORR_H
#define CORR_H

#include "OldCell.h"

// We use a code (to be used as a template parameter) to indicate which kind of data we
// are using for a particular use. 
// NData means just count the point.
// TData means use a scalar.  Nominally temperature, but works with any scalar.
// EData means use a shear. 
enum DataCode { NData=1 , TData=2,  EData=3 };

template <int Data1> struct DataHelper;

template <>
struct DataHelper<1> // NData
{
    typedef NCell CellType;
};

template <>
struct DataHelper<2> // TData
{
    typedef TCell CellType;
};

template <>
struct DataHelper<3> // EData
{
    typedef Cell CellType;
};

template <int Data1, int Data2> struct BinData2;

// NNData
template <>
struct BinData2<1,1>
{
    BinData2() : npair(0.) {}

    double npair;
};

// EEData
template <>
struct BinData2<3,3> {
    BinData2() : xiplus(0.), ximinus(0.), varxi(0.), meanlogr(0.), weight(0.), npair(0.) {}

    std::complex<double> xiplus,ximinus;
    double varxi, meanlogr, weight, npair;
};

template <>
struct BinData2<1,3> {
    BinData2() : meangammat(0.), vargammat(0.), meanlogr(0.), weight(0.), npair(0.) {}

    std::complex<double> meangammat;
    double vargammat, meanlogr, weight, npair;
};

template <int Data1, int Data2, int Data3> struct BinData3;

// NNNData
template <>
struct BinData3<1,1,1> {
    BinData3() : ntri(0.) {}

    double ntri;
};

// EEEData
template <>
struct BinData3<3,3,3> {
    BinData3() :
        gam0(0.), gam1(0.), gam2(0.), gam3(0.), vargam(0.),
        meanr(0.), meanu(0.), meanv(0.), weight(0.), ntri(0.) 
    {}

    std::complex<double> gam0, gam1, gam2, gam3;
    double vargam, meanr, meanu, meanv, weight, ntri;
};

extern const double minsep;
extern const double maxsep;
extern const double minsepsq;
extern const double maxsepsq;
extern const double binsize;
extern const double b;
extern const double bsq;

template <class CellType1, class CellType2>
inline void CalcSplit(
    bool& split1, bool& split2, const CellType1& c1, 
    const CellType2& c2, const double d3, const double b)
{
    // This function determines whether either input cell needs to be
    // split.  It is written as a template so that the second cell
    // can be either a Cell or a PairCell.  (The same rules apply.)
    // If you already know that c1 needs to be split, then split1 can
    // be input as true, and we only check c2.  (and vice versa)
    // In normal operation, both are input as false, and we check
    // whether they need to be split.  
    //
    // If (s1+s2)/d > b, then we need to split one or both.
    //
    // If s1 > b*d, then it doesn't matter what s2 is -- we should
    // definitely split c1.  
    // Likewise if s2 > b*d
    //
    // If neither of these conditions is true, then we test
    // to see if s1+s2 > b*d
    // If we're close to the threshold, it will generally be quicker
    // to only split the larger one.  But if both s1 and s2 are 
    // reasonably large (compared to b/d) then we will probably end 
    // up needing to split both, so go ahead and do so now.
    // This small vs. large test is quantified by the parameter
    // splitfactor.  I varied split factor with the 2-point
    // correlation code until it ran fastest.  The result is 
    // given above.  I don't know if this value is also best for
    // 3 point uses, but it's probably reasonably close.


    const double splitfactor = 0.585;
    // The split factor helps determine whether to split
    // both cells or just one when the factor (s1+s2)/d 
    // is too large.  
    // If s1+s2 > f*d*b then split both.
    // Otherwise just split the large Cell.
    // The value of f was determined empirically by seeing 
    // when the code ran fastest.  This may be specific
    // to the data I was testing it on, but I would guess
    // that this value is close enough to optimal for most
    // datasets.

    if (split1) {
        if (split2) {
            // do nothing
        } else {
            const double s2 = c2.getSize();
            const double maxs = d3*b;
            split2 = s2 > maxs;
        }
    } else {
        if (split2) {
            const double s1 = c1.getSize();
            const double maxs = d3*b;
            split1 = s1 > maxs;
        } else {
            const double s1 = c1.getSize();
            const double s2 = c2.getSize();
            const double maxs = d3*b;
            split1 = s1 > maxs;
            split2 = s2 > maxs;
            if (!split1 && !split2) {
                const double sum = s1+s2;
                if (sum > maxs) {
                    double modmax = splitfactor*maxs;
                    if (s1 > s2) {
                        split1 = true;
                        split2 = (s2 > modmax);
                    } else {
                        split2 = true;
                        split1 = (s1 > modmax);
                    }
                }
            }
        }
    }
}

template <class CellType1, class CellType2>
inline void CalcSplitSq(
    bool& split1, bool& split2, const CellType1& c1, 
    const CellType2& c2, const double d3sq, const double bsq)
{
    const double splitfactorsq = 0.3422;

    // The same as above, but when we know the distance squared rather
    // than just the distance.  We get some speed up by saving the 
    // square roots in some parts of the code.
    if (split1) {
        if (split2) {
            // do nothing
        } else {
            const double s2sq = c2.getSizeSq();
            const double maxssq = bsq*d3sq;
            split2 = s2sq > maxssq;
        }
    } else {
        if (split2) {
            const double s1sq = c1.getSizeSq();
            const double maxssq = bsq*d3sq;
            split1 = s1sq > maxssq;
        } else {
            const double s1sq = c1.getSizeSq();
            const double s2sq = c2.getSizeSq();
            const double maxssq = bsq*d3sq;
            split1 = s1sq > maxssq;
            split2 = s2sq > maxssq;
            if (!split1 && !split2) {
                double sumsq = c1.getSize()+c2.getSize();
                sumsq *= sumsq;
                if (sumsq > maxssq) {
                    double modmax = splitfactorsq*maxssq;
                    if (s1sq > s2sq) {
                        split1 = true;
                        split2 = (s2sq > modmax);
                    } else {
                        split2 = true;
                        split1 = (s1sq > modmax);
                    }
                }
            }
        }
    }
}

inline bool NoSplit(const Cell& c2, const Cell& c3, const double d1, const double b)
{
    static const double altb = b/(1.-b);
    // A debugging routine.  Usually of the form:
    // XAssert(NoSplit(c2,c3,d1,b))
    // This just asserts that the cells obey the non-splitting eqn:
    // (s1 + s2)/d < b
    // Technically we use altb = b/(1-b) which = b for small b.
    if (c2.getSize() + c3.getSize() < d1*altb+0.0001) {
        return true;
    } else {
        std::cerr<<c2.getSize()<<" + "<<c3.getSize()<<" > "<<
            d1<<" * "<<altb<<std::endl;
        return false;
    }
}

inline bool Check(
    const Cell& c1, const Cell& c2, const Cell& c3,
    const double d1, const double d2, const double d3)
{
    // Checks that d1,d2,d3 are correct for the three Cells given.
    // Used as a debugging check.
    bool ok=true;
    if (Dist(c3.getMeanPos(),c2.getMeanPos())-d1 > 0.0001) 
    { std::cerr<<"d1\n"; ok = false; }
    if (Dist(c1.getMeanPos(),c3.getMeanPos())-d2 > 0.0001) 
    { std::cerr<<"d2\n"; ok = false; }
    if (Dist(c2.getMeanPos(),c1.getMeanPos())-d3 > 0.0001) 
    { std::cerr<<"d3\n"; ok = false; }
    if (d1 > d2+d3+0.0001) { std::cerr<<"sum d1\n"; ok = false; }
    if (d2 > d1+d3+0.0001) { std::cerr<<"sum d2\n"; ok = false; }
    if (d3 > d1+d2+0.0001) { std::cerr<<"sum d3\n"; ok = false; }
    return ok;
}

#endif
