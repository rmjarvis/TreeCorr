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

#ifndef TreeCorr_Split_H
#define TreeCorr_Split_H

#include "Cell.h"

inline void CalcSplit(
    bool& split1, bool& split2,
    const double s1, const double s2, const double s1ps2, const double b)
{
    // This function determines whether either input cell needs to be
    // split.  It is written as a template so that the second cell
    // can be either a Cell or a PairCell.  (The same rules apply.)
    // If you already know that c1 needs to be split, then split1 can
    // be input as true, and we only check c2.  (and vice versa)
    // In normal operation, both are input as false, and we check
    // whether they need to be split.
    //
    // For Log, the criterion is
    //      (s1+s2)/d > b
    // For Linear, the criterion is
    //      (s1+s2) > b
    // in which case, we need to split one or both.  This function
    // models the latter formula, so for Log, b*d should be given as
    // the input parameter b.
    //
    // If s1 > b, then it doesn't matter what s2 is -- we should
    // definitely split c1.
    // Likewise if s2 > b
    //
    // If neither of these conditions is true, then we test
    // to see if s1+s2 > b
    // If we're close to the threshold, it will generally be quicker
    // to only split the larger one.  But if both s1 and s2 are
    // reasonably large (compared to b) then we will probably end
    // up needing to split both, so go ahead and do so now.
    // This small vs. large test is quantified by the parameter
    // splitfactor.  I varied split factor with the 2-point
    // correlation code until it ran fastest.  The result is
    // given above.  I don't know if this value is also best for
    // 3 point uses, but it's probably reasonably close.

    // Quick return
    if (split1 && split2) return;

    const double splitfactor = 0.585;
    // The split factor helps determine whether to split both
    // cells or just one when the factor (s1+s2) is too large.
    // Split either cell if s > f*b.
    // The value of f was determined empirically by seeing
    // when the code ran fastest.  This may be specific
    // to the data I was testing it on, but I would guess
    // that this value is close enough to optimal for most
    // datasets.

    if (s2 > s1) {
        // Make s1 the larger value.
        CalcSplit(split2,split1,s2,s1,s1ps2,b);
    } else if (s1 > 2.*s2) {
        // If one cell is more than 2x the size of the other, only split that one.
        split1 = true;
    } else {
        // If both are comparable, still always split the larger one.
        split1 = true;
        split2 |= s2 > splitfactor*b;  // And maybe the smaller one.
    }
}

inline void CalcSplitSq(
    bool& split1, bool& split2,
    const double s1, const double s2, const double s1ps2, const double bsq)
{
    // The same as above, but when we know the distance squared rather
    // than just the distance.  We get some speed up by saving the
    // square roots in some parts of the code.
    const double splitfactorsq = 0.3422;
    if (split1 && split2) return;
    if (s2 > s1) {
        CalcSplitSq(split2,split1,s2,s1,s1ps2,bsq);
    } else if (s1 > 2.*s2) {
        // If one cell is more than 2x the size of the other, only split that one.
        split1 = true;
    } else {
        // If both are comparable, still always split the larger one.
        split1 = true;
        split2 = s2*s2 > splitfactorsq*bsq;  // And maybe the smaller one.
    }
}

template <class CellType1, class CellType2>
inline bool NoSplit(const CellType1& c1, const CellType2& c2, const double d, const double b)
{
    static const double altb = b/(1.-b);
    // A debugging routine.  Usually of the form:
    // XAssert(NoSplit(c1,c2,d,b))
    // This just asserts that the cells obey the non-splitting eqn:
    // (s1 + s2)/d < b
    // Technically we use altb = b/(1-b) which = b for small b.
    if (c1.getSize() + c2.getSize() < d*altb+0.0001) {
        return true;
    } else {
        std::cerr<<c1.getSize()<<" + "<<c2.getSize()<<" > "<<
            d<<" * "<<altb<<std::endl;
        return false;
    }
}

template <class CellType1, class CellType2, class CellType3>
inline bool Check(
    const CellType1& c1, const CellType2& c2, const CellType3& c3,
    const double d1, const double d2, const double d3)
{
    // Checks that d1,d2,d3 are correct for the three Cells given.
    // Used as a debugging check.
    bool ok=true;
    if (Dist(c3.getData().getPos(),c2.getData().getPos())-d1 > 0.0001)
    { std::cerr<<"d1\n"; ok = false; }
    if (Dist(c1.getData().getPos(),c3.getData().getPos())-d2 > 0.0001)
    { std::cerr<<"d2\n"; ok = false; }
    if (Dist(c2.getData().getPos(),c1.getData().getPos())-d3 > 0.0001)
    { std::cerr<<"d3\n"; ok = false; }
    if (d1 > d2+d3+0.0001) { std::cerr<<"sum d1\n"; ok = false; }
    if (d2 > d1+d3+0.0001) { std::cerr<<"sum d2\n"; ok = false; }
    if (d3 > d1+d2+0.0001) { std::cerr<<"sum d3\n"; ok = false; }
    return ok;
}

#endif
