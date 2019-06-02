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

#ifndef TreeCorr_BinType_H
#define TreeCorr_BinType_H

// The BinType enum is defined here:
#include "BinType_C.h"
#include <limits>
#include <cmath>


template <int M>
struct BinTypeHelper;

template <>
struct BinTypeHelper<Log>
{
    static bool doReverse() { return false; }

    // For Log binning, the test for when to stop splitting is s1+s2 < b*d.
    // This b*d is the "effective" b used by CalcSplit.
    static double getEffectiveB(double r, double b)
    { return r*b; }

    static double getEffectiveBSq(double rsq, double bsq)
    { return rsq*bsq; }

    // Some binnings (e.g. TwoD) have a larger maximum separation than just maxsep.
    // So this function calculates that.  But here, we just use maxsep.
    static double calculateFullMaxSep(double minsep, double maxsep, int nbins, double binsize)
    { return maxsep; }

    // Check if the given rsq (but maybe also with consideration of the two positions)
    // is within the range that we are including in the binning.
    // In this case it is simply to check if minsep <= r < maxsep.
    template <int C>
    static bool isRSqInRange(double rsq, const Position<C>& p1, const Position<C>& p2,
                           double minsep, double minsepsq, double maxsep, double maxsepsq)
    {
        return rsq >= minsepsq && rsq < maxsepsq;
    }

    // Check if all posisble pairs for a two cells (given s1 + s2) necessarily have too small
    // or too large a distance to be accumulated, so we can stop considering these cells.
    static bool tooSmallDist(double rsq, double s1ps2, double minsep, double minsepsq)
    {
        // r + s1ps2 < minsep
        // r < minsep - s1ps2
        // rsq < (minsep - s1ps2)^2  and  s1ps2 < minsep
        return rsq < minsepsq && s1ps2 < minsep && rsq < SQR(minsep - s1ps2);
    }
    static bool tooLargeDist(double rsq, double s1ps2, double maxsep, double maxsepsq)
    {
        // r - s1ps2 > maxsep
        // rsq > (maxsep + s1ps2)^2
        return rsq >= maxsepsq && rsq >= SQR(maxsep + s1ps2);
    }

    // Calculate the appropriate index for the given pair of cells.
    template <int C>
    static int calculateBinK(const Position<C>& p1, const Position<C>& p2,
                             double r, double logr, double binsize,
                             double minsep, double maxsep, double logminsep)
    { return int((logr - logminsep) / binsize); }

    // Check if we can stop recursing the tree and drop the given pair of cells
    // into a single bin.
    template <int C>
    static bool singleBin(double rsq, double s1ps2,
                          const Position<C>& p1, const Position<C>& p2,
                          double binsize, double b, double bsq,
                          double minsep, double maxsep, double logminsep,
                          int& ik, double& r, double& logr)
    {
        xdbg<<"singleBin: "<<rsq<<"  "<<s1ps2<<std::endl;

        // If two leaves, stop splitting.
        if (s1ps2 == 0.) return true;

        // Standard stop splitting criterion.
        // s1 + s2 <= b * r
        double s1ps2sq = s1ps2 * s1ps2;
        if (s1ps2sq <= bsq*rsq) return true;

        // If s1+s2 > 0.5 * (binsize + b) * r, then the total leakage (on both sides perhaps)
        // will be more than b.  I.e. too much slop.
        if (s1ps2sq > 0.25 * SQR(binsize + b) * rsq) return false;

        // Now there is a chance they could fit, depending on the exact position relative
        // to the bin center.
        logr = 0.5*std::log(rsq);
        double kk = (logr - logminsep) / binsize;
        ik = int(kk);
        double frackk = kk - ik;
        xdbg<<"kk, ik, frackk = "<<kk<<", "<<ik<<", "<<frackk<<std::endl;
        xdbg<<"ik1 = "<<int( (std::log(sqrt(rsq) - s1ps2) - logminsep) / binsize )<<std::endl;
        xdbg<<"ik2 = "<<int( (std::log(sqrt(rsq) + s1ps2) - logminsep) / binsize )<<std::endl;

        // Check how much kk can change for r +- s1ps2
        // If it can change by more than frackk+binslop/2 down or (1-frackk)+binslop/2 up,
        // then too big.
        // Use log(r+x) ~ log(r) + x/r
        // So delta(kk) = s/r / binsize
        // s/r / binsize > f + binslop
        // s/r > f*binsize + binsize*binslop = f*binsize + b
        // s > (f*binsize + b) * r
        // s^2 > (f*binsize + b)^2 * rsq
        double f = std::min(frackk, 1.-frackk);
        if (s1ps2sq > SQR(f*binsize + b) * rsq) return false;

        // This test is safe for the up direction, but slightly too liberal in the down
        // direction.  log(r-x) < log(r) - x/r
        // So refine the test slightly to make sure we have a conservative check here.
        // log(r-x) > log(r) - x/r - x^2/r^2   (if x<r)
        // s/r + s^2/r^2 > f*binsize + b
        if (s1ps2sq > SQR(frackk*binsize + b - s1ps2sq/rsq) * rsq) return false;

        xdbg<<"Both checks passed.\n";
        XAssert(int( (std::log(sqrt(rsq) - s1ps2) - logminsep) / binsize ) == ik);
        XAssert(int( (std::log(sqrt(rsq) + s1ps2) - logminsep) / binsize ) == ik);
        r = sqrt(rsq); // Didn't need this above, but if we set ik, then this needs to be set too.
        return true;
    }

};

template <>
struct BinTypeHelper<Linear>
{
    static bool doReverse() { return false; }

    // For Linear binning, the test for when to stop splitting is s1+s2 < b.
    // So the effective b is just b itself.
    static double getEffectiveB(double r, double b)
    { return b; }

    static double getEffectiveBSq(double rsq, double bsq)
    { return bsq; }

    static double calculateFullMaxSep(double minsep, double maxsep, int nbins, double binsize)
    { return maxsep; }

    // These next few calculations are the same as Log.
    template <int C>
    static bool isRSqInRange(double rsq, const Position<C>& p1, const Position<C>& p2,
                           double minsep, double minsepsq, double maxsep, double maxsepsq)
    {
        return rsq >= minsepsq && rsq < maxsepsq;
    }
    static bool tooSmallDist(double rsq, double s1ps2, double minsep, double minsepsq)
    {
        return rsq < minsepsq && s1ps2 < minsep && rsq < SQR(minsep - s1ps2);
    }
    static bool tooLargeDist(double rsq, double s1ps2, double maxsep, double maxsepsq)
    {
        return rsq >= maxsepsq && rsq >= SQR(maxsep + s1ps2);
    }

    // Binning in r, not logr here, natch.
    template <int C>
    static int calculateBinK(const Position<C>& p1, const Position<C>& p2,
                             double r, double logr, double binsize,
                             double minsep, double maxsep, double logminsep)
    { return int((r - minsep) / binsize); }

    template <int C>
    static bool singleBin(double rsq, double s1ps2,
                          const Position<C>& p1, const Position<C>& p2,
                          double binsize, double b, double bsq,
                          double minsep, double maxsep, double logminsep,
                          int& ik, double& r, double& logr)
    {
        xdbg<<"singleBin: "<<rsq<<"  "<<s1ps2<<std::endl;

        // Standard stop splitting criterion.
        // s1 + s2 <= b
        // Note: this automatically includes the s1ps2 == 0 case, so don't do it separately.
        if (s1ps2 <= b) return true;

        // If s1+s2 > 0.5 * (binsize + b), then the total leakage (on both sides perhaps)
        // will be more than b.  I.e. too much slop.
        if (s1ps2 > 0.5 * (binsize + b)) return false;

        // Now there is a chance they could fit, depending on the exact position relative
        // to the bin center.
        r = sqrt(rsq);
        double kk = (r - minsep) / binsize;
        ik = int(kk);
        double frackk = kk - ik;
        xdbg<<"kk, ik, frackk = "<<kk<<", "<<ik<<", "<<frackk<<std::endl;
        xdbg<<"ik1 = "<<(r - s1ps2 - minsep) / binsize<<std::endl;
        xdbg<<"ik2 = "<<(r + s1ps2 - minsep) / binsize<<std::endl;

        // Check how much kk can change for r +- s1ps2
        // If it can change by more than frackk+binslop down or (1-frackk)+binslop up,
        // then too big.
        // delta(kk) = s / binsize
        // s / binsize > f + binslop
        // s > f*binsize + binsize*binslop
        // s > f*binsize + b
        double f = std::min(frackk, 1.-frackk);
        xdbg<<"f = "<<f<<std::endl;
        xdbg<<"s1ps2 > "<<f*binsize + b<<std::endl;

        if (s1ps2 > f*binsize + b) return false;

        XAssert(int( (r - s1ps2 - minsep + b) / binsize ) == ik);
        XAssert(int( (r + s1ps2 - minsep - b) / binsize ) == ik);
        logr = std::log(r);
        return true;
    }

};

// Note: The TwoD bin_type is only valid for the Flat Coord.
template <>
struct BinTypeHelper<TwoD>
{
    static bool doReverse() { return true; }

    // Like Linear binning, the effective b is just b itself.
    static double getEffectiveB(double r, double b)
    { return b; }

    static double getEffectiveBSq(double rsq, double bsq)
    { return bsq; }

    // The corners of the grid are at sqrt(2) * maxsep.  This is real maximum separation to use
    // for any checks related to the maxsep in the calling code.  We call this fullmaxsep.
    static double calculateFullMaxSep(double minsep, double maxsep, int nbins, double binsize)
    { return maxsep * std::sqrt(2.); }

    // Only C=Flat is valid here.
    template <int C>
    static bool isRSqInRange(double rsq, const Position<C>& p1, const Position<C>& p2,
                           double minsep, double minsepsq, double maxsep, double maxsepsq)
    {
        // Separately check for r==0, since minsep might be 0, but we still don't want to
        // include "pair"s that are really the same object.
        if (rsq == 0. || rsq < minsepsq) return false;
        else {
            Position<C> diff = p1-p2;
            double r = std::max(std::abs(diff.getX()), std::abs(diff.getY()));
            return r < maxsep;
        }
    }

    static bool tooSmallDist(double rsq, double s1ps2, double minsep, double minsepsq)
    {
        return rsq < minsepsq && s1ps2 < minsep && rsq < SQR(minsep - s1ps2);
    }
    static bool tooLargeDist(double rsq, double s1ps2, double maxsep, double maxsepsq)
    {
        // This one needs to use 2*maxsepsq, because the corners of the grid are at sqrt(2)*maxsep.
        return rsq >= 2.*maxsepsq && rsq >= SQR(sqrt(2.)*maxsep + s1ps2);
    }

    template <int C>
    static int calculateBinK(const Position<C>& p1, const Position<C>& p2,
                             double r, double logr, double binsize,
                             double minsep, double maxsep, double logminsep)
    {
        // Binning is separately in i,j, but then we combine into a single index as k=j*n+i.
        double dx = p2.getX() - p1.getX();
        double dy = p2.getY() - p1.getY();
        int i = int((dx + maxsep) / binsize);
        int j = int((dy + maxsep) / binsize);
        int n = int(2*maxsep / binsize+0.5);
        return j*n + i;
    }

    template <int C>
    static bool singleBin(double rsq, double s1ps2,
                          const Position<C>& p1, const Position<C>& p2,
                          double binsize, double b, double bsq,
                          double minsep, double maxsep, double logminsep,
                          int& k, double& r, double& logr)
    {
        xdbg<<"singleBin: "<<rsq<<"  "<<s1ps2<<std::endl;

        // Standard stop splitting criterion.
        // s1 + s2 <= b
        if (s1ps2 <= b) return true;

        // If s1+s2 > 0.5 * (binsize + b), then the total leakage (on both sides perhaps)
        // will be more than b.  I.e. too much slop.
        if (s1ps2 > 0.5 * (binsize + b)) {
            xdbg<<s1ps2<<" > 0.5 * ("<<binsize<<" + "<<b<<")\n";
            return false;
        }
        xdbg<<"Possible single bin case: "<<std::endl;

        // Now there is a chance they could fit, depending on the exact position relative
        // to the bin center.
        double dx = p2.getX() - p1.getX();
        double dy = p2.getY() - p1.getY();
        double ii = (dx + maxsep) / binsize;
        double jj = (dy + maxsep) / binsize;
        xdbg<<"dx,dy = "<<dx<<"  "<<dy<<std::endl;
        xdbg<<"ii,jj = "<<ii<<"  "<<jj<<std::endl;

        int i = int(ii);
        int j = int(jj);
        xdbg<<"ii, i = "<<ii<<", "<<i<<std::endl;
        xdbg<<"i1 = "<<int(ii - s1ps2/binsize)<<std::endl;
        xdbg<<"i2 = "<<int(ii + s1ps2/binsize)<<std::endl;
        xdbg<<"jj, j = "<<jj<<", "<<j<<std::endl;
        xdbg<<"j1 = "<<int(jj - s1ps2/binsize)<<std::endl;
        xdbg<<"j2 = "<<int(jj + s1ps2/binsize)<<std::endl;

        // With TwoD, we need to be careful about the central bin, which includes r==0.
        // We want to make sure to exclude pairs that are really one point repeated.
        // So if s1ps2 > 0 (which is the case at this point) and we are in this bin, then
        // return false so it can recurse down to exclude these non-pairs.
        int mid = int(maxsep/binsize);
        if (i == mid && j == mid) return false;

        // Check how much ii,jj can change for x,y +- s1ps2
        // This is simpler than the Log case, because we don't have to try to avoid
        // gratuitous log function calls.
        if (ii - s1ps2/binsize < i) return false;
        if (ii + s1ps2/binsize >= i+1) return false;
        if (jj - s1ps2/binsize < j) return false;
        if (jj + s1ps2/binsize >= j+1) return false;

        int n = int(2*maxsep / binsize+0.5);
        k = j*n + i;
        logr = 0.5*std::log(rsq);
        xdbg<<"Single bin returning true: "<<dx<<','<<dy<<','<<s1ps2<<','<<binsize<<std::endl;
        return true;
    }
};


#endif

