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
    static double getEffectiveB(double d, double b)
    { return d*b; }

    static double getEffectiveBSq(double dsq, double bsq)
    { return dsq*bsq; }

    static double calculateFullMaxSep(double minsep, double maxsep, int nbins, double binsize)
    { return maxsep; }

    template <int C>
    static bool isDSqInRange(double dsq, const Position<C>& p1, const Position<C>& p2,
                           double minsep, double minsepsq, double maxsep, double maxsepsq)
    {
        return dsq >= minsepsq && dsq < maxsepsq;
    }

    static bool tooSmallDist(double dsq, double s1ps2, double minsep, double minsepsq)
    {
        // d + s1ps2 < minsep
        // d < minsep - s1ps2
        // dsq < (minsep - s1ps2)^2  and  s1ps2 < minsep
        return dsq < minsepsq && s1ps2 < minsep && dsq < SQR(minsep - s1ps2);
    }
    static bool tooLargeDist(double dsq, double s1ps2, double maxsep, double maxsepsq)
    {
        // d - s1ps2 > maxsep
        // dsq > (maxsep + s1ps2)^2
        return dsq >= maxsepsq && dsq >= SQR(maxsep + s1ps2);
    }

    template <int C>
    static int calculateBinK(const Position<C>& p1, const Position<C>& p2,
                             double logr, double logminsep, double binsize,
                             double minsep, double maxsep)
    { return int((logr - logminsep) / binsize); }

    template <int C>
    static bool singleBin(double dsq, double s1ps2,
                          const Position<C>& p1, const Position<C>& p2,
                          double binsize, double b, double bsq,
                          double logminsep, double minsep, double maxsep,
                          int& ik, double& logr)
    {
        xdbg<<"singleBin: "<<dsq<<"  "<<s1ps2<<std::endl;

        // If two leaves, stop splitting.
        if (s1ps2 == 0.) {
            logr = 0.5*std::log(dsq);
            ik = int((logr - logminsep) / binsize);
            return true;
        }

        // Standard stop splitting criterion.
        // s1 + s2 <= b * r
        double s1ps2sq = s1ps2 * s1ps2;
        if (s1ps2sq <= bsq*dsq) {
            logr = 0.5*std::log(dsq);
            ik = int((logr - logminsep) / binsize);
            return true;
        }

        // b = 0 means recurse all the way to the leaves.
        if (b == 0) return false;

        // If s1+s2 > 0.5 * (binsize + b) * r, then the total leakage (on both sides perhaps)
        // will be more than b.  I.e. too much slop.
        if (s1ps2sq > 0.25 * SQR(binsize + b) * dsq) {
            return false;
        }

        // Now there is a chance they could fit, depending on the exact position relative
        // to the bin center.
        logr = 0.5*std::log(dsq);
        double kk = (logr - logminsep) / binsize;
        ik = int(kk);
        double frackk = kk - ik;
        xdbg<<"kk, ik, frackk = "<<kk<<", "<<ik<<", "<<frackk<<std::endl;
        xdbg<<"ik1 = "<<int( (std::log(sqrt(dsq) - s1ps2) - logminsep) / binsize )<<std::endl;
        xdbg<<"ik2 = "<<int( (std::log(sqrt(dsq) + s1ps2) - logminsep) / binsize )<<std::endl;

        // Check how much kk can change for r +- s1ps2
        // If it can change by more than frackk+binslop/2 down or (1-frackk)+binslop/2 up,
        // then too big.
        // Use log(r+x) ~ log(r) + x/r
        // So delta(kk) = s/r / binsize
        // s/r / binsize > f + binslop
        // s/r > f*binsize + binsize*binslop = f*binsize + b
        // s > (f*binsize + b) * r
        // s^2 > (f*binsize + b)^2 * dsq
        double f = std::min(frackk, 1.-frackk);
        if (s1ps2sq > SQR(f*binsize + b) * dsq) {
            return false;
        }

        // This test is safe for the up direction, but slightly too liberal in the down
        // direction.  log(r-x) < log(r) - x/r
        // So refine the test slightly to make sure we have a conservative check here.
        // log(r-x) > log(r) - x/r - x^2/r^2   (if x<r)
        // s/r + s^2/r^2 > f*binsize + b
        if (s1ps2sq > SQR(frackk*binsize + b - s1ps2sq/dsq) * dsq) {
            return false;
        }

        xdbg<<"Both checks passed.\n";
        XAssert(int( (std::log(sqrt(dsq) - s1ps2) - logminsep) / binsize ) == ik);
        XAssert(int( (std::log(sqrt(dsq) + s1ps2) - logminsep) / binsize ) == ik);
        return true;
    }

};

// Note: The TwoD bin_type is only valid for the Flat Coord.
template <>
struct BinTypeHelper<TwoD>
{
    static bool doReverse() { return true; }

    // Like Linear binning, the effective b is just b itself.
    static double getEffectiveB(double d, double b)
    { return b; }

    static double getEffectiveBSq(double dsq, double bsq)
    { return bsq; }

    static double calculateFullMaxSep(double minsep, double maxsep, int nbins, double binsize)
    { return maxsep * std::sqrt(2.); }

    // Only C=Flat is valid here.
    template <int C>
    static bool isDSqInRange(double dsq, const Position<C>& p1, const Position<C>& p2,
                           double minsep, double minsepsq, double maxsep, double maxsepsq)
    {
        if (dsq == 0. || dsq < minsepsq) return false;
        else {
            Position<C> r = p1-p2;
            double d = std::max(std::abs(r.getX()), std::abs(r.getY()));
            return d < maxsep;
        }
    }

    static bool tooSmallDist(double dsq, double s1ps2, double minsep, double minsepsq)
    {
        return dsq < minsepsq && s1ps2 < minsep && dsq < SQR(minsep - s1ps2);
    }
    static bool tooLargeDist(double dsq, double s1ps2, double maxsep, double maxsepsq)
    {
        return dsq >= 2.*maxsepsq && dsq >= SQR(sqrt(2.)*maxsep + s1ps2);
    }

    template <int C>
    static int calculateBinK(const Position<C>& p1, const Position<C>& p2,
                             double logr, double logminsep, double binsize,
                             double minsep, double maxsep)
    {
        double dx = p2.getX() - p1.getX();
        double dy = p2.getY() - p1.getY();
        int i = int((dx + maxsep) / binsize);
        int j = int((dy + maxsep) / binsize);
        int n = int(2*maxsep / binsize+0.5);
        return j*n + i;
    }

    template <int C>
    static bool singleBin(double dsq, double s1ps2,
                          const Position<C>& p1, const Position<C>& p2,
                          double binsize, double b, double bsq,
                          double logminsep, double minsep, double maxsep,
                          int& k, double& logr)
    {
        // Standard stop splitting criterion.
        if (s1ps2 <= b) {
            logr = 0.5*std::log(dsq);
            k = int((logr - logminsep) / binsize);
            return true;
        }

        return false;
    }


};


#endif

