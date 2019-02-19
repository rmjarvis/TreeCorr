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
    static int calculateBinK(const Position<C>& , const Position<C>& ,
                             double logr, double logminsep, double binsize,
                             double, double, double)
    { return int((logr - logminsep) / binsize); }

    template <int C>
    static bool singleBin(double dsq, double s1ps2,
                          const Position<C>& p1, const Position<C>& p2,
                          double logminsep, double binsize, double minsep, double maxsep,
                          int *xk, double* xr, double* xlogr)
    {
        xdbg<<"singleBin: "<<dsq<<"  "<<s1ps2<<std::endl;
        const double bsq = binsize * binsize;
        const double maxssq = bsq*dsq;

        if (s1ps2 == 0.) {
            // Trivial return, but need to set k, r, logr
            *xr = std::sqrt(dsq);
            *xlogr = std::log(*xr);
            *xk = int((*xlogr - logminsep) / binsize);
            return true;
        }

        // If s1+s2 > bin_size * r, then too big.
        if (s1ps2*s1ps2 > maxssq) return false;
        xdbg<<"not s1ps2 > maxss\n";

        // Now there is a chance they could fit, depending on the exact position relative
        // to the bin center.
        double r = std::sqrt(dsq);
        double logr = std::log(r);
        double kk = (logr - logminsep) / binsize;
        int ik = int(kk);
        double frackk = kk - ik;
        xdbg<<"kk, ik, frackk = "<<kk<<", "<<ik<<", "<<frackk<<std::endl;
        xdbg<<"ik1 = "<<int( (std::log(r - s1ps2) - logminsep) / binsize )<<std::endl;
        xdbg<<"ik2 = "<<int( (std::log(r + s1ps2) - logminsep) / binsize )<<std::endl;

        // Check how much kk can change for r +- s1ps2
        // If it can change by more than frackk down or (1-frackk) up, then too big.
        // log(r - x) < log(r) - x/r
        // log(r + x) > log(r) + x/r + x^2/2r^2
        if (s1ps2 > frackk * r) return false;
        xdbg<<"not s1ps2 > frackk * r\n";
        if (s1ps2 * r + 0.5 * s1ps2 * s1ps2 > (1.-frackk) * dsq) return false;
        xdbg<<"not s1ps2/r + s1ps2^2/2r^2  > 1-frackk\n";

        // Now we know it's close.  So worth it to do a couple extra logs to check whether
        // we can stop splitting here.
        int ik1 = int( (std::log(r - s1ps2) - logminsep) / binsize );
        if (ik1 < ik) return false;
        int ik2 = int( (std::log(r + s1ps2) - logminsep) / binsize );
        if (ik2 > ik) return false;
        xdbg<<"ik1 == ik2 == ik\n";

        *xk = ik;
        *xr = r;
        *xlogr = logr;
        return true;
    }

};

// Note: The TwoD bin_type is only valid for the Flat Coord.
template <>
struct BinTypeHelper<TwoD>
{
    static bool doReverse() { return true; }

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
                             double , double , double binsize,
                             double , double , double maxsep)
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
                          double _logminsep, double _binsize, double _minsep, double _maxsep,
                          int *k, double* r, double* logr)
    {
        return false;
    }

};


#endif

