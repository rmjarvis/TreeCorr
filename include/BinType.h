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
};


#endif

