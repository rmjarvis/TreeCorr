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
    static bool doReversePair() { return false; }

    template <int C>
    static bool DSqInRange(double dsq, const Position<C>& p1, const Position<C>& p2,
                           double minsep, double minsepsq, double maxsep, double maxsepsq)
    {
        return dsq >= minsepsq && dsq < maxsepsq;
    }

    template <int C>
    static int CalculateBinK(const Position<C>& , const Position<C>& ,
                             double logr, double logminr, double binsize,
                             double, double, double)
    { return int((logr - logminr) / binsize); }
};

// The TwoD metric is only valid for the Flat Coord.
template <>
struct BinTypeHelper<TwoDx>
{
    static bool doReversePair() { return true; }

    // Note: Only C=Flat is valid here.
    template <int C>
    static bool DSqInRange(double dsq, const Position<C>& p1, const Position<C>& p2,
                           double minsep, double minsepsq, double maxsep, double maxsepsq)
    {
        if (dsq == 0. || dsq < minsepsq) return false;
        else {
            Position<C> r = p1-p2;
            double d = std::max(std::abs(r.getX()), std::abs(r.getY()));
            return d < maxsep;
        }
    }

    template <int C>
    static int CalculateBinK(const Position<C>& p1, const Position<C>& p2,
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

