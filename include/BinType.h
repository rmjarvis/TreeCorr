/* Copyright (c) 2003-2024 by Mike Jarvis
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

#ifdef _WIN32
#define _USE_MATH_DEFINES  // To get M_PI
#endif

#include <limits>
#include <cmath>
#include "Metric.h"
#include "ProjectHelper.h"

// We use a code for the type of binning to use:
// Log is logarithmic spacing in r
// Linear is linear spacing in r
// TwoD is linear spacing in x,y
// LogRUV is logarithmic spacing in r=d2, linear in u=d3/d2 and v=(d1-d2)/d3
// LogSAS is logarithmic spacing in r1, r2, linear in phi
// LogMultipole is logarithmic spacing in r1, r2, and stores multipole values.

enum BinType { Log, Linear, TwoD, LogRUV, LogSAS, LogMultipole };

template <int M>
struct BinTypeHelper;

template <>
struct BinTypeHelper<Log>
{
    enum { do_reverse = false };

    // For Log binning, the test for when to stop splitting is s1+s2 < b*d.
    // This b*d is the "effective" b used by CalcSplit.
    static double getEffectiveB(double r, double b, double a)
    { return r*std::min(b, a); }

    static double getEffectiveBSq(double rsq, double bsq, double asq)
    { return rsq*std::min(bsq, asq); }

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
                          double binsize, double b, double bsq, double a, double asq,
                          double minsep, double maxsep, double logminsep,
                          int& ik, double& r, double& logr)
    {
        xdbg<<"singleBin: "<<rsq<<"  "<<s1ps2<<std::endl;

        // If two leaves, stop splitting.
        if (s1ps2 == 0.) return true;

        // Check for angle being too large.
        double s1ps2sq = s1ps2 * s1ps2;
        if (s1ps2sq > asq*rsq) return false;

        // Standard stop splitting criterion.
        // s1 + s2 <= b * r
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

    static double oneBinLessThan(double r, double binsize)
    { return (1.-binsize) * r; }

};

template <>
struct BinTypeHelper<Linear>
{
    enum { do_reverse = false };

    // For Linear binning, the test for when to stop splitting is s1+s2 < b.
    // So the effective b is just b itself.
    static double getEffectiveB(double r, double b, double a)
    { return std::min(b, a*r); }

    static double getEffectiveBSq(double rsq, double bsq, double asq)
    { return std::min(bsq, asq*rsq); }

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
                          double binsize, double b, double bsq, double a, double asq,
                          double minsep, double maxsep, double logminsep,
                          int& ik, double& r, double& logr)
    {
        xdbg<<"singleBin: "<<rsq<<"  "<<s1ps2<<std::endl;

        // Check for angle being too large.
        if (SQR(s1ps2) > asq*rsq) return false;

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

    static double oneBinLessThan(double r, double binsize)
    { return r-binsize; }
};

// Note: The TwoD bin_type is only valid for the Flat Coord.
template <>
struct BinTypeHelper<TwoD>
{
    enum { do_reverse = true };

    // Like Linear binning, the effective b is just b itself.
    static double getEffectiveB(double r, double b, double a)
    { return std::min(b, a*r); }

    static double getEffectiveBSq(double rsq, double bsq, double asq)
    { return std::min(bsq, asq*rsq); }

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
            Position<C> diff(p1-p2);
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
        Assert(i<=n);
        if (i == n) --i;
        Assert(j<=n);
        if (j == n) --j;
        return j*n + i;
    }

    template <int C>
    static bool singleBin(double rsq, double s1ps2,
                          const Position<C>& p1, const Position<C>& p2,
                          double binsize, double b, double bsq, double a, double asq,
                          double minsep, double maxsep, double logminsep,
                          int& k, double& r, double& logr)
    {
        xdbg<<"singleBin: "<<rsq<<"  "<<s1ps2<<std::endl;

        // Check for angle being too large.
        if (SQR(s1ps2) > asq*rsq) return false;

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

    static double oneBinLessThan(double r, double binsize)
    { return r-binsize; }
};


template <>
struct BinTypeHelper<LogRUV>
{
    enum { sort_d123 = true, swap_23 = true };

    static int calculateNTot(int nbins, int nubins, int nvbins)
    { return nbins * nubins * nvbins * 2; }

    static bool tooSmallS2(double s2, double halfminsep, double minu, double minv)
    {
        // When still doing process12, if the s2 cell is smaller than the minimum
        // possible triangle side length, then we can stop early.
        return (s2 == 0. || s2 < halfminsep * minu);
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

    // If the user has set a minu > 0, then we may be able to stop for that.
    template <int O>
    static bool noAllowedAngles(double rsq, double s1ps2, double s1, double s2,
                                double halfminsep,
                                double minu, double minusq, double maxu, double maxusq,
                                double minv, double minvsq, double maxv, double maxvsq)
    {
        // The maximum possible u value at this point is 2s2 / (r - s1 - s2)
        // If this is less than minu, we can stop.
        // 2s2 < minu * (r - s1 - s2)
        // minu * r > 2s2 + minu * (s1 + s2)
        return rsq > SQR(s1ps2) && minusq * rsq > SQR(2.*s2 + minu * (s1ps2));
    }

    // Once we have all the distances, see if it's possible to stop
    // For this BinType, if return value is false, d2 is set on output.
    template <int O, int M, int C>
    static bool stop111(
        double d1sq, double d2sq, double d3sq,
        double s1, double s2, double s3,
        const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
        const MetricHelper<M,0>& metric,
        double& d1, double& d2, double& d3, double& u, double& v,
        double minsep, double minsepsq, double maxsep, double maxsepsq,
        double minu, double minusq, double maxu, double maxusq,
        double minv, double minvsq, double maxv, double maxvsq)
    {
        xdbg<<"Stop111: "<<std::sqrt(d1sq)<<"  "<<std::sqrt(d2sq)<<"  "<<std::sqrt(d3sq)<<std::endl;
        xdbg<<"sizes = "<<s1<<"  "<<s2<<"  "<<s3<<std::endl;
        xdbg<<"sep range = "<<minsep<<"  "<<maxsep<<std::endl;

        if (!O) {
            Assert(d1sq >= d2sq);
            Assert(d2sq >= d3sq);
        }

        // If all possible triangles will have d2 < minsep, then abort the recursion here.
        // This means at least two sides must have d + (s+s) < minsep.
        // Probably if d2 + s1+s3 < minsep, we can stop, but also check d3.
        // If one of these don't pass, then it's pretty unlikely that d1 will, so don't bother
        // checking that one.
        if (d2sq < minsepsq && s1+s3 < minsep && s1+s2 < minsep &&
            (s1+s3 == 0. || d2sq < SQR(minsep - s1-s3)) &&
            (s1+s2 == 0. || d3sq < SQR(minsep - s1-s2)) ) {
            xdbg<<"d2 cannot be as large as minsep\n";
            return true;
        }

        // Similarly, we can abort if all possible triangles will have d2 > maxsep.
        // This means at least two sides must have d - (s+s) > maxsep.
        // Again, d2 - s1 - s3 >= maxsep is not sufficient.  Also check d1.
        // And again, it's pretty unlikely that d3 needs to be checked if one of the first
        // two don't pass.
        if (d2sq >= maxsepsq &&
            (s1+s3 == 0. || d2sq >= SQR(maxsep + s1+s3)) &&
            (s2+s3 == 0. || d1sq >= SQR(maxsep + s2+s3))) {
            xdbg<<"d2 cannot be as small as maxsep\n";
            return true;
        }

        d2 = sqrt(d2sq);
        if (O) {
            // If not sorting, then we need to check if we have a configuration where
            // d1 cannot be the largest or d3 cannot be the smallest.

            // (d2 + s1+s3) < (d3 - s1-s2)
            if (O == 3 && d3sq > SQR(d2 + 2*s1+s2+s3)) {
                xdbg<<"d2 cannot be larger than d3\n";
                return true;
            }
            // (d1 + s2+s3) < (d2 - s1-s3)
            double ss = s1+s2+2*s3;
            if (ss < d2 && d1sq < SQR(d2 - ss)) {
                xdbg<<"d1 cannot be larger than d2\n";
                return true;
            }
            d1 = sqrt(d1sq);
            // (d1 + s2+s3) < (d3 - s1-s2)
            if (d3sq > SQR(d1 + s1+2*s2+s3)) {
                xdbg<<"d1 cannot be larger than d3\n";
                return true;
            }
        }

        // If the user sets minu > 0, then we can abort if no possible triangle can have
        // u = d3/d2 as large as this.
        // The maximum possible u from our triangle is (d3+s1+s2) / (d2-s1-s3).
        // Abort if (d3+s1+s2) / (d2-s1-s3) < minu
        // (d3+s1+s2) < minu * (d2-s1-s3)
        // d3 < minu * (d2-s1-s3) - (s1+s2)
        if (minu > 0. && d3sq < minusq*d2sq && d2 > s1+s3) {
            double temp = minu * (d2-s1-s3);
            if (temp > s1+s2 && d3sq < SQR(temp - s1-s2)) {
                // However, d2 might not really be the middle leg.  So check d1 as well.
                double minusq_d1sq = minusq * d1sq;
                if (d3sq < minusq_d1sq && d1sq > 2.*SQR(s2+s3) &&
                    minusq_d1sq > 2.*d3sq + 2.*SQR(s1+s2 + minu * (s2+s3))) {
                    xdbg<<"u cannot be as large as minu\n";
                    return true;
                }
            }
        }

        // If the user sets a maxu < 1, then we can abort if no possible triangle can have
        // u as small as this.
        // The minimum possible u from our triangle is (d3-s1-s2) / (d2+s1+s3).
        // Abort if (d3-s1-s2) / (d2+s1+s3) > maxu
        // (d3-s1-s2) > maxu * (d2+s1+s3)
        // d3 > maxu * (d2+s1+s3) + (s1+s2)
        if (maxu < 1. && d3sq >= maxusq*d2sq && d3sq >= SQR(maxu * (d2+s1+s3) + s1+s2)) {
            // This time, just make sure no other side could become the smallest side.
            // d3 - s1-s2 < d2 - s1-s3
            // d3 - s1-s2 < d1 - s2-s3
            if ( d2sq > SQR(s1+s3) && d1sq > SQR(s2+s3) &&
                 (s2 > s3 || d3sq <= SQR(d2 - s3 + s2)) &&
                 (s1 > s3 || d1sq >= 2.*d3sq + 2.*SQR(s3 - s1)) ) {
                xdbg<<"u cannot be as small as maxu\n";
                return true;
            }
        }

        // If the user sets minv, maxv to be near 0, then we can abort if no possible triangle
        // can have v = (d1-d2)/d3 as small in absolute value as either of these.
        // d1 > maxv d3 + d2+s1+s2+s3 + maxv*(s1+s2)
        // As before, use the fact that d3 < d2, so check
        // d1 > maxv d2 + d2+s1+s2+s3 + maxv*(s1+s2)
        double sums = s1+s2+s3;
        if (maxv < 1. && d1sq > SQR((1.+maxv)*d2 + sums + maxv * (s1+s2))) {
            // We don't need any extra checks here related to the possibility of the sides
            // switching roles, since if this condition is true, than d1 has to be the largest
            // side no matter what.  d1-s2 > d2+s1
            xdbg<<"v cannot be as small as maxv\n";
            return true;
        }

        // It will unusual, but if minv > 0, then we can also potentially stop if no triangle
        // can have |v| as large as minv.
        // d1-d2 < minv d3 - (s1+s2+s3) - minv*(s1+s2)
        // d1^2-d2^2 < (minv d3 - (s1+s2+s3) - minv*(s1+s2)) (d1+d2)
        // This is most relevant when d1 ~= d2, so make this more restrictive with d1->d2 on rhs.
        // d1^2-d2^2 < (minv d3 - (s1+s2+s3) - minv*(s1+s2)) 2d2
        // minv d3 > (d1^2-d2^2)/(2d2) + (s1+s2+s3) + minv*(s1+s2)
        if (minv > 0. && d3sq > SQR(s1+s2) &&
            minvsq*d3sq > SQR((d1sq-d2sq)/(2.*d2) + sums + minv*(s1+s2))) {
            // And again, we don't need anything else here, since it's fine if d1,d2 swap or
            // even if d2,d3 swap.
            xdbg<<"|v| cannot be as large as minv\n";
            return true;
        }

        // Stop if any side is exactly 0 and elements are leaves
        // (This is unusual, but we want to make sure to stop if it happens.)
        if (s2==0 && s3==0 && d1sq == 0) return true;
        if (s1==0 && s3==0 && d2sq == 0) return true;
        if (s1==0 && s2==0 && d3sq == 0) return true;

        return false;
    }

    // If return value is false, split1, split2, split3 will be set on output.
    // If return value is true, d1, d2, d3, u, v will be set on output.
    // (For this BinType, d2 is already set coming in.)
    static bool singleBin(double d1sq, double d2sq, double d3sq,
                          double s1, double s2, double s3,
                          double b, double a, double bu, double bv,
                          double bsq, double asq, double busq, double bvsq,
                          bool& split1, bool& split2, bool& split3,
                          double& d1, double& d2, double& d3,
                          double& u, double& v)
    {
        // First decide whether to split c3

        // There are a few places we do a calculation akin to the splitfactor thing for 2pt.
        // That one was determined empirically to optimize the running time for a particular
        // (albeit intended to be fairly typical) use case.  Similarly, this factor was found
        // empirically on a particular (GGG) use case with a reasonable choice of separations
        // and binning.
        const double splitfactor = 0.7;

        // These are set correctly before they are used.
        double s1ps2=0., s1ps3=0.;
        bool d2split=false;

        split3 = s3 > 0 && (
            // Check if d2 solution needs a split
            // This is the same as the normal 2pt splitting check.
            (s3 > d2 * b) ||
            ((s1ps3=s1+s3) > 0. && (s1ps3 > d2 * b) && (d2split=true, s3 >= s1)) ||

            // Check if u solution needs a split
            // u = d3/d2
            // max u = d3 / (d2-s3) ~= d3/d2 * (1+s3/d2)
            // delta u = d3 s3 / d2^2
            // Split if delta u > b
            //          d3 s3 > b d2^2
            // Note: if bu >= b, then this is degenerate with above d2 check (since d3 < d2).
            (bu < b && (SQR(s3) * d3sq > SQR(bu*d2sq))) ||

            // Check angles
            (SQR(s3) > asq * d2sq) ||

            // For the v check, it turns out that the triangle where s3 has the maximum effect
            // on v is when the triangle is nearly equilateral.  Both larger d1 and smaller d3
            // reduce the potential impact of s3 on v.
            // Furthermore, for an equilateral triangle, the maximum change in v is very close
            // to s3/d.  So this is the same check as we already did for d2 above, but using
            // bv rather than b.
            // Since bv is usually not much smaller than b, don't bother being more careful
            // than this.
            (bv < b && s3 > d2 * bv));

        if (split3) {
            // If splitting c3, then usually also split c1 and c2.
            // The s3 checks are less calculation-intensive than the later s1,s2 checks.  So it
            // turns out (empirically) that unless s1 or s2 is a lot smaller than s3, we pretty much
            // always want to split them.  This is especially true if d3 << d2.
            // Thus, the decision is split if s > f (d3/d2) s3, where f is an empirical factor.
            const double temp = splitfactor * SQR(s3) * d3sq;
            split1 = SQR(s1) * d2sq > temp;
            split2 = SQR(s2) * d2sq > temp;
            return false;

        } else if (s1 > 0 || s2 > 0) {
            // Now figure out if c1 or c2 needs to be split.

            split1 = (s1 > 0.) && (
                // Apply the d2split that we saved from above.  If we didn't split c3, split c1.
                d2split ||
                s1 > b * d2 ||

                // Check angles.  Relevant check is s1 > a * d3
                (SQR(s1) > asq * d3sq));

            split2 = (s2 > 0.) && (
                (SQR(s2) > asq * d3sq) ||

                // Split c2 if it's possible for d3 to become larger than the largest possible d2
                // or if d1 could become smaller than the current smallest possible d2.
                // i.e. if d3 + s1 + s2 > d2 + s1 + s3 => d3 > d2 - s2 + s3
                //      or d1 - s2 - s3 < d2 - s1 - s3 => d1 < d2 + s2 - s1
                (s2>s3 && (d3sq > SQR(d2 - s2 + s3))) ||
                (s2>s1 && (d1sq < SQR(d2 + s2 - s1))));

            // All other checks mean split at least one of c1 or c2.
            // Done with ||, so it will stop checking if anything is true.
            bool split =
                // Don't bother doing further calculations if already splitting something.
                split1 || split2 ||

                // Check splitting c1,c2 for u calculation.
                // u = d3 / d2
                // u_max = (d3 + s1ps2) / (d2 - s1+s3) ~= u + s1ps2/d2 + s1ps3 u/d2
                // du < bu
                // (s1ps2 + u s1ps3) < bu * d2
                (d3=sqrt(d3sq), u=d3/d2, SQR((s1ps2=s1+s2) + s1ps3*u) > d2sq * busq) ||

                // Check how v changes for different pairs of points within c1,c2?
                //
                // d1-d2 can change by s1+s2, and also d3 can change by s1+s2 the other way.
                // minv = (d1-d2-s1-s2) / (d3+s1+s2) ~= v - (s1+s2)/d3 - (s1+s2)v/d3
                // maxv = (d1-d2+s1+s2) / (d3-s1-s2) ~= v + (s1+s2)/d3 + (s1+s2)v/d3
                // So require (s1+s2)(1+v) < bv d3
                (d1=sqrt(d1sq), v=(d1-d2)/d3, SQR(s1ps2 * (1.+v)) > d3sq * bvsq);

            if (split) {
                // If splitting either one, also do the other if it's close.
                // Because we were so aggressive in splitting c1,c2 above during the c3 splits,
                // it turns out that here we usually only want to split one, not both.
                // The above only entails a split if it's the larger one of s1,s2.
                split1 = split1 || s1 >= s2;
                split2 = split2 || s2 >= s1;
                return false;
            } else {
                return true;
            }
        } else {
            // s1==s2==0 and not splitting s3.
            // Just need to calculate the terms we guarantee to be set when split=false
            d1 = sqrt(d1sq);
            d3 = sqrt(d3sq);
            u = d3/d2;
            v = (d1-d2)/d3;
            return true;
        }
    }

    template <int O, int M, int C>
    static bool isTriangleInRange(const BaseCell<C>& c1, const BaseCell<C>& c2,
                                  const BaseCell<C>& c3,
                                  const MetricHelper<M,0>& metric,
                                  double d1sq, double d2sq, double d3sq,
                                  double d1, double d2, double d3, double& u, double& v,
                                  double logminsep,
                                  double minsep, double maxsep, double binsize, int nbins,
                                  double minu, double maxu, double ubinsize, int nubins,
                                  double minv, double maxv, double vbinsize, int nvbins,
                                  double& logd1, double& logd2, double& logd3,
                                  int ntot, int& index)
    {
        // Make sure all the quantities we thought should be set have been.
        Assert(d1 > 0.);
        Assert(d3 > 0.);
        Assert(u > 0.);

        if (O && !(d1 >= d2 && d2 >= d3)) {
            xdbg<<"Sides are not in correct size ordering d1 >= d2 >= d3\n";
            return false;
        }

        Assert(v >= 0.);  // v can potentially == 0.

        if (d2 < minsep || d2 >= maxsep) {
            xdbg<<"d2 not in minsep .. maxsep\n";
            return false;
        }

        if (u < minu || u >= maxu) {
            xdbg<<"u not in minu .. maxu\n";
            return false;
        }

        if (v < minv || v >= maxv) {
            xdbg<<"v not in minv .. maxv\n";
            return false;
        }

        logd2 = log(d2);
        xdbg<<"            logr = "<<logd2<<std::endl;
        xdbg<<"            u = "<<u<<std::endl;
        xdbg<<"            v = "<<v<<std::endl;

        int kr = int(floor((logd2-logminsep)/binsize));
        Assert(kr >= 0);
        Assert(kr <= nbins);
        if (kr == nbins) --kr;  // This is rare, but can happen with numerical differences
                                // between the math for log and for non-log checks.
        Assert(kr < nbins);

        int ku = int(floor((u-minu)/ubinsize));
        if (ku >= nubins) {
            // Rounding error can allow this.
            XAssert((u-minu)/ubinsize - ku < 1.e-10);
            Assert(ku==nubins);
            --ku;
        }
        Assert(ku >= 0);
        Assert(ku < nubins);

        int kv = int(floor((v-minv)/vbinsize));

        if (kv >= nvbins) {
            // Rounding error can allow this.
            XAssert((v-minv)/vbinsize - kv < 1.e-10);
            Assert(kv==nvbins);
            --kv;
        }
        Assert(kv >= 0);
        Assert(kv < nvbins);

        // Now account for negative v
        if (!metric.CCW(c1.getPos(), c2.getPos(), c3.getPos())) {
            v = -v;
            kv = nvbins - kv - 1;
        } else {
            kv += nvbins;
        }

        Assert(kv >= 0);
        Assert(kv < nvbins * 2);

        xdbg<<"d1,d2,d3 = "<<d1<<", "<<d2<<", "<<d3<<std::endl;
        xdbg<<"r,u,v = "<<d2<<", "<<u<<", "<<v<<std::endl;
        xdbg<<"kr,ku,kv = "<<kr<<", "<<ku<<", "<<kv<<std::endl;
        index = (kr * nubins + ku) * nvbins * 2 + kv;
        Assert(index >= 0);
        Assert(index < ntot);
        // Just to make extra sure we don't get seg faults (since the above
        // asserts aren't active in normal operations), do a real check that
        // index is in the allowed range.
        if (index < 0 || index >= ntot) {
            return false;
        }
        // Finish the other log(d) values.
        logd1 = log(d1);
        logd3 = log(d3);
        return true;
    }

    static double oneBinLessThan(double r, double binsize)
    { return (1.-binsize)*r; }
};

template <>
struct BinTypeHelper<LogSAS>
{
    enum { sort_d123 = false, swap_23 = true };

    static int calculateNTot(int nbins, int nphibins, int )
    { return nbins * nbins * nphibins; }

    static bool tooSmallS2(double s2, double halfminsep, double minphi, double )
    {
        // When still doing process12, if the s2 cell is smaller than the minimum
        // possible triangle side length, then we can stop early.
        // 2s < sin(min_phi) * minsep < min_phi * minsep
        return (s2 == 0. || s2 < halfminsep * minphi);
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

    // If the user has set a minphi > 0, then we may be able to stop for that.
    template <int O>
    static bool noAllowedAngles(double rsq, double s1ps2, double s1, double s2,
                                double halfminsep,
                                double minphi, double , double maxphi, double ,
                                double mincosphi, double , double maxcosphi, double )
    {
        xdbg<<"Check noAllowedAngles\n";
        // Maximum phi/2 is for right triangle with legs d and s2, and hypotenuse r-s1.
        // d^2 = (r-s1)^2 + s2^2
        //     = r^2 - 2 r s1 + s2^2
        // Let alpha be the opening angle for this triangle, so phi = 2 alpha
        // cos(alpha) = d / r-s1
        // sin(alpha) = s2 / r-s1
        // cos(phi) = cos(alpha)^2 - sin(alpha)^2
        //          = 1 - 2 (s2/(r-s1))^2
        // Note: if not ordered, we still might be able to do this check if 2*s2 < minsep,
        // since then the only allowed orientation will be with d1 fully in c2.
        if (maxcosphi < 1 && (O || s2 < halfminsep) && (SQR(s1) < rsq)) {
            double h = sqrt(rsq) - s1;  // h = r-s1
            double cosphi = 1. - 2*SQR(s2/h);
            if (cosphi > maxcosphi) {
                xdbg<<"noAllowedAngles: "<<sqrt(rsq)<<"  "<<s1ps2<<"  "<<s1<<"  "<<s2<<std::endl;
                xdbg<<"cosphi = "<<cosphi<<std::endl;
                xdbg<<"maxcosphi = "<<maxcosphi<<std::endl;
                return true;
            }
        }
        return false;
    }

    // Once we have all the distances, see if it's possible to stop
    // For this BinType, if return value is false, d1,d2,d3,cosphi are set on output.
    template <int O, int M, int C>
    static bool stop111(
        double d1sq, double d2sq, double d3sq,
        double s1, double s2, double s3,
        const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
        const MetricHelper<M,0>& metric,
        double& d1, double& d2, double& d3, double& phi, double& cosphi,
        double minsep, double minsepsq, double maxsep, double maxsepsq,
        double minphi, double , double maxphi, double ,
        double mincosphi, double mincosphisq, double maxcosphi, double maxcosphisq)
    {
        xdbg<<"Stop111: "<<std::sqrt(d1sq)<<"  "<<std::sqrt(d2sq)<<"  "<<std::sqrt(d3sq)<<std::endl;
        xdbg<<"sizes = "<<s1<<"  "<<s2<<"  "<<s3<<std::endl;
        xdbg<<"sep range = "<<minsep<<"  "<<maxsep<<std::endl;
        xdbg<<"phi range = "<<minphi<<"  "<<maxphi<<std::endl;

        // If all possible triangles will have either d2 or d3 < minsep, then abort the recursion.
        if (d2sq < minsepsq && s1+s3 < minsep && (s1+s3 == 0. || d2sq < SQR(minsep - s1-s3))) {
            xdbg<<"d2 cannot be as large as minsep\n";
            return true;
        }
        if (d3sq < minsepsq && s1+s2 < minsep && (s1+s2 == 0. || d3sq < SQR(minsep - s1-s2))) {
            xdbg<<"d3 cannot be as large as minsep\n";
            return true;
        }

        // Similarly, we can abort if all possible triangles will have d2 or d3 > maxsep.
        if (d2sq >= maxsepsq && (s1+s3 == 0. || d2sq >= SQR(maxsep + s1+s3))) {
            xdbg<<"d2 cannot be as small as maxsep\n";
            return true;
        }
        if (d3sq >= maxsepsq && (s1+s2 == 0. || d3sq >= SQR(maxsep + s1+s2))) {
            xdbg<<"d3 cannot be as small as maxsep\n";
            return true;
        }

        // Stop if any side is exactly 0 and elements are leaves
        // (This is unusual, but we want to make sure to stop if it happens.)
        if (s2==0 && s3==0 && d1sq == 0) return true;
        if (s1==0 && s3==0 && d2sq == 0) return true;
        if (s1==0 && s2==0 && d3sq == 0) return true;

        d3 = sqrt(d3sq);
        if (s1 + s2 >= d3) {
            xdbg<<"s1,s2 are bigger than d3, continue splitting.\n";
            return false;
        }
        d2 = sqrt(d2sq);
        if (s1 + s3 >= d2) {
            xdbg<<"s1,s3 are bigger than d2, continue splitting.\n";
            return false;
        }

        cosphi = metric.calculateCosPhi(c1,c2,c3,d1sq,d2sq,d3sq,d1,d2,d3);

        // If we are not swapping 2,3, stop if orientation cannot be counter-clockwise.
        if (O > 1 &&
            !metric.CCW(c1.getPos(), c3.getPos(), c2.getPos())) {
            // For skinny triangles, be careful that the points can't flip to the other side.
            // This is similar to the calculation below.  We effecively check that cosphi can't
            // increase to 1.
            // First check if either side on its own can cause a flip of orientation.
            double sindphi2=0., sindphi3=0.;
            double cosdphi2sq=0., cosdphi3sq=0.;
            if (s1+s2 > 0) {
                sindphi2 = (s1+s2)/d3;
                cosdphi2sq = 1-SQR(sindphi2);
                if (cosdphi2sq < SQR(cosphi)) {
                    xdbg<<"CW but dphi2 could be larger than phi\n";
                    return false;
                }
            }
            if (s1+s3 > 0) {
                sindphi3 = (s1+s3)/d2;
                cosdphi3sq = 1-SQR(sindphi3);
                if (cosdphi3sq < SQR(cosphi)) {
                    xdbg<<"CW but dphi3 could be larger than phi\n";
                    return false;
                }
            }
            if (sindphi2 > 0 && sindphi3 > 0) {
                // Add them together.
                double cosdphi2 = sqrt(cosdphi2sq);
                double cosdphi3 = sqrt(cosdphi3sq);
                double cosdphi = cosdphi2 * cosdphi3 - sindphi3 * sindphi2;
                if (cosdphi < abs(cosphi)) {
                    xdbg<<"CW but dphi could be larger than phi\n";
                    return false;
                }
            }
            xdbg<<"triangle is wrong orientation\n";
            return true;
        }

        // If the user sets minphi > 0, then we can abort if no possible triangle can have
        // phi as large as this.
        // The most phi can increase from the current phi is dphi2 + dphi3, where
        // sin(dphi2) = (s1+s2)/d3
        // sin(dphi3) = (s1+s3)/d2
        double s23sq = SQR(s2+s3);
        if (minphi > 0 && cosphi > maxcosphi) {
            // First a quick check for when s1+s3 is rather large.
            if (s23sq >= d1sq && (d2sq + d3sq - s23sq) > (2*d2*d3) * maxcosphi) {
                xdbg<<"s2+s3 > d1 on their own imply small enough phi, continue splitting.\n";
                return false;
            }
            // Start with the current value of cosphi, and adjust by dphi2, dphi3
            double cosphimax = cosphi;
            if (s1+s2 > 0) {
                double sindphi2 = (s1+s2)/d3;
                double cosdphi2sq = 1-SQR(sindphi2);
                if (cosdphi2sq < maxcosphisq) return false;
                double cosdphi2 = sqrt(cosdphi2sq);
                if (s1+s3 > 0) {
                    double sindphi3 = (s1+s3)/d2;
                    double cosdphi3sq = 1-SQR(sindphi3);
                    if (cosdphi3sq < maxcosphisq) return false;
                    double cosdphi3 = sqrt(cosdphi3sq);

                    double cosdphi = cosdphi2 * cosdphi3 - sindphi3 * sindphi2;
                    if (cosdphi < maxcosphi) return false;
                    double sindphi = sindphi2 * cosdphi3 + sindphi3 * cosdphi2;
                    double sinphi = sqrt(1-cosphi*cosphi);
                    cosphimax = cosphi * cosdphi - sinphi * sindphi;
                } else {
                    double sinphi = sqrt(1-cosphi*cosphi);
                    cosphimax = cosphi * cosdphi2 - sinphi * sindphi2;
                }
            } else if (s1+s3 > 0) {
                double sindphi3 = (s1+s3)/d2;
                double cosdphi3sq = 1-SQR(sindphi3);
                if (cosdphi3sq < maxcosphisq) return false;
                double cosdphi3 = sqrt(cosdphi3sq);
                double sinphi = sqrt(1-cosphi*cosphi);
                cosphimax = cosphi * cosdphi3 - sinphi * sindphi3;
            }

            // phimax is the largest possible phi we can possibly get.
            // If it's still less than minphi then stop.
            // i.e. cos(phimax) > cos(minphi) = maxcosphi
            if (cosphimax > maxcosphi) {
                xdbg<<"phi cannot be as large as minphi\n";
                return true;
            }
        }

        if (s23sq >= d1sq) {
            xdbg<<"s2,s3 are bigger than d1, continue splitting.\n";
            return false;
        }

        // If the user sets maxphi < pi, then we can abort if no possible triangle can have
        // phi as small as this.
        if (maxphi < M_PI && cosphi < mincosphi) {
            // Start with the current value of cosphi, and adjust by dphi2, dphi3
            double cosphimin = cosphi;
            if (s1+s2 > 0) {
                double sindphi2 = (s1+s2)/d3;
                double cosdphi2sq = 1-SQR(sindphi2);
                // Note mincosphisq is really sgn(mincosphi) mincosphi^2
                // So this test will only pass if dphi2 > Pi-phi_max.
                // i.e. when dphi2 is large enough on its own to get phi < phi_max.
                if (cosdphi2sq < -mincosphisq) return false;
                double cosdphi2 = sqrt(cosdphi2sq);
                if (s1+s3 > 0) {
                    double sindphi3 = (s1+s3)/d2;
                    double cosdphi3sq = 1-SQR(sindphi3);
                    if (cosdphi3sq < -mincosphisq) return false;
                    double cosdphi3 = sqrt(cosdphi3sq);

                    double cosdphi = cosdphi2 * cosdphi3 - sindphi3 * sindphi2;
                    if (cosdphi < -mincosphi) return false;
                    double sindphi = sindphi2 * cosdphi3 + sindphi3 * cosdphi2;
                    double sinphi = sqrt(1-cosphi*cosphi);
                    cosphimin = cosphi * cosdphi + sinphi * sindphi;
                } else {
                    double sinphi = sqrt(1-cosphi*cosphi);
                    cosphimin = cosphi * cosdphi2 + sinphi * sindphi2;
                }
            } else if (s1+s3 > 0) {
                double sindphi3 = (s1+s3)/d2;
                double cosdphi3sq = 1-SQR(sindphi3);
                if (cosdphi3sq < maxcosphisq) return false;
                double cosdphi3 = sqrt(cosdphi3sq);
                double sinphi = sqrt(1-cosphi*cosphi);
                cosphimin = cosphi * cosdphi3 + sinphi * sindphi3;
            }

            // phimin is the smallest possible phi we can possibly get.
            // If it's still more than maxphi, then stop.
            // i.e. cos(phimin) < cos(maxphi) = mincosphi
            if (cosphimin < mincosphi) {
                xdbg<<"phi cannot be as small as maxphi\n";
                return true;
            }
        }

        return false;
    }

    // If return value is false, split1, split2, split3 will be set on output.
    // If return value is true, d1, d2, d3, cosphi will be set on output.
    // (For this BinType, d2,d3,cosphi are already set coming in.)
    static bool singleBin(double d1sq, double d2sq, double d3sq,
                          double s1, double s2, double s3,
                          double b, double a, double bphi, double ,
                          double bsq, double asq, double bphisq, double ,
                          bool& split1, bool& split2, bool& split3,
                          double& d1, double& d2, double& d3,
                          double& phi, double& cosphi)
    {
        xdbg<<"singleBin: "<<sqrt(d1sq)<<"  "<<d2<<"  "<<d3<<std::endl;
        // First decide whether to split c3

        // There are a few places we do a calculation akin to the splitfactor thing for 2pt.
        // That one was determined empirically to optimize the running time for a particular
        // (albeit intended to be fairly typical) use case.  Similarly, this factor was found
        // empirically on a particular (GGG) use case with a reasonable choice of separations
        // and binning.
        // TODO: Recalculate this for BinSAS.
        const double splitfactor = 0.7;

        // These are set correctly before they are used.
        double s1ps2=s2, s1ps3=s3;
        bool d2split=false, d3split=false;

        // Same logic for phi bin slop as angle slop.
        bphi = std::min(bphi, a);

        if (s1 > 0) {
            split1 = (
                // Check if either d2 or d3 needs a split
                // This is the same as the normal 2pt splitting check.
                (s1 > d2 * b) ||
                (s1 > d3 * b) ||
                (((s1ps2=s1+s2) > d3 * b) && (d2split=true, s1 >= s2)) ||
                (((s1ps3=s1+s3) > d2 * b) && (d3split=true, s1 >= s3)) ||

                // Check if phi binning needs a split
                // phi_max when d2 -> d2-s1, d3 -> d3-s1
                // phi_max - phi ~= (1/sinphi) d(cosphi/ds1) s1
                // d(cosphi/ds1) = (1+cosphi) (d2+d3)/d2d3
                // So the threshold for splitting is
                // (1+cosphi)/sinphi (d2+d3)/d2d3 s1 > bphi
                ((1.+cosphi) * (d2+d3)/(sqrt(1-SQR(cosphi))*d2*d3) * s1 > bphi));
        } else {
            split1 = false;
        }
        xdbg<<"split1 = "<<split1<<std::endl;

        if (split1) {
            // If splitting c1, then usually also split c2 and c3.
            split2 = s2 > splitfactor * s1;
            split3 = s3 > splitfactor * s1;
            xdbg<<"split12 = "<<split1<<"  "<<split2<<std::endl;
            return false;
        } else if (s2 > 0 || s3 > 0) {
            xdbg<<"Don't split 1\n";
            // Now figure out if c1 or c2 needs to be split.

            d1 = sqrt(d1sq);
            double ad1 = a*d1;
            split2 = (s2 > 0.) && (
                // Apply the d2split that we saved from above.  If we didn't split c1, split c2.
                // Note: if s1 was 0, then still need to check here.
                d2split ||
                (s1==0. && s2 > d3 * b) ||

                // Split for angle slop
                (s2 > ad1) ||
                (s2+s3 > ad1 && s2 >= s3) ||

                // Split c2 if the maximum dphi from pivoting d3 is > bphi
                // (s1+s2)/d3 > bphi
                (s1ps2 > bphi * d3));
            xdbg<<"split1 = "<<split1<<std::endl;

            split3 = (s3 > 0.) && (
                // Likewise for c3
                d3split ||
                (s1==0. && s3 > d2 * b) ||
                (s3 > ad1) ||
                (s2+s3 > ad1 && s3 >= s2) ||
                (s1ps3 > bphi * d2));
            xdbg<<"split2 = "<<split2<<std::endl;

            if (split2 || split3) {
                // If splitting either one, also do the other if it's close.
                // Because we were so aggressive in splitting c2,c3 above during the c1 splits,
                // it turns out that here we usually only want to split one, not both.
                // The above only entails a split if it's the larger one of s2,s3.
                split2 = split2 || s2 >= s3;
                split3 = split3 || s3 >= s2;
                return false;
            } else {
                xdbg<<"Don't split 1 or 2\n";
                return true;
            }
        } else {
            // s1==s2==0 and not splitting s3.
            // Just calculate d1, which we haven't done yet.
            d1 = sqrt(d1sq);
            xdbg<<"Don't split\n";
            return true;
        }
    }

    // This BinType finally sets phi here.
    template <int O, int M, int C>
    static bool isTriangleInRange(const BaseCell<C>& c1, const BaseCell<C>& c2,
                                  const BaseCell<C>& c3,
                                  const MetricHelper<M,0>& metric,
                                  double d1sq, double d2sq, double d3sq,
                                  double d1, double d2, double d3, double& phi, double& cosphi,
                                  double logminsep,
                                  double minsep, double maxsep, double binsize, int nbins,
                                  double minphi, double maxphi, double phibinsize, int nphibins,
                                  double , double , double , int ,
                                  double& logd1, double& logd2, double& logd3,
                                  int ntot, int& index)
    {
        // Make sure all the quantities we thought should be set have been.
        xdbg<<"isTriangleInRange: "<<d1<<" "<<d2<<" "<<d3<<" "<<cosphi<<std::endl;
        Assert(d1 > 0.);
        Assert(d2 > 0.);
        Assert(d3 > 0.);

        if (cosphi > -1 && cosphi < 1) phi = std::acos(cosphi);
        else if (cosphi <= -1) phi = M_PI;
        else if (cosphi >= 1) phi = 0.;
        xdbg<<"phi = "<<phi<<std::endl;
        Assert(phi >= 0.);
        Assert(phi <= M_PI);

        if (d2 < minsep || d2 >= maxsep) {
            xdbg<<"d2 not in minsep .. maxsep\n";
            return false;
        }

        if (d3 < minsep || d3 >= maxsep) {
            xdbg<<"d3 not in minsep .. maxsep\n";
            return false;
        }

        if (O > 1 &&
            !metric.CCW(c1.getPos(), c3.getPos(), c2.getPos())) {
            xdbg<<"Triangle is not CCW.\n";
            return false;
        }

        XAssert(metric.CCW(c1.getPos(), c3.getPos(), c2.getPos()));

        if (phi < minphi || phi >= maxphi) {
            xdbg<<"phi not in minphi .. maxphi\n";
            return false;
        }

        logd2 = log(d2);
        logd3 = log(d3);
        xdbg<<"            logr2 = "<<logd2<<std::endl;
        xdbg<<"            logr3 = "<<logd3<<std::endl;
        xdbg<<"            phi = "<<phi<<std::endl;

        int kr2 = int(floor((logd2-logminsep)/binsize));
        int kr3 = int(floor((logd3-logminsep)/binsize));
        Assert(kr2 >= 0);
        Assert(kr3 <= nbins);
        if (kr2 == nbins) --kr2;  // This is rare, but can happen with numerical differences
                                  // between the math for log and for non-log checks.
        Assert(kr2 < nbins);

        Assert(kr3 >= 0);
        Assert(kr3 <= nbins);
        if (kr3 == nbins) --kr3;
        Assert(kr3 < nbins);

        int kphi = int(floor((phi-minphi)/phibinsize));
        if (kphi >= nphibins) {
            // Rounding error can allow this.
            XAssert((phi-minphi)/phibinsize - kphi < 1.e-10);
            Assert(kphi==nphibins);
            --kphi;
        }
        Assert(kphi >= 0);
        Assert(kphi < nphibins);

        xdbg<<"d1,d2,d3,phi = "<<d1<<", "<<d2<<", "<<d3<<",  "<<phi<<std::endl;
        index = (kr2 * nbins + kr3) * nphibins + kphi;
        xdbg<<"kr2,kr3,kphi = "<<kr2<<", "<<kr3<<", "<<kphi<<":  "<<index<<std::endl;
        Assert(index >= 0);
        Assert(index < ntot);
        // Just to make extra sure we don't get seg faults (since the above
        // asserts aren't active in normal operations), do a real check that
        // index is in the allowed range.
        if (index < 0 || index >= ntot) {
            return false;
        }
        // Also calculate logd1 for the meanlogd1 output.
        logd1 = log(d1);
        return true;
    }

    static double oneBinLessThan(double r, double binsize)
    { return (1.-binsize)*r; }
};

template <>
struct BinTypeHelper<LogMultipole>
{
    // Note: I didn't try too hard to optimize the functions here, since most of them are
    // only used when DIRECT_MULTPOLE is enabled, which is really just for debugging.

    enum { sort_d123 = false, swap_23 = false };

    static int calculateNTot(int nbins, int maxn, int )
    { return nbins * nbins * (2*maxn+1); }

    static bool tooSmallS2(double s2, double halfminsep, double , double )
    { return (s2 == 0.); }

    static bool tooSmallDist(double rsq, double s1ps2, double minsep, double minsepsq)
    { return rsq < minsepsq && s1ps2 < minsep && rsq < SQR(minsep - s1ps2); }
    static bool tooLargeDist(double rsq, double s1ps2, double maxsep, double maxsepsq)
    { return rsq >= maxsepsq && rsq >= SQR(maxsep + s1ps2); }

    template <int O>
    static bool noAllowedAngles(double rsq, double s1ps2, double s1, double s2,
                                double halfminsep,
                                double , double , double , double ,
                                double , double , double , double )
    { return false; }

    // Once we have all the distances, see if it's possible to stop
    template <int O, int M, int C>
    static bool stop111(
        double d1sq, double d2sq, double d3sq,
        double s1, double s2, double s3,
        const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
        const MetricHelper<M,0>& metric,
        double& , double& , double& , double& , double& ,
        double minsep, double minsepsq, double maxsep, double maxsepsq,
        double , double , double , double ,
        double , double , double , double )
    {
        xdbg<<"Stop111: "<<std::sqrt(d1sq)<<"  "<<std::sqrt(d2sq)<<"  "<<std::sqrt(d3sq)<<std::endl;
        xdbg<<"sizes = "<<s1<<"  "<<s2<<"  "<<s3<<std::endl;
        xdbg<<"sep range = "<<minsep<<"  "<<maxsep<<std::endl;

        // If all possible triangles will have either d2 or d3 < minsep, then abort the recursion.
        if (d2sq < minsepsq && s1+s3 < minsep && (s1+s3 == 0. || d2sq < SQR(minsep - s1-s3))) {
            xdbg<<"d2 cannot be as large as minsep\n";
            return true;
        }
        if (d3sq < minsepsq && s1+s2 < minsep && (s1+s2 == 0. || d3sq < SQR(minsep - s1-s2))) {
            xdbg<<"d3 cannot be as large as minsep\n";
            return true;
        }

        // Similarly, we can abort if all possible triangles will have d2 or d3 > maxsep.
        if (d2sq >= maxsepsq && (s1+s3 == 0. || d2sq >= SQR(maxsep + s1+s3))) {
            xdbg<<"d2 cannot be as small as maxsep\n";
            return true;
        }
        if (d3sq >= maxsepsq && (s1+s2 == 0. || d3sq >= SQR(maxsep + s1+s2))) {
            xdbg<<"d3 cannot be as small as maxsep\n";
            return true;
        }

        // Stop if any side is exactly 0 and elements are leaves
        // (This is unusual, but we want to make sure to stop if it happens.)
        if (s2==0 && s3==0 && d1sq == 0) return true;
        if (s1==0 && s3==0 && d2sq == 0) return true;
        if (s1==0 && s2==0 && d3sq == 0) return true;

        return false;
    }

    // If return value is false, split1, split2, split3 will be set on output.
    static bool singleBin(double d1sq, double d2sq, double d3sq,
                          double s1, double s2, double s3,
                          double b, double a, double , double ,
                          double bsq, double asq, double , double ,
                          bool& split1, bool& split2, bool& split3,
                          double& d1, double& d2, double& d3,
                          double& phi, double& cosphi)
    {
        xdbg<<"singleBin: "<<sqrt(d1sq)<<"  "<<d2<<"  "<<d3<<std::endl;
        const double splitfactor = 0.7;

        // These are set correctly before they are used.
        double s1ps2=s2, s1ps3=s3;
        bool d2split=false, d3split=false;

        // First decide whether to split c3
        if (s1 > 0) {
            split1 = (
                // Check if either d2 or d3 needs a split
                // This is the same as the normal 2pt splitting check.
                (SQR(s1) > d2sq * bsq) ||
                (SQR(s1) > d3sq * bsq) ||
                ((SQR(s1ps2=s1+s2) > d3sq * bsq) && (d2split=true, s1 >= s2)) ||
                ((SQR(s1ps3=s1+s3) > d2sq * bsq) && (d3split=true, s1 >= s3)));
        } else {
            split1 = false;
        }
        xdbg<<"split1 = "<<split1<<std::endl;

        if (split1) {
            // If splitting c1, then usually also split c2 and c3.
            split2 = s2 > splitfactor * s1;
            split3 = s3 > splitfactor * s1;
            xdbg<<"split12 = "<<split1<<"  "<<split2<<std::endl;
            return false;
        } else if (s2 > 0 || s3 > 0) {
            xdbg<<"Don't split 1\n";
            // Now figure out if c1 or c2 needs to be split.
            double s2sq = SQR(s2);
            split2 = (s2 > 0.) && (
                // Apply the d2split that we saved from above.  If we didn't split c1, split c2.
                // Note: if s1 was 0, then still need to check here.
                d2split ||
                (s1==0. && s2sq > d3sq * bsq) ||
                (SQR(s1+s2) > asq * d3sq) ||

                // Also, definitely split if s2 > d1
                (s2sq > d1sq));
            xdbg<<"split1 = "<<split1<<std::endl;

            double s3sq = SQR(s3);
            split3 = (s3 > 0.) && (
                d3split ||
                (s1==0. && s3sq > d2sq * bsq) ||
                (SQR(s1+s3) > asq * d2sq) ||
                (s3sq > d1sq));
            xdbg<<"split2 = "<<split2<<std::endl;

            if (split2 || split3) {
                split2 = split2 || s2 >= s3;
                split3 = split3 || s3 >= s2;
                return false;
            } else {
                xdbg<<"Don't split 1 or 2\n";
            }
        } else {
            // s1==s2==0 and not splitting s3.
            xdbg<<"Don't split\n";
        }
        d1 = sqrt(d1sq);
        d2 = sqrt(d2sq);
        d3 = sqrt(d3sq);
        return true;
    }

    template <int O, int M, int C>
    static bool isTriangleInRange(const BaseCell<C>& c1, const BaseCell<C>& c2,
                                  const BaseCell<C>& c3,
                                  const MetricHelper<M,0>& metric,
                                  double d1sq, double d2sq, double d3sq,
                                  double d1, double d2, double d3,
                                  double& sinphi, double& cosphi,
                                  double logminsep,
                                  double minsep, double maxsep, double binsize, int nbins,
                                  double , double , double , int maxn,
                                  double , double , double , int ,
                                  double& logd1, double& logd2, double& logd3,
                                  int ntot, int& index)
    {
        xdbg<<"isTriangleInRange: "<<d1<<" "<<d2<<" "<<d3<<std::endl;
        Assert(d1 > 0.);
        Assert(d2 > 0.);
        Assert(d3 > 0.);

        if (d2 < minsep || d2 >= maxsep) {
            xdbg<<"d2 not in minsep .. maxsep\n";
            return false;
        }

        if (d3 < minsep || d3 >= maxsep) {
            xdbg<<"d3 not in minsep .. maxsep\n";
            return false;
        }

        logd2 = log(d2);
        logd3 = log(d3);
        xdbg<<"            logr2 = "<<logd2<<std::endl;
        xdbg<<"            logr3 = "<<logd3<<std::endl;

        int kr2 = int(floor((logd2-logminsep)/binsize));
        int kr3 = int(floor((logd3-logminsep)/binsize));
        Assert(kr2 >= 0);
        Assert(kr3 <= nbins);
        if (kr2 == nbins) --kr2;  // This is rare, but can happen with numerical differences
                                  // between the math for log and for non-log checks.
        Assert(kr2 < nbins);

        Assert(kr3 >= 0);
        Assert(kr3 <= nbins);
        if (kr3 == nbins) --kr3;
        Assert(kr3 < nbins);

        // Calculate cosphi, sinphi for this triangle.
        // (We use the u variable for sinphi in this class.)
        std::complex<double> r3 = ProjectHelper<C>::ExpIPhi(c1.getPos(), c2.getPos(), d3);
        std::complex<double> r2 = ProjectHelper<C>::ExpIPhi(c1.getPos(), c3.getPos(), d2);
        std::complex<double> expiphi = r3 * std::conj(r2);
        cosphi = std::real(expiphi);
        sinphi = std::imag(expiphi);

        xdbg<<"d1,d2,d3 = "<<d1<<", "<<d2<<", "<<d3<<std::endl;
        // index is the index for this d2,d3 at n=0.
        index = (kr2 * nbins + kr3) * (2*maxn+1) + maxn;
        xdbg<<"kr2,kr3 = "<<kr2<<", "<<kr3<<":  "<<index<<std::endl;
        Assert(index >= 0);
        Assert(index < ntot);
        // Just to make extra sure we don't get seg faults (since the above
        // asserts aren't active in normal operations), do a real check that
        // index is in the allowed range.
        if (index < 0 || index >= ntot) {
            return false;
        }
        // Also calculate logd1 for the meanlogd1 output.
        logd1 = log(d1);
        return true;
    }

    static double oneBinLessThan(double r, double binsize)
    { return (1.-binsize)*r; }
};


#endif

