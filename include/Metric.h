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

#ifndef TreeCorr_Metric_H
#define TreeCorr_Metric_H

// The Metric enum is defined here:
#include "Metric_C.h"
#include <limits>


template <int M>
struct MetricHelper;

//
//
// Euclidean is valid for Coord == Flat, ThreeD, or Sphere
//
//

template <>
struct MetricHelper<Euclidean>
{
    ///
    //
    // Flat
    //
    ///

    static double DistSq(const Position<Flat>& p1, const Position<Flat>& p2,
                         double& s1, double& s2)
    {
        Position<Flat> r = p1-p2;
        return r.normSq();
    }
    static double Dist(const Position<Flat>& p1, const Position<Flat>& p2)
    {
        double s=0.;
        return sqrt(DistSq(p1,p2,s,s));
    }

    static bool CCW(const Position<Flat>& p1, const Position<Flat>& p2, const Position<Flat>& p3)
    {
        // If cross product r21 x r31 > 0, then the points are counter-clockwise.
        Position<Flat> r21 = p2 - p1;
        Position<Flat> r31 = p3 - p1;
        return r21.cross(r31) > 0.;
    }

    static bool TooSmallDist(const Position<Flat>& , const Position<Flat>& , double s1ps2,
                             double dsq, double minsep, double minsepsq, double minrpar)
    { return dsq < minsepsq && s1ps2 < minsep && dsq < SQR(minsep - s1ps2); }

    static bool TooLargeDist(const Position<Flat>& , const Position<Flat>& , double s1ps2,
                             double dsq, double maxsep, double maxsepsq, double maxrpar)
    { return dsq >= maxsepsq && dsq >= SQR(maxsep + s1ps2); }



    ///
    //
    // ThreeD
    // Note: Position<Sphere> *isa* Position<ThreeD>, we don't need to do any overloads for Sphere.
    //
    ///

    static double DistSq(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                         double& s1, double& s2)
    {
        Position<ThreeD> r = p1-p2;
        return r.normSq();
    }
    static double Dist(const Position<ThreeD>& p1, const Position<ThreeD>& p2)
    {
        double s=0.;
        return sqrt(DistSq(p1,p2,s,s));
    }

    static bool CCW(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                    const Position<ThreeD>& p3)
    {
        // Now it's slightly more complicated, since the points are in three dimensions.  We do
        // the same thing, computing the cross product with respect to point p1.  Then if the
        // cross product points back toward Earth, the points are viewed as counter-clockwise.
        // We check this last point by the dot product with p1.
        Position<ThreeD> r21 = p2-p1;
        Position<ThreeD> r31 = p3-p1;
        return r21.cross(r31).dot(p1) < 0.;
    }

    static bool TooSmallDist(const Position<ThreeD>& , const Position<ThreeD>& , double s1ps2,
                             double dsq, double minsep, double minsepsq, double minrpar)
    { return dsq < minsepsq && s1ps2 < minsep && dsq < SQR(minsep - s1ps2); }

    static bool TooLargeDist(const Position<ThreeD>& , const Position<ThreeD>& , double s1ps2,
                             double dsq, double maxsep, double maxsepsq, double maxrpar)
    { return dsq >= maxsepsq && dsq >= SQR(maxsep + s1ps2); }

};

//
//
// Perp is only valid for Coord == ThreeD
//
//

template <>
struct MetricHelper<Perp>
{
    static double DistSq(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                         double& s1, double& s2)
    {
        // r_perp^2 + r_parallel^2 = d^2
        Position<ThreeD> r = p1-p2;
        double dsq = r.normSq();
        double r1sq = p1.normSq();
        double r2sq = p2.normSq();
        // r_parallel^2 = (r1-r2)^2 = r1^2 + r2^2 - 2r1r2
        // Numerically more stable value if rpar << r1,r2:
        // r_par^2 = ((r1^2 + r2^2)^2 - 4r1^2 r2^2) / (r1^2 + r2^2 + 2r1r2)
        //         = (r1^2 - r2^2)^2 / (r1^2 + r2^2 + 2r1r2)
        double rparsq = SQR(r1sq - r2sq) / (r1sq + r2sq + 2.*sqrt(r1sq*r2sq));

        // For the usual case that the angle theta between p1 and p2 is << 1, the above
        // calculation reduces to
        //
        //      r_perp ~= sqrt(r1 r2) theta
        //
        // The effect of s1 is essentially s1 = r1 dtheta, which means that its effect on r_perp
        // is approximately
        //
        //      s1' ~= sqrt(r1 r2) s/r1
        //          ~= sqrt(r2/r1) s
        //
        // So if r1 < r2, the effective sphere is larger than the nominal s1 by a factor
        // sqrt(r2/r1).  We don't really want to take another sqrt here, nevermind two sqrts
        // to get this result exactly, so we approximate this as
        //
        //      s1' ~= sqrt(1 + dr/r1) s
        //          ~= (1 + 1/4 dr^2 / r1^2) s
        //
        // We also take the conservative approach of only increasing s for the closer point, not
        // decreasing it for the larger one.

        if (r1sq < r2sq) {
            if (s1 != 0.)
                s1 *= (1. + 0.25 * (r2sq-r1sq)/r1sq);
        } else {
            if (s2 != 0.)
                s2 *= (1. + 0.25 * (r2sq-r1sq)/r1sq);
        }

        // This can end up negative with rounding errors.  So take the abs value to be safe.
        return std::abs(dsq - rparsq);
    }
    static double Dist(const Position<ThreeD>& p1, const Position<ThreeD>& p2)
    {
        double s=0.;
        return sqrt(DistSq(p1,p2,s,s));
    }

    // This is the same as Euclidean
    static bool CCW(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                    const Position<ThreeD>& p3)
    { return MetricHelper<Euclidean>::CCW(p1,p2,p3); }

    // This one is a bit subtle.  The maximum possible rp can be larger than just (rp + s1ps2).
    // The most extreme case is if the two cells are in opposite directions from Earth.
    // Of course, this won't happen too often in practice, but might as well use the
    // most conservative case here.  In this case, the cell size can serve both to
    // increase d by s1ps2 and decrease |r1-r2| by s1ps2.  So rp can become
    // rp'^2 = (d+s1ps2)^2 - (rpar-s1ps2)^2
    //       = d^2 + 2d s1ps2 + s1ps2^2 - rpar^2 + 2rpar s1ps2 - s1ps2^2
    //       = rp^2 + 2(d+rpar) s1ps2
    // rp'^2 < minsep^2
    // rp^2 + 2(d + rpar) s1ps2 < minsepsq
    // 4 (d^2 + 2d rpar + rpar^2) s1ps2^2 < (minsepsq - rp^2)^2
    static bool TooSmallDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2, double s1ps2,
                             double dsq, double minsep, double minsepsq, double minrpar)
    {
        // First a simple check that will work most of the time.
        bool easy_test = dsq >= minsepsq || s1ps2 >= minsep || dsq >= SQR(minsep - s1ps2);
        // If this is false, and there is no rpar check, then we're good.
        if (easy_test && (minrpar == -std::numeric_limits<double>::max())) return false;

        // If we need to check rpar, do that now.
        double r1 = p1.norm();
        double r2 = p2.norm();
        double rpar = r2-r1;  // Positive if p2 is in background of p1.
        // If max possible rpar < minrpar, then return true.
        if (rpar + s1ps2 < minrpar) return true;

        // Redo the easy test again.
        if (easy_test) return false;

        // Now check the subtle case.
        double rparsq = SQR(rpar);
        double d3sq = rparsq + dsq;  // The 3d distance.  Remember dsq is really rp^2.
        return (d3sq + 2.*sqrt(d3sq * rparsq) + rparsq) * SQR(2.*s1ps2) < SQR(minsepsq - dsq);
    }

    // This one is similar.  The minimum possible rp can be smaller than just (rp - s1ps2).
    // rp'^2 = (d-s1ps2)^2 - (rpar+s1ps2)^2
    //       = d^2 - 2d s1ps2 + s1ps2^2 - rpar^2 - 2rpar s1ps2 - s1ps2^2
    //       = rp^2 - 2(d+rpar) s1ps2
    // rp'^2 > maxsep^2
    // rp^2 - 2(d + rpar) s1ps2 > maxsepsq
    // 4 (d^2 + 2d rpar + rpar^2) s1ps2^2 < (rp^2 - maxsepsq)^2
    static bool TooLargeDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2, double s1ps2,
                             double dsq, double maxsep, double maxsepsq, double maxrpar)
    {
        // First a simple check that will work most of the time.
        bool easy_test = dsq < maxsepsq || dsq < SQR(maxsep + s1ps2);
        // If this is false, and there is no rpar check, then we're good.
        if (easy_test && (maxrpar == std::numeric_limits<double>::max())) return false;

        // If we need to check rpar, do that now.
        double r1 = p1.norm();
        double r2 = p2.norm();
        double rpar = r2-r1;  // Positive if p2 is in background of p1.
        // If min possible rpar > maxrpar, then return true.
        if (rpar - s1ps2 > maxrpar) return true;

        // Redo the easy test again.
        if (easy_test) return false;

        // Now check the subtle case.
        double rparsq = SQR(rpar);
        double d3sq = rparsq + dsq;  // The 3d distance.  Remember dsq is really rp^2.
        return (d3sq + 2.*sqrt(d3sq * rparsq) + rparsq) * SQR(2.*s1ps2) <= SQR(dsq - maxsepsq);
    }

};


//
//
// Lens is only valid for Coord == ThreeD
//
//

template <>
struct MetricHelper<Lens>
{
#if 1
    // The first option uses the chord distance at the distance of r1
    static double DistSq(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                         double& s1, double& s2)
    {
        // theta = angle between p1, p2
        // L/r1 = 2 sin(theta/2) = sin(theta) / cos(theta/2)
        //      = sin(theta) sqrt(2/(1+cos(theta)))
        // | p1 x p2 | = r1 r2 sin(theta)
        // p1 . p2 = r1 r2 cos(theta)
        //
        // L^2 = r1^2 |p1xp2|^2/(r1^2 r2^2) (2 / (1+(p1.p2)/(r1 r2)))
        //     = 2 * |p1xp2|^2 / r2^2 / (1 + (p1.p2)/r1r2)
        double r1 = p1.norm();
        double r2 = p2.norm();
        double costheta = p1.dot(p2) / (r1*r2);

        // The effect of s2 needs to be modified here.  Its effect on rlens is
        //      s2' = r1/r2 s2
        s2 *= r1/r2;

        double dsq = 2. * p1.cross(p2).normSq() / (r2*r2) / (1.+costheta);
        // I don't think rounding can make this negative, but just to be safe...
        return std::abs(dsq);
    }
#else
    // The second option uses the direction perpendiculat to r1
    static double DistSq(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                         double& s1, double& s2)
    {
        // theta = angle between p1, p2
        // L/r1 = tan(theta)
        // | p1 x p2 | = r1 r2 sin(theta)
        // p1 . p2 = r1 r2 cos(theta)
        // L = r1 |p1 x p2| / (p1 . p2)
        double r1sq = p1.normSq();
        double r2sq = p2.normsq();
        s2 *= sqrt(r1sq/r2sq);
        return r1sq * p1.cross(p2).normSq() / SQR(p1.dot(p2));
    }
#endif

    static double Dist(const Position<ThreeD>& p1, const Position<ThreeD>& p2)
    {
        double s=0.;
        return sqrt(DistSq(p1,p2,s,s));
    }

    // This is the same as Euclidean
    static bool CCW(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                    const Position<ThreeD>& p3)
    { return MetricHelper<Euclidean>::CCW(p1,p2,p3); }

    // We've already accounted for the way that the raw s1+s2 may not be sufficient in DistSq
    // where we update s2 according to the relative distances.  So these two functions are
    // the same as the Euclidean versions.
    static bool TooSmallDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2, double s1ps2,
                             double dsq, double minsep, double minsepsq, double minrpar)

    {
        if (dsq < minsepsq && s1ps2 < minsep && dsq < SQR(minsep - s1ps2)) return true;
        // Not too small from just d, s considerations.  Maybe need to check rpar.
        if (minrpar == -std::numeric_limits<double>::max()) return false;
        // rpar is not -inf, so need to check.
        double r1 = p1.norm();
        double r2 = p2.norm();
        double rpar = r2-r1;  // Positive if p2 is in background of p1.
        // If max possible rpar < minrpar, then return true.
        if (rpar + s1ps2 < minrpar) return true;
        else return false;
    }

    // This one is similar.  The minimum possible L if p2 is larger is L - s1ps2 * r2/r1
    static bool TooLargeDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2, double s1ps2,
                             double dsq, double maxsep, double maxsepsq, double maxrpar)
    {
        if (dsq >= maxsepsq && dsq >= SQR(maxsep + s1ps2)) return true;
        if (maxrpar == std::numeric_limits<double>::max()) return false;
        double r1 = p1.norm();
        double r2 = p2.norm();
        double rpar = r2-r1;  // Positive if p2 is in background of p1.
        // If min possible rpar > maxrpar, then return true.
        if (rpar - s1ps2 > maxrpar) return true;
        else return false;
    }

};


//
//
// Arc is only valid for Coord == Sphere
//
//

template <>
struct MetricHelper<Arc>
{
    static double DistSq(const Position<Sphere>& p1, const Position<Sphere>& p2,
                         double& s1, double& s2)
    { return SQR(Dist(p1,p2)); }

    static double Dist(const Position<Sphere>& p1, const Position<Sphere>& p2)
    {
        // theta = angle between p1, p2
        // L = 2 sin(theta/2)
        // L is the normal Euclidean distance.
        // theta is the Arc distance.
        double L = MetricHelper<Euclidean>::Dist(p1,p2);
        double theta = 2. * std::asin(L/2.);
        return theta;
    }

    // These are the same as Euclidean.
    static bool CCW(const Position<Sphere>& p1, const Position<Sphere>& p2,
                    const Position<Sphere>& p3)
    { return MetricHelper<Euclidean>::CCW(p1,p2,p3); }

    static bool TooSmallDist(const Position<Sphere>& p1, const Position<Sphere>& p2, double s1ps2,
                             double dsq, double minsep, double minsepsq, double minrpar)
    { return MetricHelper<Euclidean>::TooSmallDist(p1,p2,s1ps2,dsq,minsep,minsepsq,minrpar); }

    static bool TooLargeDist(const Position<Sphere>& p1, const Position<Sphere>& p2, double s1ps2,
                             double dsq, double maxsep, double maxsepsq, double maxrpar)
    { return MetricHelper<Euclidean>::TooLargeDist(p1,p2,s1ps2,dsq,maxsep,maxsepsq,maxrpar); }

};

#endif

