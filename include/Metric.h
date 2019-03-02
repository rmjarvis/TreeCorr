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
#include <cmath>


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

    // Some metrics keep track of whether the r_parallel distance is within some range.
    // This function checks if two cells are fully outside the range, in which case we can
    // throw them out and not recurse futher.  For Euclidean, this is trivially false.
    static bool isRParOutsideRange(const Position<Flat>& p1, const Position<Flat>& p2,
                                   double s1ps2, double minrpar, double maxrpar, double& rpar)
    { return false; }

    // When we get to a point where we think all possible pairs fall into a single bin,
    // then we also need to make sure we are fully in the range for rpar as well.  For
    // Euclidean (and others that don't use rpar), this is always true, but Rper and Rlens
    // have a check here.
    static bool isRParInsideRange(double rpar, double s1ps2, double minrpar, double maxrpar)
    { return true; }

    // The normal tests about whether a given distance is inside the binning range happen
    // in BinTypeHelper.  However, Rperp needs to do an additional check to make sure we
    // don't reject cell pairs prematurely.  For this and most metrics, these two checks
    // are trivially true (since only get here if the regular check passes).
    static bool tooSmallDist(const Position<Flat>& p1, const Position<Flat>& p2,
                             double rsq, double rpar, double s1ps2, double minsepsq)
    { return true; }

    static bool tooLargeDist(const Position<Flat>& p1, const Position<Flat>& p2,
                             double rsq, double rpar, double s1ps2, double maxsepsq)
    { return true; }


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

    static bool isRParOutsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                                   double s1ps2, double minrpar, double maxrpar, double& rpar)
    { return false; }

    static bool tooSmallDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                             double rsq, double rpar, double s1ps2, double minsepsq)
    { return true; }

    static bool tooLargeDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                             double rsq, double rpar, double s1ps2, double maxsepsq)
    { return true; }

};

//
//
// Rperp is only valid for Coord == ThreeD
//
//

// For now, this is still Rperp in the python layer, and the next one is FisherRperp.
// At version 4, Rperp will become an alias for FisherRperp, and this will be OldRperp.
template <>
struct MetricHelper<OldRperp>
{
    static double DistSq(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                         double& s1, double& s2)
    {
        // r_perp^2 + r_parallel^2 = d^2
        Position<ThreeD> r = p1-p2;
        double d3sq = r.normSq();
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
        return std::abs(d3sq - rparsq);
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

    static double calculateRPar(const Position<ThreeD>& p1, const Position<ThreeD>& p2)
    {
        double r1 = p1.norm();
        double r2 = p2.norm();
        return r2-r1;  // Positive if p2 is in background of p1.
    }

    static bool isRParOutsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                                   double s1ps2, double minrpar, double maxrpar, double& rpar)
    {
        // Quick return if no min/max rpar
        if (minrpar == -std::numeric_limits<double>::max() &&
            maxrpar == std::numeric_limits<double>::max()) return false;

        rpar = calculateRPar(p1,p2);
        if (rpar + s1ps2 < minrpar) return true;
        else if (rpar - s1ps2 > maxrpar) return true;
        else return false;
    }

    static bool isRParInsideRange(double rpar, double s1ps2, double minrpar, double maxrpar)
    {
        // Quick return if no min/max rpar
        if (minrpar == -std::numeric_limits<double>::max() &&
            maxrpar == std::numeric_limits<double>::max()) return true;
        if (rpar - s1ps2 < minrpar) return false;
        else if (rpar + s1ps2 > maxrpar) return false;
        else return true;
    }

    // This one is a bit subtle.  The maximum possible rp can be larger than just (rp + s1ps2).
    // The most extreme case is if the two cells are in nearly opposite directions from Earth.
    // Of course, this won't happen too often in practice, but might as well use the
    // most conservative case here.  In this case, the cell size can serve both to
    // increase d by s1ps2 and decrease |r1-r2| by s1ps2.  So rp can become
    // rp'^2 = (d+s1ps2)^2 - (|rpar|-s1ps2)^2
    //       = d^2 + 2d s1ps2 + s1ps2^2 - rpar^2 + 2|rpar| s1ps2 - s1ps2^2
    //       = rp^2 + 2(d+|rpar|) s1ps2
    // rp'^2 < minsep^2
    // rp^2 + 2(d + |rpar|) s1ps2 < minsepsq
    static bool tooSmallDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                             double rsq, double& rpar, double s1ps2, double minsepsq)
    {
        if (rpar == 0.) rpar = calculateRPar(p1,p2); // This might not have been calculated.
        double d3 = sqrt(SQR(rpar) + rsq);  // The 3d distance.  Remember rsq is really rp^2.
        return rsq + 2.*(d3 + std::abs(rpar)) * s1ps2 < minsepsq;
    }

    // This one is similar.  The minimum possible rp can be smaller than just (rp - s1ps2).
    // rp'^2 = (d-s1ps2)^2 - (|rpar|+s1ps2)^2
    //       = d^2 - 2d s1ps2 + s1ps2^2 - rpar^2 - 2|rpar| s1ps2 - s1ps2^2
    //       = rp^2 - 2(d+|rpar|) s1ps2
    // rp'^2 > maxsep^2
    // rp^2 - 2(d + |rpar|) s1ps2 > maxsepsq
    static bool tooLargeDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                             double rsq, double rpar, double s1ps2, double maxsepsq)
    {
        if (rpar == 0.) rpar = calculateRPar(p1,p2); // This might not have been calculated.
        double d3 = sqrt(SQR(rpar) + rsq);  // The 3d distance.  Remember rsq is really rp^2.
        return rsq - 2.*(d3 + std::abs(rpar)) * s1ps2 > maxsepsq;
    }

};

template <>
struct MetricHelper<Rperp>
{
    static double DistSq(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                         double& s1, double& s2)
    {
        // Follow Fisher et al, 1994 (MNRAS, 267, 927) to define:
        //   L = (p1 + p2)/2
        //   r = (p2 - p1)
        //   r_par = |L . r| / |L|
        //   r_perp^2 = |r|^2 - |r_par|^2
        //
        // It turns out that this simplifies to
        //
        //   r_perp = |p1 x p2| / |L|
        //
        Position<ThreeD> cross = p1.cross(p2);
        Position<ThreeD> L = (p1+p2)*0.5;
        double normLsq = L.normSq();

        double rperpsq = cross.normSq() / normLsq;

        // The maximum effect of a displacement s1 on the value of rperp is (normally)
        //
        //      delta r_perp = r2 s / |L|
        //
        // Assuming r1 < r2, this increases the impact of the cell radius s on this distance
        // metric.  We thus need to scale up the input s1 by the factor r2/|L|.
        //
        // Similarly, the effect of s2 is decreased by the factor r1/|L|.
        //
        // However, we take the conservative approach of only increasing s for the closer point,
        // not decreasing it for the farther one.
        //
        // Also note: for extreme geometries, this calculation isn't sufficient. The change
        // in L can also be important.  Hence the below tooLargeDist and tooSmallDist functions.

        double r1sq = p1.normSq();
        double r2sq = p2.normSq();
        if (r1sq < r2sq) {
            if (s1 != 0.)
                s1 *= sqrt(r2sq / normLsq);
        } else {
            if (s2 != 0.)
                s2 *= sqrt(r1sq / normLsq);
        }

        return rperpsq;
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

    static double calculateRPar(const Position<ThreeD>& p1, const Position<ThreeD>& p2)
    {
        Position<ThreeD> r = p2-p1;
        Position<ThreeD> L = (p1+p2)*0.5;
        return r.dot(L) / L.norm();
    }

    static bool isRParOutsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                                   double s1ps2, double minrpar, double maxrpar, double& rpar)
    {
        // Quick return if no min/max rpar
        if (minrpar == -std::numeric_limits<double>::max() &&
            maxrpar == std::numeric_limits<double>::max()) return false;

        rpar = calculateRPar(p1,p2);
        if (rpar + s1ps2 < minrpar) return true;
        else if (rpar - s1ps2 > maxrpar) return true;
        else return false;
    }

    static bool isRParInsideRange(double rpar, double s1ps2, double minrpar, double maxrpar)
    {
        // Quick return if no min/max rpar
        if (minrpar == -std::numeric_limits<double>::max() &&
            maxrpar == std::numeric_limits<double>::max()) return true;
        if (rpar - s1ps2 < minrpar) return false;
        else if (rpar + s1ps2 > maxrpar) return false;
        else return true;
    }

    // This one is a bit subtle.  The maximum possible rp can be larger than just (rp + s1ps2).
    // The most extreme case is if the two cells are in nearly opposite directions from Earth.
    // Of course, this won't happen too often in practice, but might as well use the
    // most conservative case here.  In this case, the cell size can serve both to
    // increase d by s1ps2 and decrease |r1-r2| by s1ps2.  So rp can become
    // rp'^2 = (d+s1ps2)^2 - (|rpar|-s1ps2)^2
    //       = d^2 + 2d s1ps2 + s1ps2^2 - rpar^2 + 2|rpar| s1ps2 - s1ps2^2
    //       = rp^2 + 2(d+|rpar|) s1ps2
    // rp'^2 < minsep^2
    // rp^2 + 2(d + |rpar|) s1ps2 < minsepsq
    static bool tooSmallDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                             double rsq, double& rpar, double s1ps2, double minsepsq)
    {
        if (rpar == 0.) rpar = calculateRPar(p1,p2); // This might not have been calculated.
        double d3 = sqrt(SQR(rpar) + rsq);  // The 3d distance.  Remember rsq is really rp^2.
        return rsq + 2.*(d3 + std::abs(rpar)) * s1ps2 < minsepsq;
    }

    // This one is similar.  The minimum possible rp can be smaller than just (rp - s1ps2).
    // rp'^2 = (d-s1ps2)^2 - (|rpar|+s1ps2)^2
    //       = d^2 - 2d s1ps2 + s1ps2^2 - rpar^2 - 2|rpar| s1ps2 - s1ps2^2
    //       = rp^2 - 2(d+|rpar|) s1ps2
    // rp'^2 > maxsep^2
    // rp^2 - 2(d + |rpar|) s1ps2 > maxsepsq
    static bool tooLargeDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                             double rsq, double rpar, double s1ps2, double maxsepsq)
    {
        if (rpar == 0.) rpar = calculateRPar(p1,p2); // This might not have been calculated.
        double d3 = sqrt(SQR(rpar) + rsq);  // The 3d distance.  Remember rsq is really rp^2.
        return rsq - 2.*(d3 + std::abs(rpar)) * s1ps2 > maxsepsq;
    }

};


//
//
// Rlens is only valid for Coord == ThreeD
//
//

template <>
struct MetricHelper<Rlens>
{
    // The distance is measured perpendicular to the p2 direction at the distance of p1.
    static double DistSq(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                         double& s1, double& s2)
    {
        // Rlens = |p1 x p2| / |p2|
        double r2sq = p2.normSq();
        double rsq = p1.cross(p2).normSq() / r2sq;

        // The effect of s2 needs to be modified here.  Its effect on Rlens is approximately
        //      s2' = r1/r2 s2
        s2 *= sqrt(p1.normSq() / r2sq);

        return rsq;
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

    static double calculateRPar(const Position<ThreeD>& p1, const Position<ThreeD>& p2)
    {
        Position<ThreeD> r = p2-p1;
        Position<ThreeD> L = (p1+p2)*0.5;
        return r.dot(L) / L.norm();
    }

    static bool isRParOutsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                                   double s1ps2, double minrpar, double maxrpar, double& rpar)
    {
        // Quick return if no min/max rpar
        if (minrpar == -std::numeric_limits<double>::max() &&
            maxrpar == std::numeric_limits<double>::max()) return false;

        rpar = calculateRPar(p1,p2);
        if (rpar + s1ps2 < minrpar) return true;
        else if (rpar - s1ps2 > maxrpar) return true;
        else return false;
    }

    static bool isRParInsideRange(double rpar, double s1ps2, double minrpar, double maxrpar)
    {
        // Quick return if no min/max rpar
        if (minrpar == -std::numeric_limits<double>::max() &&
            maxrpar == std::numeric_limits<double>::max()) return true;
        if (rpar - s1ps2 < minrpar) return false;
        else if (rpar + s1ps2 > maxrpar) return false;
        else return true;
    }

    // We've already accounted for the way that the raw s1+s2 may not be sufficient in DistSq
    // where we update s2 according to the relative distances.  So there is nothing further to
    // do here.
    static bool tooSmallDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                             double rsq, double rpar, double s1ps2, double minsepsq)
    { return true; }

    static bool tooLargeDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                             double rsq, double rpar, double s1ps2, double maxsepsq)
    { return true; }
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

    static bool isRParOutsideRange(const Position<Sphere>& p1, const Position<Sphere>& p2,
                                   double s1ps2, double minrpar, double maxrpar, double& rpar)
    { return false; }

    static bool isRParInsideRange(double rpar, double s1ps2, double minrpar, double maxrpar)
    { return true; }

    static bool tooSmallDist(const Position<Sphere>& p1, const Position<Sphere>& p2,
                             double rsq, double rpar, double s1ps2, double minsepsq)
    { return true; }

    static bool tooLargeDist(const Position<Sphere>& p1, const Position<Sphere>& p2,
                             double rsq, double rpar, double s1ps2, double maxsepsq)
    { return true; }

};

#endif

