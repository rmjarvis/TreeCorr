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
    // For each metric, we have enum values with allowed coordinate systems mapped to
    // the normal enum value for that system, but disallowed coordinates mapped to one
    // of the allowed ones, so that when we instantiate the templates, we don't get
    // compiler errors from trying to instantiate functions that don't exist.
    //
    // For Euclidean, all coordinate systems are allowed.
    enum { _Flat=Flat, _ThreeD=ThreeD, _Sphere=Sphere };

    // We always have the constructor take all arguments that someone might need.  But
    // we only save them as values in the struct if this metric will actually need them.
    MetricHelper(double minrpar=0, double maxrpar=0, double xp=0, double yp=0, double zp=0) {}

    ///
    //
    // Flat
    //
    ///

    double DistSq(const Position<Flat>& p1, const Position<Flat>& p2, double& s1, double& s2) const
    {
        Position<Flat> r = p1-p2;
        return r.normSq();
    }
    double Dist(const Position<Flat>& p1, const Position<Flat>& p2) const
    {
        double s=0.;
        return sqrt(DistSq(p1,p2,s,s));
    }

    bool CCW(const Position<Flat>& p1, const Position<Flat>& p2, const Position<Flat>& p3) const
    {
        // If cross product r21 x r31 > 0, then the points are counter-clockwise.
        Position<Flat> r21 = p2 - p1;
        Position<Flat> r31 = p3 - p1;
        return r21.cross(r31) > 0.;
    }

    // Some metrics keep track of whether the r_parallel distance is within some range.
    // This function checks if two cells are fully outside the range, in which case we can
    // throw them out and not recurse futher.  For Euclidean, this is trivially false.
    bool isRParOutsideRange(const Position<Flat>& p1, const Position<Flat>& p2,
                            double s1ps2, double& rpar) const
    { return false; }

    // When we get to a point where we think all possible pairs fall into a single bin,
    // then we also need to make sure we are fully in the range for rpar as well.  For
    // Euclidean (and others that don't use rpar), this is always true, but Rper and Rlens
    // have a check here.
    bool isRParInsideRange(const Position<Flat>& p1, const Position<Flat>& p2,
                           double s1ps2, double rpar) const
    { return true; }

    // The normal tests about whether a given distance is inside the binning range happen
    // in BinTypeHelper.  However, Rperp needs to do an additional check to make sure we
    // don't reject cell pairs prematurely.  For this and most metrics, these two checks
    // are trivially true (since only get here if the regular check passes).
    bool tooSmallDist(const Position<Flat>& p1, const Position<Flat>& p2,
                      double rsq, double rpar, double s1ps2, double minsep, double minsepsq) const
    { return true; }

    bool tooLargeDist(const Position<Flat>& p1, const Position<Flat>& p2,
                      double rsq, double rpar, double s1ps2, double maxsep, double maxsepsq) const
    { return true; }


    ///
    //
    // ThreeD
    // Note: Position<Sphere> *isa* Position<ThreeD>, we don't need to do any overloads for Sphere.
    //
    ///

    double DistSq(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                  double& s1, double& s2) const
    {
        Position<ThreeD> r = p1-p2;
        return r.normSq();
    }
    double Dist(const Position<ThreeD>& p1, const Position<ThreeD>& p2) const
    {
        double s=0.;
        return sqrt(DistSq(p1,p2,s,s));
    }

    bool CCW(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
             const Position<ThreeD>& p3) const
    {
        // Now it's slightly more complicated, since the points are in three dimensions.  We do
        // the same thing, computing the cross product with respect to point p1.  Then if the
        // cross product points back toward Earth, the points are viewed as counter-clockwise.
        // We check this last point by the dot product with p1.
        Position<ThreeD> r21 = p2-p1;
        Position<ThreeD> r31 = p3-p1;
        return r21.cross(r31).dot(p1) < 0.;
    }

    bool isRParOutsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                            double s1ps2, double& rpar) const
    { return false; }

    bool isRParInsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                           double s1ps2, double rpar) const
    { return true; }

    bool tooSmallDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                      double rsq, double rpar, double s1ps2, double minsep, double minsepsq) const
    { return true; }

    bool tooLargeDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                      double rsq, double rpar, double s1ps2, double maxsep, double maxsepsq) const
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
    enum { _Flat=ThreeD, _ThreeD=ThreeD, _Sphere=ThreeD };

    double minrpar, maxrpar;

    MetricHelper(double _minrpar, double _maxrpar, double xp=0, double yp=0, double zp=0) :
        minrpar(_minrpar), maxrpar(_maxrpar) {}

    double DistSq(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                  double& s1, double& s2) const
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
    double Dist(const Position<ThreeD>& p1, const Position<ThreeD>& p2) const
    {
        double s=0.;
        return sqrt(DistSq(p1,p2,s,s));
    }

    // This is the same as Euclidean
    bool CCW(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
             const Position<ThreeD>& p3) const
    {
        Position<ThreeD> r21 = p2-p1;
        Position<ThreeD> r31 = p3-p1;
        return r21.cross(r31).dot(p1) < 0.;
    }

    double calculateRPar(const Position<ThreeD>& p1, const Position<ThreeD>& p2) const
    {
        double r1 = p1.norm();
        double r2 = p2.norm();
        return r2-r1;  // Positive if p2 is in background of p1.
    }

    bool isRParOutsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                            double s1ps2, double& rpar) const
    {
        // Quick return if no min/max rpar
        if (minrpar == -std::numeric_limits<double>::max() &&
            maxrpar == std::numeric_limits<double>::max()) return false;

        rpar = calculateRPar(p1,p2);
        if (rpar + s1ps2 < minrpar) return true;
        else if (rpar - s1ps2 > maxrpar) return true;
        else return false;
    }

    bool isRParInsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                           double s1ps2, double rpar) const
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
    bool tooSmallDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                      double rsq, double& rpar, double s1ps2, double minsep, double minsepsq) const
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
    bool tooLargeDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                      double rsq, double rpar, double s1ps2, double maxsep, double maxsepsq) const
    {
        if (rpar == 0.) rpar = calculateRPar(p1,p2); // This might not have been calculated.
        double d3 = sqrt(SQR(rpar) + rsq);  // The 3d distance.  Remember rsq is really rp^2.
        return rsq - 2.*(d3 + std::abs(rpar)) * s1ps2 > maxsepsq;
    }

};

template <>
struct MetricHelper<Rperp>
{
    enum { _Flat=ThreeD, _ThreeD=ThreeD, _Sphere=ThreeD };

    double minrpar, maxrpar;
    mutable double _normLsq;  // Variable that can be saved and used across multiple functions.

    MetricHelper(double _minrpar, double _maxrpar, double xp=0, double yp=0, double zp=0) :
        minrpar(_minrpar), maxrpar(_maxrpar) {}

    double DistSq(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                  double& s1, double& s2) const
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
        // Save this so we can use it later in tooLargeDist.
        _normLsq = normLsq;

        // It's extremely unlikely to happen, but if p1 = -p2, then L = 0, and the above
        // math fails.  In this case, r_par = 0 and r_perp = |r| = 2|p1|.
        double rperpsq = normLsq > 0. ? cross.normSq() / normLsq : 4.*p1.normSq();

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
        if (r2sq > normLsq && s1 != 0) {
            s1 *= sqrt(r2sq / normLsq);
        }
        if (r1sq > normLsq && s2 != 0) {
            s2 *= sqrt(r1sq / normLsq);
        }

        return rperpsq;
    }
    double Dist(const Position<ThreeD>& p1, const Position<ThreeD>& p2) const
    {
        double s=0.;
        return sqrt(DistSq(p1,p2,s,s));
    }

    // This is the same as Euclidean
    bool CCW(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
             const Position<ThreeD>& p3) const
    {
        Position<ThreeD> r21 = p2-p1;
        Position<ThreeD> r31 = p3-p1;
        return r21.cross(r31).dot(p1) < 0.;
    }

    double calculateRPar(const Position<ThreeD>& p1, const Position<ThreeD>& p2) const
    {
        Position<ThreeD> r = p2-p1;
        Position<ThreeD> L = (p1+p2)*0.5;
        return r.dot(L) / L.norm();
    }

    bool isRParOutsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                            double s1ps2, double& rpar) const
    {
        // Quick return if no min/max rpar
        if (minrpar == -std::numeric_limits<double>::max() &&
            maxrpar == std::numeric_limits<double>::max()) return false;

        rpar = calculateRPar(p1,p2);
        if (rpar + s1ps2 < minrpar) return true;
        else if (rpar - s1ps2 > maxrpar) return true;
        else return false;
    }

    bool isRParInsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                           double s1ps2, double rpar) const
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
    // increase |p1xp2| and decrease |L|. The effect is largest if all of s1ps2 is on the closer
    // of the two points, so assume |p2| > |p1| and all the s is on p1.
    //
    // rp' = |(p1+s) x p2| / (|L| - s/2)
    //     = (|p1xp2| + s|p2|) / (|L| (1-s/2|L|))
    //     = (rp + s') / (1 - s/2|L|)
    // where s' is the adjusted s we already did in DistSq.
    bool tooSmallDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                      double rsq, double& rpar, double s1ps2, double minsep, double minsepsq) const
    {
        // If on the same side of the origin, then no additional check is required.
        if (rsq < _normLsq) return true;
        // (rp + s) / (1 - s/2L) < minsep
        // (rp + s) < minsep (1-s/2L)
        // rp < minsep (1-s/2L) - s
        if (SQR(s1ps2) > 4.*_normLsq) return false;
        double twoL = 2.*sqrt(_normLsq);
        return rsq < SQR(minsep * (1 - s1ps2/twoL) - s1ps2);
    }

    // Likewise, we need to account for the possibility of L decreasing here.
    bool tooLargeDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                      double rsq, double rpar, double s1ps2, double maxsep, double maxsepsq) const
    {
        // If on the same side of the origin, then no additional check is required.
        if (rsq < _normLsq) return true;
        // (rp - s) / (1 + s/2L) > maxsep
        // (rp - s) > maxsep (1+s/2L)
        // rp > maxsep (1+s/2L) + s
        double twoL = 2.*sqrt(_normLsq);
        return rsq > SQR(maxsep * (1 + s1ps2/twoL) + s1ps2);
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
    enum { _Flat=ThreeD, _ThreeD=ThreeD, _Sphere=ThreeD };

    double minrpar, maxrpar;

    MetricHelper(double _minrpar, double _maxrpar, double xp=0, double yp=0, double zp=0) :
        minrpar(_minrpar), maxrpar(_maxrpar) {}

    // The distance is measured perpendicular to the p2 direction at the distance of p1.
    double DistSq(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                  double& s1, double& s2) const
    {
        // Rlens = |p1 x p2| / |p2|
        double r2sq = p2.normSq();
        double rsq = p1.cross(p2).normSq() / r2sq;

        // The effect of s2 needs to be modified here.  Its effect on Rlens is approximately
        //      s2' = r1/r2 s2
        s2 *= sqrt(p1.normSq() / r2sq);

        return rsq;
    }

    double Dist(const Position<ThreeD>& p1, const Position<ThreeD>& p2) const
    {
        double s=0.;
        return sqrt(DistSq(p1,p2,s,s));
    }

    // This is the same as Euclidean
    bool CCW(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
             const Position<ThreeD>& p3) const
    {
        Position<ThreeD> r21 = p2-p1;
        Position<ThreeD> r31 = p3-p1;
        return r21.cross(r31).dot(p1) < 0.;
    }

    double calculateRPar(const Position<ThreeD>& p1, const Position<ThreeD>& p2) const
    {
        Position<ThreeD> r = p2-p1;
        Position<ThreeD> L = (p1+p2)*0.5;
        return r.dot(L) / L.norm();
    }

    bool isRParOutsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                            double s1ps2, double& rpar) const
    {
        // Quick return if no min/max rpar
        if (minrpar == -std::numeric_limits<double>::max() &&
            maxrpar == std::numeric_limits<double>::max()) return false;

        rpar = calculateRPar(p1,p2);
        if (rpar + s1ps2 < minrpar) return true;
        else if (rpar - s1ps2 > maxrpar) return true;
        else return false;
    }

    bool isRParInsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                           double s1ps2, double rpar) const
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
    bool tooSmallDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                      double rsq, double rpar, double s1ps2, double minsep, double minsepsq) const
    { return true; }

    bool tooLargeDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                      double rsq, double rpar, double s1ps2, double maxsep, double maxsepsq) const
    { return true; }
};


//
//
// Arc is only valid for Coord == Sphere and Coord == ThreeD
//
//

template <>
struct MetricHelper<Arc>
{
    enum { _Flat=ThreeD, _ThreeD=ThreeD, _Sphere=Sphere };

    double minrpar, maxrpar;

    MetricHelper(double _minrpar, double _maxrpar, double xp=0, double yp=0, double zp=0) :
        minrpar(_minrpar), maxrpar(_maxrpar) {}

    double DistSq(const Position<Sphere>& p1, const Position<Sphere>& p2,
                  double& s1, double& s2) const
    { return SQR(Dist(p1,p2)); }

    double Dist(const Position<Sphere>& p1, const Position<Sphere>& p2) const
    {
        // theta = angle between p1, p2
        // L = 2 sin(theta/2)
        // L is the normal Euclidean distance.
        // theta is the Arc distance.
        Position<ThreeD> r = p1-p2;
        double L = r.norm();
        double theta = 2. * std::asin(L/2.);
        return theta;
    }

    // These are the same as Euclidean.
    bool CCW(const Position<Sphere>& p1, const Position<Sphere>& p2,
             const Position<Sphere>& p3) const
    {
        Position<ThreeD> r21 = p2-p1;
        Position<ThreeD> r31 = p3-p1;
        return r21.cross(r31).dot(p1) < 0.;
    }

    bool tooSmallDist(const Position<Sphere>& p1, const Position<Sphere>& p2,
                      double rsq, double rpar, double s1ps2, double minsep, double minsepsq) const
    { return true; }

    bool tooLargeDist(const Position<Sphere>& p1, const Position<Sphere>& p2,
                      double rsq, double rpar, double s1ps2, double maxsep, double maxsepsq) const
    { return true; }

    // For 3d coordinates, use the cross product to get sin(theta)
    double DistSq(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                  double& s1, double& s2) const
    {
        double dsq = SQR(Dist(p1,p2));
        // Also the sizes need to be converted to the effective size on the unit sphere
        s1 /= p1.norm();
        s2 /= p2.norm();
        return dsq;
    }

    double Dist(const Position<ThreeD>& p1, const Position<ThreeD>& p2) const
    {
        // sin(theta) = |p1 x p2| / |p1| |p2|
        double sintheta = p1.cross(p2).norm() / (p1.norm() * p2.norm());
        double theta = std::asin(sintheta);
        return theta;
    }

    // This is the same as Euclidean
    bool CCW(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
             const Position<ThreeD>& p3) const
    {
        Position<ThreeD> r21 = p2-p1;
        Position<ThreeD> r31 = p3-p1;
        return r21.cross(r31).dot(p1) < 0.;
    }

    double calculateRPar(const Position<ThreeD>& p1, const Position<ThreeD>& p2) const
    {
        Position<ThreeD> r = p2-p1;
        Position<ThreeD> L = (p1+p2)*0.5;
        return r.dot(L) / L.norm();
    }

    bool isRParOutsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                            double s1ps2, double& rpar) const
    {
        // Quick return if no min/max rpar
        if (minrpar == -std::numeric_limits<double>::max() &&
            maxrpar == std::numeric_limits<double>::max()) return false;

        // Remember that s1ps2 has been scaled to be the values on the unit circle.
        // So scale them back up here.
        s1ps2 *= std::max(p1.norm(), p2.norm());
        rpar = calculateRPar(p1,p2);
        if (rpar + s1ps2 < minrpar) return true;
        else if (rpar - s1ps2 > maxrpar) return true;
        else return false;
    }

    bool isRParInsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                           double s1ps2, double rpar) const
    {
        // Quick return if no min/max rpar
        if (minrpar == -std::numeric_limits<double>::max() &&
            maxrpar == std::numeric_limits<double>::max()) return true;
        s1ps2 *= std::max(p1.norm(), p2.norm());
        if (rpar - s1ps2 < minrpar) return false;
        else if (rpar + s1ps2 > maxrpar) return false;
        else return true;
    }

    bool tooSmallDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                      double rsq, double rpar, double s1ps2, double minsep, double minsepsq) const
    { return true; }

    bool tooLargeDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                      double rsq, double rpar, double s1ps2, double maxsep, double maxsepsq) const
    { return true; }
};

//
// Periodic is like Euclidean, except that the edges of the range wrap around, and the
// distance is the smaller of (x2-x1) % xperiod and (x1-x2) % xperiod.
//

template <>
struct MetricHelper<Periodic>
{
    enum { _Flat=Flat, _ThreeD=ThreeD, _Sphere=ThreeD };

    // The period in each direction.
    double xp, yp, zp;

    MetricHelper(double minrpar, double maxrpar, double _xp, double _yp, double _zp) :
        xp(_xp), yp(_yp), zp(_zp) {}

    ///
    //
    // Flat
    //
    ///

    double DistSq(const Position<Flat>& p1, const Position<Flat>& p2,
                  double& s1, double& s2) const
    {
        // Mostly the changes here are just to wrap each difference given the periods.
        Position<Flat> r = p1-p2;
        r.wrap(xp, yp);
        return r.normSq();
    }
    double Dist(const Position<Flat>& p1, const Position<Flat>& p2) const
    {
        double s=0.;
        return sqrt(DistSq(p1,p2,s,s));
    }

    bool CCW(const Position<Flat>& p1, const Position<Flat>& p2, const Position<Flat>& p3) const
    {
        // If cross product r21 x r31 > 0, then the points are counter-clockwise.
        Position<Flat> r21 = p2 - p1;
        Position<Flat> r31 = p3 - p1;
        r21.wrap(xp, yp);
        r31.wrap(xp, yp);
        return r21.cross(r31) > 0.;
    }

    // Some metrics keep track of whether the r_parallel distance is within some range.
    // This function checks if two cells are fully outside the range, in which case we can
    // throw them out and not recurse futher.  For Euclidean, this is trivially false.
    bool isRParOutsideRange(const Position<Flat>& p1, const Position<Flat>& p2,
                            double s1ps2, double& rpar) const
    { return false; }

    // When we get to a point where we think all possible pairs fall into a single bin,
    // then we also need to make sure we are fully in the range for rpar as well.  For
    // Euclidean (and others that don't use rpar), this is always true, but Rper and Rlens
    // have a check here.
    bool isRParInsideRange(const Position<Flat>& p1, const Position<Flat>& p2,
                           double s1ps2, double rpar) const
    { return true; }

    // The normal tests about whether a given distance is inside the binning range happen
    // in BinTypeHelper.  However, Rperp needs to do an additional check to make sure we
    // don't reject cell pairs prematurely.  For this and most metrics, these two checks
    // are trivially true (since only get here if the regular check passes).
    bool tooSmallDist(const Position<Flat>& p1, const Position<Flat>& p2,
                      double rsq, double rpar, double s1ps2, double minsep, double minsepsq) const
    { return true; }

    bool tooLargeDist(const Position<Flat>& p1, const Position<Flat>& p2,
                      double rsq, double rpar, double s1ps2, double maxsep, double maxsepsq) const
    { return true; }


    ///
    //
    // ThreeD
    //
    ///

    double DistSq(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                  double& s1, double& s2) const
    {
        Position<ThreeD> r = p1-p2;
        r.wrap(xp, yp, zp);
        return r.normSq();
    }
    double Dist(const Position<ThreeD>& p1, const Position<ThreeD>& p2) const
    {
        double s=0.;
        return sqrt(DistSq(p1,p2,s,s));
    }

    bool CCW(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
             const Position<ThreeD>& p3) const
    {
        // Now it's slightly more complicated, since the points are in three dimensions.  We do
        // the same thing, computing the cross product with respect to point p1.  Then if the
        // cross product points back toward Earth, the points are viewed as counter-clockwise.
        // We check this last point by the dot product with p1.
        Position<ThreeD> r21 = p2-p1;
        Position<ThreeD> r31 = p3-p1;
        r21.wrap(xp, yp, zp);
        r31.wrap(xp, yp, zp);
        return r21.cross(r31).dot(p1) < 0.;
    }

    bool isRParOutsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                            double s1ps2, double& rpar) const
    { return false; }

    bool isRParInsideRange(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                           double s1ps2, double rpar) const
    { return true; }

    bool tooSmallDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                      double rsq, double rpar, double s1ps2, double minsep, double minsepsq) const
    { return true; }

    bool tooLargeDist(const Position<ThreeD>& p1, const Position<ThreeD>& p2,
                      double rsq, double rpar, double s1ps2, double maxsep, double maxsepsq) const
    { return true; }

};

#endif

