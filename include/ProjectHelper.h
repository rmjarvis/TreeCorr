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

#ifndef TreeCorr_ProjectHelper_H
#define TreeCorr_ProjectHelper_H

// For the direct processing, we need a helper struct to handle some of the projections
// we need to do to the shear values.
template <int C>
struct ProjectHelper;

inline double safe_norm(const std::complex<double>& z)
{
    // When dividing by a norm, if z = 0, that can lead to nans.
    // This can happen in various places for extremal, degenerate triangles.
    // So always use this for any place where that might be a problem.
    double n = std::norm(z);
    return n > 0. ? n : 1.;
}

template <>
struct ProjectHelper<Flat>
{
    template <int D1>
    static void ProjectShear(
        const Cell<D1,Flat>& c1, const Cell<GData,Flat>& c2, std::complex<double>& g2)
    {
        // Project given shear to the line connecting them.
        std::complex<double> cr(c2.getData().getPos() - c1.getData().getPos());
        std::complex<double> expm2iarg = conj(cr*cr)/safe_norm(cr);
        g2 = c2.getData().getWG() * expm2iarg;
    }

    static void ProjectShears(
        const Cell<GData,Flat>& c1, const Cell<GData,Flat>& c2,
        std::complex<double>& g1, std::complex<double>& g2)
    {
        // Project given shears to the line connecting them.
        std::complex<double> cr(c2.getData().getPos() - c1.getData().getPos());
        std::complex<double> expm2iarg = conj(cr*cr)/safe_norm(cr);
        g1 = c1.getData().getWG() * expm2iarg;
        g2 = c2.getData().getWG() * expm2iarg;
    }

    static void ProjectShears(
        const Cell<GData,Flat>& c1, const Cell<GData,Flat>& c2, const Cell<GData,Flat>& c3,
        std::complex<double>& g1, std::complex<double>& g2, std::complex<double>& g3)
    {
        // Project given shears to the line connecting each to the centroid.
        const Position<Flat>& p1 = c1.getData().getPos();
        const Position<Flat>& p2 = c2.getData().getPos();
        const Position<Flat>& p3 = c3.getData().getPos();
        Position<Flat> cen = (p1 + p2 + p3)/3.;
        std::complex<double> cr1(cen - p1);
        std::complex<double> cr2(cen - p2);
        std::complex<double> cr3(cen - p3);
        g1 = c1.getData().getWG() * conj(cr1*cr1)/safe_norm(cr1);
        g2 = c2.getData().getWG() * conj(cr2*cr2)/safe_norm(cr2);
        g3 = c3.getData().getWG() * conj(cr3*cr3)/safe_norm(cr3);
    }

    template <int D1>
    static void ProjectVector(
        const Cell<D1,Flat>& c1, const Cell<VData,Flat>& c2, std::complex<double>& v2)
    {
        // Project given vector to the line connecting them.
        std::complex<double> cr(c2.getData().getPos() - c1.getData().getPos());
        std::complex<double> expmiarg = conj(cr)/sqrt(safe_norm(cr));
        v2 = c2.getData().getWV() * expmiarg;
    }

    static void ProjectVectors(
        const Cell<VData,Flat>& c1, const Cell<VData,Flat>& c2,
        std::complex<double>& v1, std::complex<double>& v2)
    {
        // Project given shears to the line connecting them.
        std::complex<double> cr(c2.getData().getPos() - c1.getData().getPos());
        std::complex<double> expmiarg = conj(cr)/sqrt(safe_norm(cr));
        v1 = c1.getData().getWV() * expmiarg;
        v2 = c2.getData().getWV() * expmiarg;
    }

    static void ProjectVectors(
        const Cell<VData,Flat>& c1, const Cell<VData,Flat>& c2, const Cell<VData,Flat>& c3,
        std::complex<double>& v1, std::complex<double>& v2, std::complex<double>& v3)
    {
        // Project given shears to the line connecting each to the centroid.
        const Position<Flat>& p1 = c1.getData().getPos();
        const Position<Flat>& p2 = c2.getData().getPos();
        const Position<Flat>& p3 = c3.getData().getPos();
        Position<Flat> cen = (p1 + p2 + p3)/3.;
        std::complex<double> cr1(cen - p1);
        std::complex<double> cr2(cen - p2);
        std::complex<double> cr3(cen - p3);
        v1 = c1.getData().getWV() * conj(cr1)/sqrt(safe_norm(cr1));
        v2 = c2.getData().getWV() * conj(cr2)/sqrt(safe_norm(cr2));
        v3 = c3.getData().getWV() * conj(cr3)/sqrt(safe_norm(cr3));
    }
};

template <>
struct ProjectHelper<Sphere>
{
    static void ProjectShear2(
        const Position<Sphere>& p1, const Position<Sphere>& p2, std::complex<double>& g2)
    {
        // For spherical triangles, it's a bit trickier, since the angles aren't equal.
        // In this function we just project the shear at p2.
        // This will be used by both NG and GG projections below.

        // We need the angle at each point between north and the line connecting the
        // two points.
        //
        // Use the spherical law of cosines:
        //
        // cos(a) = cos(b) cos(c) + sin(b) sin(c) cos(A)
        //
        // In our case:
        //   a = distance from pole to p1 = Pi/2 - dec1
        //   b = distance from pole to p2 = Pi/2 - dec2
        //   c = the great circle distance between the two points.
        //   A = angle between c and north at p2
        //   B = angle between c and north at p1
        //   C = angle between meridians = ra1 - ra2
        //

        // cos(C) = cos(ra1 - ra2) = cos(ra1)cos(ra2) + sin(ra1)sin(ra2)
        //        = (x1/cos(dec1)) (x2/cos(dec2)) + (y1/cos(dec1)) (y2/cos(dec2))
        //        = (x1 x2 + y1 y2) / (cos(dec1) cos(dec2))
        // sin(C) = sin(ra1 - ra2) = sin(ra1)cos(ra2) - cos(ra1)sin(ra2)
        //        = (y1/cos(dec1)) (x2/cos(dec2)) - (x1/cos(dec1)) (y2/cos(dec2))
        //        = (y1 x2 - x1 y2) / (cos(dec1) cos(dec2))

        // cos(A) = (sin(dec1) - sin(dec2) cos(c)) / (cos(dec2) sin(c))
        //
        // The division is fairly unstable if cos(dec2) or sin(c) is small.
        // And sin(c) is often small, so we want to manipulate this a bit.
        // cos(c) = sin(dec1) sin(dec2) + cos(dec1) cos(dec2) cos(C)
        // cos(A) = (sin(dec1) cos(dec2)^2 - sin(dec2) cos(dec1) cos(dec2) cos(C)) /
        //                (cos(dec2) sin(c))
        //        = (sin(dec1) cos(dec2) - sin(dec2) cos(dec1) cos(C)) / sin(c)
        //        = (sin(dec1) cos(dec2)^2 - sin(dec2) (x1 x2 + y1 y2)) / (cos(dec2) sin(c))
        //        = (z1 (1-z2^2) - z2 (x1 x2 + y1 y2)) / (cos(dec2) sin(c))
        //        = (z1 - z2 (x1 x2 + y1 y2 + z1 z2)) / (cos(dec2) sin(c))
        //
        // Note:  dsq = (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2
        //            = (x1^2+y1^2+z1^2) + (x2^2+y2^2+z2^2) -2x1x2 -2y1y2 - 2z1z2
        //            = 2 - 2 (x1 x2 + y1 y2 + z1 z2)
        //            = 2 - 2 dot  ^^^^ This is the dot product of the positions.
        //
        // cos(A) = ( z1 - z2*dot ) / (cos(dec2) sin(c))
        //        = ( (z1-z2) + z2*(1-dot) ) / (cos(dec2) sin(c))
        //        = ( (z1-z2) + z2*(dsq/2) ) / (cos(dec2) sin(c))
        //
        // sin(A) / sin(a) = sin(C) / sin(c)
        // sin(A) = cos(dec1) sin(C) / sin(c)
        //        = (y1 x2 - x1 y2) / (cos(dec2) sin(c))
        //
        // We ignore the denominator for now, and then figure out what it is from
        // sin(A)^2 + cos(A)^2 = 1
        double z1 = p1.getZ();
        double z2 = p2.getZ();
        double dsq = (p1-p2).normSq();
        double cosA = (z1-z2) + 0.5*z2*dsq;  // These are unnormalized.
        double sinA = p1.getY()*p2.getX() - p1.getX()*p2.getY();
        double cosAsq = cosA*cosA;
        double sinAsq = sinA*sinA;
        double normAsq = cosAsq + sinAsq;
        if (normAsq == 0.) normAsq = 1.;  // This happens if p1==p2, which is possible for 3pt.
        Assert(normAsq > 0.);
        double cos2A = (cosAsq - sinAsq) / normAsq; // These are now correct.
        double sin2A = 2.*sinA*cosA / normAsq;

        // In fact, A is not really the angles by which we want to rotate the shear.
        // We really want to rotate by the angle between due _east_ and c, not _north_.
        //
        // exp(-2ialpha) = exp(-2i (A - Pi/2) )
        //               = exp(iPi) * exp(-2iA)
        //               = - exp(-2iA)

        std::complex<double> expm2ialpha(-cos2A, sin2A);
        g2 *= expm2ialpha;
    }

    template <int D1>
    static void ProjectShear(
        const Cell<D1,Sphere>& c1, const Cell<GData,Sphere>& c2, std::complex<double>& g2)
    {
        const Position<Sphere>& p1 = c1.getData().getPos();
        const Position<Sphere>& p2 = c2.getData().getPos();
        g2 = c2.getData().getWG();
        ProjectShear2(p1, p2, g2);
    }

    static void ProjectShears(
        const Cell<GData,Sphere>& c1, const Cell<GData,Sphere>& c2,
        std::complex<double>& g1, std::complex<double>& g2)
    {
        const Position<Sphere>& p1 = c1.getData().getPos();
        const Position<Sphere>& p2 = c2.getData().getPos();
        g1 = c1.getData().getWG();
        g2 = c2.getData().getWG();
        ProjectShear2(p2, p1, g1);
        ProjectShear2(p1, p2, g2);
    }
    static void ProjectShears(
        const Cell<GData,Sphere>& c1, const Cell<GData,Sphere>& c2, const Cell<GData,Sphere>& c3,
        std::complex<double>& g1, std::complex<double>& g2, std::complex<double>& g3)
    {
        const Position<Sphere>& p1 = c1.getData().getPos();
        const Position<Sphere>& p2 = c2.getData().getPos();
        const Position<Sphere>& p3 = c3.getData().getPos();
        Position<Sphere> cen((p1 + p2 + p3)/3.);
        g1 = c1.getData().getWG();
        g2 = c2.getData().getWG();
        g3 = c3.getData().getWG();

        ProjectShear2(cen, p1, g1);
        ProjectShear2(cen, p2, g2);
        ProjectShear2(cen, p3, g3);
    }

    static void ProjectVector2(
        const Position<Sphere>& p1, const Position<Sphere>& p2, std::complex<double>& v2)
    {
        // This is the same idea as the math in ProjectShear.
        // We just stop at cosA, sinA and normalize that rather than continue to cos2A, sin2A.
        double z1 = p1.getZ();
        double z2 = p2.getZ();
        double dsq = (p1-p2).normSq();
        double cosA = (z1-z2) + 0.5*z2*dsq;  // These are unnormalized.
        double sinA = p1.getY()*p2.getX() - p1.getX()*p2.getY();
        double cosAsq = cosA*cosA;
        double sinAsq = sinA*sinA;
        double normAsq = cosAsq + sinAsq;
        if (normAsq == 0.) normAsq = 1.;  // This happens if p1==p2, which is possible for 3pt.
        Assert(normAsq > 0.);
        double normA = sqrt(normAsq);
        cosA /= normA;  // These are now correct.
        sinA /= normA;

        // Again, A is not actually the angle we want.
        // We really want to rotate by the angle between due _east_ and c, not _north_.
        //
        // exp(-ialpha) = exp(-i (A - Pi/2) )
        //               = exp(iPi/2) * exp(-iA)
        //               = i exp(-iA) = i (cosA - isinA)
        //               = sinA + icosA

        std::complex<double> expmialpha(sinA, cosA);
        v2 *= expmialpha;
    }

    template <int D1>
    static void ProjectVector(
        const Cell<D1,Sphere>& c1, const Cell<VData,Sphere>& c2, std::complex<double>& v2)
    {
        const Position<Sphere>& p1 = c1.getData().getPos();
        const Position<Sphere>& p2 = c2.getData().getPos();
        v2 = c2.getData().getWV();
        ProjectVector2(p1, p2, v2);
    }

    static void ProjectVectors(
        const Cell<VData,Sphere>& c1, const Cell<VData,Sphere>& c2,
        std::complex<double>& v1, std::complex<double>& v2)
    {
        const Position<Sphere>& p1 = c1.getData().getPos();
        const Position<Sphere>& p2 = c2.getData().getPos();
        // Note the minus sign here.  This is so both points are projected onto the
        // coordinate system with p1 and p2 arranged horizontally with p1 on the left.
        // The normal project function for v1 is 180 degrees rotated from this, so
        // need an extra minus sign.
        v1 = -c1.getData().getWV();
        v2 = c2.getData().getWV();
        ProjectVector2(p2, p1, v1);
        ProjectVector2(p1, p2, v2);
    }
    static void ProjectVectors(
        const Cell<VData,Sphere>& c1, const Cell<VData,Sphere>& c2, const Cell<VData,Sphere>& c3,
        std::complex<double>& v1, std::complex<double>& v2, std::complex<double>& v3)
    {
        const Position<Sphere>& p1 = c1.getData().getPos();
        const Position<Sphere>& p2 = c2.getData().getPos();
        const Position<Sphere>& p3 = c3.getData().getPos();
        Position<Sphere> cen((p1 + p2 + p3)/3.);
        v1 = c1.getData().getWV();
        v2 = c2.getData().getWV();
        v3 = c3.getData().getWV();

        ProjectVector2(cen, p1, v1);
        ProjectVector2(cen, p2, v2);
        ProjectVector2(cen, p3, v3);
    }
};

// The projections for ThreeD are basically the same as for Sphere.
// We just need to normalize the positions to be on the unit sphere, then we can call
// the ProjectHelper<Sphere> methods.
template <>
struct ProjectHelper<ThreeD>
{
    template <int D1>
    static void ProjectShear(
        const Cell<D1,ThreeD>& c1, const Cell<GData,ThreeD>& c2, std::complex<double>& g2)
    {
        const Position<ThreeD>& p1 = c1.getData().getPos();
        const Position<ThreeD>& p2 = c2.getData().getPos();
        Position<Sphere> sp1(p1);
        Position<Sphere> sp2(p2);
        g2 = c2.getData().getWG();
        ProjectHelper<Sphere>::ProjectShear2(sp1, sp2, g2);
    }

    static void ProjectShears(
        const Cell<GData,ThreeD>& c1, const Cell<GData,ThreeD>& c2,
        std::complex<double>& g1, std::complex<double>& g2)
    {
        const Position<ThreeD>& p1 = c1.getData().getPos();
        const Position<ThreeD>& p2 = c2.getData().getPos();
        Position<Sphere> sp1(p1);
        Position<Sphere> sp2(p2);
        g1 = c1.getData().getWG();
        g2 = c2.getData().getWG();
        ProjectHelper<Sphere>::ProjectShear2(sp2, sp1, g1);
        ProjectHelper<Sphere>::ProjectShear2(sp1, sp2, g2);
    }

    static void ProjectShears(
        const Cell<GData,ThreeD>& c1, const Cell<GData,ThreeD>& c2, const Cell<GData,ThreeD>& c3,
        std::complex<double>& g1, std::complex<double>& g2, std::complex<double>& g3)
    {
        const Position<ThreeD>& p1 = c1.getData().getPos();
        const Position<ThreeD>& p2 = c2.getData().getPos();
        const Position<ThreeD>& p3 = c3.getData().getPos();
        Position<Sphere> sp1(p1);
        Position<Sphere> sp2(p2);
        Position<Sphere> sp3(p3);
        Position<Sphere> cen((sp1 + sp2 + sp3)/3.);
        cen.normalize();
        g1 = c1.getData().getWG();
        g2 = c2.getData().getWG();
        g3 = c3.getData().getWG();

        ProjectHelper<Sphere>::ProjectShear2(cen, sp1, g1);
        ProjectHelper<Sphere>::ProjectShear2(cen, sp2, g2);
        ProjectHelper<Sphere>::ProjectShear2(cen, sp3, g3);
    }

    template <int D1>
    static void ProjectVector(
        const Cell<D1,ThreeD>& c1, const Cell<VData,ThreeD>& c2, std::complex<double>& v2)
    {
        const Position<ThreeD>& p1 = c1.getData().getPos();
        const Position<ThreeD>& p2 = c2.getData().getPos();
        Position<Sphere> sp1(p1);
        Position<Sphere> sp2(p2);
        v2 = c2.getData().getWV();
        ProjectHelper<Sphere>::ProjectVector2(sp1, sp2, v2);
    }

    static void ProjectVectors(
        const Cell<VData,ThreeD>& c1, const Cell<VData,ThreeD>& c2,
        std::complex<double>& v1, std::complex<double>& v2)
    {
        const Position<ThreeD>& p1 = c1.getData().getPos();
        const Position<ThreeD>& p2 = c2.getData().getPos();
        Position<Sphere> sp1(p1);
        Position<Sphere> sp2(p2);
        v1 = c1.getData().getWV();
        v2 = c2.getData().getWV();
        ProjectHelper<Sphere>::ProjectVector2(sp2, sp1, v1);
        ProjectHelper<Sphere>::ProjectVector2(sp1, sp2, v2);
    }

    static void ProjectVectors(
        const Cell<VData,ThreeD>& c1, const Cell<VData,ThreeD>& c2, const Cell<VData,ThreeD>& c3,
        std::complex<double>& v1, std::complex<double>& v2, std::complex<double>& v3)
    {
        const Position<ThreeD>& p1 = c1.getData().getPos();
        const Position<ThreeD>& p2 = c2.getData().getPos();
        const Position<ThreeD>& p3 = c3.getData().getPos();
        Position<Sphere> sp1(p1);
        Position<Sphere> sp2(p2);
        Position<Sphere> sp3(p3);
        Position<Sphere> cen((sp1 + sp2 + sp3)/3.);
        cen.normalize();
        v1 = c1.getData().getWV();
        v2 = c2.getData().getWV();
        v3 = c3.getData().getWV();

        ProjectHelper<Sphere>::ProjectVector2(cen, sp1, v1);
        ProjectHelper<Sphere>::ProjectVector2(cen, sp2, v2);
        ProjectHelper<Sphere>::ProjectVector2(cen, sp3, v3);
    }
};

#endif
