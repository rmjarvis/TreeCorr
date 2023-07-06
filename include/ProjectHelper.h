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

template <int s>
std::complex<double> calculate_expmsialpha(const std::complex<double>& r);

template <>
inline std::complex<double> calculate_expmsialpha<1>(const std::complex<double>& r)
{ return conj(r) / sqrt(safe_norm(r)); }

template <>
inline std::complex<double> calculate_expmsialpha<2>(const std::complex<double>& r)
{ return conj(r*r) / safe_norm(r); }

template <>
inline std::complex<double> calculate_expmsialpha<3>(const std::complex<double>& r)
{
    std::complex<double> r3 = r*r*r;
    return conj(r3) / sqrt(safe_norm(r3));
}

template <>
inline std::complex<double> calculate_expmsialpha<4>(const std::complex<double>& r)
{
    std::complex<double> r2 = r*r;
    return conj(r2*r2) / safe_norm(r2);
}

template <int D>
inline std::complex<double> _expmsialpha(const std::complex<double>& r)
{
    const int s = (D==VData ? 1 :
                   D==GData ? 2 : 0);
    return calculate_expmsialpha<s>(r);
}

template <>
struct ProjectHelper<Flat>
{
    template <int D1, int D>
    static void Project(
        const Cell<D1,Flat>& c1, const Cell<D,Flat>& c2, std::complex<double>& z2)
    {
        // Project given spin-s quantity to the line connecting them.
        std::complex<double> r(c2.getData().getPos() - c1.getData().getPos());
        z2 *= _expmsialpha<D>(r);
    }

    template <int D>
    static void Project(
        const Cell<D,Flat>& c1, const Cell<D,Flat>& c2,
        std::complex<double>& z1, std::complex<double>& z2)
    {
        // Project given spin-s quantities to the line connecting them.
        std::complex<double> r(c2.getData().getPos() - c1.getData().getPos());
        std::complex<double> expmsialpha = _expmsialpha<D>(r);
        z1 *= expmsialpha;
        z2 *= expmsialpha;
    }

    template <int D>
    static void Project(
        const Cell<D,Flat>& c1, const Cell<D,Flat>& c2, const Cell<D,Flat>& c3,
        std::complex<double>& z1, std::complex<double>& z2, std::complex<double>& z3)
    {
        // Project given spin-s quantities to the line connecting each to the centroid.
        const Position<Flat>& p1 = c1.getData().getPos();
        const Position<Flat>& p2 = c2.getData().getPos();
        const Position<Flat>& p3 = c3.getData().getPos();
        Position<Flat> cen = (p1 + p2 + p3)/3.;
        std::complex<double> r1(cen - p1);
        std::complex<double> r2(cen - p2);
        std::complex<double> r3(cen - p3);
        z1 *= _expmsialpha<D>(r1);
        z2 *= _expmsialpha<D>(r2);
        z3 *= _expmsialpha<D>(r3);
    }
};

template <>
struct ProjectHelper<Sphere>
{
    static std::complex<double> calculate_direction(
        const Position<Sphere>& p1, const Position<Sphere>& p2)
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

        // In fact, A is not really the angles by which we want to rotate the shear.
        // We really want to rotate by the angle between due _east_ and c, not _north_.
        //
        // exp(ialpha) = exp(i (A - Pi/2) )
        //             = exp(iA) * exp(-iPi/2)
        //             = -i (cosA + isinA)
        //             = sinA - icosA

        // Note: the final normalization is done by the calling routine now.
        return std::complex<double>(sinA, -cosA);
    }

    template <int D1, int D>
    static void Project(
        const Cell<D1,Sphere>& c1, const Cell<D,Sphere>& c2, std::complex<double>& z2)
    {
        const Position<Sphere>& p1 = c1.getData().getPos();
        const Position<Sphere>& p2 = c2.getData().getPos();
        std::complex<double> r = calculate_direction(p1,p2);
        z2 *= _expmsialpha<D>(r);
    }

    template <int D>
    static void Project(
        const Cell<D,Sphere>& c1, const Cell<D,Sphere>& c2,
        std::complex<double>& z1, std::complex<double>& z2)
    {
        const Position<Sphere>& p1 = c1.getData().getPos();
        const Position<Sphere>& p2 = c2.getData().getPos();
        std::complex<double> r12 = calculate_direction(p1,p2);
        std::complex<double> expmsialpha = _expmsialpha<D>(r12);
        z2 *= expmsialpha;
        std::complex<double> r21 = calculate_direction(p2,p1);
        std::complex<double> expmsibeta = _expmsialpha<D>(r21);
        z1 *= expmsibeta;
        // Note there should be a minus sign here on z1 if s is odd.
        // This is so both points are projected onto the coordinate system with p1 and p2
        // arranged horizontally with p1 on the left. The normal project function for z1 is
        // 180 degrees rotated from this, so need an extra minus sign.
        if (D == VData) z1 = -z1;
    }

    template <int D>
    static void Project(
        const Cell<D,Sphere>& c1, const Cell<D,Sphere>& c2, const Cell<D,Sphere>& c3,
        std::complex<double>& z1, std::complex<double>& z2, std::complex<double>& z3)
    {
        const Position<Sphere>& p1 = c1.getData().getPos();
        const Position<Sphere>& p2 = c2.getData().getPos();
        const Position<Sphere>& p3 = c3.getData().getPos();
        Position<Sphere> cen((p1 + p2 + p3)/3.);
        z1 *= _expmsialpha<D>(calculate_direction(cen,p1));
        z2 *= _expmsialpha<D>(calculate_direction(cen,p2));
        z3 *= _expmsialpha<D>(calculate_direction(cen,p3));
    }
};

// The projections for ThreeD are basically the same as for Sphere.
// We just need to normalize the positions to be on the unit sphere, then we can call
// the ProjectHelper<Sphere> methods.
template <>
struct ProjectHelper<ThreeD>
{
    template <int D1, int D>
    static void Project(
        const Cell<D1,ThreeD>& c1, const Cell<D,ThreeD>& c2, std::complex<double>& z2)
    {
        const Position<ThreeD>& p1 = c1.getData().getPos();
        const Position<ThreeD>& p2 = c2.getData().getPos();
        Position<Sphere> sp1(p1);
        Position<Sphere> sp2(p2);
        std::complex<double> r = ProjectHelper<Sphere>::calculate_direction(sp1,sp2);
        z2 *= _expmsialpha<D>(r);
    }

    template <int D>
    static void Project(
        const Cell<D,ThreeD>& c1, const Cell<D,ThreeD>& c2,
        std::complex<double>& z1, std::complex<double>& z2)
    {
        const Position<ThreeD>& p1 = c1.getData().getPos();
        const Position<ThreeD>& p2 = c2.getData().getPos();
        Position<Sphere> sp1(p1);
        Position<Sphere> sp2(p2);
        std::complex<double> r12 = ProjectHelper<Sphere>::calculate_direction(sp1,sp2);
        std::complex<double> expmsialpha = _expmsialpha<D>(r12);
        z2 *= expmsialpha;
        std::complex<double> r21 = ProjectHelper<Sphere>::calculate_direction(sp2,sp1);
        std::complex<double> expmsibeta = _expmsialpha<D>(r21);
        z1 *= expmsibeta;
        if (D == VData) z1 = -z1;
    }

    template <int D>
    static void Project(
        const Cell<D,ThreeD>& c1, const Cell<D,ThreeD>& c2, const Cell<D,ThreeD>& c3,
        std::complex<double>& z1, std::complex<double>& z2, std::complex<double>& z3)
    {
        const Position<ThreeD>& p1 = c1.getData().getPos();
        const Position<ThreeD>& p2 = c2.getData().getPos();
        const Position<ThreeD>& p3 = c3.getData().getPos();
        Position<Sphere> sp1(p1);
        Position<Sphere> sp2(p2);
        Position<Sphere> sp3(p3);
        Position<Sphere> cen((sp1 + sp2 + sp3)/3.);
        cen.normalize();
        z1 *= _expmsialpha<D>(ProjectHelper<Sphere>::calculate_direction(cen,sp1));
        z2 *= _expmsialpha<D>(ProjectHelper<Sphere>::calculate_direction(cen,sp2));
        z3 *= _expmsialpha<D>(ProjectHelper<Sphere>::calculate_direction(cen,sp3));
    }
};

#endif
