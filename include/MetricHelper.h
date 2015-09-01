/* Copyright (c) 2003-2014 by Mike Jarvis
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

#ifndef TreeCorr_MetricHelper_H
#define TreeCorr_MetricHelper_H

// For the direct processing, we need a helper struct to handle some of the manipulations
// we need to do to the shear values.
template <int M>
struct MetricHelper;

template <>
struct MetricHelper<Flat>
{
    template <int DC1>
    static void ProjectShear(
        const Cell<DC1,Flat>& c1, const Cell<GData,Flat>& c2,
        double dsq, std::complex<double>& g2)
    {
        // Project given shear to the line connecting them.
        std::complex<double> cr(c2.getData().getPos() - c1.getData().getPos());
        Assert(dsq != 0.);
        std::complex<double> expm2iarg = conj(cr*cr)/dsq;
        g2 = c2.getData().getWG() * expm2iarg;
    }

    static void ProjectShears(
        const Cell<GData,Flat>& c1, const Cell<GData,Flat>& c2,
        double dsq, std::complex<double>& g1, std::complex<double>& g2)
    {
        // Project given shears to the line connecting them.
        std::complex<double> cr(c2.getData().getPos() - c1.getData().getPos());
        Assert(dsq != 0.);
        std::complex<double> expm2iarg = conj(cr*cr)/dsq;
        g1 = c1.getData().getWG() * expm2iarg;
        g2 = c2.getData().getWG() * expm2iarg;
    }
};

template <>
struct MetricHelper<Sphere>
{
    static void ProjectShear2(
        const Position<Sphere>& p1, const Position<Sphere>& p2,
        double dsq, double cross, double crosssq, std::complex<double>& g2)
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
        double cosA = (z1-z2) + 0.5*z2*dsq;  // These are unnormalized.
        double sinA = cross;
        double cosAsq = cosA*cosA;
        double sinAsq = crosssq;
        double normAsq = cosAsq + sinAsq;
        Assert(normAsq > 0.);
        double cos2A = (cosAsq - sinAsq) / normAsq; // These are now correct.
        double sin2A = 2.*sinA*cosA / normAsq;

        // In fact, A is not really the angles by which we want to rotate the shear.
        // We really want to rotae by the angle between due _east_ and c, not _north_.
        //
        // exp(-2ialpha) = exp(-2i (A - Pi/2) )
        //               = exp(iPi) * exp(-2iA)
        //               = - exp(-2iA)

        std::complex<double> expm2ialpha(-cos2A,sin2A);
        g2 *= expm2ialpha;
    }

    static void ProjectShear1(
        const Position<Sphere>& p1, const Position<Sphere>& p2,
        double dsq, double cross, double crosssq, std::complex<double>& g1)
    {
        // It is similar for the shear at p1:

        // cos(B) = (sin(dec2) - sin(dec1) cos(c)) / (cos(dec1) sin(c))
        //        = (sin(dec2) cos(dec1)^2 - sin(dec1) (x1 x2 + y1 y2)) / (cos(dec1) sin(c))
        //        = (z2 (1-z1^2) - z1 (x1 x2 + y1 y2)) / (cos(dec1) sin(c))
        //        = (z2 - z1 (x1 x2 + y1 y2 + z1 z2)) / (cos(dec1) sin(c))
        //        = (z2-z1 + z1*dsq/2) / (cos(dec1) sin(c))
        // sin(B) / sin(b) = sin(C) / sin(c)
        // sin(B) = cos(dec2) sin(C) / sin(c)
        //        = (y1 x2 - x1 y2) / (cos(dec1) sin(c))
        double z1 = p1.getZ();
        double z2 = p2.getZ();
        double cosB = (z2-z1) + 0.5*z1*dsq;  // These are unnormalized.
        double sinB = cross;  
        double cosBsq = cosB*cosB;
        double sinBsq = crosssq;
        double normBsq = cosBsq + sinBsq;
        Assert(normBsq != 0.);
        double cos2B = (cosBsq - sinBsq) / normBsq;
        double sin2B = 2.*sinB*cosB / normBsq;

        // exp(-2ibeta)  = exp(-2i (Pi/2 - B) )
        //               = exp(-iPi) * exp(2iB)
        //               = - exp(2iB)

        std::complex<double> expm2ibeta(-cos2B,-sin2B); 
        g1 *= expm2ibeta;
    }

    template <int DC1>
    static void ProjectShear(
        const Cell<DC1,Sphere>& c1, const Cell<GData,Sphere>& c2,
        double dsq, std::complex<double>& g2)
    {
        const Position<Sphere>& p1 = c1.getData().getPos();
        const Position<Sphere>& p2 = c2.getData().getPos();
        g2 = c2.getData().getWG();
        double cross = p1.getY()*p2.getX() - p1.getX()*p2.getY();
        double crosssq = cross*cross;
        ProjectShear2(p1,p2,dsq,cross,crosssq,g2);
    }

    static void ProjectShears(
        const Cell<GData,Sphere>& c1, const Cell<GData,Sphere>& c2,
        double dsq, std::complex<double>& g1, std::complex<double>& g2)
    {
        const Position<Sphere>& p1 = c1.getData().getPos();
        const Position<Sphere>& p2 = c2.getData().getPos();
        g1 = c1.getData().getWG();
        g2 = c2.getData().getWG();
        double cross = p1.getY()*p2.getX() - p1.getX()*p2.getY();
        double crosssq = cross*cross;
        ProjectShear1(p1,p2,dsq,cross,crosssq,g1);
        ProjectShear2(p1,p2,dsq,cross,crosssq,g2);
    }
};

// The projections for Perp are the same as for Sphere.
template <>
struct MetricHelper<Perp>
{
    static void ProjectShear2(
        const Position<Perp>& p1, const Position<Perp>& p2,
        double dsq, double cross, double crosssq, std::complex<double>& g2)
    { MetricHelper<Sphere>::ProjectShear2(p1,p2,dsq,cross,crosssq,g2); }

    static void ProjectShear1(
        const Position<Perp>& p1, const Position<Perp>& p2,
        double dsq, double cross, double crosssq, std::complex<double>& g1)
    { MetricHelper<Sphere>::ProjectShear1(p1,p2,dsq,cross,crosssq,g1); }

    template <int DC1>
    static void ProjectShear(
        const Cell<DC1,Perp>& c1, const Cell<GData,Perp>& c2,
        double dsq, std::complex<double>& g2)
    {
        const Position<Perp>& p1 = c1.getData().getPos();
        const Position<Perp>& p2 = c2.getData().getPos();
        g2 = c2.getData().getWG();
        double cross = p1.getY()*p2.getX() - p1.getX()*p2.getY();
        double crosssq = cross*cross;
        ProjectShear2(p1,p2,dsq,cross,crosssq,g2);
    }

    static void ProjectShears(
        const Cell<GData,Perp>& c1, const Cell<GData,Perp>& c2,
        double dsq, std::complex<double>& g1, std::complex<double>& g2)
    {
        const Position<Perp>& p1 = c1.getData().getPos();
        const Position<Perp>& p2 = c2.getData().getPos();
        g1 = c1.getData().getWG();
        g2 = c2.getData().getWG();
        double cross = p1.getY()*p2.getX() - p1.getX()*p2.getY();
        double crosssq = cross*cross;
        ProjectShear1(p1,p2,dsq,cross,crosssq,g1);
        ProjectShear2(p1,p2,dsq,cross,crosssq,g2);
    }
};

#endif
