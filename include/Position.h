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

#ifndef TreeCorr_Position_H
#define TreeCorr_Position_H

#include <complex>

// The Coord enum is defined here:
#include "Position_C.h"
#include "dbg.h"

template <typename T>
inline T SQR(T x) { return x * x; }

template <int C>
class Position;

//
// Flat defines 2-D coordinates (x,y)
//

template <>
class Position<Flat>
{

public:
    Position() : _x(0.), _y(0.), _normsq(0.), _norm(0.) {}
    Position(const Position<Flat>& rhs) :
        _x(rhs._x), _y(rhs._y), _normsq(rhs._normsq), _norm(rhs._norm) {}
    ~Position() {}
    Position(double x, double y) : _x(x), _y(y), _normsq(0.), _norm(0.) {}
    Position& operator=(const Position<Flat>& rhs)
    {
        _x = rhs._x; _y = rhs._y;
        _normsq = rhs._normsq; _norm = rhs._norm;
        return *this;
    }

    // A convenience constructor to be parallel with 3d positions so I can do things like
    // Position<C> pos(x,y,z) when z=0 for Flat.
    Position(double x, double y, double z) : _x(x), _y(y), _normsq(0.), _norm(0.)
    { Assert(z==0.); }

    double getX() const { return _x; }
    double getY() const { return _y; }
    double get(int split) const { return split==1 ? _y : _x; }
    operator std::complex<double>() const { return std::complex<double>(_x,_y); }

    double normSq() const
    {
        if (_normsq == 0.) _normsq = _x*_x + _y*_y;
        return _normsq;
    }
    double norm() const
    {
        if (_norm == 0.) _norm = sqrt(normSq());
        return _norm;
    }
    void normalize() {}

    double dot(const Position<Flat>& p2) const
    { return _x*p2._x + _y*p2._y; }
    double cross(const Position<Flat>& p2) const
    { return _x*p2._y - _y*p2._x; }

    Position<Flat>& operator+=(const Position<Flat>& p2)
    { _x += p2.getX(); _y += p2.getY(); resetNorm(); return *this; }
    Position<Flat>& operator-=(const Position<Flat>& p2)
    { _x -= p2.getX(); _y -= p2.getY(); resetNorm(); return *this; }
    Position<Flat>& operator*=(double a)
    { _x *= a; _y *= a; resetNorm(); return *this; }
    Position<Flat>& operator/=(double a)
    { _x /= a; _y /= a; resetNorm(); return *this; }

    Position<Flat> operator+(const Position<Flat>& p2) const
    { Position<Flat> p1 = *this; p1 += p2; return p1; }
    Position<Flat> operator-(const Position<Flat>& p2) const
    { Position<Flat> p1 = *this; p1 -= p2; return p1; }
    Position<Flat> operator*(double a) const
    { Position<Flat> p1 = *this; p1 *= a; return p1; }
    Position<Flat> operator/(double a) const
    { Position<Flat> p1 = *this; p1 /= a; return p1; }

    void read(std::istream& fin) { fin >> _x >> _y; resetNorm(); }
    void write(std::ostream& fout) const
    { fout << _x << " " << _y << " "; }

    void wrap(const double xp, const double yp)
    {
        while (_x > xp/2.) _x -= xp;
        while (_x < -xp/2.) _x += xp;
        while (_y > yp/2.) _y -= yp;
        while (_y < -yp/2.) _y += yp;
    }

protected:
    void resetNorm() { _normsq = _norm = 0.; }

private:
    double _x,_y;
    mutable double _normsq, _norm;

}; // Position<Flat>

template <int C>
inline std::ostream& operator<<(std::ostream& os, const Position<C>& pos)
{ pos.write(os); return os; }

template <int C>
inline std::istream& operator>>(std::istream& os, Position<C>& pos)
{ pos.read(os); return os; }


//
// ThreeD defines 3-D coordinates (x,y,z)
//

template <>
class Position<ThreeD>
{

public:
    Position() : _x(0.), _y(0.), _z(0.), _normsq(0.), _norm(0.) {}
    Position(const Position<ThreeD>& rhs) :
        _x(rhs._x), _y(rhs._y), _z(rhs._z), _normsq(rhs._normsq), _norm(rhs._norm) {}
    ~Position() {}
    Position(double x, double y, double z) :
        _x(x), _y(y), _z(z), _normsq(0.), _norm(0.) {}
    Position<ThreeD>& operator=(const Position<ThreeD>& rhs)
    {
        _x = rhs._x; _y = rhs._y; _z = rhs._z;
        _normsq = rhs._normsq; _norm = rhs._norm;
        return *this;
    }

    double getX() const { return _x; }
    double getY() const { return _y; }
    double getZ() const { return _z; }
    double get(int split) const { return split==2 ? _z : split==1 ? _y : _x; }

    double normSq() const
    {
        if (_normsq == 0.) _normsq = _x*_x + _y*_y + _z*_z;
        return _normsq;
    }
    double norm() const
    {
        if (_norm == 0.) _norm = sqrt(normSq());
        return _norm;
    }
    void normalize() {}

    double dot(const Position<ThreeD>& p2) const
    { return _x*p2._x + _y*p2._y + _z*p2._z; }
    Position<ThreeD> cross(const Position<ThreeD>& p2) const
    {
        return Position<ThreeD>(_y*p2._z - _z*p2._y,
                                _z*p2._x - _x*p2._z,
                                _x*p2._y - _y*p2._x);
    }

    Position<ThreeD>& operator+=(const Position<ThreeD>& p2)
    { _x += p2.getX(); _y += p2.getY(); _z += p2.getZ(); resetNorm(); return *this; }
    Position<ThreeD>& operator-=(const Position<ThreeD>& p2)
    { _x -= p2.getX(); _y -= p2.getY(); _z -= p2.getZ(); resetNorm(); return *this; }
    Position<ThreeD>& operator*=(double a)
    { _x *= a; _y *= a; _z *= a; resetNorm(); return *this; }
    Position<ThreeD>& operator/=(double a)
    { _x /= a; _y /= a; _z /= a; resetNorm(); return *this; }

    Position<ThreeD> operator+(const Position<ThreeD>& p2) const
    { Position<ThreeD> p1 = *this; p1 += p2; return p1; }
    Position<ThreeD> operator-(const Position<ThreeD>& p2) const
    { Position<ThreeD> p1 = *this; p1 -= p2; return p1; }
    Position<ThreeD> operator*(double a) const
    { Position<ThreeD> p1 = *this; p1 *= a; return p1; }
    Position<ThreeD> operator/(double a) const
    { Position<ThreeD> p1 = *this; p1 /= a; return p1; }

    void wrap(const double xp, const double yp, const double zp)
    {
        while (_x > xp/2.) _x -= xp;
        while (_x < -xp/2.) _x += xp;
        while (_y > yp/2.) _y -= yp;
        while (_y < -yp/2.) _y += yp;
        while (_z > zp/2.) _z -= zp;
        while (_z < -zp/2.) _z += zp;
    }

    void read(std::istream& fin)
    { fin >> _x >> _y >> _z; resetNorm(); }
    void write(std::ostream& fout) const
    { fout << _x << " " << _y << " " << _z << " "; }

protected:
    void resetNorm() { _normsq = _norm = 0.; }

private:
    double _x,_y,_z;
    mutable double _normsq, _norm;

}; // Position<ThreeD>


// Spherical coordinates are stored as (x,y,z) with
// x = cos(dec) * cos(ra)
// y = cos(dec) * sin(ra)
// z = sin(dec)
// These values are computed in the python layer and passed as such to the C++ layer.
// The only difference then for Sphere in the C++ layer is the normalize() function, which
// had been a no-op for the above two classes, but now renormalizes positions to make sure
// x^2 + y^2 + z^2 == 1.

template <>
class Position<Sphere> : public Position<ThreeD>
{
public:
    Position() : Position<ThreeD>() {}
    Position(const Position<Sphere>& rhs) : Position<ThreeD>(rhs) {}
    Position(const Position<ThreeD>& rhs) : Position<ThreeD>(rhs) { normalize(); }
    ~Position() {}
    Position(double x, double y, double z) : Position<ThreeD>(x,y,z) { normalize(); }
    Position<Sphere>& operator=(const Position<Sphere>& rhs)
    { Position<ThreeD>::operator=(rhs); return *this; }

    double normSq() const { return 1.; }
    double norm() const { return 1.; }

    // If appropriate, put the position back on the unit sphere.
    void normalize() { *this /= Position<ThreeD>::norm(); resetNorm(); }

    Position<Sphere>& operator+=(const Position<Sphere>& p2)
    { Position<ThreeD>::operator+=(p2); return *this; }
    Position<Sphere>& operator-=(const Position<Sphere>& p2)
    { Position<ThreeD>::operator-=(p2); return *this; }
    Position<Sphere>& operator*=(double a)
    { Position<ThreeD>::operator*=(a); return *this; }
    Position<Sphere>& operator/=(double a)
    { Position<ThreeD>::operator/=(a); return *this; }

    Position<ThreeD> operator+(const Position<Sphere>& p2) const
    { Position<ThreeD> p1 = *this; p1 += p2; return p1; }
    Position<ThreeD> operator-(const Position<Sphere>& p2) const
    { Position<ThreeD> p1 = *this; p1 -= p2; return p1; }
    Position<ThreeD> operator*(double a) const
    { Position<ThreeD> p1 = *this; p1 *= a; return p1; }
    Position<ThreeD> operator/(double a) const
    { Position<ThreeD> p1 = *this; p1 /= a; return p1; }

}; // Position<Sphere>


#endif
