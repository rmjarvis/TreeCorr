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

#ifndef TreeCorr_Position_H
#define TreeCorr_Position_H

#include <complex>

template <typename T>
inline T SQR(T x) { return x * x; }

// There are three kinds of coordinate systems we can use:
// 1 = Flat = (x,y) coordinates
// 2 = ThreeD (called 3d in python) = (x,y,z) coordinates
// 3 = Sphere = (ra,dec)  These are stored as (x,y,z), but normalized to have |r| = 1.
enum Coord { Flat=1, ThreeD=2, Sphere=3 };

template <int C>
class Position;

//
// Flat defines 2-D coordinates (x,y)
//

template <>
class Position<Flat>
{

public:
    Position() : _x(0.), _y(0.) {}
    Position(const Position<Flat>& rhs) :  _x(rhs._x), _y(rhs._y) {}
    ~Position() {}
    Position(double x, double y) : _x(x), _y(y) {}
    Position& operator=(const Position<Flat>& rhs) 
    { _x = rhs._x; _y = rhs._y; return *this; }

    double getX() const { return _x; }
    double getY() const { return _y; }
    double get(int split) const { return split==1 ? _y : _x; }
    operator std::complex<double>() const { return std::complex<double>(_x,_y); }

    double normSq() const { return _x*_x+_y*_y; }
    double norm() const { return sqrt(normSq()); }
    void normalize() {}

    Position<Flat>& operator+=(const Position<Flat>& p2)
    { _x += p2.getX(); _y += p2.getY(); return *this; }
    Position<Flat>& operator-=(const Position<Flat>& p2)
    { _x -= p2.getX(); _y -= p2.getY(); return *this; }
    Position<Flat>& operator*=(double a)
    { _x *= a; _y *= a; return *this; }
    Position<Flat>& operator/=(double a)
    { _x /= a; _y /= a; return *this; }

    Position<Flat> operator+(const Position<Flat>& p2) const
    { Position<Flat> p1 = *this; p1 += p2; return p1; }
    Position<Flat> operator-(const Position<Flat>& p2) const
    { Position<Flat> p1 = *this; p1 -= p2; return p1; }
    Position<Flat> operator*(double a) const
    { Position<Flat> p1 = *this; p1 *= a; return p1; }
    Position<Flat> operator/(double a) const
    { Position<Flat> p1 = *this; p1 /= a; return p1; }

    void read(std::istream& fin) { fin >> _x >> _y; }
    void write(std::ostream& fout) const
    { fout << _x << " " << _y << " "; }

private:

    double _x,_y;

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
    Position() : _x(0.), _y(0.), _z(0.) {}
    Position(const Position<ThreeD>& rhs) : 
        _x(rhs._x), _y(rhs._y), _z(rhs._z) {}
    ~Position() {}
    Position(double x, double y, double z) :
        _x(x), _y(y), _z(z) {}
    Position<ThreeD>& operator=(const Position<ThreeD>& rhs) 
    { _x = rhs.getX(); _y = rhs.getY(); _z = rhs.getZ(); return *this; }

    double getX() const { return _x; }
    double getY() const { return _y; }
    double getZ() const { return _z; }
    double get(int split) const { return split==2 ? _z : split==1 ? _y : _x; }

    double normSq() const { return _x*_x + _y*_y + _z*_z; }
    double norm() const { return sqrt(normSq()); }
    void normalize() {}

    Position<ThreeD>& operator+=(const Position<ThreeD>& p2)
    { _x += p2.getX(); _y += p2.getY(); _z += p2.getZ(); return *this; }
    Position<ThreeD>& operator-=(const Position<ThreeD>& p2)
    { _x -= p2.getX(); _y -= p2.getY(); _z -= p2.getZ(); return *this; }
    Position<ThreeD>& operator*=(double a)
    { _x *= a; _y *= a; _z *= a; return *this; }
    Position<ThreeD>& operator/=(double a)
    { _x /= a; _y /= a; _z /= a; return *this; }

    Position<ThreeD> operator+(const Position<ThreeD>& p2) const
    { Position<ThreeD> p1 = *this; p1 += p2; return p1; }
    Position<ThreeD> operator-(const Position<ThreeD>& p2) const
    { Position<ThreeD> p1 = *this; p1 -= p2; return p1; }
    Position<ThreeD> operator*(double a) const
    { Position<ThreeD> p1 = *this; p1 *= a; return p1; }
    Position<ThreeD> operator/(double a) const
    { Position<ThreeD> p1 = *this; p1 /= a; return p1; }

    void read(std::istream& fin) 
    { fin >> _x >> _y >> _z; }
    void write(std::ostream& fout) const
    { fout << _x << " " << _y << " " << _z << " "; }

private:
    double _x,_y,_z;

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
    ~Position() {}
    Position(double x, double y, double z) : Position<ThreeD>(x,y,z) {}
    Position<Sphere>& operator=(const Position<Sphere>& rhs) 
    { Position<ThreeD>::operator=(rhs); return *this; }

    // If appropriate, put the position back on the unit sphere.
    void normalize() { *this /= norm(); }

    Position<Sphere>& operator+=(const Position<Sphere>& p2)
    { Position<ThreeD>::operator+=(p2); return *this; }
    Position<Sphere>& operator-=(const Position<Sphere>& p2)
    { Position<ThreeD>::operator-=(p2); return *this; }
    Position<Sphere>& operator*=(double a)
    { Position<ThreeD>::operator*=(a); return *this; }
    Position<Sphere>& operator/=(double a)
    { Position<ThreeD>::operator/=(a); return *this; }

    Position<Sphere> operator+(const Position<Sphere>& p2) const
    { Position<Sphere> p1 = *this; p1 += p2; return p1; }
    Position<Sphere> operator-(const Position<Sphere>& p2) const
    { Position<Sphere> p1 = *this; p1 -= p2; return p1; }
    Position<Sphere> operator*(double a) const
    { Position<Sphere> p1 = *this; p1 *= a; return p1; }
    Position<Sphere> operator/(double a) const
    { Position<Sphere> p1 = *this; p1 /= a; return p1; }

}; // Position<Sphere>


#endif
