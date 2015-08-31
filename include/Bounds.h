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
//---------------------------------------------------------------------------
#ifndef BoundsH
#define BoundsH

#include <complex>

//---------------------------------------------------------------------------

// We use a code for the metric to use:
// Flat = flat-sky approximation using (x,y)
// Sphere = spherical geometry using (ra,dec,r) or (x,y,z)
// Perp = 3D, but using the perpendicular component of the separation as the distance
enum Metric { Flat=1, Sphere=2, Perp=3 };

template <int M>
class Position;

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

inline double DistSq(const Position<Flat>& p1, const Position<Flat>& p2)
{ 
    Position<Flat> r = p1-p2;
    return r.getX()*r.getX() + r.getY()*r.getY(); 
}
inline double Dist(const Position<Flat>& p1, const Position<Flat>& p2)
{ return sqrt(DistSq(p1,p2)); }

inline bool CCW(const Position<Flat>& p1, const Position<Flat>& p2, const Position<Flat>& p3)
{
    // If cross product r21 x r31 > 0, then the points are counter-clockwise.
    double x2 = p2.getX() - p1.getX();
    double y2 = p2.getY() - p1.getY();
    double x3 = p3.getX() - p1.getX();
    double y3 = p3.getY() - p1.getY();
    return (x2*y3 - x3*y2) > 0.;
}
template <int M>
inline std::ostream& operator<<(std::ostream& os, const Position<M>& pos)
{ pos.write(os); return os; }

template <int M>
inline std::istream& operator>>(std::istream& os, Position<M>& pos)
{ pos.read(os); return os; }

template <int M>
class Bounds;

template <>
class Bounds<Flat> 
{
    // Basically just a rectangle.  This is used to keep track of the bounds of
    // catalogs and fields.  You can set values, but generally you just keep
    // including positions of each galaxy or the bounds of each catalog
    // respectively using the += operators

public:
    Bounds(double x1, double x2, double y1, double y2) :
        _defined(1), _xmin(x1), _xmax(x2), _ymin(y1), _ymax(y2) {}
    Bounds(const Position<Flat>& pos) :
        _defined(1), _xmin(pos.getX()), _xmax(pos.getX()),
        _ymin(pos.getY()), _ymax(pos.getY()) {}
    Bounds() : _defined(0), _xmin(0.), _xmax(0.), _ymin(0.), _ymax(0.) {}
    ~Bounds() {}
    double getXMin() const { return _xmin; }
    double getXMax() const { return _xmax; }
    double getYMin() const { return _ymin; }
    double getYMax() const { return _ymax; }
    bool isDefined() const { return _defined; }

    // Expand the bounds to include the given position.
    void operator+=(const Position<Flat>& pos)
    {
        if (_defined) {
            if (pos.getX() < _xmin) _xmin = pos.getX();
            else if (pos.getX() > _xmax) _xmax = pos.getX();
            if (pos.getY() < _ymin) _ymin = pos.getY();
            else if (pos.getY() > _ymax) _ymax = pos.getY();
        } else {
            _xmin = _xmax = pos.getX();
            _ymin = _ymax = pos.getY();
            _defined = 1;
        }
    }

    void write(std::ostream& fout) const
    { fout << _xmin << ' ' << _xmax << ' ' << _ymin << ' ' << _ymax << ' '; }
    void read(std::istream& fin)
    { fin >> _xmin >> _xmax >> _ymin >> _ymax; _defined = true; }

    int getSplit()
    { return (_ymax-_ymin) > (_xmax-_xmin) ? 1 : 0; }
    double getMiddle(int split)
    { return split==1 ? (_ymax+_ymin)/2. : (_xmax+_xmin)/2.; }

private:
    bool _defined;
    double _xmin,_xmax,_ymin,_ymax;

};

template <int M>
inline std::ostream& operator<<(std::ostream& fout, const Bounds<M>& b)
{ b.write(fout); return fout;}

template <int M>
inline std::istream& operator>>(std::istream& fin, Bounds<M>& b)
{ b.read(fin); return fin;}


// For spherical metric, we store things as (x,y,z)
// x = cos(dec) cos(ra)
// y = cos(dec) sin(ra)
// z = sin(dec)
template <>
class Position<Sphere> 
{

public:
    Position() : _x(0.), _y(0.), _z(0.), _spher(false) {}
    Position(const Position<Sphere>& rhs) : 
        _x(rhs._x), _y(rhs._y), _z(rhs._z), _spher(rhs._spher) {}
    ~Position() {}
    Position(double x, double y, double z, bool spher) :
        _x(x), _y(y), _z(z), _spher(spher) {}
    Position<Sphere>& operator=(const Position<Sphere>& rhs) 
    { _x = rhs.getX(); _y = rhs.getY(); _z = rhs.getZ(); _spher = rhs.isSpher(); return *this; }

    double getX() const { return _x; }
    double getY() const { return _y; }
    double getZ() const { return _z; }
    bool isSpher() const { return _spher; }
    double get(int split) const { return split==2 ? _z : split==1 ? _y : _x; }

    double normSq() const { return _x*_x + _y*_y + _z*_z; }
    double norm() const { return sqrt(normSq()); }

    // If appropriate, put the position back on the unit sphere.
    void normalize() { if (_spher) *this /= norm(); }

    Position<Sphere>& operator+=(const Position<Sphere>& p2)
    { _x += p2.getX(); _y += p2.getY(); _z += p2.getZ(); return *this; }
    Position<Sphere>& operator-=(const Position<Sphere>& p2)
    { _x -= p2.getX(); _y -= p2.getY(); _z -= p2.getZ(); return *this; }
    Position<Sphere>& operator*=(double a)
    { _x *= a; _y *= a; _z *= a; return *this; }
    Position<Sphere>& operator/=(double a)
    { _x /= a; _y /= a; _z /= a; return *this; }

    Position<Sphere> operator+(const Position<Sphere>& p2) const
    { Position<Sphere> p1 = *this; p1 += p2; return p1; }
    Position<Sphere> operator-(const Position<Sphere>& p2) const
    { Position<Sphere> p1 = *this; p1 -= p2; return p1; }
    Position<Sphere> operator*(double a) const
    { Position<Sphere> p1 = *this; p1 *= a; return p1; }
    Position<Sphere> operator/(double a) const
    { Position<Sphere> p1 = *this; p1 /= a; return p1; }

    void read(std::istream& fin) 
    { fin >> _x >> _y >> _z >> _spher; }
    void write(std::ostream& fout) const
    { fout << _x << " " << _y << " " << _z << " " << _spher << " "; }

private:
    double _x,_y,_z;
    bool _spher;

}; // Position<Sphere>

template <>
class Bounds<Sphere>
{

public:
    Bounds(double x1, double x2, double y1, double y2, double z1, double z2) :
        _defined(1), _xmin(x1), _xmax(x2), _ymin(y1), _ymax(y2), _zmin(z1), _zmax(z2) {}
    Bounds(const Position<Sphere>& pos) :
        _defined(1), _xmin(pos.getX()), _xmax(pos.getX()),
        _ymin(pos.getY()), _ymax(pos.getY()),
        _zmin(pos.getZ()), _zmax(pos.getZ()) {}
    Bounds() : 
        _defined(0), _xmin(0.), _xmax(0.), _ymin(0.), _ymax(0.), _zmin(0.), _zmax(0.) {}
    ~Bounds() {}
    double getXMin() const { return _xmin; }
    double getXMax() const { return _xmax; }
    double getYMin() const { return _ymin; }
    double getYMax() const { return _ymax; }
    double getZMin() const { return _zmin; }
    double getZMax() const { return _zmax; }
    bool isDefined() const { return _defined; }

    // Expand the bounds to include the given position.
    void operator+=(const Position<Sphere>& pos)
    {
        if (_defined) {
            if (pos.getX() < _xmin) _xmin = pos.getX();
            else if (pos.getX() > _xmax) _xmax = pos.getX();
            if (pos.getY() < _ymin) _ymin = pos.getY();
            else if (pos.getY() > _ymax) _ymax = pos.getY();
            if (pos.getZ() < _zmin) _zmin = pos.getZ();
            else if (pos.getZ() > _zmax) _zmax = pos.getZ();
        } else {
            _xmin = _xmax = pos.getX();
            _ymin = _ymax = pos.getY();
            _zmin = _zmax = pos.getZ();
            _defined = 1;
        }
    }

    void write(std::ostream& fout) const
    { 
        fout << _xmin << ' ' << _xmax << ' ' << _ymin << ' ' << _ymax << 
            ' ' << _zmin << ' ' << _zmax << ' '; 
    }
    void read(std::istream& fin)
    { fin >> _xmin >> _xmax >> _ymin >> _ymax >> _zmin >> _zmax; _defined = true; }

    int getSplit()
    { 
        double xrange = _xmax-_xmin;
        double yrange = _ymax-_ymin;
        double zrange = _zmax-_zmin;
        return yrange > xrange ?
            ( zrange > yrange ? 2 : 1 ) : 
            ( zrange > xrange ? 2 : 0 );
    }

    double getMiddle(int split)
    { return split==2 ? (_zmax+_zmin)/2. : split==1 ? (_ymax+_ymin)/2. : (_xmax+_xmin)/2.; }

private:
    bool _defined;
    double _xmin,_xmax,_ymin,_ymax,_zmin,_zmax;

};

inline double DistSq(const Position<Sphere>& p1, const Position<Sphere>& p2)
{ 
    Position<Sphere> r = p1-p2;
    return r.getX()*r.getX() + r.getY()*r.getY() + r.getZ()*r.getZ(); 
}
inline double Dist(const Position<Sphere>& p1, const Position<Sphere>& p2)
{ return sqrt(DistSq(p1,p2)); }

inline bool CCW(const Position<Sphere>& p1, const Position<Sphere>& p2, const Position<Sphere>& p3)
{
    // Now it's slightly more complicated, since the points are in three dimensions.  We do
    // the same thing, computing the cross product with respect to point p1.  Then if the 
    // cross product points back toward Earth, the points are viewed as counter-clockwise.
    // We check this last point by the dot product with p1.
    double x2 = p2.getX() - p1.getX();
    double y2 = p2.getY() - p1.getY();
    double z2 = p2.getZ() - p1.getZ();
    double x3 = p3.getX() - p1.getX();
    double y3 = p3.getY() - p1.getY();
    double z3 = p3.getZ() - p1.getZ();
    double cx = y2*z3 - y3*z2;
    double cy = z2*x3 - z3*x2;
    double cz = x2*y3 - x3*y2;
    return cx*p1.getX() + cy*p1.getY() + cz*p1.getZ() < 0.;
}

template <>
class Position<Perp> : public Position<Sphere> 
{
public:
    Position() : Position<Sphere>(0.,0.,0.,true) {}
    Position(const Position<Perp>& rhs) : Position<Sphere>(rhs) {}
    ~Position() {}
    Position(double x, double y, double z, bool spher) : Position<Sphere>(x,y,z,spher) {}
    Position<Perp>& operator=(const Position<Perp>& rhs) 
    { Position<Sphere>::operator=(rhs); return *this; }

    Position<Perp>& operator+=(const Position<Perp>& p2)
    { Position<Sphere>::operator+=(p2); return *this; }
    Position<Perp>& operator-=(const Position<Perp>& p2)
    { Position<Sphere>::operator-=(p2); return *this; }
    Position<Perp>& operator*=(double a)
    { Position<Sphere>::operator*=(a); return *this; }
    Position<Perp>& operator/=(double a)
    { Position<Sphere>::operator/=(a); return *this; }

    Position<Perp> operator+(const Position<Perp>& p2) const
    { Position<Perp> p1 = *this; p1 += p2; return p1; }
    Position<Perp> operator-(const Position<Perp>& p2) const
    { Position<Perp> p1 = *this; p1 -= p2; return p1; }
    Position<Perp> operator*(double a) const
    { Position<Perp> p1 = *this; p1 *= a; return p1; }
    Position<Perp> operator/(double a) const
    { Position<Perp> p1 = *this; p1 /= a; return p1; }

}; // Position<Perp>

template <>
class Bounds<Perp> : public Bounds<Sphere>
{

public:
    Bounds() {}
    Bounds(double x1, double x2, double y1, double y2, double z1, double z2) :
        Bounds<Sphere>(x1,x2,y1,y2,z1,z2) {}
    Bounds(const Position<Perp>& pos) : Bounds<Sphere>(pos) {}
    ~Bounds() {}

    // Expand the bounds to include the given position.
    void operator+=(const Position<Perp>& pos)
    { Bounds<Sphere>::operator+=(pos); }

};

inline double DistSq(const Position<Perp>& p1, const Position<Perp>& p2)
{ 
    // r_perp^2 + r_parallel^2 = d^2
    Position<Perp> r = p1-p2;
    double dsq = r.getX()*r.getX() + r.getY()*r.getY() + r.getZ()*r.getZ(); 
    double r1sq = p1.getX()*p1.getX() + p1.getY()*p1.getY() + p1.getZ()*p1.getZ(); 
    double r2sq = p2.getX()*p2.getX() + p2.getY()*p2.getY() + p2.getZ()*p2.getZ(); 
    // r_parallel^2 = (r1-r2)^2 = r1^2 + r2^2 - 2r1r2
    double rparsq = r1sq + r2sq - 2.*sqrt(r1sq*r2sq);
    return dsq - rparsq;
}
inline double Dist(const Position<Perp>& p1, const Position<Perp>& p2)
{ return sqrt(DistSq(p1,p2)); }

inline bool CCW(const Position<Perp>& p1, const Position<Perp>& p2, const Position<Perp>& p3)
{
    // This is the same as Sphere
    double x2 = p2.getX() - p1.getX();
    double y2 = p2.getY() - p1.getY();
    double z2 = p2.getZ() - p1.getZ();
    double x3 = p3.getX() - p1.getX();
    double y3 = p3.getY() - p1.getY();
    double z3 = p3.getZ() - p1.getZ();
    double cx = y2*z3 - y3*z2;
    double cy = z2*x3 - z3*x2;
    double cz = x2*y3 - x3*y2;
    return cx*p1.getX() + cy*p1.getY() + cz*p1.getZ() < 0.;
}

#endif
