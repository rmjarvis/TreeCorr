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

#ifndef TreeCorr_Bounds_H
#define TreeCorr_Bounds_H

#include "Position.h"

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


template <>
class Bounds<ThreeD>
{

public:
    Bounds(double x1, double x2, double y1, double y2, double z1, double z2) :
        _defined(1), _xmin(x1), _xmax(x2), _ymin(y1), _ymax(y2), _zmin(z1), _zmax(z2) {}
    Bounds(const Position<ThreeD>& pos) :
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
    void operator+=(const Position<ThreeD>& pos)
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

template <>
class Bounds<Sphere> : public Bounds<ThreeD>
{

public:
    Bounds() {}
    Bounds(double x1, double x2, double y1, double y2, double z1, double z2) :
        Bounds<ThreeD>(x1,x2,y1,y2,z1,z2) {}
    Bounds(const Position<Sphere>& pos) : Bounds<ThreeD>(pos) {}
    ~Bounds() {}

    // Expand the bounds to include the given position.
    void operator+=(const Position<Sphere>& pos)
    { Bounds<ThreeD>::operator+=(pos); }

};

#endif
