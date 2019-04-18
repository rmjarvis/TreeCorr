//---------------------------------------------------------------------------
#ifndef BoundsH
#define BoundsH

#include <complex>

//---------------------------------------------------------------------------

class Position2D 
{

public:
    Position2D() : x(0.),y(0.) {}
    Position2D(const Position2D& rhs) : x(rhs.x), y(rhs.y) {}
    ~Position2D() {}
    Position2D(double xin,double yin) : x(xin), y(yin) {}
    Position2D& operator=(const Position2D& rhs) 
    { x = rhs.x; y = rhs.y; return *this; }

    double getX() const { return x; }
    double getY() const { return y; }
    operator std::complex<double>() const { return std::complex<double>(x,y); }

    Position2D& operator+=(const Position2D& p2)
    { x += p2.getX(); y += p2.getY(); return *this; }
    Position2D& operator-=(const Position2D& p2)
    { x -= p2.getX(); y -= p2.getY(); return *this; }
    Position2D& operator*=(double a)
    { x *= a; y *= a; return *this; }
    Position2D& operator/=(double a)
    { x /= a; y /= a; return *this; }

    Position2D operator+(const Position2D& p2) const
    { return Position2D(x+p2.getX(),y+p2.getY()); }
    Position2D operator-(const Position2D& p2) const
    { return Position2D(x-p2.getX(),y-p2.getY()); }
    Position2D operator*(double a) const
    { return Position2D(x*a,y*a); }
    Position2D operator/(double a) const
    { return Position2D(x/a,y/a); }

    void read(std::istream& fin) { fin >> x >> y; }
    void write(std::ostream& fout) const
    { fout << getX() << " " << getY() << " "; }

private:

    double x,y;

}; // Position2D

inline double DistSq(const Position2D& p1, const Position2D& p2)
{ 
    Position2D r = p1-p2;
    return r.getX()*r.getX() + r.getY()*r.getY(); 
}
inline double Dist(const Position2D& p1, const Position2D& p2)
{ return sqrt(DistSq(p1,p2)); }

inline std::ostream& operator<<(std::ostream& os, const Position2D& pos)
{ pos.write(os); return os; }

inline std::istream& operator>>(std::istream& os, Position2D& pos)
{ pos.read(os); return os; }

class Bounds2D 
{
    // Basically just a rectangle.  This is used to keep track of the bounds of
    // catalogs and fields.  You can set values, but generally you just keep
    // including positions of each galaxy or the bounds of each catalog
    // respectively using the += operators

public:
    Bounds2D(double x1, double x2, double y1, double y2):
        defined(1),xmin(x1),xmax(x2),ymin(y1),ymax(y2) {}
    Bounds2D(const Position2D& pos):
        defined(1),xmin(pos.getX()),xmax(pos.getX()),
        ymin(pos.getY()),ymax(pos.getY()) {}
    Bounds2D(): defined(0),xmin(0.),xmax(0.),ymin(0.),ymax(0.) {}
    ~Bounds2D() {}
    double getXMin() const { return xmin; }
    double getXMax() const { return xmax; }
    double getYMin() const { return ymin; }
    double getYMax() const { return ymax; }
    bool isDefined() const { return defined; }
    void operator+=(const Position2D& pos)
        // Expand the bounds to include the given position.
    {
        if(defined) {
            if(pos.getX() < xmin) xmin = pos.getX();
            else if (pos.getX() > xmax) xmax = pos.getX();
            if(pos.getY() < ymin) ymin = pos.getY();
            else if (pos.getY() > ymax) ymax = pos.getY();
        } else {
            xmin = xmax = pos.getX();
            ymin = ymax = pos.getY();
            defined = 1;
        }
    }

    void write(std::ostream& fout) const
    { fout << xmin << ' ' << xmax << ' ' << ymin << ' ' << ymax << ' '; }
    void read(std::istream& fin)
    { fin >> xmin >> xmax >> ymin >> ymax; defined = true; }

private:
    bool defined;
    double xmin,xmax,ymin,ymax;

};

inline std::ostream& operator<<(std::ostream& fout, const Bounds2D& b)
{ b.write(fout); return fout;}

inline std::istream& operator>>(std::istream& fin, Bounds2D& b)
{ b.read(fin); return fin;}

class Position3D 
{

public:
    Position3D() : x(0.),y(0.),z(0.) {}
    Position3D(const Position3D& rhs) : x(rhs.x),y(rhs.y),z(rhs.z) {}
    ~Position3D() {}
    Position3D(double xin,double yin,double zin) : x(xin),y(yin),z(zin) {}
    Position3D& operator=(const Position3D& rhs) 
    { x = rhs.getX(); y = rhs.getY(); z = rhs.getZ(); return *this; }

    double getX() const { return(x); }
    double getY() const { return(y); }
    double getZ() const { return(z); }

    Position3D& operator+=(const Position3D& p2)
    { x += p2.getX(); y += p2.getY(); z += p2.getZ(); return *this; }
    Position3D& operator-=(const Position3D& p2)
    { x -= p2.getX(); y -= p2.getY(); z -= p2.getZ(); return *this; }
    Position3D& operator*=(double a)
    { x *= a; y *= a; z *= a; return *this; }
    Position3D& operator/=(double a)
    { x /= a; y /= a; z /= a; return *this; }

    Position3D operator+(const Position3D& p2) const
    { return Position3D(x+p2.getX(),y+p2.getY(),z+p2.getZ()); }
    Position3D operator-(const Position3D& p2) const
    { return Position3D(x-p2.getX(),y-p2.getY(),z-p2.getZ()); }
    Position3D operator*(double a) const
    { return Position3D(x*a,y*a,z*a); }
    Position3D operator/(double a) const
    { return Position3D(x/a,y/a,z/a); }

    void read(std::istream& fin) 
    { fin >> x >> y >> z; }
    void write(std::ostream& fout) const
    { fout << x << " " << y << " " << z << " "; }

private:
    double x,y,z;

}; // Position3D

inline std::ostream& operator<<(std::ostream& os, const Position3D& pos)
{ pos.write(os); return os; }

inline std::istream& operator>>(std::istream& os, Position3D& pos)
{ pos.read(os); return os; }

class Bounds3D 
{

public:
    Bounds3D(double x1, double x2, double y1, double y2, double z1, double z2) :
        defined(1),xmin(x1),xmax(x2),ymin(y1),ymax(y2),zmin(z1),zmax(z2) {}
    Bounds3D(const Position3D& pos) :
        defined(1),xmin(pos.getX()),xmax(pos.getX()),
        ymin(pos.getY()),ymax(pos.getY()),
        zmin(pos.getZ()),zmax(pos.getZ()) {}
    Bounds3D() : 
        defined(0),xmin(0.),xmax(0.),ymin(0.),ymax(0.), zmin(0.),zmax(0.) {}
    ~Bounds3D() {}
    double getXMin() const { return xmin; }
    double getXMax() const { return xmax; }
    double getYMin() const { return ymin; }
    double getYMax() const { return ymax; }
    double getZMin() const { return zmin; }
    double getZMax() const { return zmax; }
    bool isDefined() const { return defined; }
    void operator+=(const Position3D& pos)
        // Expand the bounds to include the given position.
    {
        if(defined) {
            if(pos.getX() < xmin) xmin = pos.getX();
            else if (pos.getX() > xmax) xmax = pos.getX();
            if(pos.getY() < ymin) ymin = pos.getY();
            else if (pos.getY() > ymax) ymax = pos.getY();
            if(pos.getZ() < zmin) zmin = pos.getZ();
            else if (pos.getZ() > zmax) zmax = pos.getZ();
        } else {
            xmin = xmax = pos.getX();
            ymin = ymax = pos.getY();
            zmin = zmax = pos.getZ();
            defined = 1;
        }
    }

    void write(std::ostream& fout) const
    { 
        fout << xmin << ' ' << xmax << ' ' << ymin << ' ' << ymax << 
            ' ' << zmin << ' ' << zmax << ' '; 
    }
    void read(std::istream& fin)
    { fin >> xmin >> xmax >> ymin >> ymax >> zmin >> zmax; defined = true; }

private:
        bool defined;
        double xmin,xmax,ymin,ymax,zmin,zmax;

};

inline double DistSq(const Position3D& p1, const Position3D& p2)
{ 
    Position3D r = p1-p2;
    return r.getX()*r.getX() + r.getY()*r.getY() + r.getZ()*r.getZ(); 
}
inline double Dist(const Position3D& p1, const Position3D& p2)
{ return sqrt(DistSq(p1,p2)); }

inline std::ostream& operator<<(std::ostream& fout, const Bounds3D& b)
{ b.write(fout); return fout;}

inline std::istream& operator>>(std::istream& fin, Bounds3D& b)
{ b.read(fin); return fin;}

#endif
