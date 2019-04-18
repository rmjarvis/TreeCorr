//---------------------------------------------------------------------------
#ifndef BoundsH
#define BoundsH

#include <complex>

//---------------------------------------------------------------------------

// We use a code for the metric to use -- flat-sky or spherical geometry.
enum Metric { Flat=1, Sphere=2 };

template <int M>
class Position;

template <>
class Position<Flat>
{

public:
    Position() : x(0.),y(0.) {}
    Position(const Position<Flat>& rhs) : x(rhs.x), y(rhs.y) {}
    ~Position() {}
    Position(double xin,double yin) : x(xin), y(yin) {}
    Position& operator=(const Position<Flat>& rhs) 
    { x = rhs.x; y = rhs.y; return *this; }

    double getX() const { return x; }
    double getY() const { return y; }
    double get(int split) const { return split==1 ? y : x; }
    operator std::complex<double>() const { return std::complex<double>(x,y); }

    double normSq() const { return x*x+y*y; }
    double norm() const { return sqrt(normSq()); }

    Position<Flat>& operator+=(const Position<Flat>& p2)
    { x += p2.getX(); y += p2.getY(); return *this; }
    Position<Flat>& operator-=(const Position<Flat>& p2)
    { x -= p2.getX(); y -= p2.getY(); return *this; }
    Position<Flat>& operator*=(double a)
    { x *= a; y *= a; return *this; }
    Position<Flat>& operator/=(double a)
    { x /= a; y /= a; return *this; }

    Position<Flat> operator+(const Position<Flat>& p2) const
    { return Position<Flat>(x+p2.getX(),y+p2.getY()); }
    Position<Flat> operator-(const Position<Flat>& p2) const
    { return Position<Flat>(x-p2.getX(),y-p2.getY()); }
    Position<Flat> operator*(double a) const
    { return Position<Flat>(x*a,y*a); }
    Position<Flat> operator/(double a) const
    { return Position<Flat>(x/a,y/a); }

    void read(std::istream& fin) { fin >> x >> y; }
    void write(std::ostream& fout) const
    { fout << getX() << " " << getY() << " "; }

private:

    double x,y;

}; // Position<Flat>

inline double DistSq(const Position<Flat>& p1, const Position<Flat>& p2)
{ 
    Position<Flat> r = p1-p2;
    return r.getX()*r.getX() + r.getY()*r.getY(); 
}
inline double Dist(const Position<Flat>& p1, const Position<Flat>& p2)
{ return sqrt(DistSq(p1,p2)); }

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
    Bounds(double x1, double x2, double y1, double y2):
        defined(1),xmin(x1),xmax(x2),ymin(y1),ymax(y2) {}
    Bounds(const Position<Flat>& pos):
        defined(1),xmin(pos.getX()),xmax(pos.getX()),
        ymin(pos.getY()),ymax(pos.getY()) {}
    Bounds(): defined(0),xmin(0.),xmax(0.),ymin(0.),ymax(0.) {}
    ~Bounds() {}
    double getXMin() const { return xmin; }
    double getXMax() const { return xmax; }
    double getYMin() const { return ymin; }
    double getYMax() const { return ymax; }
    bool isDefined() const { return defined; }
    void operator+=(const Position<Flat>& pos)
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

    int getSplit()
    { return (ymax-ymin) > (xmax-xmin) ? 1 : 0; }
    double getMiddle(int split)
    { return split==1 ? (ymax+ymin)/2. : (xmax+xmin)/2.; }

private:
    bool defined;
    double xmin,xmax,ymin,ymax;

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
    Position() : x(0.),y(0.),z(0.) {}
    Position(const Position<Sphere>& rhs) : x(rhs.x),y(rhs.y),z(rhs.z) {}
    ~Position() {}
    Position(double xin,double yin,double zin) : x(xin),y(yin),z(zin) {}
    Position<Sphere>& operator=(const Position<Sphere>& rhs) 
    { x = rhs.getX(); y = rhs.getY(); z = rhs.getZ(); return *this; }

    // Position<Sphere> can also be initialized with a Postion<Flat> object, which is 
    // taken to be RA, Dec, both in radians.  The <Sphere> position is then the 
    // corresponding point on the unit sphere.
    Position(const Position<Flat>& rhs) 
    {
        const double ra = rhs.getX();
        const double dec = rhs.getY();
        const double cosra = cos(ra);
        const double sinra = sin(ra);
        const double cosdec = cos(dec);
        const double sindec = sin(dec);
        x = cosdec * cosra;
        y = cosdec * sinra;
        z = sindec;
    }

    double getX() const { return x; }
    double getY() const { return y; }
    double getZ() const { return z; }
    double get(int split) const { return split==2 ? z : split==1 ? y : x; }

    double normSq() const { return x*x+y*y+z*z; }
    double norm() const { return sqrt(normSq()); }

    Position<Sphere>& operator+=(const Position<Sphere>& p2)
    { x += p2.getX(); y += p2.getY(); z += p2.getZ(); return *this; }
    Position<Sphere>& operator-=(const Position<Sphere>& p2)
    { x -= p2.getX(); y -= p2.getY(); z -= p2.getZ(); return *this; }
    Position<Sphere>& operator*=(double a)
    { x *= a; y *= a; z *= a; return *this; }
    Position<Sphere>& operator/=(double a)
    { x /= a; y /= a; z /= a; return *this; }

    Position<Sphere> operator+(const Position<Sphere>& p2) const
    { return Position<Sphere>(x+p2.getX(),y+p2.getY(),z+p2.getZ()); }
    Position<Sphere> operator-(const Position<Sphere>& p2) const
    { return Position<Sphere>(x-p2.getX(),y-p2.getY(),z-p2.getZ()); }
    Position<Sphere> operator*(double a) const
    { return Position<Sphere>(x*a,y*a,z*a); }
    Position<Sphere> operator/(double a) const
    { return Position<Sphere>(x/a,y/a,z/a); }

    void read(std::istream& fin) 
    { fin >> x >> y >> z; }
    void write(std::ostream& fout) const
    { fout << x << " " << y << " " << z << " "; }

private:
    double x,y,z;

}; // Position<Sphere>

template <>
class Bounds<Sphere>
{

public:
    Bounds(double x1, double x2, double y1, double y2, double z1, double z2) :
        defined(1),xmin(x1),xmax(x2),ymin(y1),ymax(y2),zmin(z1),zmax(z2) {}
    Bounds(const Position<Sphere>& pos) :
        defined(1),xmin(pos.getX()),xmax(pos.getX()),
        ymin(pos.getY()),ymax(pos.getY()),
        zmin(pos.getZ()),zmax(pos.getZ()) {}
    Bounds() : 
        defined(0),xmin(0.),xmax(0.),ymin(0.),ymax(0.), zmin(0.),zmax(0.) {}
    ~Bounds() {}
    double getXMin() const { return xmin; }
    double getXMax() const { return xmax; }
    double getYMin() const { return ymin; }
    double getYMax() const { return ymax; }
    double getZMin() const { return zmin; }
    double getZMax() const { return zmax; }
    bool isDefined() const { return defined; }
    void operator+=(const Position<Sphere>& pos)
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

    int getSplit()
    { 
        double xrange = xmax-xmin;
        double yrange = ymax-ymin;
        double zrange = zmax-zmin;
        return yrange > xrange ?
            ( zrange > yrange ? 2 : 1 ) : 
            ( zrange > xrange ? 2 : 0 );
    }

    double getMiddle(int split)
    { return split==2 ? (zmax+zmin)/2. : split==1 ? (ymax+ymin)/2. : (xmax+xmin)/2.; }

private:
    bool defined;
    double xmin,xmax,ymin,ymax,zmin,zmax;

};

inline double DistSq(const Position<Sphere>& p1, const Position<Sphere>& p2)
{ 
    Position<Sphere> r = p1-p2;
    return r.getX()*r.getX() + r.getY()*r.getY() + r.getZ()*r.getZ(); 
}
inline double Dist(const Position<Sphere>& p1, const Position<Sphere>& p2)
{ return sqrt(DistSq(p1,p2)); }


#endif
