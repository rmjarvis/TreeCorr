#ifndef FormH
#define FormH

// Formatting for simplified stream output
// see Stroustrup(2000), p. 635

#include <complex>
#include <sstream>
#include <iostream>
#include <string>

template <class T> 
struct Bound_form;

class Form {
    template <class T> 
    friend std::ostream& operator<<(std::ostream&, const Bound_form<T>&);
    friend void FloatSetup(std::ostream&,const Form&);
    friend void IntSetup(std::ostream&,const Form&);

    int prc; // precision
    int wdt; // width, 0 means as wide as necessary
    std::ios_base::fmtflags fmt; // general sci, or fixed
    std::ios_base::fmtflags base; // dec, hex, oct
    std::ios_base::fmtflags just; // left, right, internal fill
    char newfillch; // fill character
    int doupper; // +1 to have uppercase E,X, -1 turn off, (0 leave as is)
    int doplus; // +1 to have explicit plus for positive values, -1 off, 0 same
    int dotrail; // +1 to write trailing zeros, -1 off, 0 same
    int doboolalpha; // +1 to write "true","false", -1 off, 0 same
    int ntrail; // number of spaces after output
    char trailch; // character of trailing "spaces"

public:
    Form(): prc(6), wdt(0), fmt(), base(std::ios_base::dec),
    just(std::ios_base::left), newfillch(0),
    doupper(0), doplus(0), dotrail(0), doboolalpha(0),
    ntrail(1), trailch(' ')  {}
    template <class T> Bound_form<T> operator()(T val) const;

    Form& prec(int p) {prc = p; return *this;}

    Form& sci() {fmt = std::ios_base::scientific; return *this;}
    Form& fix() {fmt = std::ios_base::fixed; return *this;}
    Form& gen() {fmt = ~std::ios_base::floatfield; return *this;}

    Form& width(int w) {wdt = w; return *this;}
    Form& fill(char c) {newfillch = c; return *this;}

    Form& dec() {base = std::ios_base::dec; return *this;}
    Form& oct() {base = std::ios_base::oct; return *this;}
    Form& hex() {base = std::ios_base::hex; return *this;}

    Form& left() {just = std::ios_base::left; return *this;}
    Form& right() {just = std::ios_base::right; return *this;}
    Form& internal() {just = std::ios_base::internal; return *this;}

    Form& uppercase(bool b=true) {doupper = b?1:-1; return *this;}
    Form& showpos(bool b=true) {doplus = b?1:-1; return *this;}
    Form& showpoint(bool b=true) {dotrail = b?1:-1; return *this;}
    Form& boolalpha(bool b=true) {doboolalpha = b?1:-1; return *this;}

    Form& trail(int n,char ch=' ') {ntrail = n; trailch = ch; return *this;}
};

template <class T>
struct Bound_form 
{
    const Form& f;
    T val;
    Bound_form(const Form& ff, T v) : f(ff), val(v) {}
};

template <class T>
inline Bound_form<T> Form::operator()(T val) const 
{ return Bound_form<T>(*this,val); }

inline void FloatSetup(std::ostream& s, const Form& f)
{
    s.precision(f.prc);
    s.setf(f.fmt,std::ios_base::floatfield);
    s.setf(f.just,std::ios_base::adjustfield);
    if (f.wdt) s.width(f.wdt);
    if (f.newfillch) s.fill(f.newfillch);
    if (f.doupper && f.fmt == std::ios_base::scientific) {
        if (f.doupper>0) s.setf(std::ios_base::uppercase);
        else s.unsetf(std::ios_base::uppercase); 
    }
    if (f.doplus) {
        if (f.doplus>0) s.setf(std::ios_base::showpos); 
        else s.unsetf(std::ios_base::showpos); 
    }
    if (f.dotrail) {
        if (f.dotrail>0) s.setf(std::ios_base::showpoint); 
        else s.unsetf(std::ios_base::showpoint); 
    }
}

inline void IntSetup(std::ostream& s, const Form& f)
{
    s.setf(f.just,std::ios_base::adjustfield);
    s.setf(f.base,std::ios_base::basefield);
    if (f.wdt) s.width(f.wdt);
    if (f.newfillch) s.fill(f.newfillch);
    if (f.doupper && f.base == std::ios_base::hex) {
        if (f.doupper>0) s.setf(std::ios_base::uppercase); 
        else s.unsetf(std::ios_base::uppercase); 
    }
    if (f.doplus) {
        if (f.doplus>0) s.setf(std::ios_base::showpos); 
        else s.unsetf(std::ios_base::showpos); 
    }
    if (f.base != std::ios_base::dec) s.setf(std::ios_base::showbase);
}

inline void Setup(std::ostream& os, const Bound_form<double>& bf)
{ FloatSetup(os,bf.f); }
inline void Setup(std::ostream& os, const Bound_form<long double>& bf)
{ FloatSetup(os,bf.f); }
inline void Setup(std::ostream& os, const Bound_form<float>& bf)
{ FloatSetup(os,bf.f); }
inline void Setup(std::ostream& os, const Bound_form<std::complex<double> >& bf)
{ FloatSetup(os,bf.f); }
inline void Setup(std::ostream& os, const Bound_form<std::complex<long double> >& bf)
{ FloatSetup(os,bf.f); }
inline void Setup(std::ostream& os, const Bound_form<std::complex<float> >& bf)
{ FloatSetup(os,bf.f); }
inline void Setup(std::ostream& os, const Bound_form<int>& bf)
{ IntSetup(os,bf.f); }
inline void Setup(std::ostream& os, const Bound_form<short>& bf)
{ IntSetup(os,bf.f); }
inline void Setup(std::ostream& os, const Bound_form<long>& bf)
{ IntSetup(os,bf.f); }
inline void Setup(std::ostream& os, const Bound_form<unsigned int>& bf)
{ IntSetup(os,bf.f); }
inline void Setup(std::ostream& os, const Bound_form<unsigned short>& bf)
{ IntSetup(os,bf.f); }
inline void Setup(std::ostream& os, const Bound_form<unsigned long>& bf)
{ IntSetup(os,bf.f); }
    template<class T>
    inline void Setup(std::ostream& os, const Bound_form<T>& bf)
{ FloatSetup(os,bf.f); }

template <class T>
inline std::ostream& operator<<(std::ostream& os, const Bound_form<T>& bf)
{
    std::ostringstream s;
    Setup(s,bf);
    s << bf.val;
    if (bf.f.ntrail>0) s << std::string(bf.f.ntrail,bf.f.trailch);
    os << s.str();
    return os;
}

#endif
