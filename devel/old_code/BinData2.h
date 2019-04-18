#ifndef BinData2_H
#define BinData2_H

#include <complex>
#include "Cell.h"

// BinData2 stores the information for each bin of the accumulated correlation function.
// Since the details of this is different for each kind of correlation function, we
// have specializations for each pair of data types.


template <int DC1, int DC2> struct BinData2;

template <>
struct BinData2<NData,NData>
{
    BinData2() : meanlogr(0.), npair(0.) {}

    template <int M>
    void directProcess11(const Cell<NData,M>& c1, const Cell<NData,M>& c2,
                         const double dsq, const double logr);
    // The argument here is ignored.  It is present to be compatible with the other 
    // BinData varieties.
    void finalize(double, double );

    void clear() 
    { meanlogr = npair = 0.; }

    void operator+=(const BinData2<NData,NData>& rhs) 
    {
        meanlogr += rhs.meanlogr; 
        npair += rhs.npair;
    }

    double meanlogr, npair;
};

template <>
struct BinData2<NData,GData> 
{
    BinData2() : meangammat(0.), vargammat(0.), meanlogr(0.), weight(0.), npair(0.) {}

    template <int M>
    void directProcess11(const Cell<NData,M>& c1, const Cell<GData,M>& c2,
                         const double dsq, const double logr);
    void finalize(double, double varg);

    void clear() 
    { meangammat = vargammat = meanlogr = weight = npair = 0.; }

    void operator+=(const BinData2<NData,GData>& rhs) 
    {
        meangammat += rhs.meangammat;
        vargammat += rhs.vargammat;
        meanlogr += rhs.meanlogr; 
        weight += rhs.weight;
        npair += rhs.npair;
    }

    std::complex<double> meangammat;
    double vargammat, meanlogr, weight, npair;
};


template <>
struct BinData2<GData,GData> 
{
    BinData2() : xiplus(0.), ximinus(0.), varxi(0.), meanlogr(0.), weight(0.), npair(0.) {}

    template <int M>
    void directProcess11(const Cell<GData,M>& c1, const Cell<GData,M>& c2,
                         const double dsq, const double logr);
    void finalize(double varg1, double varg2);

    void clear() 
    { xiplus = ximinus = varxi = meanlogr = weight = npair = 0.; }

    void operator+=(const BinData2<GData,GData>& rhs) 
    {
        xiplus += rhs.xiplus;
        ximinus += rhs.ximinus;
        varxi += rhs.varxi;
        meanlogr += rhs.meanlogr; 
        weight += rhs.weight;
        npair += rhs.npair;
    }

    std::complex<double> xiplus,ximinus;
    double varxi, meanlogr, weight, npair;
};

template <>
struct BinData2<NData,KData> 
{
    BinData2() : meankappa(0.), varkappa(0.), meanlogr(0.), weight(0.), npair(0.) {}

    template <int M>
    void directProcess11(const Cell<NData,M>& c1, const Cell<KData,M>& c2,
                         const double dsq, const double logr);
    void finalize(double, double vark);

    void clear() 
    { meankappa = varkappa = meanlogr = weight = npair = 0.; }

    void operator+=(const BinData2<NData,KData>& rhs) 
    {
        meankappa += rhs.meankappa;
        varkappa += rhs.varkappa;
        meanlogr += rhs.meanlogr; 
        weight += rhs.weight;
        npair += rhs.npair;
    }

    double meankappa, varkappa, meanlogr, weight, npair;
};


template <>
struct BinData2<KData,KData> 
{
    BinData2() : xi(0.), varxi(0.), meanlogr(0.), weight(0.), npair(0.) {}

    template <int M>
    void directProcess11(const Cell<KData,M>& c1, const Cell<KData,M>& c2,
                         const double dsq, const double logr);
    void finalize(double vark1, double vark2);

    void clear() 
    { xi = varxi = meanlogr = weight = npair = 0.; }

    void operator+=(const BinData2<KData,KData>& rhs) 
    {
        xi += rhs.xi;
        varxi += rhs.varxi;
        meanlogr += rhs.meanlogr; 
        weight += rhs.weight;
        npair += rhs.npair;
    }

    double xi, varxi, meanlogr, weight, npair;
};

template <>
struct BinData2<KData,GData> 
{
    BinData2() : meankgammat(0.), varkgammat(0.), meanlogr(0.), weight(0.), npair(0.) {}

    template <int M>
    void directProcess11(const Cell<KData,M>& c1, const Cell<GData,M>& c2,
                         const double dsq, const double logr);
    void finalize(double vark, double varg);

    void clear() 
    { meankgammat = varkgammat = meanlogr = weight = npair = 0.; }

    void operator+=(const BinData2<KData,GData>& rhs) 
    {
        meankgammat += rhs.meankgammat;
        varkgammat += rhs.varkgammat;
        meanlogr += rhs.meanlogr; 
        weight += rhs.weight;
        npair += rhs.npair;
    }

    std::complex<double> meankgammat;
    double varkgammat, meanlogr, weight, npair;
};



#endif
