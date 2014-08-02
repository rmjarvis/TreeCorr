#ifndef Corr2_H
#define Corr2_H

#include <vector>
#include <string>

#include "Cell.h"
#include "Field.h"

template <int DC1, int DC2>
struct XiData
{
    XiData(double* xi0, double*, double*, double*) : xi(xi0) {}
    void Copy(const XiData<DC1,DC2>& rhs,int n) 
    { for (int i=0; i<n; ++i) xi[i] = rhs.xi[i]; }
    void Add(const XiData<DC1,DC2>& rhs,int n) 
    { for (int i=0; i<n; ++i) xi[i] += rhs.xi[i]; }

    double* xi;
};

template <int DC1>
struct XiData<DC1, GData>
{
    XiData(double* xi0, double* xi1, double*, double*) : xi(xi0), xi_im(xi1) {}
    void Copy(const XiData<DC1,GData>& rhs,int n) 
    { 
        for (int i=0; i<n; ++i) xi[i] = rhs.xi[i]; 
        for (int i=0; i<n; ++i) xi_im[i] = rhs.xi_im[i]; 
    }
    void Add(const XiData<DC1,GData>& rhs,int n) 
    {
        for (int i=0; i<n; ++i) xi[i] += rhs.xi[i]; 
        for (int i=0; i<n; ++i) xi_im[i] += rhs.xi_im[i]; 
    }

    double* xi;
    double* xi_im;
};

template <>
struct XiData<GData, GData>
{
    XiData(double* xi0, double* xi1, double* xi2, double* xi3) :
        xip(xi0), xip_im(xi1), xim(xi2), xim_im(xi3) {}
    void Copy(const XiData<GData,GData>& rhs,int n) 
    { 
        for (int i=0; i<n; ++i) xip[i] = rhs.xip[i]; 
        for (int i=0; i<n; ++i) xip_im[i] = rhs.xip_im[i]; 
        for (int i=0; i<n; ++i) xim[i] = rhs.xim[i]; 
        for (int i=0; i<n; ++i) xim_im[i] = rhs.xim_im[i]; 
    }
    void Add(const XiData<GData,GData>& rhs,int n) 
    {
        for (int i=0; i<n; ++i) xip[i] += rhs.xip[i]; 
        for (int i=0; i<n; ++i) xip_im[i] += rhs.xip_im[i]; 
        for (int i=0; i<n; ++i) xim[i] += rhs.xim[i]; 
        for (int i=0; i<n; ++i) xim_im[i] += rhs.xim_im[i]; 
    }
    double* xip;
    double* xip_im;
    double* xim;
    double* xim_im;
};

template <>
struct XiData<NData, NData>
{
    XiData(double* , double* , double* , double* ) {}
    void Copy(const XiData<NData,NData>& rhs,int n) {}
    void Add(const XiData<NData,NData>& rhs,int n) {}
};


// BinnedCorr2 encapsulates a binned correlation function.
template <int DC1, int DC2>
class BinnedCorr2
{

public:

    BinnedCorr2(double minsep, double maxsep, int nbins, double binsize, double b,
                double* xi, double* xi1, double* xi2, double* xi3,
                double* varxi, double* meanlogr, double* weight, double* npairs);

    template <int M>
    void process(const Field<DC1, M>& field);
    template <int M>
    void process(const Field<DC1, M>& field1, const Field<DC2, M>& field2);
    //void processPairwise(const InputFile& file1, const InputFile& file2);

    // Main worker functions for calculating the result
    template <int M>
    void process2(const Cell<DC1,M>& c12);

    template <int M>
    void process11(const Cell<DC1,M>& c1, const Cell<DC2,M>& c2);

    template <int M>
    void directProcess11(const Cell<DC1,M>& c1, const Cell<DC2,M>& c2, const double dsq);

    // Note: op= only copies _data.  Not all the params.
    void operator=(const BinnedCorr2<DC1,DC2>& rhs);
    void operator+=(const BinnedCorr2<DC1,DC2>& rhs);

protected:

    //template <int M>
    //void doProcessPairwise(const InputFile& file1, const InputFile& file2);

    double _minsep;
    double _maxsep;
    int _nbins;
    double _binsize;
    double _b;
    double _logminsep;
    double _halfminsep;
    double _minsepsq;
    double _maxsepsq;
    double _bsq;
    int _metric; // Stores which Metric is being used for the analysis.

    // These are all allocated in the python layer and just built up here.
    // So all we have here is a bare pointer for each of them.
    // The different correlation functions have different numbers of arrays for xi, 
    // so encapsulate that difference with a templated XiData class.
    XiData<DC1,DC2> _xi;
    double* _varxi;
    double* _meanlogr;
    double* _weight;
    double* _npairs;
};


#endif
