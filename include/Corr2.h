/* Copyright (c) 2003-2024 by Mike Jarvis
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

#ifndef TreeCorr_Corr2_H
#define TreeCorr_Corr2_H

#include <vector>
#include <string>

#include "Cell.h"
#include "Field.h"
#include "BinType.h"
#include "Metric.h"

template <int D1, int D2>
struct XiData;

// Things that don't need to access the data vectors can live in the base class.
class BaseCorr2
{
public:

    BaseCorr2(BinType bin_type, double minsep, double maxsep, int nbins, double binsize,
              double b, double a,
              double minrpar, double maxrpar, double xp, double yp, double zp);
    BaseCorr2(const BaseCorr2& rhs);
    virtual ~BaseCorr2() {}

    BinType getBinType() const { return _bin_type; }

    virtual std::shared_ptr<BaseCorr2> duplicate() =0;

    virtual void addData(const BaseCorr2& rhs) =0;

    // Sample a random subset of pairs in a given range
    template <int B, int M, int P, int C>
    long samplePairs(const BaseField<C>& field1, const BaseField<C>& field2,
                     double min_sep, double max_sep, long* i1, long* i2, double* sep, int n);
    template <int B, int M, int P, int C>
    void samplePairs(const BaseCell<C>& c1, const BaseCell<C>& c2, const MetricHelper<M,P>& m,
                     double min_sep, double min_sepsq, double max_sep, double max_sepsq,
                     long* i1, long* i2, double* sep, int n, long& k);
    template <int B, int C>
    void sampleFrom(const BaseCell<C>& c1, const BaseCell<C>& c2, double rsq, double r,
                    long* i1, long* i2, double* sep, int n, long& k);

    bool nontrivialRPar() const
    {
        return (_minrpar != -std::numeric_limits<double>::max() ||
                _maxrpar != std::numeric_limits<double>::max());
    }

    template <int B, int M, int C>
    bool triviallyZero(Position<C> p1, Position<C> p2, double s1, double s2);

    template <int B, int M, int P, int C>
    void process(const BaseField<C>& field, bool dots, bool quick);

    template <int B, int M, int P, int C>
    void process(const BaseField<C>& field1, const BaseField<C>& field2, bool dots, bool quick);

    // Main worker functions for calculating the result
    template <int B, int M, int P, int C>
    void process2(const BaseCell<C>& c12, const MetricHelper<M,P>& m, bool quick);

    template <int B, int M, int P, int Q, int R, int C>
    void process11(const BaseCell<C>& c1, const BaseCell<C>& c2, const MetricHelper<M,P>& m);

    template <int B, int Q, int R, int C>
    void directProcess11(const BaseCell<C>& c1, const BaseCell<C>& c2, const double rsq,
                         int k=-1, double r=0., double logr=0.);

    template <int Q, int R, int C>
    void finishProcess(const BaseCell<C>& c1, const BaseCell<C>& c2,
                       double rsq, double r, double logr, int k, int k2)
    { doFinishProcess(c1, c2, rsq, r, logr, k, k2, R1<Q,R>()); }

protected:
    template <int Q, int R>
    struct R1 {};

    // This bit is a workaround for the the fact that virtual functions cannot be templates.
    virtual void doFinishProcess(const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
                                 double rsq, double r, double logr, int k, int k2, R1<1,0>)=0;
    virtual void doFinishProcess(const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
                                 double rsq, double r, double logr, int k, int k2, R1<1,0>)=0;
    virtual void doFinishProcess(const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
                                 double rsq, double r, double logr, int k, int k2, R1<1,0>)=0;
    virtual void doFinishProcess(const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
                                 double rsq, double r, double logr, int k, int k2, R1<0,0>)=0;
    virtual void doFinishProcess(const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
                                 double rsq, double r, double logr, int k, int k2, R1<0,0>)=0;
    virtual void doFinishProcess(const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
                                 double rsq, double r, double logr, int k, int k2, R1<0,0>)=0;
    virtual void doFinishProcess(const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
                                 double rsq, double r, double logr, int k, int k2, R1<1,1>)=0;
    virtual void doFinishProcess(const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
                                 double rsq, double r, double logr, int k, int k2, R1<1,1>)=0;
    virtual void doFinishProcess(const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
                                 double rsq, double r, double logr, int k, int k2, R1<1,1>)=0;
    virtual void doFinishProcess(const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
                                 double rsq, double r, double logr, int k, int k2, R1<0,1>)=0;
    virtual void doFinishProcess(const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
                                 double rsq, double r, double logr, int k, int k2, R1<0,1>)=0;
    virtual void doFinishProcess(const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
                                 double rsq, double r, double logr, int k, int k2, R1<0,1>)=0;

    BinType _bin_type;
    double _minsep;
    double _maxsep;
    int _nbins;
    double _binsize;
    double _b;
    double _a;
    double _minrpar, _maxrpar;
    double _xp, _yp, _zp;
    double _logminsep;
    double _halfminsep;
    double _minsepsq;
    double _maxsepsq;
    double _bsq;
    double _asq;
    double _fullmaxsep;
    double _fullmaxsepsq;
    int _coords; // Stores the kind of coordinates being used for the analysis.
};

// Corr2 encapsulates a binned correlation function.
template <int D1, int D2>
class Corr2 : public BaseCorr2
{

public:

    Corr2(BinType bin_type, double minsep, double maxsep, int nbins, double binsize,
          double b, double a,
          double minrpar, double maxrpar, double xp, double yp, double zp,
          double* xi0, double* xi1, double* xi2, double* xi3,
          double* meanr, double* meanlogr, double* weight, double* npairs);
    Corr2(const Corr2& rhs, bool copy_data=true);
    ~Corr2();

    std::shared_ptr<BaseCorr2> duplicate()
    { return std::make_shared<Corr2<D1,D2> >(*this, false); }

    void addData(const BaseCorr2& rhs)
    { *this += static_cast<const Corr2<D1,D2>&>(rhs); }

    void clear();  // Set all data to 0.

    template <int Q, int R, int C>
    void finishProcess(const BaseCell<C>& c1, const BaseCell<C>& c2,
                       double rsq, double r, double logr, int k, int k2);

    // Note: op= only copies _data.  Not all the params.
    void operator=(const Corr2<D1,D2>& rhs);
    void operator+=(const Corr2<D1,D2>& rhs);

protected:

    void doFinishProcess(const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<1,0>)
    { finishProcess<1,0>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<1,0>)
    { finishProcess<1,0>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<1,0>)
    { finishProcess<1,0>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<0,0>)
    { finishProcess<0,0>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<0,0>)
    { finishProcess<0,0>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<0,0>)
    { finishProcess<0,0>(c1, c2, rsq, r, logr, k, k2); }

    void doFinishProcess(const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<1,1>)
    { finishProcess<1,1>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<1,1>)
    { finishProcess<1,1>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<1,1>)
    { finishProcess<1,1>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<0,1>)
    { finishProcess<0,1>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<0,1>)
    { finishProcess<0,1>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<0,1>)
    { finishProcess<0,1>(c1, c2, rsq, r, logr, k, k2); }

    // These are usually allocated in the python layer and just built up here.
    // So all we have here is a bare pointer for each of them.
    // However, for the OpenMP stuff, we do create copies that we need to delete.
    // So keep track if we own the data and need to delete the memory ourselves.
    bool _owns_data;

    // The different correlation functions have different numbers of arrays for xi,
    // so encapsulate that difference with a templated XiData class.
    XiData<D1,D2> _xi;
    double* _meanr;
    double* _meanlogr;
    double* _weight;
    double* _npairs;
};

template <int D1, int D2>
struct XiData // This works for NK, KK
{
    XiData(double* xi0, double*, double*, double*) : xi(xi0) {}

    void new_data(int n) { xi = new double[n]; }
    void delete_data(int n) { delete [] xi; xi = 0; }
    void copy(const XiData<D1,D2>& rhs,int n)
    { for (int i=0; i<n; ++i) xi[i] = rhs.xi[i]; }
    void add(const XiData<D1,D2>& rhs,int n)
    { for (int i=0; i<n; ++i) xi[i] += rhs.xi[i]; }
    void clear(int n)
    { for (int i=0; i<n; ++i) xi[i] = 0.; }
    void write(std::ostream& os) const // Just used for debugging.  Print the first value.
    { os << xi[0]; }

    double* xi;
};

template <int D1, int D2>
inline std::ostream& operator<<(std::ostream& os, const XiData<D1, D2>& xi)
{ xi.write(os); return os; }

template <int D1>
struct XiData<D1, GData> // This works for NG, KG
{
    XiData(double* xi0, double* xi1, double*, double*) : xi(xi0), xi_im(xi1) {}

    void new_data(int n)
    {
        xi = new double[n];
        xi_im = new double[n];
    }
    void delete_data(int n)
    {
        delete [] xi; xi = 0;
        delete [] xi_im; xi_im = 0;
    }
    void copy(const XiData<D1,GData>& rhs,int n)
    {
        for (int i=0; i<n; ++i) xi[i] = rhs.xi[i];
        for (int i=0; i<n; ++i) xi_im[i] = rhs.xi_im[i];
    }
    void add(const XiData<D1,GData>& rhs,int n)
    {
        for (int i=0; i<n; ++i) xi[i] += rhs.xi[i];
        for (int i=0; i<n; ++i) xi_im[i] += rhs.xi_im[i];
    }
    void clear(int n)
    {
        for (int i=0; i<n; ++i) xi[i] = 0.;
        for (int i=0; i<n; ++i) xi_im[i] = 0.;
    }
    void write(std::ostream& os) const
    { os << xi[0]<<','<<xi_im[0]; }

    double* xi;
    double* xi_im;
};

template <>
struct XiData<GData, GData>
{
    XiData(double* xi0, double* xi1, double* xi2, double* xi3) :
        xip(xi0), xip_im(xi1), xim(xi2), xim_im(xi3) {}

    void new_data(int n)
    {
        xip = new double[n];
        xip_im = new double[n];
        xim = new double[n];
        xim_im = new double[n];
    }
    void delete_data(int n)
    {
        delete [] xip; xip = 0;
        delete [] xip_im; xip_im = 0;
        delete [] xim; xim = 0;
        delete [] xim_im; xim_im = 0;
    }
    void copy(const XiData<GData,GData>& rhs,int n)
    {
        for (int i=0; i<n; ++i) xip[i] = rhs.xip[i];
        for (int i=0; i<n; ++i) xip_im[i] = rhs.xip_im[i];
        for (int i=0; i<n; ++i) xim[i] = rhs.xim[i];
        for (int i=0; i<n; ++i) xim_im[i] = rhs.xim_im[i];
    }
    void add(const XiData<GData,GData>& rhs,int n)
    {
        for (int i=0; i<n; ++i) xip[i] += rhs.xip[i];
        for (int i=0; i<n; ++i) xip_im[i] += rhs.xip_im[i];
        for (int i=0; i<n; ++i) xim[i] += rhs.xim[i];
        for (int i=0; i<n; ++i) xim_im[i] += rhs.xim_im[i];
    }
    void clear(int n)
    {
        for (int i=0; i<n; ++i) xip[i] = 0.;
        for (int i=0; i<n; ++i) xip_im[i] = 0.;
        for (int i=0; i<n; ++i) xim[i] = 0.;
        for (int i=0; i<n; ++i) xim_im[i] = 0.;
    }
    void write(std::ostream& os) const
    { os << xip[0]<<','<<xip_im[0]<<','<<xim[0]<<','<<xim_im; }

    double* xip;
    double* xip_im;
    double* xim;
    double* xim_im;
};

template <>
struct XiData<NData, NData>
{
    XiData(double* , double* , double* , double* ) {}
    void new_data(int n) {}
    void delete_data(int n) {}
    void copy(const XiData<NData,NData>& rhs,int n) {}
    void add(const XiData<NData,NData>& rhs,int n) {}
    void clear(int n) {}
    void write(std::ostream& os) const {}
};

// All complex valued work the same as GData
// So just make them sub-types of the GData versions.
template <int D1>
struct XiData<D1, ZData> : public XiData<D1, GData>
{
    XiData(double* xi0, double* xi1, double*, double*) :
        XiData<D1,GData>(xi0,xi1,0,0) {}
};
template <>
struct XiData<ZData, ZData> : public XiData<GData, GData>
{
    XiData(double* xi0, double* xi1, double* xi2, double* xi3) :
        XiData<GData,GData>(xi0,xi1,xi2,xi3) {}
};

template <int D1>
struct XiData<D1, VData> : public XiData<D1, GData>
{
    XiData(double* xi0, double* xi1, double*, double*) :
        XiData<D1,GData>(xi0,xi1,0,0) {}
};
template <>
struct XiData<VData, VData> : public XiData<GData, GData>
{
    XiData(double* xi0, double* xi1, double* xi2, double* xi3) :
        XiData<GData,GData>(xi0,xi1,xi2,xi3) {}
};

template <int D1>
struct XiData<D1, TData> : public XiData<D1, GData>
{
    XiData(double* xi0, double* xi1, double*, double*) :
        XiData<D1,GData>(xi0,xi1,0,0) {}
};
template <>
struct XiData<TData, TData> : public XiData<GData, GData>
{
    XiData(double* xi0, double* xi1, double* xi2, double* xi3) :
        XiData<GData,GData>(xi0,xi1,xi2,xi3) {}
};

template <int D1>
struct XiData<D1, QData> : public XiData<D1, GData>
{
    XiData(double* xi0, double* xi1, double*, double*) :
        XiData<D1,GData>(xi0,xi1,0,0) {}
};
template <>
struct XiData<QData, QData> : public XiData<GData, GData>
{
    XiData(double* xi0, double* xi1, double* xi2, double* xi3) :
        XiData<GData,GData>(xi0,xi1,xi2,xi3) {}
};


class Sampler : public BaseCorr2
{

public:

    Sampler(const BaseCorr2& base_corr2, double minsep, double maxsep,
            long* i1, long* i2, double* sep, int n);
    // Use default copy/destr/op=

    std::shared_ptr<BaseCorr2> duplicate()
    { return std::make_shared<Sampler>(*this); }

    // We only use this with a single sampler, so addData just needs to
    // copy things back to the original sampler.
    void addData(const BaseCorr2& rhs)
    { *this = static_cast<const Sampler&>(rhs); }

    template <int Q, int R, int C>
    void finishProcess(const BaseCell<C>& c1, const BaseCell<C>& c2,
                       double rsq, double r, double logr, int k, int k2);

    long getK() const { return _k; }

protected:

    void doFinishProcess(const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<1,0>)
    { finishProcess<1,0>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<1,0>)
    { finishProcess<1,0>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<1,0>)
    { finishProcess<1,0>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<0,0>)
    { finishProcess<0,0>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<0,0>)
    { finishProcess<0,0>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<0,0>)
    { finishProcess<0,0>(c1, c2, rsq, r, logr, k, k2); }

    void doFinishProcess(const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<1,1>)
    { finishProcess<1,1>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<1,1>)
    { finishProcess<1,1>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<1,1>)
    { finishProcess<1,1>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<0,1>)
    { finishProcess<0,1>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<0,1>)
    { finishProcess<0,1>(c1, c2, rsq, r, logr, k, k2); }
    void doFinishProcess(const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
                         double rsq, double r, double logr, int k, int k2, R1<0,1>)
    { finishProcess<0,1>(c1, c2, rsq, r, logr, k, k2); }

    long* _i1;
    long* _i2;
    double* _sep;
    int _n;
    long _k;
};

#endif
