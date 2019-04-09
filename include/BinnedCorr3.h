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

#ifndef TreeCorr_Corr3_H
#define TreeCorr_Corr3_H

#include <vector>
#include <string>

#include "Cell.h"
#include "Field.h"
#include "BinType.h"
#include "Metric.h"

template <int DC1, int DC2, int DC3>
struct ZetaData;

// BinnedCorr3 encapsulates a binned correlation function.
template <int DC1, int DC2, int DC3, int B>
class BinnedCorr3
{

public:

    BinnedCorr3(double minsep, double maxsep, int nbins, double binsize, double b,
                double minu, double maxu, int nubins, double ubinsize, double bu,
                double minv, double maxv, int nvbins, double vbinsize, double bv,
                double minrpar, double maxrpar, double xp, double yp, double zp,
                double* zeta0, double* zeta1, double* zeta2, double* zeta3,
                double* zeta4, double* zeta5, double* zeta6, double* zeta7,
                double* meand1, double* meanlogd1, double* meand2, double* meanlogd2,
                double* meand3, double* meanlogd3, double* meanu, double* meanv,
                double* weight, double* ntri);
    BinnedCorr3(const BinnedCorr3& rhs, bool copy_data=true);
    ~BinnedCorr3();

    void clear();  // Set all data to 0.

    template <int C, int M>
    void process(const Field<DC1, C>& field, bool dots);
    template <int C, int M>
    void process(const Field<DC1, C>& field1, const Field<DC2, C>& field2,
                 const Field<DC3, C>& field3, bool dots);

    // Main worker functions for calculating the result
    template <int C, int M>
    void process3(const Cell<DC1,C>* c123, const MetricHelper<M>& metric);

    template <bool sort, int C, int M>
    void process21(const Cell<DC1,C>* c12, const Cell<DC3,C>* c3, const MetricHelper<M>& metric);

    template <bool sort, int C, int M>
    void process111(const Cell<DC1,C>* c1, const Cell<DC2,C>* c2, const Cell<DC3,C>* c3,
                    const MetricHelper<M>& metric,
                    double d1sq=0., double d2sq=0., double d3sq=0.);

    template <int C, int M>
    void directProcess111(const Cell<DC1,C>& c1, const Cell<DC2,C>& c2, const Cell<DC3,C>& c3,
                          const double d1, const double d2, const double d3,
                          const double logr, const double u, const double v, const int index);

    // Note: op= only copies _data.  Not all the params.
    void operator=(const BinnedCorr3<DC1,DC2,DC3,B>& rhs);
    void operator+=(const BinnedCorr3<DC1,DC2,DC3,B>& rhs);

protected:

    double _minsep;
    double _maxsep;
    int _nbins;
    double _binsize;
    double _b;
    double _minu;
    double _maxu;
    int _nubins;
    double _ubinsize;
    double _bu;
    double _minv;
    double _maxv;
    int _nvbins;
    double _vbinsize;
    double _bv;
    double _minrpar, _maxrpar;
    double _xp, _yp, _zp;
    double _logminsep;
    double _halfminsep;
    double _halfmind3;
    double _minsepsq;
    double _maxsepsq;
    double _minusq;
    double _maxusq;
    double _minvsq;
    double _maxvsq;
    double _bsq;
    double _busq;
    double _bvsq;
    double _sqrttwobv;
    int _coords; // Stores the kind of coordinates being used for the analysis.
    int _nvbins2; // = nvbins * 2
    int _nuv; // = nubins * nvbins2
    int _ntot; // = nbins * nubins2 * nvbins

    // These are usually allocated in the python layer and just built up here.
    // So all we have here is a bare pointer for each of them.
    // However, for the OpenMP stuff, we do create copies that we need to delete.
    // So keep track if we own the data and need to delete the memory ourselves.
    bool _owns_data;

    // The different correlation functions have different numbers of arrays for zeta,
    // so encapsulate that difference with a templated ZetaData class.
    ZetaData<DC1,DC2,DC3> _zeta;
    double* _meand1;
    double* _meanlogd1;
    double* _meand2;
    double* _meanlogd2;
    double* _meand3;
    double* _meanlogd3;
    double* _meanu;
    double* _meanv;
    double* _weight;
    double* _ntri;
};

template <int DC1, int DC2, int DC3>
struct ZetaData // This works for NNK, NKK, KKK
{
    ZetaData(double* zeta0, double*, double*, double*, double*, double*, double*, double*) :
        zeta(zeta0) {}

    void new_data(int n) { zeta = new double[n]; }
    void delete_data() { delete [] zeta; zeta = 0; }
    void copy(const ZetaData<DC1,DC2,DC3>& rhs, int n)
    { for (int i=0; i<n; ++i) zeta[i] = rhs.zeta[i]; }
    void add(const ZetaData<DC1,DC2,DC3>& rhs, int n)
    { for (int i=0; i<n; ++i) zeta[i] += rhs.zeta[i]; }
    void clear(int n)
    { for (int i=0; i<n; ++i) zeta[i] = 0.; }
    void write(std::ostream& os) const // Just used for debugging.  Print the first value.
    { os << zeta[0]; }
    void write_full(std::ostream& os, int n) const
    { for(int i=0;i<n;++i) os << zeta[i] <<" "; }

    double* zeta;
};

template <int DC1, int DC2, int DC3>
inline std::ostream& operator<<(std::ostream& os, const ZetaData<DC1, DC2, DC3>& zeta)
{ zeta.write(os); return os; }

template <int DC1, int DC2>
struct ZetaData<DC1, DC2, GData> // This works for NNG, NKG, KKG
{
    ZetaData(double* z0, double* z1, double*, double*, double*, double*, double*, double*) :
        zeta(z0), zeta_im(z1) {}

    void new_data(int n)
    {
        zeta = new double[n];
        zeta_im = new double[n];
    }
    void delete_data()
    {
        delete [] zeta; zeta = 0;
        delete [] zeta_im; zeta_im = 0;
    }
    void copy(const ZetaData<DC1,DC2,GData>& rhs, int n)
    {
        for (int i=0; i<n; ++i) zeta[i] = rhs.zeta[i];
        for (int i=0; i<n; ++i) zeta_im[i] = rhs.zeta_im[i];
    }
    void add(const ZetaData<DC1,DC2,GData>& rhs, int n)
    {
        for (int i=0; i<n; ++i) zeta[i] += rhs.zeta[i];
        for (int i=0; i<n; ++i) zeta_im[i] += rhs.zeta_im[i];
    }
    void clear(int n)
    {
        for (int i=0; i<n; ++i) zeta[i] = 0.;
        for (int i=0; i<n; ++i) zeta_im[i] = 0.;
    }
    void write(std::ostream& os) const
    { os << zeta[0]<<','<<zeta_im[0]; }
    void write_full(std::ostream& os, int n) const
    { for(int i=0;i<n;++i) os << zeta[i] <<" "; }

    double* zeta;
    double* zeta_im;
};

template <int DC1>
struct ZetaData<DC1, GData, GData> // This works for NGG, KGG
{
    ZetaData(double* z0, double* z1, double* z2, double* z3, double*, double*, double*, double*) :
        zetap(z0), zetap_im(z1), zetam(z2), zetam_im(z3) {}

    void new_data(int n)
    {
        zetap = new double[n];
        zetap_im = new double[n];
        zetam = new double[n];
        zetam_im = new double[n];
    }
    void delete_data()
    {
        delete [] zetap; zetap = 0;
        delete [] zetap_im; zetap_im = 0;
        delete [] zetam; zetam = 0;
        delete [] zetam_im; zetam_im = 0;
    }
    void copy(const ZetaData<DC1,GData,GData>& rhs, int n)
    {
        for (int i=0; i<n; ++i) zetap[i] = rhs.zetap[i];
        for (int i=0; i<n; ++i) zetap_im[i] = rhs.zetap_im[i];
        for (int i=0; i<n; ++i) zetam[i] = rhs.zetam[i];
        for (int i=0; i<n; ++i) zetam_im[i] = rhs.zetam_im[i];
    }
    void add(const ZetaData<DC1,GData,GData>& rhs, int n)
    {
        for (int i=0; i<n; ++i) zetap[i] += rhs.zetap[i];
        for (int i=0; i<n; ++i) zetap_im[i] += rhs.zetap_im[i];
        for (int i=0; i<n; ++i) zetam[i] += rhs.zetam[i];
        for (int i=0; i<n; ++i) zetam_im[i] += rhs.zetam_im[i];
    }
    void clear(int n)
    {
        for (int i=0; i<n; ++i) zetap[i] = 0.;
        for (int i=0; i<n; ++i) zetap_im[i] = 0.;
        for (int i=0; i<n; ++i) zetam[i] = 0.;
        for (int i=0; i<n; ++i) zetam_im[i] = 0.;
    }
    void write(std::ostream& os) const
    { os << zetap[0]<<','<<zetap_im[0]<<','<<zetam[0]<<','<<zetam_im; }
    void write_full(std::ostream& os, int n) const
    { for(int i=0;i<n;++i) os << zetap[i] <<" "; }

    double* zetap;
    double* zetap_im;
    double* zetam;
    double* zetam_im;
};


template <>
struct ZetaData<GData, GData, GData>
{
    ZetaData(double* z0, double* z1, double* z2, double* z3,
             double* z4, double* z5, double* z6, double* z7) :
        gam0r(z0), gam0i(z1), gam1r(z2), gam1i(z3),
        gam2r(z4), gam2i(z5), gam3r(z6), gam3i(z7) {}

    void new_data(int n)
    {
        gam0r = new double[n];
        gam0i = new double[n];
        gam1r = new double[n];
        gam1i = new double[n];
        gam2r = new double[n];
        gam2i = new double[n];
        gam3r = new double[n];
        gam3i = new double[n];
    }
    void delete_data()
    {
        delete [] gam0r; gam0r = 0;
        delete [] gam0i; gam0i = 0;
        delete [] gam1r; gam1r = 0;
        delete [] gam1i; gam1i = 0;
        delete [] gam2r; gam2r = 0;
        delete [] gam2i; gam2i = 0;
        delete [] gam3r; gam3r = 0;
        delete [] gam3i; gam3i = 0;
    }
    void copy(const ZetaData<GData,GData,GData>& rhs, int n)
    {
        for (int i=0; i<n; ++i) gam0r[i] = rhs.gam0r[i];
        for (int i=0; i<n; ++i) gam0i[i] = rhs.gam0i[i];
        for (int i=0; i<n; ++i) gam1r[i] = rhs.gam1r[i];
        for (int i=0; i<n; ++i) gam1i[i] = rhs.gam1i[i];
        for (int i=0; i<n; ++i) gam2r[i] = rhs.gam2r[i];
        for (int i=0; i<n; ++i) gam2i[i] = rhs.gam2i[i];
        for (int i=0; i<n; ++i) gam3r[i] = rhs.gam3r[i];
        for (int i=0; i<n; ++i) gam3i[i] = rhs.gam3i[i];
    }
    void add(const ZetaData<GData,GData,GData>& rhs, int n)
    {
        for (int i=0; i<n; ++i) gam0r[i] += rhs.gam0r[i];
        for (int i=0; i<n; ++i) gam0i[i] += rhs.gam0i[i];
        for (int i=0; i<n; ++i) gam1r[i] += rhs.gam1r[i];
        for (int i=0; i<n; ++i) gam1i[i] += rhs.gam1i[i];
        for (int i=0; i<n; ++i) gam2r[i] += rhs.gam2r[i];
        for (int i=0; i<n; ++i) gam2i[i] += rhs.gam2i[i];
        for (int i=0; i<n; ++i) gam3r[i] += rhs.gam3r[i];
        for (int i=0; i<n; ++i) gam3i[i] += rhs.gam3i[i];
    }
    void clear(int n)
    {
        for (int i=0; i<n; ++i) gam0r[i] = 0.;
        for (int i=0; i<n; ++i) gam0i[i] = 0.;
        for (int i=0; i<n; ++i) gam1r[i] = 0.;
        for (int i=0; i<n; ++i) gam1i[i] = 0.;
        for (int i=0; i<n; ++i) gam2r[i] = 0.;
        for (int i=0; i<n; ++i) gam2i[i] = 0.;
        for (int i=0; i<n; ++i) gam3r[i] = 0.;
        for (int i=0; i<n; ++i) gam3i[i] = 0.;
    }
    void write(std::ostream& os) const
    {
        os << gam0r[0]<<','<<gam0i[0]<<','<<gam1r[0]<<','<<gam1i[0]<<','<<
            gam2r[0]<<','<<gam2i[0]<<','<<gam3r[0]<<','<<gam3i[0];
    }
    void write_full(std::ostream& os, int n) const
    { for(int i=0;i<n;++i) os << gam0r[i] <<" "; }

    double* gam0r;
    double* gam0i;
    double* gam1r;
    double* gam1i;
    double* gam2r;
    double* gam2i;
    double* gam3r;
    double* gam3i;
};

template <>
struct ZetaData<NData, NData, NData>
{
    ZetaData(double* , double* , double* , double*, double*, double*, double*, double * ) {}
    void new_data(int n) {}
    void delete_data() {}
    void copy(const ZetaData<NData,NData,NData>& rhs, int n) {}
    void add(const ZetaData<NData,NData,NData>& rhs, int n) {}
    void clear(int n) {}
    void write(std::ostream& os) const {}
    void write_full(std::ostream& os, int n) const {}
};


#endif
