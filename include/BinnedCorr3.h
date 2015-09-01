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

#ifndef TreeCorr_Corr3_H
#define TreeCorr_Corr3_H

#include <vector>
#include <string>

#include "Cell.h"
#include "Field.h"

template <int DC1, int DC2, int DC3>
struct ZetaData;

// BinnedCorr3 encapsulates a binned correlation function.
template <int DC1, int DC2, int DC3>
class BinnedCorr3
{

public:

    BinnedCorr3(double minsep, double maxsep, int nbins, double binsize, double b,
                double minu, double maxu, int nubins, double ubinsize, double bu,
                double minv, double maxv, int nvbins, double vbinsize, double bv,
                double* zeta0, double* zeta1, double* zeta2, double* zeta3,
                double* zeta4, double* zeta5, double* zeta6, double* zeta7,
                double* meand1, double* meanlogd1, double* meand2, double* meanlogd2,
                double* meand3, double* meanlogd3, double* meanu, double* meanv,
                double* weight, double* ntri);
    BinnedCorr3(const BinnedCorr3& rhs, bool copy_data=true);
    ~BinnedCorr3();

    void clear();  // Set all data to 0.

    template <int M>
    void process(const Field<DC1, M>& field, bool dots);
    template <int M>
    void process(const Field<DC1, M>& field1, const Field<DC2, M>& field2, 
                 const Field<DC3, M>& field3, bool dots);

    // Main worker functions for calculating the result
    template <int M>
    void process3(const Cell<DC1,M>* c123);

    template <bool sort, int M>
    void process21(const Cell<DC1,M>* c12, const Cell<DC3,M>* c3);

    template <bool sort, int M>
    void process111(const Cell<DC1,M>* c1, const Cell<DC2,M>* c2, const Cell<DC3,M>* c3,
                    double d1sq=0., double d2sq=0., double d3sq=0.);

    template <int M>
    void directProcess111(const Cell<DC1,M>& c1, const Cell<DC2,M>& c2, const Cell<DC3,M>& c3,
                          const double d1, const double d2, const double d3,
                          const double logr, const double u, const double v, const int index);

    // Note: op= only copies _data.  Not all the params.
    void operator=(const BinnedCorr3<DC1,DC2,DC3>& rhs);
    void operator+=(const BinnedCorr3<DC1,DC2,DC3>& rhs);

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
    double _logminsep;
    double _halfminsep;
    double _halfmind3;
    double _minsepsq;
    double _maxsepsq;
    double _minusq;
    double _maxusq;
    double _minabsv;
    double _maxabsv;
    double _minabsvsq;
    double _maxabsvsq;
    double _bsq;
    double _busq;
    double _bvsq;
    double _sqrttwobv;
    int _metric; // Stores which Metric is being used for the analysis.
    int _nuv; // = nubins * nvbins
    int _ntot; // = nbins * nubins * nvbins

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
        gam2r(z0), gam2i(z1), gam3r(z2), gam3i(z3) {}

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
        os << gam0r[0]<<','<<gam0i[0]<<','<<gam1r[0]<<','<<gam1i<<','<<
            gam2r[0]<<','<<gam2i[0]<<','<<gam3r[0]<<','<<gam3i; 
    }

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
};


// The C interface for python
extern "C" {

    extern void* BuildNNNCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                              double minu, double maxu, int nubins, double ubinsize, double bu,
                              double minv, double maxv, int nvbins, double vbinsize, double bv,
                              double* meand1, double* meanlogd1, double* meand2, double* meanlogd2, 
                              double* meand3, double* meanlogd3, double* meanu, double* meanv,
                              double* ntri);
    extern void* BuildKKKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                              double minu, double maxu, int nubins, double ubinsize, double bu,
                              double minv, double maxv, int nvbins, double vbinsize, double bv,
                              double* zeta,
                              double* meand1, double* meanlogd1, double* meand2, double* meanlogd2, 
                              double* meand3, double* meanlogd3, double* meanu, double* meanv,
                              double* weight, double* ntri);
    extern void* BuildGGGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                              double minu, double maxu, int nubins, double ubinsize, double bu,
                              double minv, double maxv, int nvbins, double vbinsize, double bv,
                              double* gam0, double* gam0_im, double* gam1, double* gam1_im,
                              double* gam2, double* gam2_im, double* gam3, double* gam3_im,
                              double* meand1, double* meanlogd1, double* meand2, double* meanlogd2, 
                              double* meand3, double* meanlogd3, double* meanu, double* meanv,
                              double* weight, double* ntri);
#if 0
    extern void* BuildNNKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                              double minu, double maxu, int nubins, double ubinsize, double bu,
                              double minv, double maxv, int nvbins, double vbinsize, double bv,
                              double* zeta,
                              double* meand1, double* meanlogd1, double* meand2, double* meanlogd2, 
                              double* meand3, double* meanlogd3, double* meanu, double* meanv,
                              double* weight, double* ntri);
    extern void* BuildNNGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                              double minu, double maxu, int nubins, double ubinsize, double bu,
                              double minv, double maxv, int nvbins, double vbinsize, double bv,
                              double* zeta, double* zeta_im,
                              double* meand1, double* meanlogd1, double* meand2, double* meanlogd2, 
                              double* meand3, double* meanlogd3, double* meanu, double* meanv,
                              double* weight, double* ntri);
    extern void* BuildKKGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                              double minu, double maxu, int nubins, double ubinsize, double bu,
                              double minv, double maxv, int nvbins, double vbinsize, double bv,
                              double* zeta, double* zeta_im,
                              double* meand1, double* meanlogd1, double* meand2, double* meanlogd2, 
                              double* meand3, double* meanlogd3, double* meanu, double* meanv,
                              double* weight, double* ntri);
#endif

    extern void DestroyNNNCorr(void* corr);
    extern void DestroyKKKCorr(void* corr);
    extern void DestroyGGGCorr(void* corr);
#if 0
    extern void DestroyNNKCorr(void* corr);
    extern void DestroyNNGCorr(void* corr);
    extern void DestroyKKGCorr(void* corr);
#endif

    extern void ProcessAutoNNNFlat(void* corr, void* field, int dots);
    extern void ProcessAutoNNN3D(void* corr, void* field, int dots);
    extern void ProcessAutoNNNPerp(void* corr, void* field, int dots);
    extern void ProcessAutoKKKFlat(void* corr, void* field, int dots);
    extern void ProcessAutoKKK3D(void* corr, void* field, int dots);
    extern void ProcessAutoKKKPerp(void* corr, void* field, int dots);
    extern void ProcessAutoGGGFlat(void* corr, void* field, int dots);
    extern void ProcessAutoGGG3D(void* corr, void* field, int dots);
    extern void ProcessAutoGGGPerp(void* corr, void* field, int dots);

    extern void ProcessCrossNNNFlat(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossNNN3D(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossNNNPerp(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossKKKFlat(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossKKK3D(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossKKKPerp(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossGGGFlat(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossGGG3D(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossGGGPerp(void* corr, void* field1, void* field2, void* field3, int dots);
#if 0
    extern void ProcessCrossNNKFlat(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossNNK3D(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossNNKPerp(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossNNGFlat(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossNNG3D(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossNNGPerp(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossKKGFlat(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossKKG3D(void* corr, void* field1, void* field2, void* field3, int dots);
    extern void ProcessCrossKKGPerp(void* corr, void* field1, void* field2, void* field3, int dots);
#endif
}

#endif
