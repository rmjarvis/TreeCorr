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

#ifndef TreeCorr_Corr3_H
#define TreeCorr_Corr3_H

#include <vector>
#include <string>

#include "Cell.h"
#include "Field.h"
#include "BinType.h"
#include "Metric.h"
#include "MultipoleScratch.h"

template <int Z>
struct ZetaData;

// Things that don't need to access the data vectors can live in the base class.
// Unlike for Corr2, there's not much here, but use the same structure anyway
// in case something turns up that would be useful to put it here.
class BaseCorr3
{
public:

    BaseCorr3(BinType bin_type, int d1, int d2, int d3,
              double minsep, double maxsep, int nbins, double binsize,
              double b, double a,
              double minu, double maxu, int nubins, double ubinsize, double bu,
              double minv, double maxv, int nvbins, double vbinsize, double bv,
              double minrpar, double maxrpar, double xp, double yp, double zp);
    BaseCorr3(const BaseCorr3& rhs);
    virtual ~BaseCorr3() {}

    BinType getBinType() const { return _bin_type; }

    virtual std::shared_ptr<BaseCorr3> duplicate() =0;
    virtual void writeZeta(std::ostream& os, int n) const = 0;

    virtual void addData(const BaseCorr3& rhs) =0;

    template <int C>
    bool validCellTypes(
        const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3)
    { return _d1 == c1.getD() && _d2 == c2.getD() && _d3 == c3.getD(); }

    bool nontrivialRPar() const
    {
        return (_minrpar != -std::numeric_limits<double>::max() ||
                _maxrpar != std::numeric_limits<double>::max());
    }

    template <int B, int M, int C>
    bool triviallyZero(Position<C> p1, Position<C> p2, Position<C> p3,
                       double s1, double s2, double s3, int ordered, bool p13);

    template <int B, int M, int P, int C>
    void process3(const BaseField<C>& field, bool dots, bool quick);
    template <int B, int O, int M, int P, int C>
    void process12(const BaseField<C>& field1, const BaseField<C>& field2, bool dots, bool quick);
    template <int B, int O, int M, int P, int C>
    void process21(const BaseField<C>& field1, const BaseField<C>& field2, bool dots, bool quick);
    template <int B, int O, int M, int P, int C>
    void process111(const BaseField<C>& field1, const BaseField<C>& field2,
                    const BaseField<C>& field3, bool dots, bool quick);

    template <int B, int M, int P, int C>
    void multipole(const BaseField<C>& field, bool dots, bool quick);
    template <int B, int M, int P, int C>
    void multipole(const BaseField<C>& field1,  const BaseField<C>& field2, bool dots, bool quick);
    template <int B, int M, int P, int C>
    void multipole(const BaseField<C>& field1,  const BaseField<C>& field2,
                   const BaseField<C>& field3, bool dots, int ordered, bool quick);

    // Main worker functions for calculating the result
    template <int B, int M, int P, int C>
    void process3(const BaseCell<C>& c1, const MetricHelper<M,P>& metric, bool quick);

    template <int B, int O, int M, int P, int C>
    void process12(const BaseCell<C>& c1, const BaseCell<C>& c2, const MetricHelper<M,P>& metric,
                   bool quick);
    template <int B, int O, int M, int P, int C>
    void process21(const BaseCell<C>& c1, const BaseCell<C>& c2, const MetricHelper<M,P>& metric,
                   bool quick);

    template <int B, int O, int Q, int M, int P, int C>
    void process111(const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
                    const MetricHelper<M,P>& metric,
                    double d1sq=0., double d2sq=0., double d3sq=0.);

    template <int B, int O, int Q, int M, int P, int C>
    void process111Sorted(const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
                          const MetricHelper<M,P>& metric,
                          double d1sq=0., double d2sq=0., double d3sq=0.);

    template <int B, int Q, int C>
    void directProcess111(const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
                          const double d1, const double d2, const double d3,
                          const double u, const double v,
                          const double logd1, const double logd2, const double logd3,
                          const int index);


    template <int B, int M, int P, int C>
    void splitC2Cells(const BaseCell<C>& c1, const std::vector<const BaseCell<C>*>& c2list,
                      const MetricHelper<M,P>& metric, std::vector<const BaseCell<C>*>& newc2list);

    template <int B, int M, int P, int C>
    void multipoleSplit1(const BaseCell<C>& c1, const std::vector<const BaseCell<C>*>& c2list,
                         const MetricHelper<M,P>& metric, bool quick,
                         BaseMultipoleScratch& mp);

    template <int B, int M, int P, int C>
    void multipoleSplit1(const BaseCell<C>& c1,
                         const std::vector<const BaseCell<C>*>& c2list,
                         const std::vector<const BaseCell<C>*>& c3list,
                         const MetricHelper<M,P>& metric, int ordered, bool quick,
                         BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3);

    template <int B, int M, int P, int C>
    double splitC2CellsOrCalculateGn(
        const BaseCell<C>& c1, const std::vector<const BaseCell<C>*>& c2list,
        const MetricHelper<M,P>& metric,
        std::vector<const BaseCell<C>*>& newc2list, bool& anysplit1,
        BaseMultipoleScratch& mp, double maxr);

    template <int B, int Q, int M, int P, int C>
    void multipoleFinish(const BaseCell<C>& c1, const std::vector<const BaseCell<C>*>& c2list,
                         const MetricHelper<M,P>& metric, BaseMultipoleScratch& mp,
                         int mink_zeta, double maxr);

    template <int B, int Q, int M, int P, int C>
    void multipoleFinish(const BaseCell<C>& c1,
                         const std::vector<const BaseCell<C>*>& c2list,
                         const std::vector<const BaseCell<C>*>& c3list,
                         const MetricHelper<M,P>& metric, int ordered,
                         BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3,
                         int mink_zeta, double maxr2, double maxr3);

    template <int Q, int C>
    void finishProcess(const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
                       const double d1, const double d2, const double d3,
                       const double u, const double v,
                       const double logd1, const double logd2, const double logd3,
                       const int index)
    { doFinishProcess(c1, c2, c3, d1, d2, d3, u, v, logd1, logd2, logd3, index, Q1<Q>()); }

    template <int Q, int C>
    void finishProcessMP(const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
                         const double d1, const double d2, const double d3,
                         const double sinphi, const double cosphi,
                         const double logd1, const double logd2, const double logd3,
                         const int index)
    {
        doFinishProcessMP(c1, c2, c3, d1, d2, d3, sinphi, cosphi, logd1, logd2, logd3, index,
                          Q1<Q>());
    }

    template <int C>
    void calculateGn(const BaseCell<C>& c1, const BaseCell<C>& c2,
                     double rsq, double r, double logr, int k,
                     BaseMultipoleScratch& mp)
    { doCalculateGn(c1, c2, rsq, r, logr, k, mp); }

    template <int Q, int C>
    void calculateZeta(const BaseCell<C>& c1, BaseMultipoleScratch& mp,
                       int kstart, int mink_zeta)
    { doCalculateZeta(c1, mp, kstart, mink_zeta, Q1<Q>()); }

    template <int Q, int C>
    void calculateZeta(const BaseCell<C>& c1, int ordered,
                       BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3,
                       int kstart, int mink_zeta)
    { doCalculateZeta(c1, ordered, mp2, mp3, kstart, mink_zeta, Q1<Q>()); }

protected:

    template <int Q>
    struct Q1 {};

    virtual std::unique_ptr<BaseMultipoleScratch> getMP2(bool use_ww) =0;
    virtual std::unique_ptr<BaseMultipoleScratch> getMP3(bool use_ww) =0;

    // This bit is a workaround for the the fact that virtual functions cannot be templates.
    virtual void doFinishProcess(
        const BaseCell<Flat>& c1, const BaseCell<Flat>& c2, const BaseCell<Flat>& c3,
        const double d1, const double d2, const double d3, const double u, const double v,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<0>) =0;
    virtual void doFinishProcess(
        const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2, const BaseCell<Sphere>& c3,
        const double d1, const double d2, const double d3, const double u, const double v,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<0>) =0;
    virtual void doFinishProcess(
        const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2, const BaseCell<ThreeD>& c3,
        const double d1, const double d2, const double d3, const double u, const double v,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<0>) =0;

    virtual void doFinishProcess(
        const BaseCell<Flat>& c1, const BaseCell<Flat>& c2, const BaseCell<Flat>& c3,
        const double d1, const double d2, const double d3, const double u, const double v,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<1>) =0;
    virtual void doFinishProcess(
        const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2, const BaseCell<Sphere>& c3,
        const double d1, const double d2, const double d3, const double u, const double v,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<1>) =0;
    virtual void doFinishProcess(
        const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2, const BaseCell<ThreeD>& c3,
        const double d1, const double d2, const double d3, const double u, const double v,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<1>) =0;

    virtual void doFinishProcessMP(
        const BaseCell<Flat>& c1, const BaseCell<Flat>& c2, const BaseCell<Flat>& c3,
        const double d1, const double d2, const double d3, const double sinphi, const double cosphi,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<0>) =0;
    virtual void doFinishProcessMP(
        const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2, const BaseCell<Sphere>& c3,
        const double d1, const double d2, const double d3, const double sinphi, const double cosphi,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<0>) =0;
    virtual void doFinishProcessMP(
        const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2, const BaseCell<ThreeD>& c3,
        const double d1, const double d2, const double d3, const double sinphi, const double cosphi,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<0>) =0;

    virtual void doFinishProcessMP(
        const BaseCell<Flat>& c1, const BaseCell<Flat>& c2, const BaseCell<Flat>& c3,
        const double d1, const double d2, const double d3, const double sinphi, const double cosphi,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<1>) =0;
    virtual void doFinishProcessMP(
        const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2, const BaseCell<Sphere>& c3,
        const double d1, const double d2, const double d3, const double sinphi, const double cosphi,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<1>) =0;
    virtual void doFinishProcessMP(
        const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2, const BaseCell<ThreeD>& c3,
        const double d1, const double d2, const double d3, const double sinphi, const double cosphi,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<1>) =0;

    virtual void doCalculateGn(
        const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
        double rsq, double r, double logr, int k, BaseMultipoleScratch& mp) =0;
    virtual void doCalculateGn(
        const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
        double rsq, double r, double logr, int k, BaseMultipoleScratch& mp) =0;
    virtual void doCalculateGn(
        const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
        double rsq, double r, double logr, int k, BaseMultipoleScratch& mp) =0;

    virtual void doCalculateZeta(const BaseCell<Flat>& c1, BaseMultipoleScratch& mp,
                                 int kstart, int mink_zeta, Q1<0>) =0;
    virtual void doCalculateZeta(const BaseCell<Sphere>& c1, BaseMultipoleScratch& mp,
                                 int kstart, int mink_zeta, Q1<0>) =0;
    virtual void doCalculateZeta(const BaseCell<ThreeD>& c1, BaseMultipoleScratch& mp,
                                 int kstart, int mink_zeta, Q1<0>) =0;

    virtual void doCalculateZeta(const BaseCell<Flat>& c1, BaseMultipoleScratch& mp,
                                 int kstart, int mink_zeta, Q1<1>) =0;
    virtual void doCalculateZeta(const BaseCell<Sphere>& c1, BaseMultipoleScratch& mp,
                                 int kstart, int mink_zeta, Q1<1>) =0;
    virtual void doCalculateZeta(const BaseCell<ThreeD>& c1, BaseMultipoleScratch& mp,
                                 int kstart, int mink_zeta, Q1<1>) =0;

    virtual void doCalculateZeta(const BaseCell<Flat>& c1, int ordered,
                                 BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3,
                                 int kstart, int mink_zeta, Q1<0>) =0;
    virtual void doCalculateZeta(const BaseCell<Sphere>& c1, int ordered,
                                 BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3,
                                 int kstart, int mink_zeta, Q1<0>) =0;
    virtual void doCalculateZeta(const BaseCell<ThreeD>& c1, int ordered,
                                 BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3,
                                 int kstart, int mink_zeta, Q1<0>) =0;

    virtual void doCalculateZeta(const BaseCell<Flat>& c1, int ordered,
                                 BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3,
                                 int kstart, int mink_zeta, Q1<1>) =0;
    virtual void doCalculateZeta(const BaseCell<Sphere>& c1, int ordered,
                                 BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3,
                                 int kstart, int mink_zeta, Q1<1>) =0;
    virtual void doCalculateZeta(const BaseCell<ThreeD>& c1, int ordered,
                                 BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3,
                                 int kstart, int mink_zeta, Q1<1>) =0;

protected:

    BinType _bin_type;
    int _d1;
    int _d2;
    int _d3;
    double _minsep;
    double _maxsep;
    int _nbins;
    double _binsize;
    double _b;
    double _a;
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
    double _minsepsq;
    double _maxsepsq;
    double _minusq;
    double _maxusq;
    double _minvsq;
    double _maxvsq;
    double _bsq;
    double _asq;
    double _busq;
    double _bvsq;
    int _ntot; // Total number of bins (e.g. nbins * nubins * nvbins * 2 for LogRUV)
    int _coords; // Stores the kind of coordinates being used for the analysis.
};

// Corr3 encapsulates a binned correlation function.
template <int D1, int D2, int D3>
class Corr3 : public BaseCorr3
{
public:

    Corr3(BinType bin_type, double minsep, double maxsep, int nbins, double binsize,
          double b, double a,
          double minu, double maxu, int nubins, double ubinsize, double bu,
          double minv, double maxv, int nvbins, double vbinsize, double bv,
          double minrpar, double maxrpar, double xp, double yp, double zp,
          double* zeta0, double* zeta1, double* zeta2, double* zeta3,
          double* zeta4, double* zeta5, double* zeta6, double* zeta7,
          double* meand1, double* meanlogd1, double* meand2, double* meanlogd2,
          double* meand3, double* meanlogd3, double* meanu, double* meanv,
          double* weight, double* weight_im, double* ntri);
    Corr3(const Corr3& rhs, bool copy_data=true);
    ~Corr3();

    std::shared_ptr<BaseCorr3> duplicate()
    { return std::make_shared<Corr3<D1,D2,D3> >(*this, false); }

    void addData(const BaseCorr3& rhs)
    { *this += static_cast<const Corr3<D1,D2,D3>&>(rhs); }

    void clear();  // Set all data to 0.

    void writeZeta(std::ostream& os, int n=1) const
    { _zeta.write_full(os, n); }

    template <int Q, int C>
    void finishProcess(
        const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
        const double d1, const double d2, const double d3, const double u, const double v,
        const double logd1, const double logd2, const double logd3, const int index);

    template <int Q, int C>
    void finishProcessMP(
        const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
        const double d1, const double d2, const double d3, const double sinphi, const double cosphi,
        const double logd1, const double logd2, const double logd3, const int index);

    std::unique_ptr<BaseMultipoleScratch> getMP2(bool use_ww);
    std::unique_ptr<BaseMultipoleScratch> getMP3(bool use_ww);

    template <int C>
    void calculateGn(const BaseCell<C>& c1, const BaseCell<C>& c2,
                     double rsq, double r, double logr, int k, BaseMultipoleScratch& mp);

    template <int Q, int C>
    void calculateZeta(const BaseCell<C>& c1, BaseMultipoleScratch& mp,
                       int kstart, int mink_zeta);

    template <int Q, int C>
    void calculateZeta(const BaseCell<C>& c1, int ordered,
                       BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3,
                       int kstart, int mink_zeta);

    // Note: op= only copies _data.  Not all the params.
    void operator=(const Corr3<D1,D2,D3>& rhs);
    void operator+=(const Corr3<D1,D2,D3>& rhs);

protected:

    void doFinishProcess(
        const BaseCell<Flat>& c1, const BaseCell<Flat>& c2, const BaseCell<Flat>& c3,
        const double d1, const double d2, const double d3, const double u, const double v,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<0>)
    { finishProcess<0>(c1, c2, c3, d1, d2, d3, u, v, logd1, logd2, logd3, index); }
    void doFinishProcess(
        const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2, const BaseCell<Sphere>& c3,
        const double d1, const double d2, const double d3, const double u, const double v,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<0>)
    { finishProcess<0>(c1, c2, c3, d1, d2, d3, u, v, logd1, logd2, logd3, index); }
    void doFinishProcess(
        const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2, const BaseCell<ThreeD>& c3,
        const double d1, const double d2, const double d3, const double u, const double v,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<0>)
    { finishProcess<0>(c1, c2, c3, d1, d2, d3, u, v, logd1, logd2, logd3, index); }

    void doFinishProcess(
        const BaseCell<Flat>& c1, const BaseCell<Flat>& c2, const BaseCell<Flat>& c3,
        const double d1, const double d2, const double d3, const double u, const double v,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<1>)
    { finishProcess<1>(c1, c2, c3, d1, d2, d3, u, v, logd1, logd2, logd3, index); }
    void doFinishProcess(
        const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2, const BaseCell<Sphere>& c3,
        const double d1, const double d2, const double d3, const double u, const double v,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<1>)
    { finishProcess<1>(c1, c2, c3, d1, d2, d3, u, v, logd1, logd2, logd3, index); }
    void doFinishProcess(
        const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2, const BaseCell<ThreeD>& c3,
        const double d1, const double d2, const double d3, const double u, const double v,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<1>)
    { finishProcess<1>(c1, c2, c3, d1, d2, d3, u, v, logd1, logd2, logd3, index); }

    void doFinishProcessMP(
        const BaseCell<Flat>& c1, const BaseCell<Flat>& c2, const BaseCell<Flat>& c3,
        const double d1, const double d2, const double d3, const double sinphi, const double cosphi,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<0>)
    { finishProcessMP<0>(c1, c2, c3, d1, d2, d3, sinphi, cosphi, logd1, logd2, logd3, index); }
    void doFinishProcessMP(
        const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2, const BaseCell<Sphere>& c3,
        const double d1, const double d2, const double d3, const double sinphi, const double cosphi,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<0>)
    { finishProcessMP<0>(c1, c2, c3, d1, d2, d3, sinphi, cosphi, logd1, logd2, logd3, index); }
    void doFinishProcessMP(
        const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2, const BaseCell<ThreeD>& c3,
        const double d1, const double d2, const double d3, const double sinphi, const double cosphi,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<0>)
    { finishProcessMP<0>(c1, c2, c3, d1, d2, d3, sinphi, cosphi, logd1, logd2, logd3, index); }

    void doFinishProcessMP(
        const BaseCell<Flat>& c1, const BaseCell<Flat>& c2, const BaseCell<Flat>& c3,
        const double d1, const double d2, const double d3, const double sinphi, const double cosphi,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<1>)
    { finishProcessMP<1>(c1, c2, c3, d1, d2, d3, sinphi, cosphi, logd1, logd2, logd3, index); }
    void doFinishProcessMP(
        const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2, const BaseCell<Sphere>& c3,
        const double d1, const double d2, const double d3, const double sinphi, const double cosphi,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<1>)
    { finishProcessMP<1>(c1, c2, c3, d1, d2, d3, sinphi, cosphi, logd1, logd2, logd3, index); }
    void doFinishProcessMP(
        const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2, const BaseCell<ThreeD>& c3,
        const double d1, const double d2, const double d3, const double sinphi, const double cosphi,
        const double logd1, const double logd2, const double logd3, const int index,
        Q1<1>)
    { finishProcessMP<1>(c1, c2, c3, d1, d2, d3, sinphi, cosphi, logd1, logd2, logd3, index); }

    void doCalculateGn(
        const BaseCell<Flat>& c1, const BaseCell<Flat>& c2,
        double rsq, double r, double logr, int k, BaseMultipoleScratch& mp)
    { calculateGn(c1, c2, rsq, r, logr, k, mp); }
    void doCalculateGn(
        const BaseCell<Sphere>& c1, const BaseCell<Sphere>& c2,
        double rsq, double r, double logr, int k, BaseMultipoleScratch& mp)
    { calculateGn(c1, c2, rsq, r, logr, k, mp); }
    void doCalculateGn(
        const BaseCell<ThreeD>& c1, const BaseCell<ThreeD>& c2,
        double rsq, double r, double logr, int k, BaseMultipoleScratch& mp)
    { calculateGn(c1, c2, rsq, r, logr, k, mp); }

    void doCalculateZeta(const BaseCell<Flat>& c1, BaseMultipoleScratch& mp,
                         int kstart, int mink_zeta, Q1<0>)
    { calculateZeta<0>(c1, mp, kstart, mink_zeta); }
    void doCalculateZeta(const BaseCell<Sphere>& c1, BaseMultipoleScratch& mp,
                         int kstart, int mink_zeta, Q1<0>)
    { calculateZeta<0>(c1, mp, kstart, mink_zeta); }
    void doCalculateZeta(const BaseCell<ThreeD>& c1, BaseMultipoleScratch& mp,
                         int kstart, int mink_zeta, Q1<0>)
    { calculateZeta<0>(c1, mp, kstart, mink_zeta); }

    void doCalculateZeta(const BaseCell<Flat>& c1, BaseMultipoleScratch& mp,
                         int kstart, int mink_zeta, Q1<1>)
    { calculateZeta<1>(c1, mp, kstart, mink_zeta); }
    void doCalculateZeta(const BaseCell<Sphere>& c1, BaseMultipoleScratch& mp,
                         int kstart, int mink_zeta, Q1<1>)
    { calculateZeta<1>(c1, mp, kstart, mink_zeta); }
    void doCalculateZeta(const BaseCell<ThreeD>& c1, BaseMultipoleScratch& mp,
                         int kstart, int mink_zeta, Q1<1>)
    { calculateZeta<1>(c1, mp, kstart, mink_zeta); }

    void doCalculateZeta(const BaseCell<Flat>& c1, int ordered,
                         BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3,
                         int kstart, int mink_zeta, Q1<0>)
    { calculateZeta<0>(c1, ordered, mp2, mp3, kstart, mink_zeta); }
    void doCalculateZeta(const BaseCell<Sphere>& c1, int ordered,
                         BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3,
                         int kstart, int mink_zeta, Q1<0>)
    { calculateZeta<0>(c1, ordered, mp2, mp3, kstart, mink_zeta); }
    void doCalculateZeta(const BaseCell<ThreeD>& c1, int ordered,
                         BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3,
                         int kstart, int mink_zeta, Q1<0>)
    { calculateZeta<0>(c1, ordered, mp2, mp3, kstart, mink_zeta); }

    void doCalculateZeta(const BaseCell<Flat>& c1, int ordered,
                         BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3,
                         int kstart, int mink_zeta, Q1<1>)
    { calculateZeta<1>(c1, ordered, mp2, mp3, kstart, mink_zeta); }
    void doCalculateZeta(const BaseCell<Sphere>& c1, int ordered,
                         BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3,
                         int kstart, int mink_zeta, Q1<1>)
    { calculateZeta<1>(c1, ordered, mp2, mp3, kstart, mink_zeta); }
    void doCalculateZeta(const BaseCell<ThreeD>& c1, int ordered,
                         BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3,
                         int kstart, int mink_zeta, Q1<1>)
    { calculateZeta<1>(c1, ordered, mp2, mp3, kstart, mink_zeta); }

    // These are usually allocated in the python layer and just built up here.
    // So all we have here is a bare pointer for each of them.
    // However, for the OpenMP stuff, we do create copies that we need to delete.
    // So keep track if we own the data and need to delete the memory ourselves.
    bool _owns_data;

    // The different correlation functions have different numbers of arrays for zeta,
    // so encapsulate that difference with a templated ZetaData class.
    enum { Z = (
            D1 == NData && D2 == NData && D3 == NData          ? 0 : // NNN
            D1 <= KData && D2 <= KData && D3 <= KData          ? 1 : // NNK, NKK, etc. (all real)
            (D1 <= KData) + (D2 <= KData) + (D3 <= KData) == 2 ? 2 : // KKG, etc. (1 complex)
            (D1 <= KData) + (D2 <= KData) + (D3 <= KData) == 1 ? 3 : // KGG, etc. (2 complex)
                                                                 4   // GGG, etc. (3 complex)
    ) };

    ZetaData<Z> _zeta;
    double* _meand1;
    double* _meanlogd1;
    double* _meand2;
    double* _meanlogd2;
    double* _meand3;
    double* _meanlogd3;
    double* _meanu;
    double* _meanv;
    double* _weight;
    double* _weight_im;
    double* _ntri;
};

template <int Z>
struct ZetaData;

template <int Z>
inline std::ostream& operator<<(std::ostream& os, const ZetaData<Z>& zeta)
{ zeta.write(os); return os; }

template <>
struct ZetaData<0> // NNN
{
    ZetaData(double* , double* , double*, double*, double*, double*, double*, double*) {}
    void new_data(int n) {}
    void delete_data() {}
    void copy(const ZetaData<0>& rhs, int n) {}
    void add(const ZetaData<0>& rhs, int n) {}
    void clear(int n) {}
    void write(std::ostream& os) const {}
    void write_full(std::ostream& os, int n) const {}
};

template <>
struct ZetaData<1> // NNK, NKK, KKK -- all real
{
    ZetaData(double* z0, double* z1, double*, double*, double*, double*, double*, double*) :
        zeta(z0), zeta_im(z1), is_complex(z1 != nullptr)
    {}

    void new_data(int n)
    {
        zeta = new double[n];
        if (is_complex) zeta_im = new double[n];
    }
    void delete_data()
    {
        delete [] zeta; zeta = 0;
        if (is_complex) { delete [] zeta_im; zeta_im = 0; }
    }
    void copy(const ZetaData<1>& rhs, int n)
    {
        for (int i=0; i<n; ++i) zeta[i] = rhs.zeta[i];
        if (is_complex) {
            for (int i=0; i<n; ++i) zeta_im[i] = rhs.zeta_im[i];
        }
    }
    void add(const ZetaData<1>& rhs, int n)
    {
        for (int i=0; i<n; ++i) zeta[i] += rhs.zeta[i];
        if (is_complex) {
            for (int i=0; i<n; ++i) zeta_im[i] += rhs.zeta_im[i];
        }
    }
    void clear(int n)
    {
        for (int i=0; i<n; ++i) zeta[i] = 0.;
        if (is_complex) {
            for (int i=0; i<n; ++i) zeta_im[i] = 0.;
        }
    }
    void write(std::ostream& os) const // Just used for debugging.  Print the first value.
    {
        os << zeta[0];
        if (is_complex) { os << ','<<zeta_im[0]; }
    }
    void write_full(std::ostream& os, int n) const
    { for(int i=0;i<n;++i) os << zeta[i] <<" "; }

    double* zeta;
    double* zeta_im;
    bool is_complex;
};

template <>
struct ZetaData<2> // NNG, NKG, KKG, etc.  1 complex
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
    void copy(const ZetaData<2>& rhs, int n)
    {
        for (int i=0; i<n; ++i) zeta[i] = rhs.zeta[i];
        for (int i=0; i<n; ++i) zeta_im[i] = rhs.zeta_im[i];
    }
    void add(const ZetaData<2>& rhs, int n)
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

template <>
struct ZetaData<3> // NGG, KGG, etc. -- 2 complex
{
    ZetaData(double* z0, double* z1, double* z2, double* z3, double*, double*, double*, double*) :
        gam0r(z0), gam0i(z1), gam1r(z2), gam1i(z3) {}

    void new_data(int n)
    {
        gam0r = new double[n];
        gam0i = new double[n];
        gam1r = new double[n];
        gam1i = new double[n];
    }
    void delete_data()
    {
        delete [] gam0r; gam0r = 0;
        delete [] gam0i; gam0i = 0;
        delete [] gam1r; gam1r = 0;
        delete [] gam1i; gam1i = 0;
    }
    void copy(const ZetaData<3>& rhs, int n)
    {
        for (int i=0; i<n; ++i) gam0r[i] = rhs.gam0r[i];
        for (int i=0; i<n; ++i) gam0i[i] = rhs.gam0i[i];
        for (int i=0; i<n; ++i) gam1r[i] = rhs.gam1r[i];
        for (int i=0; i<n; ++i) gam1i[i] = rhs.gam1i[i];
    }
    void add(const ZetaData<3>& rhs, int n)
    {
        for (int i=0; i<n; ++i) gam0r[i] += rhs.gam0r[i];
        for (int i=0; i<n; ++i) gam0i[i] += rhs.gam0i[i];
        for (int i=0; i<n; ++i) gam1r[i] += rhs.gam1r[i];
        for (int i=0; i<n; ++i) gam1i[i] += rhs.gam1i[i];
    }
    void clear(int n)
    {
        for (int i=0; i<n; ++i) gam0r[i] = 0.;
        for (int i=0; i<n; ++i) gam0i[i] = 0.;
        for (int i=0; i<n; ++i) gam1r[i] = 0.;
        for (int i=0; i<n; ++i) gam1i[i] = 0.;
    }
    void write(std::ostream& os) const
    { os << gam0r[0]<<','<<gam0i[0]<<','<<gam1r[0]<<','<<gam1i; }
    void write_full(std::ostream& os, int n) const
    { for(int i=0;i<n;++i) os << gam0r[i] <<" "; }

    double* gam0r;
    double* gam0i;
    double* gam1r;
    double* gam1i;
};

template <>
struct ZetaData<4> // all complex
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
    void copy(const ZetaData<4>& rhs, int n)
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
    void add(const ZetaData<4>& rhs, int n)
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

#endif
