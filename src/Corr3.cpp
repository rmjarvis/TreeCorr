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

//#define DEBUGLOGGING

#ifdef _WIN32
#define _USE_MATH_DEFINES  // To get M_PI
#endif

#include "PyBind11Helper.h"

#include <cmath>
#include "dbg.h"
#include "Corr3.h"
#include "Split.h"
#include "ProjectHelper.h"

#ifdef _OPENMP
#include "omp.h"
#endif

// Some white space helper functions to make dbg output a little nicer.
#ifdef DEBUGLOGGING
#include <sstream>
int ws_count=0;
std::string ws()
{
    // With multiple threads, this can race and end up with ws_count < 0.
    // Output is garbage then anyway, so just ignore that.
    if (ws_count < 0) return "";
    else return std::string(ws_count, ' ');
}
void inc_ws() { ++ws_count; }
void dec_ws() { --ws_count; }
void reset_ws() { ws_count = 0; }
template <int C>
std::string indices(const BaseCell<C>& c) {
    std::stringstream ss;
    ss<<"[";
    for (int k : c.getAllIndices()) ss<<k<<" ";
    ss<<"]";
    return ss.str();
}
#else
std::string ws() { return ""; }
void inc_ws() {}
void dec_ws() {}
void reset_ws() {}
template <int C> std::string indices(const BaseCell<C>& c) { return ""; }
#endif


BaseCorr3::BaseCorr3(
    BinType bin_type, double minsep, double maxsep, int nbins, double binsize, double b,
    double minu, double maxu, int nubins, double ubinsize, double bu,
    double minv, double maxv, int nvbins, double vbinsize, double bv,
    double xp, double yp, double zp):
    _bin_type(bin_type),
    _minsep(minsep), _maxsep(maxsep), _nbins(nbins), _binsize(binsize), _b(b),
    _minu(minu), _maxu(maxu), _nubins(nubins), _ubinsize(ubinsize), _bu(bu),
    _minv(minv), _maxv(maxv), _nvbins(nvbins), _vbinsize(vbinsize), _bv(bv),
    _xp(xp), _yp(yp), _zp(zp), _coords(-1)
{
    // Do a few things that are specific to different bin_types.
    switch(bin_type) {
      case LogRUV:
           _ntot = BinTypeHelper<LogRUV>::calculateNTot(nbins, nubins, nvbins);
           break;
      case LogSAS:
           _ntot = BinTypeHelper<LogSAS>::calculateNTot(nbins, nubins, nvbins);
           // For LogSAS, we don't have v, and min/maxu is really min/maxphi.
           // Most of the checks can be done on cosphi instead, which is faster, so use
           // the v values to store mincosphi, maxcosphi
           if (_minu < 0 && _maxu > 0) _maxv = 1;
           else _maxv = std::max(std::cos(_minu), std::cos(_maxu));
           if (_minu < M_PI && _maxu > M_PI) _minv = -1;
           else _minv = std::min(std::cos(_minu), std::cos(_maxu));
           break;
      case LogMultipole:
           _ntot = BinTypeHelper<LogMultipole>::calculateNTot(nbins, nubins, nvbins);
           _nvbins = _nbins * (2*_nubins+1);  // Use this for size of Gn array.
           break;
      default:
           dbg<<"bin_type = "<<bin_type<<std::endl;
           Assert(false);
    }

    // Some helpful variables we can calculate once here.
    _logminsep = log(_minsep);
    _halfminsep = 0.5*_minsep;
    _bsq = _b * _b;
    _busq = _bu * _bu;
    _bvsq = _bv * _bv;
    _minsepsq = _minsep*_minsep;
    _maxsepsq = _maxsep*_maxsep;
    _minusq = _minu*_minu;
    _maxusq = _maxu*_maxu;
    _minvsq = _minv * _minv;
    _maxvsq = _maxv * _maxv;
    if (bin_type == LogSAS) {
        // For LogSAS it is more useful for mincosphisq, maxcosphisq to preserve the
        // negative sign if there is one on the unsquared version.
        _minvsq = _minv < 0 ? -_minvsq : _minvsq;
        _maxvsq = _maxv < 0 ? -_maxvsq : _maxvsq;
    }
}

template <int D1, int D2, int D3>
Corr3<D1,D2,D3>::Corr3(
    BinType bin_type, double minsep, double maxsep, int nbins, double binsize, double b,
    double minu, double maxu, int nubins, double ubinsize, double bu,
    double minv, double maxv, int nvbins, double vbinsize, double bv,
    double xp, double yp, double zp,
    double* zeta0, double* zeta1, double* zeta2, double* zeta3,
    double* zeta4, double* zeta5, double* zeta6, double* zeta7,
    double* meand1, double* meanlogd1, double* meand2, double* meanlogd2,
    double* meand3, double* meanlogd3, double* meanu, double* meanv,
    double* weight, double* ntri) :
    BaseCorr3(bin_type, minsep, maxsep, nbins, binsize, b,
              minu, maxu, nubins, ubinsize, bu,
              minv, maxv, nvbins, vbinsize, bv,
              xp, yp, zp),
    _owns_data(false),
    _zeta(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, bin_type==LogMultipole),
    _meand1(meand1), _meanlogd1(meanlogd1), _meand2(meand2), _meanlogd2(meanlogd2),
    _meand3(meand3), _meanlogd3(meanlogd3), _meanu(meanu), _meanv(meanv),
    _weight(weight), _ntri(ntri)
{}

BaseCorr3::BaseCorr3(const BaseCorr3& rhs):
    _bin_type(rhs._bin_type),
    _minsep(rhs._minsep), _maxsep(rhs._maxsep), _nbins(rhs._nbins),
    _binsize(rhs._binsize), _b(rhs._b),
    _minu(rhs._minu), _maxu(rhs._maxu), _nubins(rhs._nubins),
    _ubinsize(rhs._ubinsize), _bu(rhs._bu),
    _minv(rhs._minv), _maxv(rhs._maxv), _nvbins(rhs._nvbins),
    _vbinsize(rhs._vbinsize), _bv(rhs._bv),
    _logminsep(rhs._logminsep), _halfminsep(rhs._halfminsep),
    _minsepsq(rhs._minsepsq), _maxsepsq(rhs._maxsepsq),
    _minusq(rhs._minusq), _maxusq(rhs._maxusq),
    _minvsq(rhs._minvsq), _maxvsq(rhs._maxvsq),
    _bsq(rhs._bsq), _busq(rhs._busq), _bvsq(rhs._bvsq),
    _ntot(rhs._ntot), _coords(rhs._coords)
{}


template <int D1, int D2, int D3>
Corr3<D1,D2,D3>::Corr3(const Corr3<D1,D2,D3>& rhs, bool copy_data) :
    BaseCorr3(rhs), _owns_data(true),
    _zeta(0,0,0,0,0,0,0,0, rhs._bin_type==LogMultipole), _weight(0)
{
    _zeta.new_data(_ntot);
    _meand1 = new double[_ntot];
    _meanlogd1 = new double[_ntot];
    _meand2 = new double[_ntot];
    _meanlogd2 = new double[_ntot];
    _meand3 = new double[_ntot];
    _meanlogd3 = new double[_ntot];
    _meanu = new double[_ntot];
    _meanv = new double[_ntot];
    _weight = new double[_ntot];
    _ntri = new double[_ntot];

    if (copy_data) *this = rhs;
    else clear();
}

template <int D1, int D2, int D3>
Corr3<D1,D2,D3>::~Corr3()
{
    if (_owns_data) {
        _zeta.delete_data();
        delete [] _meand1; _meand1 = 0;
        delete [] _meanlogd1; _meanlogd1 = 0;
        delete [] _meand2; _meand2 = 0;
        delete [] _meanlogd2; _meanlogd2 = 0;
        delete [] _meand3; _meand3 = 0;
        delete [] _meanlogd3; _meanlogd3 = 0;
        delete [] _meanu; _meanu = 0;
        delete [] _meanv; _meanv = 0;
        delete [] _weight; _weight = 0;
        delete [] _ntri; _ntri = 0;
    }
}

template <int D1, int D2, int D3>
void Corr3<D1,D2,D3>::clear()
{
    _zeta.clear(_ntot);
    for (int i=0; i<_ntot; ++i) _meand1[i] = 0.;
    for (int i=0; i<_ntot; ++i) _meanlogd1[i] = 0.;
    for (int i=0; i<_ntot; ++i) _meand2[i] = 0.;
    for (int i=0; i<_ntot; ++i) _meanlogd2[i] = 0.;
    for (int i=0; i<_ntot; ++i) _meand3[i] = 0.;
    for (int i=0; i<_ntot; ++i) _meanlogd3[i] = 0.;
    for (int i=0; i<_ntot; ++i) _meanu[i] = 0.;
    for (int i=0; i<_ntot; ++i) _meanv[i] = 0.;
    for (int i=0; i<_ntot; ++i) _weight[i] = 0.;
    for (int i=0; i<_ntot; ++i) _ntri[i] = 0.;
    _coords = -1;
}

template <int B, int M, int C>
void BaseCorr3::process(const BaseField<C>& field, bool dots)
{
    dbg<<"Start process auto\n";
    reset_ws();
    Assert(_coords == -1 || _coords == C);
    _coords = C;
    const long n1 = field.getNTopLevel();
    dbg<<"field has "<<n1<<" top level nodes\n";
    Assert(n1 > 0);

    const std::vector<const BaseCell<C>*>& cells = field.getCells();

#ifdef _OPENMP
#pragma omp parallel
    {
        // Give each thread their own copy of the data vector to fill in.
        std::shared_ptr<BaseCorr3> corrp = duplicate();
        BaseCorr3& corr = *corrp;
#else
        BaseCorr3& corr = *this;
#endif

        MetricHelper<M,0> metric(0, 0, _xp, _yp, _zp);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (long i=0;i<n1;++i) {
            const BaseCell<C>& c1 = *cells[i];
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                if (dots) std::cout<<'.'<<std::flush;
#ifdef _OPENMP
                dbg<<omp_get_thread_num()<<" "<<i<<std::endl;
#endif
            }
            corr.template process3<B>(c1, metric);
            for (long j=i+1;j<n1;++j) {
                const BaseCell<C>& c2 = *cells[j];
                corr.template process12<B,0>(c1, c2, metric);
                corr.template process12<B,0>(c2, c1, metric);
                for (long k=j+1;k<n1;++k) {
                    const BaseCell<C>& c3 = *cells[k];
                    corr.template process111<B,0>(c1, c2, c3, metric);
                }
            }
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            addData(corr);
        }
    }
#endif
    if (dots) std::cout<<std::endl;
}

template <int B, int O, int M, int C>
void BaseCorr3::process(const BaseField<C>& field1, const BaseField<C>& field2,
                        bool dots)
{
    dbg<<"Start process cross12\n";
    reset_ws();
    xdbg<<"_coords = "<<_coords<<std::endl;
    xdbg<<"C = "<<C<<std::endl;
    Assert(_coords == -1 || _coords == C);
    _coords = C;
    const long n1 = field1.getNTopLevel();
    const long n2 = field2.getNTopLevel();
    dbg<<"field1 has "<<n1<<" top level nodes\n";
    dbg<<"field2 has "<<n2<<" top level nodes\n";
    Assert(n1 > 0);
    Assert(n2 > 0);

    MetricHelper<M,0> metric(0, 0, _xp, _yp, _zp);

    const std::vector<const BaseCell<C>*>& c1list = field1.getCells();
    const std::vector<const BaseCell<C>*>& c2list = field2.getCells();

#ifdef _OPENMP
#pragma omp parallel
    {
        // Give each thread their own copy of the data vector to fill in.
        std::shared_ptr<BaseCorr3> corrp = duplicate();
        BaseCorr3& corr = *corrp;
#else
        BaseCorr3& corr = *this;
#endif

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (long i=0;i<n1;++i) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                if (dots) std::cout<<'.'<<std::flush;
#ifdef _OPENMP
                dbg<<omp_get_thread_num()<<" "<<i<<std::endl;
#endif
            }
            const BaseCell<C>& c1 = *c1list[i];
            for (long j=0;j<n2;++j) {
                const BaseCell<C>& c2 = *c2list[j];
                corr.template process12<B,O>(c1, c2, metric);
                for (long k=j+1;k<n2;++k) {
                    const BaseCell<C>& c3 = *c2list[k];
                    corr.template process111<B,O>(c1, c2, c3, metric);
                }
            }
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            addData(corr);
        }
    }
#endif
    if (dots) std::cout<<std::endl;
}

template <int B, int O, int M, int C>
void BaseCorr3::process(const BaseField<C>& field1, const BaseField<C>& field2,
                        const BaseField<C>& field3, bool dots)
{
    dbg<<"Start process cross full\n";
    reset_ws();
    xdbg<<"_coords = "<<_coords<<std::endl;
    xdbg<<"C = "<<C<<std::endl;
    Assert(_coords == -1 || _coords == C);
    _coords = C;
    const long n1 = field1.getNTopLevel();
    const long n2 = field2.getNTopLevel();
    const long n3 = field3.getNTopLevel();
    dbg<<"field1 has "<<n1<<" top level nodes\n";
    dbg<<"field2 has "<<n2<<" top level nodes\n";
    dbg<<"field3 has "<<n3<<" top level nodes\n";
    Assert(n1 > 0);
    Assert(n2 > 0);
    Assert(n3 > 0);

    MetricHelper<M,0> metric(0, 0, _xp, _yp, _zp);

    const std::vector<const BaseCell<C>*>& c1list = field1.getCells();
    const std::vector<const BaseCell<C>*>& c2list = field2.getCells();
    const std::vector<const BaseCell<C>*>& c3list = field3.getCells();

#ifdef _OPENMP
#pragma omp parallel
    {
        // Give each thread their own copy of the data vector to fill in.
        std::shared_ptr<BaseCorr3> corrp = duplicate();
        BaseCorr3& corr = *corrp;
#else
        BaseCorr3& corr = *this;
#endif

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (long i=0;i<n1;++i) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                if (dots) std::cout<<'.'<<std::flush;
#ifdef _OPENMP
                dbg<<omp_get_thread_num()<<" "<<i<<std::endl;
#endif
            }
            const BaseCell<C>& c1 = *c1list[i];
            for (long j=0;j<n2;++j) {
                const BaseCell<C>& c2 = *c2list[j];
                for (long k=0;k<n3;++k) {
                    const BaseCell<C>& c3 = *c3list[k];
                    corr.template process111<B,O>(c1, c2, c3, metric);
                }
            }
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            addData(corr);
        }
    }
#endif
    if (dots) std::cout<<std::endl;
}

template <int B, int M, int C>
void BaseCorr3::process3(const BaseCell<C>& c1, const MetricHelper<M,0>& metric)
{
    // Does all triangles with 3 points in c1
    xdbg<<ws()<<"Process3: c1 = "<<c1.getPos()<<"  "<<c1.getSize()<<"  "<<c1.getN()<<std::endl;
    xdbg<<ws()<<"Process3: c1 = "<<indices(c1)<<"\n";

    if (c1.getW() == 0) {
        xdbg<<ws()<<"    w == 0.  return\n";
        return;
    }
    if (c1.getSize() < _halfminsep) {
        xdbg<<ws()<<"    size < halfminsep.  return\n";
        return;
    }

    inc_ws();
    Assert(c1.getLeft());
    Assert(c1.getRight());
    process3<B>(*c1.getLeft(), metric);
    process3<B>(*c1.getRight(), metric);
    process12<B,0>(*c1.getLeft(), *c1.getRight(), metric);
    process12<B,0>(*c1.getRight(), *c1.getLeft(), metric);
    dec_ws();
}

template <int B, int O, int M, int C>
void BaseCorr3::process12(const BaseCell<C>& c1, const BaseCell<C>& c2,
                          const MetricHelper<M,0>& metric)
{
    // Does all triangles with one point in c1 and the other two points in c2
    xdbg<<ws()<<"Process12: c1: "<<c1.getSize()<<"  "<<c1.getW()<<"  c2: "<<c2.getSize()<<"  "<<c2.getW()<<std::endl;
    xdbg<<"Process12: c1 = "<<c1.getPos()<<"  "<<c1.getSize()<<"  "<<c1.getN()<<std::endl;
    xdbg<<"           c2  = "<<c2.getPos()<<"  "<<c2.getSize()<<"  "<<c2.getN()<<std::endl;
    xdbg<<ws()<<"Process12: c1 = "<<indices(c1)<<"  c2 = "<<indices(c2)<<"  ordered="<<O<<"\n";

    // ordered=0 means that we don't care which point is called c1, c2, or c3 at the end.
    // ordered=1 means that c1 must be from the given c1 cell.
    xdbg<<"ordered = "<<O<<std::endl;

    // Some trivial stoppers:
    if (c1.getW() == 0) {
        xdbg<<ws()<<"    w1 == 0.  return\n";
        return;
    }
    if (c2.getW() == 0) {
        xdbg<<ws()<<"    w2 == 0.  return\n";
        return;
    }
    double s2 = c2.getSize();
    if (BinTypeHelper<B>::tooSmallS2(s2, _halfminsep, _minu, _minv))
    {
        xdbg<<ws()<<"    s2 smaller than minimum triangle side.  return\n";
        return;
    }

    double s1 = c1.getSize();
    double rsq = metric.DistSq(c1.getPos(), c2.getPos(), s1, s2);
    double s1ps2 = s1 + s2;

    // If all possible triangles will have d2 < minsep, then abort the recursion here.
    // i.e. if d + s1 + s2 < minsep
    if (BinTypeHelper<B>::tooSmallDist(rsq, s1ps2, _minsep, _minsepsq)) {
        xdbg<<ws()<<"    d2 cannot be as large as minsep\n";
        return;
    }

    // Similarly, we can abort if all possible triangles will have d > maxsep.
    // i.e. if  d - s1 - s2 >= maxsep
    if (BinTypeHelper<B>::tooLargeDist(rsq, s1ps2, _maxsep, _maxsepsq)) {
        xdbg<<ws()<<"    d cannot be as small as maxsep\n";
        return;
    }

    // Depending on the binning, we may be able to stop due to allowed angles.
    if (BinTypeHelper<B>::template noAllowedAngles<O>(rsq, s1ps2, s1, s2, _halfminsep,
                                                      _minu, _minusq, _maxu, _maxusq,
                                                      _minv, _minvsq, _maxv, _maxvsq)) {
        xdbg<<ws()<<"    No possible triangles with allowed angles\n";
        return;
    }

    inc_ws();
    Assert(c2.getLeft());
    Assert(c2.getRight());
    process12<B,O>(c1, *c2.getLeft(), metric);
    process12<B,O>(c1, *c2.getRight(), metric);
    // 111 order is 123, 132, 213, 231, 312, 321   Here 3->2.
    process111<B,O>(c1, *c2.getLeft(), *c2.getRight(), metric);
    dec_ws();
}

template <int B, int O, int M, int C>
void BaseCorr3::process111(
    const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
    const MetricHelper<M,0>& metric, double d1sq, double d2sq, double d3sq)
{
    xdbg<<ws()<<"Process111: c1: "<<c1.getSize()<<"  "<<c1.getW()<<"  c2: "<<c2.getSize()<<"  "<<c2.getW()<<"  c3: "<<c3.getSize()<<"  "<<c3.getW()<<std::endl;
    xdbg<<ws()<<"Process111: c1 = "<<indices(c1)<<"  c2 = "<<indices(c2)<<"  c3 = "<<indices(c3)<<"  ordered="<<O<<"\n";

    // ordered=0 means that we don't care which point is called c1, c2, or c3 at the end.
    // ordered=1 means that c1 must be from the given c1 cell.
    // ordered=3 means that c1, c2, c3 must be from the given c1,c2,c3 cells respectively.
    xdbg<<"ordered = "<<O<<std::endl;

    // Does all triangles with 1 point each in c1, c2, c3
    if (c1.getW() == 0) {
        xdbg<<ws()<<"    w1 == 0.  return\n";
        return;
    }
    if (c2.getW() == 0) {
        xdbg<<ws()<<"    w2 == 0.  return\n";
        return;
    }
    if (c3.getW() == 0) {
        xdbg<<ws()<<"    w3 == 0.  return\n";
        return;
    }

    // Calculate the distances if they aren't known yet
    double s=0.;
    if (d1sq == 0.)
        d1sq = metric.DistSq(c2.getPos(), c3.getPos(), s, s);
    if (d2sq == 0.)
        d2sq = metric.DistSq(c1.getPos(), c3.getPos(), s, s);
    if (d3sq == 0.)
        d3sq = metric.DistSq(c1.getPos(), c2.getPos(), s, s);

    inc_ws();
    if (O == 0) {
        if (BinTypeHelper<B>::sort_d123) {
            xdbg<<":sort123\n";
            xdbg<<"Before sort: d123 = "<<sqrt(d1sq)<<"  "<<sqrt(d2sq)<<"  "<<sqrt(d3sq)<<std::endl;

            // Need to end up with d1 > d2 > d3
            if (d1sq > d2sq) {
                if (d2sq > d3sq) {
                    xdbg<<"123\n";
                    // 123 -> 123
                    process111Sorted<B,O>(c1, c2, c3, metric, d1sq, d2sq, d3sq);
                } else if (d1sq > d3sq) {
                    xdbg<<"132\n";
                    // 132 -> 123
                    process111Sorted<B,O>(c1, c3, c2, metric, d1sq, d3sq, d2sq);
                } else {
                    xdbg<<"312\n";
                    // 312 -> 123
                    process111Sorted<B,O>(c3, c1, c2, metric, d3sq, d1sq, d2sq);
                }
            } else {
                if (d1sq > d3sq) {
                    xdbg<<"213\n";
                    // 213 -> 123
                    process111Sorted<B,O>(c2, c1, c3, metric, d2sq, d1sq, d3sq);
                } else if (d2sq > d3sq) {
                    xdbg<<"231\n";
                    // 231 -> 123
                    process111Sorted<B,O>(c2, c3, c1, metric, d2sq, d3sq, d1sq);
                } else {
                    xdbg<<"321\n";
                    // 321 -> 123
                    process111Sorted<B,O>(c3, c2, c1, metric, d3sq, d2sq, d1sq);
                }
            }
        } else {
            xdbg<<":set1\n";
            // If the BinType doesn't want sorting, then make sure we get all the cells
            // into the first location, and switch to ordered = 1.
            if (!metric.CCW(c1.getPos(), c3.getPos(), c2.getPos())) {
                xdbg<<"132\n";
                process111Sorted<B,1>(c1, c3, c2, metric, d1sq, d3sq, d2sq);
                xdbg<<"213\n";
                process111Sorted<B,1>(c2, c1, c3, metric, d2sq, d1sq, d3sq);
                xdbg<<"321\n";
                process111Sorted<B,1>(c3, c2, c1, metric, d3sq, d2sq, d1sq);
            } else {
                xdbg<<"123\n";
                process111Sorted<B,1>(c1, c2, c3, metric, d1sq, d2sq, d3sq);
                xdbg<<"312\n";
                process111Sorted<B,1>(c3, c1, c2, metric, d3sq, d1sq, d2sq);
                xdbg<<"231\n";
                process111Sorted<B,1>(c2, c3, c1, metric, d2sq, d3sq, d1sq);
            }
        }
    } else if (O == 1) {
        if (BinTypeHelper<B>::sort_d123) {
            // If the BinType allows sorting, but we have c1 fixed, then just check d2,d3.
            if (d2sq > d3sq) {
                xdbg<<"123\n";
                // 123 -> 123
                process111Sorted<B,O>(c1, c2, c3, metric, d1sq, d2sq, d3sq);
            } else {
                xdbg<<"132\n";
                // 132 -> 123
                process111Sorted<B,O>(c1, c3, c2, metric, d1sq, d3sq, d2sq);
            }
        } else {
            // For the non-sorting BinTypes (e.g. LogSAS), we just need to make sure
            // 1-3-2 is counter-clockwise
            if (!metric.CCW(c1.getPos(), c3.getPos(), c2.getPos())) {
                xdbg<<":swap23\n";
                // Swap 2,3
                process111Sorted<B,O>(c1, c3, c2, metric, d1sq, d3sq, d2sq);
            } else {
                xdbg<<":noswap\n";
                process111Sorted<B,O>(c1, c2, c3, metric, d1sq, d2sq, d3sq);
            }
        }
    } else {
        xdbg<<":nosort\n";
        process111Sorted<B,O>(c1, c2, c3, metric, d1sq, d2sq, d3sq);
    }
    dec_ws();
}

template <int B, int O, int M, int C>
void BaseCorr3::process111Sorted(
    const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
    const MetricHelper<M,0>& metric, double d1sq, double d2sq, double d3sq)
{
    const double s1 = c1.getSize();
    const double s2 = c2.getSize();
    const double s3 = c3.getSize();

    xdbg<<"Process111Sorted: c1 = "<<c1.getPos()<<"  "<<c1.getSize()<<"  "<<c1.getN()<<std::endl;
    xdbg<<"                  c2 = "<<c2.getPos()<<"  "<<c2.getSize()<<"  "<<c2.getN()<<std::endl;
    xdbg<<"                  c3 = "<<c3.getPos()<<"  "<<c3.getSize()<<"  "<<c3.getN()<<std::endl;
    xdbg<<"                  d123 = "<<sqrt(d1sq)<<"  "<<sqrt(d2sq)<<"  "<<sqrt(d3sq)<<std::endl;
    xdbg<<ws()<<"ProcessSorted111: c1 = "<<indices(c1)<<"  c2 = "<<indices(c2)<<"  c3 = "<<indices(c3)<<"  ordered="<<O<<"\n";

    // Various quanities that we'll set along the way if we need them.
    // At the end, if singleBin is true, then all these will be set correctly.
    double d1=-1., d2=-1., d3=-1., u=-1., v=-1.;
    if (BinTypeHelper<B>::template stop111<O>(d1sq, d2sq, d3sq, s1, s2, s3,
                                              c1, c2, c3, metric,
                                              d1, d2, d3, u, v,
                                              _minsep, _minsepsq, _maxsep, _maxsepsq,
                                              _minu, _minusq, _maxu, _maxusq,
                                              _minv, _minvsq, _maxv, _maxvsq))
    {
        xdbg<<ws()<<"Stopping early -- no possible triangles in range\n";
        return;
    }

    // Now check if these cells are small enough that it is ok to drop into a single bin.
    bool split1=false, split2=false, split3=false;
    if (BinTypeHelper<B>::singleBin(d1sq, d2sq, d3sq, s1, s2, s3,
                                    _b, _bu, _bv, _bsq, _busq, _bvsq,
                                    split1, split2, split3,
                                    d1, d2, d3, u, v))
    {
        xdbg<<"Drop into single bin.\n";

        // These get set if triangle is in range.
        double logd1, logd2, logd3;
        int index;
        if (BinTypeHelper<B>::template isTriangleInRange<O>(
                c1, c2, c3, metric,
                d1sq, d2sq, d3sq, d1, d2, d3, u, v,
                _logminsep, _minsep, _maxsep, _binsize, _nbins,
                _minu, _maxu, _ubinsize, _nubins,
                _minv, _maxv, _vbinsize, _nvbins,
                logd1, logd2, logd3,
                _ntot, index))
        {
            directProcess111<B>(c1, c2, c3, d1, d2, d3, u, v, logd1, logd2, logd3, index);
        } else {
            xdbg<<ws()<<"Triangle not in range\n";
        }
    } else {
        xdbg<<"Need to split.\n";

        XAssert(split1 == false || s1 > 0);
        XAssert(split2 == false || s2 > 0);
        XAssert(split3 == false || s3 > 0);

        if (split3) {
            if (split2) {
                if (split1) {
                    // split 1,2,3
                    XAssert(c1.getLeft());
                    XAssert(c1.getRight());
                    XAssert(c2.getLeft());
                    XAssert(c2.getRight());
                    XAssert(c3.getLeft());
                    XAssert(c3.getRight());
                    process111<B,O>(*c1.getLeft(), *c2.getLeft(), *c3.getLeft(), metric);
                    process111<B,O>(*c1.getLeft(), *c2.getLeft(), *c3.getRight(), metric);
                    process111<B,O>(*c1.getLeft(), *c2.getRight(), *c3.getLeft(), metric);
                    process111<B,O>(*c1.getLeft(), *c2.getRight(), *c3.getRight(), metric);
                    process111<B,O>(*c1.getRight(), *c2.getLeft(), *c3.getLeft(), metric);
                    process111<B,O>(*c1.getRight(), *c2.getLeft(), *c3.getRight(), metric);
                    process111<B,O>(*c1.getRight(), *c2.getRight(), *c3.getLeft(), metric);
                    process111<B,O>(*c1.getRight(), *c2.getRight(), *c3.getRight(), metric);
                } else {
                    // split 2,3
                    XAssert(c2.getLeft());
                    XAssert(c2.getRight());
                    XAssert(c3.getLeft());
                    XAssert(c3.getRight());
                    process111<B,O>(c1, *c2.getLeft(), *c3.getLeft(), metric);
                    process111<B,O>(c1, *c2.getLeft(), *c3.getRight(), metric);
                    process111<B,O>(c1, *c2.getRight(), *c3.getLeft(), metric);
                    process111<B,O>(c1, *c2.getRight(), *c3.getRight(), metric);
                }
            } else {
                if (split1) {
                    // split 1,3
                    XAssert(c1.getLeft());
                    XAssert(c1.getRight());
                    XAssert(c3.getLeft());
                    XAssert(c3.getRight());
                    process111<B,O>(*c1.getLeft(), c2, *c3.getLeft(), metric);
                    process111<B,O>(*c1.getLeft(), c2, *c3.getRight(), metric);
                    process111<B,O>(*c1.getRight(), c2, *c3.getLeft(), metric);
                    process111<B,O>(*c1.getRight(), c2, *c3.getRight(), metric);
                } else {
                    // split 3 only
                    XAssert(c3.getLeft());
                    XAssert(c3.getRight());
                    process111<B,O>(c1, c2, *c3.getLeft(), metric, 0., 0., d3sq);
                    process111<B,O>(c1, c2, *c3.getRight(), metric, 0., 0., d3sq);
                }
            }
        } else {
            if (split2) {
                if (split1) {
                    // split 1,2
                    XAssert(c1.getLeft());
                    XAssert(c1.getRight());
                    XAssert(c2.getLeft());
                    XAssert(c2.getRight());
                    process111<B,O>(*c1.getLeft(), *c2.getLeft(), c3, metric);
                    process111<B,O>(*c1.getLeft(), *c2.getRight(), c3, metric);
                    process111<B,O>(*c1.getRight(), *c2.getLeft(), c3, metric);
                    process111<B,O>(*c1.getRight(), *c2.getRight(), c3, metric);
                } else {
                    // split 2 only
                    XAssert(c2.getLeft());
                    XAssert(c2.getRight());
                    process111<B,O>(c1, *c2.getLeft(), c3, metric, 0., d2sq);
                    process111<B,O>(c1, *c2.getRight(), c3, metric, 0., d2sq);
                }
            } else {
                // split 1 only
                XAssert(c1.getLeft());
                XAssert(c1.getRight());
                process111<B,O>(*c1.getLeft(), c2, c3, metric, d1sq);
                process111<B,O>(*c1.getRight(), c2, c3, metric, d1sq);
            }
        }
    }
}

// We also set up a helper class for doing the direct processing
template <int D1, int D2, int D3>
struct DirectHelper;

template <>
struct DirectHelper<NData,NData,NData>
{
    template <int C>
    static void ProcessZeta(
        const Cell<NData,C>& , const Cell<NData,C>& , const Cell<NData,C>&,
        ZetaData<NData,NData,NData>& , int )
    {}
    template <int C>
    static void ProcessMultipole(
        const Cell<NData,C>& c1, const Cell<NData,C>& c2, const Cell<NData,C>& c3,
        const double sinphi, const double cosphi, const double www,
        ZetaData<NData,NData,NData>& zeta, int index, int maxn)
    {
        // index is the index for n=0.
        // Add www * exp(-inphi) to all -maxn <= n <= maxn

        std::complex<double> z(cosphi, sinphi);
        std::complex<double> wwwztothen = www * z;
        zeta.zeta[index] += www;
        for (int n=1; n <= maxn; ++n) {
            zeta.zeta[index + n] += wwwztothen.real();
            zeta.zeta_im[index + n] -= wwwztothen.imag();
            zeta.zeta[index - n] += wwwztothen.real();
            zeta.zeta_im[index - n] += wwwztothen.imag();
            wwwztothen *= z;
        }
    }
    template <int C>
    static void CalculateGn(
        const Cell<NData,C>& c1, const Cell<NData,C>& c2,
        double rsq, double r, int k, int maxn, double w,
        std::complex<double>* Gn)
    {
        std::complex<double> z = ProjectHelper<C>::ExpIPhi(c1.getPos(), c2.getPos(), r);
        int index = k*(2*maxn+1) + maxn;
        dbg<<"CalculateGn: "<<index<<"  "<<maxn<<"  "<<w<<"  "<<z<<std::endl;
        dbg<<"pos = "<<c1.getPos()<<"  "<<c2.getPos()<<" r = "<<r<<std::endl;
        std::complex<double> wztothen = w * z;
        Gn[index] += w;
        for (int n=1; n <= maxn; ++n) {
            // TODO: For N and K, these are symmetric conjugates.  Probably only need 1/2.
            Gn[index + n] += wztothen;
            Gn[index - n] += std::conj(wztothen);
            wztothen *= z;
        }
    }
    template <int C>
    static void CalculateZeta(const Cell<NData,C>& c1, std::complex<double>* Gn, double* weight2,
                              ZetaData<NData,NData,NData>& zeta, int nbins, int maxn)
    {
        const int nnbins = 2*maxn+1;
        const double w1 = c1.getW();
        dbg<<"CalculateZeta: "<<nbins<<"  "<<maxn<<"  "<<nnbins<<"  "<<w1<<std::endl;
        for (int k2=0; k2<nbins; ++k2) {
            // Do the k2=k3 bins.
            // We want to make sure not to include degenerate triangles with c2 = c3.
            // These would get included in the naive calculation:
            //   Zeta[k2,k3,n] = Sum w_k1 G_n[k2,n] G_n[k3,n]*
            //   G_n[k,n] = Sum w_k e^{i phi_k n)
            // The Zeta product includes terms like
            //   w_k2^2 e^{i phi_k2 n) e^{-i phi_k2 n}
            //   = w_k2^2.
            // So we need to subtract them off.
            // The /2 is so that we only count half the triangles.  In the python layer
            // (finalize) we add this to the transpose, which copies k2,k3 -> k3,k2.
            // In the process, it will double these, so only make them half the right value now.
            int iz = (k2*nbins + k2)*nnbins;
            int i2 = k2*nnbins;
            for (int n=0; n < nnbins; ++n, ++iz, ++i2) {
                zeta.zeta[iz] += w1 * (std::norm(Gn[i2]) - weight2[k2]) / 2.;
            }
            for (int k3=k2+1; k3<nbins; ++k3) {
                int i2 = k2*nnbins;
                int i3 = k3*nnbins;
                for (int n=0; n < nnbins; ++n, ++iz, ++i2, ++i3) {
                    std::complex<double> z = w1 * Gn[i2] * std::conj(Gn[i3]);
                    zeta.zeta[iz] += z.real();
                    zeta.zeta_im[iz] += z.imag();
                }
            }
        }
    }
};

template <>
struct DirectHelper<KData,KData,KData>
{
    template <int C>
    static void ProcessZeta(
        const Cell<KData,C>& c1, const Cell<KData,C>& c2, const Cell<KData,C>& c3,
        ZetaData<KData,KData,KData>& zeta, int index)
    {
        zeta.zeta[index] += c1.getData().getWK() * c2.getData().getWK() * c3.getData().getWK();
    }
    template <int C>
    static void ProcessMultipole(
        const Cell<KData,C>& c1, const Cell<KData,C>& c2, const Cell<KData,C>& c3,
        const double sinphi, const double cosphi, const double www,
        ZetaData<KData,KData,KData>& zeta, int index, int maxn)
    {
        double wk = c1.getData().getWK() * c2.getData().getWK() * c3.getData().getWK();

        std::complex<double> z(cosphi, sinphi);
        std::complex<double> wkztothen = wk * z;
        zeta.zeta[index] += wk;
        for (int n=1; n <= maxn; ++n) {
            zeta.zeta[index + n] += wkztothen.real();
            zeta.zeta_im[index + n] -= wkztothen.imag();
            zeta.zeta[index - n] += wkztothen.real();
            zeta.zeta_im[index - n] += wkztothen.imag();
            wkztothen *= z;
        }
    }
    template <int C>
    static void CalculateGn(
        const Cell<KData,C>& c1, const Cell<KData,C>& c2,
        double rsq, double r, int k, int maxn, double w, std::complex<double>* Gn)
    {
        double wk = c2.getData().getWK();
        std::complex<double> z = ProjectHelper<C>::ExpIPhi(c1.getPos(), c2.getPos(), r);
        int index = k*(2*maxn+1) + maxn;
        std::complex<double> wkztothen = wk * z;
        Gn[index] += wk;
        for (int n=1; n <= maxn; ++n) {
            Gn[index + n] += std::conj(wkztothen);
            Gn[index - n] += wkztothen;
            wkztothen *= z;
        }
    }
    template <int C>
    static void CalculateZeta(const Cell<KData,C>& c1, std::complex<double>* Gn, double* weight2,
                              ZetaData<KData,KData,KData>& zeta, int nbins, int maxn)
    {
        const int nnbins = 2*maxn+1;
        const double wk1 = c1.getData().getWK();
        for (int k2=0; k2<nbins; ++k2) {
            for (int k3=0; k3<nbins; ++k3) {
                int iz = (k2*nbins + k3)*nnbins;
                int i2 = k2*nnbins;
                int i3 = k3*nnbins;
                for (int n=0; n < nnbins; ++n, ++iz, ++i2, ++i3) {
                    std::complex<double> z = wk1 * Gn[i2] * std::conj(Gn[i3]);
                    zeta.zeta[iz] += z.real();
                    zeta.zeta_im[iz] += z.imag();
                }
            }
        }
    }
};

template <>
struct DirectHelper<GData,GData,GData>
{
    template <int C>
    static void ProcessZeta(
        const Cell<GData,C>& c1, const Cell<GData,C>& c2, const Cell<GData,C>& c3,
        ZetaData<GData,GData,GData>& zeta, int index)
    {
        std::complex<double> g1 = c1.getData().getWG();
        std::complex<double> g2 = c2.getData().getWG();
        std::complex<double> g3 = c3.getData().getWG();
        ProjectHelper<C>::Project(c1, c2, c3, g1, g2, g3);

        //std::complex<double> gam0 = g1 * g2 * g3;
        //std::complex<double> gam1 = std::conj(g1) * g2 * g3;
        //std::complex<double> gam2 = g1 * std::conj(g2) * g3;
        //std::complex<double> gam3 = g1 * g2 * std::conj(g3);

        // The complex products g1 g2 and g1 g2* share most of the calculations,
        // so faster to do this manually.
        // The above uses 32 multiplies and 16 adds.
        // We can do this with just 12 multiplies and 12 adds.
        double g1rg2r = g1.real() * g2.real();
        double g1rg2i = g1.real() * g2.imag();
        double g1ig2r = g1.imag() * g2.real();
        double g1ig2i = g1.imag() * g2.imag();

        double g1g2r = g1rg2r - g1ig2i;
        double g1g2i = g1rg2i + g1ig2r;
        double g1cg2r = g1rg2r + g1ig2i;
        double g1cg2i = g1rg2i - g1ig2r;

        double g1g2rg3r = g1g2r * g3.real();
        double g1g2rg3i = g1g2r * g3.imag();
        double g1g2ig3r = g1g2i * g3.real();
        double g1g2ig3i = g1g2i * g3.imag();
        double g1cg2rg3r = g1cg2r * g3.real();
        double g1cg2rg3i = g1cg2r * g3.imag();
        double g1cg2ig3r = g1cg2i * g3.real();
        double g1cg2ig3i = g1cg2i * g3.imag();

        zeta.gam0r[index] += g1g2rg3r - g1g2ig3i;
        zeta.gam0i[index] += g1g2rg3i + g1g2ig3r;
        zeta.gam1r[index] += g1cg2rg3r - g1cg2ig3i;
        zeta.gam1i[index] += g1cg2rg3i + g1cg2ig3r;
        zeta.gam2r[index] += g1cg2rg3r + g1cg2ig3i;
        zeta.gam2i[index] += g1cg2rg3i - g1cg2ig3r;
        zeta.gam3r[index] += g1g2rg3r + g1g2ig3i;
        zeta.gam3i[index] += -g1g2rg3i + g1g2ig3r;
    }
    template <int C>
    static void ProcessMultipole(
        const Cell<GData,C>& c1, const Cell<GData,C>& c2, const Cell<GData,C>& c3,
        const double sinphi, const double cosphi, const double www,
        ZetaData<GData,GData,GData>& zeta, int index, int maxn)
    {
        // TODO
    }
    template <int C>
    static void CalculateGn(
        const Cell<GData,C>& c1, const Cell<GData,C>& c2,
        double rsq, double r, int k, int maxn, double w, std::complex<double>* Gn)
    {
        // TODO
    }
    template <int C>
    static void CalculateZeta(const Cell<GData,C>& c1, std::complex<double>* Gn, double* weight2,
                              ZetaData<GData,GData,GData>& zeta, int nbins, int maxn)
    {
        // TODO
    }
};


template <int B, int C>
void BaseCorr3::directProcess111(
    const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
    const double d1, const double d2, const double d3, const double u, const double v,
    const double logd1, const double logd2, const double logd3, const int index)
{
    if (B == LogMultipole) {
        finishProcessMP(c1, c2, c3, d1, d2, d3, u, v, logd1, logd2, logd3, index);
    } else {
        finishProcess(c1, c2, c3, d1, d2, d3, u, v, logd1, logd2, logd3, index);
    }
}

template <int D1, int D2, int D3> template <int C>
void Corr3<D1,D2,D3>::finishProcess(
    const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
    const double d1, const double d2, const double d3, const double u, const double v,
    const double logd1, const double logd2, const double logd3, const int index)
{
    double nnn = double(c1.getN()) * c2.getN() * c3.getN();
    _ntri[index] += nnn;
    xdbg<<ws()<<"Index = "<<index<<", nnn = "<<nnn<<" => "<<_ntri[index]<<std::endl;

    double www = c1.getW() * c2.getW() * c3.getW();
    _meand1[index] += www * d1;
    _meanlogd1[index] += www * logd1;
    _meand2[index] += www * d2;
    _meanlogd2[index] += www * logd2;
    _meand3[index] += www * d3;
    _meanlogd3[index] += www * logd3;
    _meanu[index] += www * u;
    _meanv[index] += www * v;
    _weight[index] += www;

    DirectHelper<D1,D2,D3>::ProcessZeta(
        static_cast<const Cell<D1,C>&>(c1),
        static_cast<const Cell<D2,C>&>(c2),
        static_cast<const Cell<D3,C>&>(c3),
        _zeta, index);
}

template <int D1, int D2, int D3> template <int C>
void Corr3<D1,D2,D3>::finishProcessMP(
    const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
    const double d1, const double d2, const double d3, const double sinphi, const double cosphi,
    const double logd1, const double logd2, const double logd3, const int index)
{
    // Note: For multipole process, we don't have a u,v so we use those spots to store
    // sinphi, cosphi.
    // Also, the index is just the index of the first two directions of the array.
    // We will fill in all 2*maxn+1 values in the _zeta arrays that correpond to this
    // d2, d3 combination.
    double nnn = double(c1.getN()) * c2.getN() * c3.getN();
    _ntri[index] += nnn;
    dbg<<ws()<<"Index = "<<index<<", nnn = "<<nnn<<" => "<<_ntri[index]<<std::endl;

    double www = c1.getW() * c2.getW() * c3.getW();
    _meand1[index] += www * d1;
    _meanlogd1[index] += www * logd1;
    _meand2[index] += www * d2;
    _meanlogd2[index] += www * logd2;
    _meand3[index] += www * d3;
    _meanlogd3[index] += www * logd3;
    _weight[index] += www;

    DirectHelper<D1,D2,D3>::ProcessMultipole(
        static_cast<const Cell<D1,C>&>(c1),
        static_cast<const Cell<D2,C>&>(c2),
        static_cast<const Cell<D3,C>&>(c3),
        sinphi, cosphi, www, _zeta, index, _nubins);
}

template <int D1, int D2, int D3> template <int C>
void Corr3<D1,D2,D3>::calculateGn(
    const BaseCell<C>& c1, const BaseCell<C>& c2,
    double rsq, double r, double logr, int k,
    double* meanr, double* meanlogr, double* weight,
    double* meanr2, double* meanlogr2, double* weight2,
    double* npairs, std::complex<double>* Gn)
{
    // For now we only include the counts and weight from c2.
    // We'll include c1 as part of the triple in calculateZeta.
    double n = c2.getN();
    double w = c2.getW();
    double w2 = c2.getW2();
    meanr[k] += w * r;
    meanlogr[k] += w * logr;
    weight[k] += w;
    meanr2[k] += w2 * r;
    meanlogr2[k] += w2 * logr;
    weight2[k] += w2;
    npairs[k] += n;

    DirectHelper<D1,D2,D3>::CalculateGn(
        static_cast<const Cell<D1,C>&>(c1),
        static_cast<const Cell<D2,C>&>(c2),
        rsq, r, k, _nubins, w, Gn);
}

template <int D1, int D2, int D3> template <int C>
void Corr3<D1,D2,D3>::calculateZeta(
    const BaseCell<C>& c1,
    double* meanr, double* meanlogr, double* weight,
    double* meanr2, double* meanlogr2, double* weight2,
    double* npairs, std::complex<double>* Gn)
{
    // First finish the computation of meand2, etc. based on the 2pt accumulations.
    double n1 = c1.getN();
    double w1 = c1.getW();
    const int maxn = _nubins;
    const int nnbins = 2*maxn+1;
    for (int k2=0; k2<_nbins; ++k2) {
        int i = (k2 * _nbins + k2) * nnbins + maxn;
        // Do the k2=k3 bins.
        // For these, we need to be careful not to count cases where c2=c3.
        // This means we need to subtract off sums of w^2 for instance.
        // The /2 is so that we only count half the triangles.  In the python layer
        // (finalize) we add this to the transpose, which copies k2,k3 -> k3,k2.
        // In the process, it will double these, so only make them half the right value now.
        _ntri[i] += n1 * npairs[k2] * (npairs[k2]-1) / 2;
        _weight[i] += w1 * (weight[k2] * weight[k2] - weight2[k2]) / 2.;
        _meand2[i] += w1 * (weight[k2] * meanr[k2] - meanr2[k2]) / 2.;
        _meanlogd2[i] += w1 * (weight[k2] * meanlogr[k2] - meanlogr2[k2]) / 2.;
        _meand3[i] += w1 * (weight[k2] * meanr[k2] - meanr2[k2]) / 2.;
        _meanlogd3[i] += w1 * (weight[k2] * meanlogr[k2] - meanlogr2[k2]) / 2.;
        for (int k3=k2+1; k3<_nbins; ++k3) {
            i += nnbins;
            _ntri[i] += n1 * npairs[k2] * npairs[k3];
            _weight[i] += w1 * weight[k2] * weight[k3];
            _meand2[i] += w1 * weight[k3] * meanr[k2];
            _meanlogd2[i] += w1 * weight[k3] * meanlogr[k2];
            _meand3[i] += w1 * weight[k2] * meanr[k3];
            _meanlogd3[i] += w1 * weight[k2] * meanlogr[k3];
        }
    }

    // Finish the calculation for Zeta_n(d1,d2) using G_n(d).
    // In Porth et al, this is eqs. 21, 23, 24, 25 for GGG, and eqn 27 for NNN.
    // The version for KKK is obvious from these.
    DirectHelper<D1,D2,D3>::CalculateZeta(
        static_cast<const Cell<D1,C>&>(c1), Gn, weight2, _zeta, _nbins, _nubins);
}

template <int D1, int D2, int D3>
void Corr3<D1,D2,D3>::operator=(const Corr3<D1,D2,D3>& rhs)
{
    Assert(rhs._ntot == _ntot);
    _zeta.copy(rhs._zeta, _ntot);
    for (int i=0; i<_ntot; ++i) _meand1[i] = rhs._meand1[i];
    for (int i=0; i<_ntot; ++i) _meanlogd1[i] = rhs._meanlogd1[i];
    for (int i=0; i<_ntot; ++i) _meand2[i] = rhs._meand2[i];
    for (int i=0; i<_ntot; ++i) _meanlogd2[i] = rhs._meanlogd2[i];
    for (int i=0; i<_ntot; ++i) _meand3[i] = rhs._meand3[i];
    for (int i=0; i<_ntot; ++i) _meanlogd3[i] = rhs._meanlogd3[i];
    for (int i=0; i<_ntot; ++i) _meanu[i] = rhs._meanu[i];
    for (int i=0; i<_ntot; ++i) _meanv[i] = rhs._meanv[i];
    for (int i=0; i<_ntot; ++i) _weight[i] = rhs._weight[i];
    for (int i=0; i<_ntot; ++i) _ntri[i] = rhs._ntri[i];
}

template <int D1, int D2, int D3>
void Corr3<D1,D2,D3>::operator+=(const Corr3<D1,D2,D3>& rhs)
{
    Assert(rhs._ntot == _ntot);
    _zeta.add(rhs._zeta, _ntot);
    for (int i=0; i<_ntot; ++i) _meand1[i] += rhs._meand1[i];
    for (int i=0; i<_ntot; ++i) _meanlogd1[i] += rhs._meanlogd1[i];
    for (int i=0; i<_ntot; ++i) _meand2[i] += rhs._meand2[i];
    for (int i=0; i<_ntot; ++i) _meanlogd2[i] += rhs._meanlogd2[i];
    for (int i=0; i<_ntot; ++i) _meand3[i] += rhs._meand3[i];
    for (int i=0; i<_ntot; ++i) _meanlogd3[i] += rhs._meanlogd3[i];
    for (int i=0; i<_ntot; ++i) _meanu[i] += rhs._meanu[i];
    for (int i=0; i<_ntot; ++i) _meanv[i] += rhs._meanv[i];
    for (int i=0; i<_ntot; ++i) _weight[i] += rhs._weight[i];
    for (int i=0; i<_ntot; ++i) _ntri[i] += rhs._ntri[i];
}


//
// Fast multipole algorithm
// The idea here is to first recurse down to find cells that are small enough that
// at least some of the bins can use them as is.
// Then for each of these cells, we find all the cells that can contribute to the
// sums for Gn, throwing out any that are too far away.
// Next we recursively split these up, computing the parts of Gn that we can at each
// level.
// When we get to the cells that are small enough not to need to be split further, we
// finish the calculation, coputing Zeta_n from G_n.
//

template <int B, int M, int C>
void BaseCorr3::multipole(const BaseField<C>& field, bool dots)
{
    reset_ws();
    Assert(_coords == -1 || _coords == C);
    _coords = C;
    const long n1 = field.getNTopLevel();
    dbg<<"field has "<<n1<<" top level nodes\n";
    Assert(n1 > 0);

#ifdef _OPENMP
#pragma omp parallel
    {
        // Give each thread their own copy of the data vector to fill in.
        std::shared_ptr<BaseCorr3> corrp = duplicate();
        BaseCorr3& corr = *corrp;
#else
        BaseCorr3& corr = *this;
#endif

        MetricHelper<M,0> metric(0, 0, _xp, _yp, _zp);
        // Scratch arrays used for computing Gn, etc..
        std::complex<double> Gn[_nvbins];
        double meanr[_nvbins];
        double meanlogr[_nvbins];
        double weight[_nvbins];
        double meanr2[_nvbins];
        double meanlogr2[_nvbins];
        double weight2[_nvbins];
        double npairs[_nvbins];

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (long i=0;i<n1;++i) {
            const BaseCell<C>& c1 = *field.getCells()[i];
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                if (dots) std::cout<<'.'<<std::flush;
#ifdef DEBUGLOGGING
#ifdef _OPENMP
                dbg<<omp_get_thread_num()<<" "<<i<<std::endl;
#endif
                xdbg<<"field = \n";
                if (verbose_level >= 2) c1.WriteTree(get_dbgout());
#endif
            }
            corr.template multipoleSplit1<B>(c1, field.getCells(), metric,
                                             meanr, meanlogr, weight,
                                             meanr2, meanlogr2, weight2,
                                             npairs, Gn);
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            addData(corr);
        }
    }
#endif
    if (dots) std::cout<<std::endl;
}

template <int B, int M, int C>
void BaseCorr3::multipoleSplit1(
    const BaseCell<C>& c1, const std::vector<const BaseCell<C>*>& c2list,
    const MetricHelper<M,0>& metric,
    double* meanr, double* meanlogr, double* weight,
    double* meanr2, double* meanlogr2, double* weight2,
    double* npairs, std::complex<double>* Gn)
{
    dbg<<"MultipoleSplit1: c1 = "<<c1.getPos()<<"  "<<c1.getSize()<<"  "<<c1.getN()<<std::endl;
    dbg<<"           len c2  = "<<c2list.size()<<std::endl;

    const Position<C>& p1 = c1.getPos();
    double s1 = c1.getSize(); // May be modified by DistSq function.
    xdbg<<"B,M,C = "<<B<<"  "<<M<<"  "<<C<<std::endl;

    // Split cells in c2list if they will definitely need to be split.
    // And remove any that cannot contribute to the sums for this c1.
    std::vector<const BaseCell<C>*> newc2list;
    for (const BaseCell<C>* c2: c2list) {

        const Position<C>& p2 = c2->getPos();
        double s2 = c2->getSize();
        const double rsq = metric.DistSq(p1,p2,s1,s2);
        const double s1ps2 = s1+s2;
        xdbg<<"rsq = "<<rsq<<"  s = "<<s2<<"  "<<s1ps2<<std::endl;

        // This sequence mirrors the calculation in Corr2.process11.
        double rpar = 0; // Gets set to correct value by this function if appropriate
        if (metric.isRParOutsideRange(p1, p2, s1ps2, rpar)) {
            continue;
        }
        xdbg<<"RPar in range\n";

        if (BinTypeHelper<B>::tooSmallDist(rsq, s1ps2, _minsep, _minsepsq) &&
            metric.tooSmallDist(p1, p2, rsq, rpar, s1ps2, _minsep, _minsepsq)) {
            continue;
        }
        xdbg<<"Not too small separation\n";

        if (BinTypeHelper<B>::tooLargeDist(rsq, s1ps2, _maxsep, _maxsepsq) &&
            metric.tooLargeDist(p1, p2, rsq, rpar, s1ps2, _maxsep, _maxsepsq)) {
            continue;
        }
        xdbg<<"Not too large separation\n";

        int k=-1;
        double r=0,logr=0;
        // First check if the distance alone requires a split for c2.
        bool split = !BinTypeHelper<B>::singleBin(rsq, s1ps2, p1, p2, _binsize, _b, _bsq,
                                                   _minsep, _maxsep, _logminsep, k, r, logr);
        xdbg<<"split = "<<split<<std::endl;

        // If not splitting due to side length, might still need to split for angle.
        if (!split) {
            double bphisq_eff = BinTypeHelper<B>::getEffectiveBSq(rsq, _busq);
            split = SQR(s1ps2) > bphisq_eff;
            xdbg<<"split => "<<split<<std::endl;
        }

        // If we need to split something, split c2 if it's larger than c1.
        if (split && s2 > s1) {
            Assert(c2->getLeft());
            Assert(c2->getRight());
            newc2list.push_back(c2->getLeft());
            newc2list.push_back(c2->getRight());
        } else {
            newc2list.push_back(c2);
        }
    }

    // See if we can stop splitting c1
    // The criterion is that for the largest separation we will be caring about,
    // c1 is at least possibly small enough to use as is without futher splitting.
    // i.e. s1 < maxsep * b
    double maxbsq_eff = BinTypeHelper<B>::getEffectiveBSq(_maxsepsq, _bsq);
    if (SQR(s1) > maxbsq_eff) {
        multipoleSplit1<B>(*c1.getLeft(), newc2list, metric,
                           meanr, meanlogr, weight, meanr2, meanlogr2, weight2, npairs, Gn);
        multipoleSplit1<B>(*c1.getRight(), newc2list, metric,
                           meanr, meanlogr, weight, meanr2, meanlogr2, weight2, npairs, Gn);
    } else {
        // Zero out scratch arrays
        for (int i=0; i<_nvbins; ++i) {
            meanr[i] = 0.;
            meanlogr[i] = 0.;
            weight[i] = 0.;
            meanr2[i] = 0.;
            meanlogr2[i] = 0.;
            weight2[i] = 0.;
            npairs[i] = 0.;
            Gn[i] = 0.;
        }
        multipoleFinish<B>(c1, newc2list, metric,
                           meanr, meanlogr, weight, meanr2, meanlogr2, weight2, npairs, Gn);
    }
}

template <int B, int M, int C>
void BaseCorr3::multipoleFinish(
    const BaseCell<C>& c1, const std::vector<const BaseCell<C>*>& c2list,
    const MetricHelper<M,0>& metric,
    double* meanr, double* meanlogr, double* weight,
    double* meanr2, double* meanlogr2, double* weight2,
    double* npairs, std::complex<double>* Gn)
{
    // This is structured a lot like the previous function.
    // However, in this one, we will actually be filling the Gn array.
    // We fill what we can before splitting further.
    // Note: if we decide to split c1, we need to make a copy of Gn before
    // recursing.  This is why we split up the calculation into two functions like
    // this.  We want to minimize the number of copies of Gn (et al) we need to make.
    dbg<<"MultipoleFinish: c1 = "<<c1.getPos()<<"  "<<c1.getSize()<<"  "<<c1.getN()<<std::endl;
    dbg<<"           len c2  = "<<c2list.size()<<std::endl;

    const Position<C>& p1 = c1.getPos();
    double s1 = c1.getSize(); // May be modified by DistSq function.
    double s1sq = SQR(s1);
    xdbg<<"B,M,C = "<<B<<"  "<<M<<"  "<<C<<std::endl;
    bool anysplit1=false;

    // As before, split any cells in c2list that need to be split given the current c1.
    // And remove any that cannot contribute to the sums.
    // This time we also accumulate the Gn array for any that can be done.
    // Those also don't get added to newc2list.
    std::vector<const BaseCell<C>*> newc2list;
    for (const BaseCell<C>* c2: c2list) {

        const Position<C>& p2 = c2->getPos();
        double s2 = c2->getSize();
        const double rsq = metric.DistSq(p1,p2,s1,s2);
        xdbg<<"rsq = "<<rsq<<std::endl;
        const double s1ps2 = s1+s2;

        // This sequence mirrors the calculation in Corr2.process11.
        double rpar = 0; // Gets set to correct value by this function if appropriate
        if (metric.isRParOutsideRange(p1, p2, s1ps2, rpar)) {
            continue;
        }
        xdbg<<"RPar in range\n";

        if (BinTypeHelper<B>::tooSmallDist(rsq, s1ps2, _minsep, _minsepsq) &&
            metric.tooSmallDist(p1, p2, rsq, rpar, s1ps2, _minsep, _minsepsq)) {
            continue;
        }
        xdbg<<"Not too small separation\n";

        if (BinTypeHelper<B>::tooLargeDist(rsq, s1ps2, _maxsep, _maxsepsq) &&
            metric.tooLargeDist(p1, p2, rsq, rpar, s1ps2, _maxsep, _maxsepsq)) {
            continue;
        }
        xdbg<<"Not too large separation\n";

        // Now check if these cells are small enough that it is ok to drop into a single bin.
        int k=-1;
        double r=0,logr=0;

        if (metric.isRParInsideRange(p1, p2, s1ps2, rpar) &&
            BinTypeHelper<B>::singleBin(rsq, s1ps2, p1, p2, _binsize, _b, _bsq,
                                        _minsep, _maxsep, _logminsep, k, r, logr))
        {
            // Check angle
            double bphisq_eff = BinTypeHelper<B>::getEffectiveBSq(rsq, _busq);
            if (SQR(s1ps2) <= bphisq_eff) {
                // This c2 is fine to use as is with the current c1.  Neither needs to split.
                xdbg<<"s1ps2 = "<<s1ps2<<" _b = "<<_b<<"  "<<_bu<<std::endl;
                xdbg<<"Drop into single bin.\n";
                if (BinTypeHelper<B>::isRSqInRange(rsq, p1, p2, _minsep, _minsepsq,
                                                   _maxsep, _maxsepsq)) {
                    if (k < 0) {
                        // Then these aren't calculated yet.  Do that now.
                        r = sqrt(rsq);
                        logr = log(r);
                        k = BinTypeHelper<B>::calculateBinK(p1, p2, r, logr, _binsize,
                                                            _minsep, _maxsep, _logminsep);
                    }
                    calculateGn(c1, *c2, rsq, r, logr, k,
                                meanr, meanlogr, weight, meanr2, meanlogr2, weight2, npairs, Gn);
                }
                continue;
            }
        }

        // OK, need to split.  Figure out if we should split c1 or c2 or both.
        bool split1=false;
        bool split2=false;
        double bsq_eff = BinTypeHelper<B>::getEffectiveBSq(rsq,std::min(_bsq,_busq));
        CalcSplitSq(split1,split2,s1,s2,s1ps2,bsq_eff);
        xdbg<<"split2 = "<<split2<<std::endl;
        if (split1) anysplit1 = true;

        if (split2) {
            Assert(c2->getLeft());
            Assert(c2->getRight());
            newc2list.push_back(c2->getLeft());
            newc2list.push_back(c2->getRight());
        } else {
            newc2list.push_back(c2);
        }
    }
    xdbg<<"newsize = "<<newc2list.size()<<", anysplit1 = "<<anysplit1<<std::endl;
    if (newc2list.size() > 0) {
        if (anysplit1) {
            Assert(c1.getLeft());
            Assert(c1.getRight());
            // Then we need to split c1 further.  This means we need a copy of Gn and the
            // other scratch arrays, so we can pass what we have now to each child cell.
            double meanr_copy[_nvbins];
            double meanlogr_copy[_nvbins];
            double weight_copy[_nvbins];
            double meanr2_copy[_nvbins];
            double meanlogr2_copy[_nvbins];
            double weight2_copy[_nvbins];
            double npairs_copy[_nvbins];
            std::complex<double> Gn_copy[_nvbins];
            for (int i=0;i<_nvbins;++i) {
                meanr_copy[i] = meanr[i];
                meanlogr_copy[i] = meanlogr[i];
                weight_copy[i] = weight[i];
                meanr2_copy[i] = meanr2[i];
                meanlogr2_copy[i] = meanlogr2[i];
                weight2_copy[i] = weight2[i];
                npairs_copy[i] = npairs[i];
                Gn_copy[i] = Gn[i];
            }
            multipoleFinish<B>(*c1.getLeft(), newc2list, metric,
                               meanr_copy, meanlogr_copy, weight_copy,
                               meanr2_copy, meanlogr2_copy, weight2_copy,
                               npairs_copy, Gn_copy);
            multipoleFinish<B>(*c1.getRight(), newc2list, metric,
                               meanr_copy, meanlogr_copy, weight_copy,
                               meanr2_copy, meanlogr2_copy, weight2_copy,
                               npairs, Gn);
        } else {
            // If we still have c2 items to process, but don't have to split c1,
            // we don't need to make copies.
            multipoleFinish<B>(c1, newc2list, metric, meanr, meanlogr, weight,
                               meanr2, meanlogr2, weight2, npairs, Gn);
        }
    } else {
        // We finished all the calculations for Gn.
        // Turn this into Zeta_n.
        calculateZeta(c1, meanr, meanlogr, weight, meanr2, meanlogr2, weight2, npairs, Gn);
    }
}

//
//
// The functions we call from Python.
//
//

template <int D1, int D2, int D3>
Corr3<D1,D2,D3>* BuildCorr3(
    BinType bin_type, double minsep, double maxsep, int nbins, double binsize, double b,
    double minu, double maxu, int nubins, double ubinsize, double bu,
    double minv, double maxv, int nvbins, double vbinsize, double bv,
    double xp, double yp, double zp,
    py::array_t<double>& zeta0p, py::array_t<double>& zeta1p,
    py::array_t<double>& zeta2p, py::array_t<double>& zeta3p,
    py::array_t<double>& zeta4p, py::array_t<double>& zeta5p,
    py::array_t<double>& zeta6p, py::array_t<double>& zeta7p,
    py::array_t<double>& meand1p, py::array_t<double>& meanlogd1p,
    py::array_t<double>& meand2p, py::array_t<double>& meanlogd2p,
    py::array_t<double>& meand3p, py::array_t<double>& meanlogd3p,
    py::array_t<double>& meanup, py::array_t<double>& meanvp,
    py::array_t<double>& weightp, py::array_t<double>& ntrip)
{
    double* zeta0 = zeta0p.size() == 0 ? 0 : static_cast<double*>(zeta0p.mutable_data());
    double* zeta1 = zeta1p.size() == 0 ? 0 : static_cast<double*>(zeta1p.mutable_data());
    double* zeta2 = zeta2p.size() == 0 ? 0 : static_cast<double*>(zeta2p.mutable_data());
    double* zeta3 = zeta3p.size() == 0 ? 0 : static_cast<double*>(zeta3p.mutable_data());
    double* zeta4 = zeta4p.size() == 0 ? 0 : static_cast<double*>(zeta4p.mutable_data());
    double* zeta5 = zeta5p.size() == 0 ? 0 : static_cast<double*>(zeta5p.mutable_data());
    double* zeta6 = zeta6p.size() == 0 ? 0 : static_cast<double*>(zeta6p.mutable_data());
    double* zeta7 = zeta7p.size() == 0 ? 0 : static_cast<double*>(zeta7p.mutable_data());
    double* meand1 = static_cast<double*>(meand1p.mutable_data());
    double* meanlogd1 = static_cast<double*>(meanlogd1p.mutable_data());
    double* meand2 = static_cast<double*>(meand2p.mutable_data());
    double* meanlogd2 = static_cast<double*>(meanlogd2p.mutable_data());
    double* meand3 = static_cast<double*>(meand3p.mutable_data());
    double* meanlogd3 = static_cast<double*>(meanlogd3p.mutable_data());
    double* meanu = static_cast<double*>(meanup.mutable_data());
    double* meanv = static_cast<double*>(meanvp.mutable_data());
    double* weight = static_cast<double*>(weightp.mutable_data());
    double* ntri = static_cast<double*>(ntrip.mutable_data());

    dbg<<"Start BuildCorr3 "<<D1<<" "<<D2<<" "<<D3<<" "<<bin_type<<std::endl;
    Assert(D2 == D1);
    Assert(D3 == D1);

    return new Corr3<D1,D2,D3>(
            bin_type, minsep, maxsep, nbins, binsize, b,
            minu, maxu, nubins, ubinsize, bu, minv, maxv, nvbins, vbinsize, bv, xp, yp, zp,
            zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7,
            meand1, meanlogd1, meand2, meanlogd2, meand3, meanlogd3,
            meanu, meanv, weight, ntri);
}

// A quick helper struct so we only instantiate multipole for Multipole BinTypes.
// _B is actually the corresponding 2pt BinType since the algorithm will use 2pt tests.
// For now, we only have one such 3pt type, LogMultipole, so _B is always Log.
template <int B>
struct ValidMultipole
{ enum { _B = Log }; };

template <int B, int M, int C>
void ProcessAutob(BaseCorr3& corr, BaseField<C>& field, bool dots)
{
    Assert((ValidMC<M,C>::_M == M));
    // Note: The code exists to do Multipole binning with the regular processing, which
    // was useful in development.  But the alternate algorithm is really the point of that
    // binning, so I always do that now.  Not sure if there is a good reason to have an
    // option to use the normal 3pt algorithm with LogMultipole, but maybe.
    if (B == LogMultipole) {
        corr.template multipole<ValidMultipole<B>::_B,ValidMC<M,C>::_M>(field, dots);
    } else {
        corr.template process<B,ValidMC<M,C>::_M>(field, dots);
    }
}

template <int B, int C>
void ProcessAutoa(BaseCorr3& corr, BaseField<C>& field, bool dots, Metric metric)
{
    switch(metric) {
      case Euclidean:
           ProcessAutob<B,Euclidean>(corr, field, dots);
           break;
      case Arc:
           ProcessAutob<B,Arc>(corr, field, dots);
           break;
      case Periodic:
           ProcessAutob<B,Periodic>(corr, field, dots);
           break;
      default:
           Assert(false);
    }
}

template <int C>
void ProcessAuto(BaseCorr3& corr, BaseField<C>& field,
                 bool dots, BinType bin_type, Metric metric)
{
    dbg<<"Start ProcessAuto "<<bin_type<<" "<<metric<<std::endl;
    switch(bin_type) {
      case LogRUV:
           ProcessAutoa<LogRUV>(corr, field, dots, metric);
           break;
      case LogSAS:
           ProcessAutoa<LogSAS>(corr, field, dots, metric);
           break;
      case LogMultipole:
           ProcessAutoa<LogMultipole>(corr, field, dots, metric);
           break;
      default:
           Assert(false);
    }
}

template <int B, int M, int C>
void ProcessCross12b(BaseCorr3& corr, BaseField<C>& field1, BaseField<C>& field2,
                     int ordered, bool dots)
{
    Assert((ValidMC<M,C>::_M == M));
    switch(ordered) {
      case 0:
           corr.template process<B,0,ValidMC<M,C>::_M>(field1, field2, dots);
           break;
      case 1:
           corr.template process<B,1,ValidMC<M,C>::_M>(field1, field2, dots);
           break;
      default:
           Assert(false);
    }
}

template <int B, int C>
void ProcessCross12a(BaseCorr3& corr, BaseField<C>& field1, BaseField<C>& field2,
                     int ordered, bool dots, Metric metric)
{
    switch(metric) {
      case Euclidean:
           ProcessCross12b<B,Euclidean>(corr, field1, field2, ordered, dots);
           break;
      case Arc:
           ProcessCross12b<B,Arc>(corr, field1, field2, ordered, dots);
           break;
      case Periodic:
           ProcessCross12b<B,Periodic>(corr, field1, field2, ordered, dots);
           break;
      default:
           Assert(false);
    }
}

template <int C>
void ProcessCross12(BaseCorr3& corr, BaseField<C>& field1, BaseField<C>& field2,
                    int ordered, bool dots, BinType bin_type, Metric metric)
{
    dbg<<"Start ProcessCross12 "<<bin_type<<" "<<ordered<<"  "<<metric<<std::endl;
    switch(bin_type) {
      case LogRUV:
           ProcessCross12a<LogRUV>(corr, field1, field2, ordered, dots, metric);
           break;
      case LogSAS:
           ProcessCross12a<LogSAS>(corr, field1, field2, ordered, dots, metric);
           break;
      case LogMultipole:
           ProcessCross12a<LogMultipole>(corr, field1, field2, ordered, dots, metric);
           break;
      default:
           Assert(false);
    }
}

template <int B, int M, int C>
void ProcessCrossb(BaseCorr3& corr,
                   BaseField<C>& field1, BaseField<C>& field2, BaseField<C>& field3,
                   int ordered, bool dots)
{
    Assert((ValidMC<M,C>::_M == M));
    switch(ordered) {
      case 0:
           corr.template process<B,0,ValidMC<M,C>::_M>(field1, field2, field3, dots);
           break;
      case 1:
           corr.template process<B,1,ValidMC<M,C>::_M>(field1, field2, field3, dots);
           break;
      case 3:
           corr.template process<B,3,ValidMC<M,C>::_M>(field1, field2, field3, dots);
           break;
      default:
           Assert(false);
    }
}

template <int B, int C>
void ProcessCrossa(BaseCorr3& corr,
                   BaseField<C>& field1, BaseField<C>& field2, BaseField<C>& field3,
                   int ordered, bool dots, Metric metric)
{
    switch(metric) {
      case Euclidean:
           ProcessCrossb<B,Euclidean>(corr, field1, field2, field3, ordered, dots);
           break;
      case Arc:
           ProcessCrossb<B,Arc>(corr, field1, field2, field3, ordered, dots);
           break;
      case Periodic:
           ProcessCrossb<B,Periodic>(corr, field1, field2, field3, ordered, dots);
           break;
      default:
           Assert(false);
    }
}

template <int C>
void ProcessCross(BaseCorr3& corr,
                  BaseField<C>& field1, BaseField<C>& field2, BaseField<C>& field3,
                  int ordered, bool dots, BinType bin_type, Metric metric)
{
    dbg<<"Start ProcessCross3 "<<bin_type<<" "<<ordered<<"  "<<metric<<std::endl;
    switch(bin_type) {
      case LogRUV:
           ProcessCrossa<LogRUV>(corr, field1, field2, field3, ordered, dots, metric);
           break;
      case LogSAS:
           ProcessCrossa<LogSAS>(corr, field1, field2, field3, ordered, dots, metric);
           break;
      case LogMultipole:
           ProcessCrossa<LogMultipole>(corr, field1, field2, field3, ordered, dots, metric);
           break;
      default:
           Assert(false);
    }
}

// Export the above functions using pybind11

template <int C, typename W>
void WrapProcess(py::module& _treecorr, W& base_corr3)
{
    typedef void (*auto_type)(BaseCorr3& corr, BaseField<C>& field,
                              bool dots, BinType bin_type, Metric metric);
    typedef void (*cross12_type)(BaseCorr3& corr, BaseField<C>& field1, BaseField<C>& field2,
                                 int ordered, bool dots, BinType bin_type, Metric metric);
    typedef void (*cross_type)(BaseCorr3& corr,
                               BaseField<C>& field1, BaseField<C>& field2, BaseField<C>& field3,
                               int ordered, bool dots, BinType bin_type, Metric metric);

    base_corr3.def("processAuto", auto_type(&ProcessAuto));
    base_corr3.def("processCross12", cross12_type(&ProcessCross12));
    base_corr3.def("processCross", cross_type(&ProcessCross));
}

template <int D1, int D2, int D3>
void WrapCorr3(py::module& _treecorr, std::string prefix)
{
    typedef Corr3<D1,D2,D3>* (*init_type)(
        BinType bin_type, double minsep, double maxsep, int nbins, double binsize, double b,
        double minu, double maxu, int nubins, double ubinsize, double bu,
        double minv, double maxv, int nvbins, double vbinsize, double bv,
        double xp, double yp, double zp,
        py::array_t<double>& zeta0p, py::array_t<double>& zeta1p,
        py::array_t<double>& zeta2p, py::array_t<double>& zeta3p,
        py::array_t<double>& zeta4p, py::array_t<double>& zeta5p,
        py::array_t<double>& zeta6p, py::array_t<double>& zeta7p,
        py::array_t<double>& meand1p, py::array_t<double>& meanlogd1p,
        py::array_t<double>& meand2p, py::array_t<double>& meanlogd2p,
        py::array_t<double>& meand3p, py::array_t<double>& meanlogd3p,
        py::array_t<double>& meanup, py::array_t<double>& meanvp,
        py::array_t<double>& weightp, py::array_t<double>& ntrip);

    py::class_<Corr3<D1,D2,D3>, BaseCorr3> corr3(_treecorr, (prefix + "Corr").c_str());
    corr3.def(py::init(init_type(&BuildCorr3)));
}

void pyExportCorr3(py::module& _treecorr)
{
    py::class_<BaseCorr3> base_corr3(_treecorr, "BaseCorr3");

    WrapProcess<Flat>(_treecorr, base_corr3);
    WrapProcess<Sphere>(_treecorr, base_corr3);
    WrapProcess<ThreeD>(_treecorr, base_corr3);

    WrapCorr3<NData,NData,NData>(_treecorr, "NNN");
    WrapCorr3<KData,KData,KData>(_treecorr, "KKK");
    WrapCorr3<GData,GData,GData>(_treecorr, "GGG");
}
