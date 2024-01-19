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

// The point of the multipole binning is to use the fast algorithm with Gn, Wn, etc.
// However, it is useful for debugging to also have code paths that do this the direct way
// using the three cell recursive algorithm.
// Uncomment this line to switch to use that instead.
//#define DIRECT_MULTIPOLE

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
    double xp, double yp, double zp, bool nnn):
    _bin_type(bin_type), _is_multipole(false),
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
           _nvbins = _nbins * (2*_nubins+1);
           _is_multipole = true;
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
    double* weight, double* weight_im, double* ntri) :
    BaseCorr3(bin_type, minsep, maxsep, nbins, binsize, b,
              minu, maxu, nubins, ubinsize, bu,
              minv, maxv, nvbins, vbinsize, bv,
              xp, yp, zp, (D1==NData && D2==NData && D3==NData)),
    _owns_data(false),
    _zeta(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, _is_multipole),
    _meand1(meand1), _meanlogd1(meanlogd1), _meand2(meand2), _meanlogd2(meanlogd2),
    _meand3(meand3), _meanlogd3(meanlogd3), _meanu(meanu), _meanv(meanv),
    _weight(weight), _weight_im(weight_im), _ntri(ntri)
{}

BaseCorr3::BaseCorr3(const BaseCorr3& rhs):
    _bin_type(rhs._bin_type), _is_multipole(rhs._is_multipole),
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
    _zeta(0,0,0,0,0,0,0,0, _is_multipole)
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
    if (_is_multipole)
        _weight_im = new double[_ntot];
    else
        _weight_im = 0;
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
        if (_is_multipole) {
            delete [] _weight_im; _weight_im = 0;
        }
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
    if (_is_multipole) {
        for (int i=0; i<_ntot; ++i) _weight_im[i] = 0.;
    }
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
        } else if (BinTypeHelper<B>::swap_23) {
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
        } else {
            // For LogMultipole, just do all the combinations.
            process111Sorted<B,3>(c1, c3, c2, metric, d1sq, d3sq, d2sq);
            process111Sorted<B,3>(c1, c2, c3, metric, d1sq, d2sq, d3sq);
            process111Sorted<B,3>(c2, c1, c3, metric, d2sq, d1sq, d3sq);
            process111Sorted<B,3>(c2, c3, c1, metric, d2sq, d3sq, d1sq);
            process111Sorted<B,3>(c3, c2, c1, metric, d3sq, d2sq, d1sq);
            process111Sorted<B,3>(c3, c1, c2, metric, d3sq, d1sq, d2sq);
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
        } else if (BinTypeHelper<B>::swap_23) {
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
        } else {
            // For LogMultipole, this is just a debugging thing anyway, so do the most
            // straightforward thing and process both orderings.
            process111Sorted<B,3>(c1, c2, c3, metric, d1sq, d2sq, d3sq);
            process111Sorted<B,3>(c1, c3, c2, metric, d1sq, d3sq, d2sq);
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
        double d1, double d2, double d3, double sinphi, double cosphi, double www,
        double* weight, double* weight_im,
        ZetaData<NData,NData,NData>& zeta, int index, int maxn)
    {
        // index is the index for n=0.
        // Add www * exp(-inphi) to all -maxn <= n <= maxn

        std::complex<double> z(cosphi, -sinphi);
        weight[index] += www;
        std::complex<double> wwwztothen = www;
        for (int n=1; n <= maxn; ++n) {
            wwwztothen *= z;
            weight[index + n] += wwwztothen.real();
            weight_im[index + n] += wwwztothen.imag();
            weight[index - n] += wwwztothen.real();
            weight_im[index - n] -= wwwztothen.imag();
        }
    }

    template <int C>
    static void CalculateGn(
        const Cell<NData,C>& c1, const Cell<NData,C>& c2,
        double rsq, double r, int k, int maxn, double w,
        MultipoleScratch<NData, NData>& mp)
    {
        std::complex<double> z = ProjectHelper<C>::ExpIPhi(c1.getPos(), c2.getPos(), r);
        int index = k*(maxn+1);
        mp.Wn[index] += w;
        std::complex<double> wztothen = w;
        for (int n=1; n <= maxn; ++n) {
            wztothen *= z;
            mp.Wn[index + n] += wztothen;
        }
    }

    template <int C>
    static void CalculateZeta(const Cell<NData,C>& c1,
                              MultipoleScratch<NData, NData>& mp,
                              double* weight, double* weight_im,
                              ZetaData<NData,NData,NData>& zeta, int nbins, int maxn)
    {
        const double w1 = c1.getW();
        const int step23 = 2*maxn+1;
        const int step32 = nbins * step23;
        const int step22 = step32 + step23;
        int iz22 = maxn;
        for (int k2=0; k2<nbins; ++k2, iz22+=step22) {
            // Do the k2=k3 bins.
            // We want to make sure not to include degenerate triangles with c2 = c3.
            // These would get included in the naive calculation:
            //   W[k2,k3,n] = Sum w_k1 W_n[k2,n] W_n[k3,n]*
            //   W_n[k,n] = Sum w_k e^(i phi_k n)
            // The W product includes terms like
            //   (w_k2)^2 e^(i phi_k2 n) e^(-i phi_k2 n)
            //   = (w_k2)^2.
            // So we need to subtract them off.
            // sumww has been storing Sum (w_k)^2, which is the amount to subtract in each bin.

            const int i2 = k2*(maxn+1);
            for (int n=0; n<=maxn; ++n) {
                double www = w1 * (std::norm(mp.Wn[i2+n]) - mp.sumww[k2]);
                weight[iz22+n] += www;
                if (n > 0) {
                    weight[iz22-n] += www;
                }
            }
            int iz23 = iz22 + step23;
            int iz32 = iz22 + step32;
            for (int k3=k2+1; k3<nbins; ++k3, iz23+=step23, iz32+=step32) {
                XAssert(iz23 == (k2 * nbins + k3) * (2*maxn+1) + maxn);
                XAssert(iz32 == (k3 * nbins + k2) * (2*maxn+1) + maxn);
                const int i3 = k3*(maxn+1);
                for (int n=0; n<=maxn; ++n) {
                    std::complex<double> www = w1 * mp.Wn[i2+n] * std::conj(mp.Wn[i3+n]);
                    weight[iz23+n] += www.real();
                    weight_im[iz23+n] += www.imag();
                    // Given the symmetry of these, we could skip these here and copy them later,
                    // but a few extra additions in this step is a tiny fraction of the total
                    // computation, so for simplicity, just do them all here.
                    weight[iz32+n] += www.real();
                    weight_im[iz32+n] -= www.imag();
                    if (n > 0) {
                        weight[iz23-n] += www.real();
                        weight_im[iz23-n] -= www.imag();
                        weight[iz32-n] += www.real();
                        weight_im[iz32-n] += www.imag();
                    }
                }
            }
        }
    }

    template <int C>
    static void CalculateZeta(const Cell<NData,C>& c1, int ordered,
                              MultipoleScratch<NData, NData>& mp2,
                              MultipoleScratch<NData, NData>& mp3,
                              double* weight, double* weight_im,
                              ZetaData<NData,NData,NData>& zeta, int nbins, int maxn)
    {
        const double w1 = c1.getW();
        const int step = 2*maxn+1;
        int iz = maxn;
        // If ordered == 1, then also count contribution from swapping cats 2,3
        const bool swap23 = (ordered == 1);
        for (int k2=0; k2<nbins; ++k2) {
            const int i2 = k2*(maxn+1);
            for (int k3=0; k3<nbins; ++k3, iz+=step) {
                const int i3 = k3*(maxn+1);
                for (int n=0; n<=maxn; ++n) {
                    // Slighlty confusing: c2 is across from d2, so Wn2 (which used c2list)
                    // has values that correspond to d3 (distance from c1 to c2).
                    // So we use mp2.Wn with i3, but mp3.Wn with i2.
                    // The conjugate goes with Wn2, since phi sweeps from d2 to d3, and
                    // we want z = exp(-inphi).
                    std::complex<double> www = w1 * mp3.Wn[i2+n] * std::conj(mp2.Wn[i3+n]);
                    if (swap23) {
                        www += w1 * mp2.Wn[i2+n] * std::conj(mp3.Wn[i3+n]);
                    }
                    weight[iz+n] += www.real();
                    weight_im[iz+n] += www.imag();
                    if (n > 0) {
                        weight[iz-n] += www.real();
                        weight_im[iz-n] -= www.imag();
                    }
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
        double d1, double d2, double d3, double sinphi, double cosphi, double www,
        double* weight, double* weight_im,
        ZetaData<KData,KData,KData>& zeta, int index, int maxn)
    {
        double wk = c1.getData().getWK() * c2.getData().getWK() * c3.getData().getWK();

        std::complex<double> z(cosphi, -sinphi);
        weight[index] += www;
        zeta.zeta[index] += wk;
        std::complex<double> wwwztothen = www;
        std::complex<double> wkztothen = wk;
        for (int n=1; n <= maxn; ++n) {
            wwwztothen *= z;
            wkztothen *= z;
            weight[index + n] += wwwztothen.real();
            weight_im[index + n] += wwwztothen.imag();
            weight[index - n] += wwwztothen.real();
            weight_im[index - n] -= wwwztothen.imag();
            zeta.zeta[index + n] += wkztothen.real();
            zeta.zeta_im[index + n] += wkztothen.imag();
            zeta.zeta[index - n] += wkztothen.real();
            zeta.zeta_im[index - n] -= wkztothen.imag();
        }
    }

    template <int C>
    static void CalculateGn(
        const Cell<KData,C>& c1, const Cell<KData,C>& c2,
        double rsq, double r, int k, int maxn, double w,
        MultipoleScratch<KData, KData>& mp)
    {
        double wk = c2.getData().getWK();
        if (mp.ww) mp.sumwwkk[k] += wk * wk;
        std::complex<double> z = ProjectHelper<C>::ExpIPhi(c1.getPos(), c2.getPos(), r);
        int index = k*(maxn+1);
        mp.Wn[index] += w;
        mp.Gn[index] += wk;
        std::complex<double> wztothen = w;
        std::complex<double> wkztothen = wk;
        for (int n=1; n <= maxn; ++n) {
            wztothen *= z;
            wkztothen *= z;
            mp.Wn[index + n] += wztothen;
            mp.Gn[index + n] += wkztothen;
        }
    }

    template <int C>
    static void CalculateZeta(const Cell<KData,C>& c1,
                              MultipoleScratch<KData, KData>& mp,
                              double* weight, double* weight_im,
                              ZetaData<KData,KData,KData>& zeta, int nbins, int maxn)
    {
        const double w1 = c1.getW();
        const double wk1 = c1.getData().getWK();
        const int step23 = 2*maxn+1;
        const int step32 = nbins * step23;
        const int step22 = step32 + step23;
        int iz22 = maxn;
        for (int k2=0; k2<nbins; ++k2, iz22+=step22) {
            // Do the k2=k3 bins.
            // As for NNN, we subtract off the sum of w_k^2 when k2=k3.
            // We also subtract off the sum of (w_k kappa_k)^2 for the same reason.

            const int i2 = k2*(maxn+1);
            for (int n=0; n<=maxn; ++n) {
                double www = w1 * (std::norm(mp.Wn[i2+n]) - mp.sumww[k2]);
                double wwwk = wk1 * (std::norm(mp.Gn[i2+n]) - mp.sumwwkk[k2]);
                weight[iz22+n] += www;
                zeta.zeta[iz22+n] += wwwk;
                if (n > 0) {
                    weight[iz22-n] += www;
                    zeta.zeta[iz22-n] += wwwk;
                }
            }
            int iz23 = iz22 + step23;
            int iz32 = iz22 + step32;
            for (int k3=k2+1; k3<nbins; ++k3, iz23+=step23, iz32+=step32) {
                XAssert(iz23 == (k2 * nbins + k3) * (2*maxn+1) + maxn);
                XAssert(iz32 == (k3 * nbins + k2) * (2*maxn+1) + maxn);
                const int i3 = k3*(maxn+1);
                for (int n=0; n<=maxn; ++n) {
                    std::complex<double> www = w1 * mp.Wn[i2+n] * std::conj(mp.Wn[i3+n]);
                    std::complex<double> wwwk = wk1 * mp.Gn[i2+n] * std::conj(mp.Gn[i3+n]);
                    weight[iz23+n] += www.real();
                    weight_im[iz23+n] += www.imag();
                    weight[iz32+n] += www.real();
                    weight_im[iz32+n] -= www.imag();
                    zeta.zeta[iz23+n] += wwwk.real();
                    zeta.zeta_im[iz23+n] += wwwk.imag();
                    zeta.zeta[iz32+n] += wwwk.real();
                    zeta.zeta_im[iz32+n] -= wwwk.imag();
                    if (n > 0) {
                        weight[iz23-n] += www.real();
                        weight_im[iz23-n] -= www.imag();
                        weight[iz32-n] += www.real();
                        weight_im[iz32-n] += www.imag();
                        zeta.zeta[iz23-n] += wwwk.real();
                        zeta.zeta_im[iz23-n] -= wwwk.imag();
                        zeta.zeta[iz32-n] += wwwk.real();
                        zeta.zeta_im[iz32-n] += wwwk.imag();
                    }
                }
            }
        }
    }

    template <int C>
    static void CalculateZeta(const Cell<KData,C>& c1, int ordered,
                              MultipoleScratch<KData, KData>& mp2,
                              MultipoleScratch<KData, KData>& mp3,
                              double* weight, double* weight_im,
                              ZetaData<KData,KData,KData>& zeta, int nbins, int maxn)
    {
        const double w1 = c1.getW();
        const double wk1 = c1.getData().getWK();
        const int step = 2*maxn+1;
        int iz = maxn;
        // If ordered == 1, then also count contribution from swapping cats 2,3
        const bool swap23 = (ordered == 1);
        for (int k2=0; k2<nbins; ++k2) {
            const int i2 = k2*(maxn+1);
            for (int k3=0; k3<nbins; ++k3, iz+=step) {
                const int i3 = k3*(maxn+1);
                for (int n=0; n<=maxn; ++n) {
                    std::complex<double> www = w1 * mp3.Wn[i2+n] * std::conj(mp2.Wn[i3+n]);
                    std::complex<double> wwwk = wk1 * mp3.Gn[i2+n] * std::conj(mp2.Gn[i3+n]);
                    if (swap23) {
                        www += w1 * mp2.Wn[i2+n] * std::conj(mp3.Wn[i3+n]);
                        wwwk += wk1 * mp2.Gn[i2+n] * std::conj(mp3.Gn[i3+n]);
                    }
                    weight[iz+n] += www.real();
                    weight_im[iz+n] += www.imag();
                    zeta.zeta[iz+n] += wwwk.real();
                    zeta.zeta_im[iz+n] += wwwk.imag();
                    if (n > 0) {
                        weight[iz-n] += www.real();
                        weight_im[iz-n] -= www.imag();
                        zeta.zeta[iz-n] += wwwk.real();
                        zeta.zeta_im[iz-n] -= wwwk.imag();
                    }
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
        double d1, double d2, double d3, double sinphi, double cosphi, double www,
        double* weight, double* weight_im,
        ZetaData<GData,GData,GData>& zeta, int index, int maxn)
    {
        std::complex<double> g1 = c1.getData().getWG();
        std::complex<double> g2 = c2.getData().getWG();
        std::complex<double> g3 = c3.getData().getWG();
        ProjectHelper<C>::ProjectX(c1, c2, c3, d1, d2, d3, g1, g2, g3);

        std::complex<double> gam0 = g1 * g2 * g3;
        std::complex<double> gam1 = std::conj(g1) * g2 * g3;
        std::complex<double> gam2 = g1 * std::conj(g2) * g3;
        std::complex<double> gam3 = g1 * g2 * std::conj(g3);

        std::complex<double> z(cosphi, -sinphi);
        weight[index] += www;
        zeta.gam0r[index] += gam0.real();
        zeta.gam0i[index] += gam0.imag();
        zeta.gam1r[index] += gam1.real();
        zeta.gam1i[index] += gam1.imag();
        zeta.gam2r[index] += gam2.real();
        zeta.gam2i[index] += gam2.imag();
        zeta.gam3r[index] += gam3.real();
        zeta.gam3i[index] += gam3.imag();
        std::complex<double> wwwztothen = www;
        std::complex<double> gam0ztothen = gam0;
        std::complex<double> gam1ztothen = gam1;
        std::complex<double> gam2ztothen = gam2;
        std::complex<double> gam3ztothen = gam3;
        for (int n=1; n <= maxn; ++n) {
            wwwztothen *= z;
            gam0ztothen *= z;
            gam1ztothen *= z;
            gam2ztothen *= z;
            gam3ztothen *= z;
            weight[index + n] += wwwztothen.real();
            weight_im[index + n] += wwwztothen.imag();
            weight[index - n] += wwwztothen.real();
            weight_im[index - n] -= wwwztothen.imag();
            zeta.gam0r[index + n] += gam0ztothen.real();
            zeta.gam0i[index + n] += gam0ztothen.imag();
            zeta.gam1r[index + n] += gam1ztothen.real();
            zeta.gam1i[index + n] += gam1ztothen.imag();
            zeta.gam2r[index + n] += gam2ztothen.real();
            zeta.gam2i[index + n] += gam2ztothen.imag();
            zeta.gam3r[index + n] += gam3ztothen.real();
            zeta.gam3i[index + n] += gam3ztothen.imag();
        }
        // Unlike for N and K, the -n components are not complex conjugates.
        // So do a separate loop for them.
        gam0ztothen = gam0;  // These will now be gam_mu * conj(z)^n
        gam1ztothen = gam1;
        gam2ztothen = gam2;
        gam3ztothen = gam3;
        for (int n=1; n <= maxn; ++n) {
            gam0ztothen *= std::conj(z);
            gam1ztothen *= std::conj(z);
            gam2ztothen *= std::conj(z);
            gam3ztothen *= std::conj(z);
            zeta.gam0r[index - n] += gam0ztothen.real();
            zeta.gam0i[index - n] += gam0ztothen.imag();
            zeta.gam1r[index - n] += gam1ztothen.real();
            zeta.gam1i[index - n] += gam1ztothen.imag();
            zeta.gam2r[index - n] += gam2ztothen.real();
            zeta.gam2i[index - n] += gam2ztothen.imag();
            zeta.gam3r[index - n] += gam3ztothen.real();
            zeta.gam3i[index - n] += gam3ztothen.imag();
        }
    }

    template <int C>
    static void CalculateGn(
        const Cell<GData,C>& c1, const Cell<GData,C>& c2,
        double rsq, double r, int k, int maxn, double w,
        MultipoleScratch<GData, GData>& mp)
    {
        std::complex<double> wg = c2.getData().getWG();
        std::complex<double> z = ProjectHelper<C>::ExpIPhi(c1.getPos(), c2.getPos(), r);

        if (mp.ww) {
            std::complex<double> wgzc = wg * std::conj(z);
            std::complex<double> zcsq = std::conj(z*z);
            mp.sumwwgg1[k] += SQR(wgzc);
            mp.sumwwgg2[k] += std::norm(wg) * zcsq;
            mp.sumwwgg3[k] += SQR(wgzc * zcsq);
        }

        int iw = k*(maxn+1);
        mp.Wn[iw] += w;
        std::complex<double> wztothen = w;
        for (int n=1; n <= maxn; ++n) {
            wztothen *= z;
            mp.Wn[iw + n] += wztothen;
        }

        // Note: we'll need Gn values from -maxn-3 <= n <= maxn-1
        int ig = k*(2*maxn+3)+maxn+3;
        mp.Gn[ig] += wg;
        std::complex<double> wgztothen = wg;
        for (int n=1; n <= maxn-1; ++n) {
            wgztothen *= z;
            mp.Gn[ig + n] += wgztothen;
        }
        // Repeat for -n, since +/- n is not symmetric as g is complex.
        wgztothen = wg;   // Now this is really wg conj(z)^n
        for (int n=1; n <= maxn+3; ++n) {
            wgztothen *= std::conj(z);
            mp.Gn[ig - n] += wgztothen;
        }
    }

    template <int C>
    static void CalculateZeta(const Cell<GData,C>& c1,
                              MultipoleScratch<GData, GData>& mp,
                              double* weight, double* weight_im,
                              ZetaData<GData,GData,GData>& zeta, int nbins, int maxn)
    {
        const double w1 = c1.getW();
        const std::complex<double> wg1 = c1.getData().getWG();
        const int step23 = 2*maxn+1;
        const int step32 = nbins * step23;
        const int step22 = step32 + step23;
        int iz22 = maxn;
        for (int k2=0; k2<nbins; ++k2, iz22+=step22) {
            // Do the k2=k3 bins.
            // As for NNN, we subtract off the sum of w_k^2 when k2=k3.
            // The corresponding thing for the Gamma values is a bit trickier.
            // Gamma_0 includes sum_k wg_k^2 z^-3
            // Gamma_1 includes sum_k wg_k^2 z^-1
            // Gamma_2 includes sum_k |wg_k|^2 z^-2
            // Gamma_3 includes sum_k |wg_k|^2 z^-2
            // These terms are called sumwwgg3, sumwwgg1, and sumwwgg3, respectively,
            // numbered by the power of z they include.

            // These are the indices in the Wn, Gn arrays (respectively) for n=0.
            const int iw2 = k2*(maxn+1);
            const int ig2 = k2*(2*maxn+3)+maxn+3;

            for (int n=0; n<=maxn; ++n) {
                double www = w1 * (std::norm(mp.Wn[iw2+n]) - mp.sumww[k2]);
                weight[iz22+n] += www;
                if (n > 0) {
                    weight[iz22-n] += www;
                }
            }
            for (int n=-maxn; n<=maxn; ++n) {
                std::complex<double> gam0 =
                    wg1 * (mp.Gn[ig2+n-3] * mp.Gn[ig2-n-3] - mp.sumwwgg3[k2]);
                std::complex<double> gam1 =
                    std::conj(wg1) * (mp.Gn[ig2+n-1] * mp.Gn[ig2-n-1] - mp.sumwwgg1[k2]);
                std::complex<double> gam2 =
                    wg1 * (mp.Gn[ig2+n-3] * std::conj(mp.Gn[ig2+n-1]) - mp.sumwwgg2[k2]);
                std::complex<double> gam3 =
                    wg1 * (std::conj(mp.Gn[ig2-n-1]) * mp.Gn[ig2-n-3] - mp.sumwwgg2[k2]);
                zeta.gam0r[iz22+n] += gam0.real();
                zeta.gam0i[iz22+n] += gam0.imag();
                zeta.gam1r[iz22+n] += gam1.real();
                zeta.gam1i[iz22+n] += gam1.imag();
                zeta.gam2r[iz22+n] += gam2.real();
                zeta.gam2i[iz22+n] += gam2.imag();
                zeta.gam3r[iz22+n] += gam3.real();
                zeta.gam3i[iz22+n] += gam3.imag();
            }

            int iz23 = iz22 + step23;
            int iz32 = iz22 + step32;
            for (int k3=k2+1; k3<nbins; ++k3, iz23+=step23, iz32+=step32) {
                XAssert(iz23 == (k2 * nbins + k3) * (2*maxn+1) + maxn);
                XAssert(iz32 == (k3 * nbins + k2) * (2*maxn+1) + maxn);
                const int iw3 = k3*(maxn+1);
                const int ig3 = k3*(2*maxn+3) + maxn+3;
                for (int n=0; n<=maxn; ++n) {
                    std::complex<double> www = w1 * mp.Wn[iw2+n] * std::conj(mp.Wn[iw3+n]);
                    weight[iz23+n] += www.real();
                    weight_im[iz23+n] += www.imag();
                    weight[iz32+n] += www.real();
                    weight_im[iz32+n] -= www.imag();
                    if (n > 0) {
                        weight[iz23-n] += www.real();
                        weight_im[iz23-n] -= www.imag();
                        weight[iz32-n] += www.real();
                        weight_im[iz32-n] += www.imag();
                    }
                }

                // W_-n = conj(W_n), so we have only been storing n>=0.
                // But G_n does not have that property, so use the actual n<0 index.
                // cf. Porth et al eqns 21, 23, 24, 25
                // Note: they have an extra negative sign, which I'm pretty sure we just
                // incorporate into our expiphi factor.  It first shows up in their eqn 18.
                // We define our projection directions such that there is no additional minus
                // sign required to compute the Gammas.
                //
                // Note also that this is not symmetric w.r.t +/-n, so just loop
                // from -maxn to maxn for these.
                for (int n=-maxn; n<=maxn; ++n) {
                    // Notation note: What we call d2 and d3 are Theta1 and Theta2 in
                    // Porth et al.  So the ig2 factors, which refer to d2 and thus c3,
                    // correspond to G(theta_i, Theta_1) in eqns 21,23-25.  And ig3 factors
                    // correspond to G(theta_i, Theta_2).
                    std::complex<double> gam0 = wg1 * mp.Gn[ig2+n-3] * mp.Gn[ig3-n-3];
                    std::complex<double> gam1 = std::conj(wg1) * mp.Gn[ig2+n-1] * mp.Gn[ig3-n-1];
                    std::complex<double> gam2 = wg1 * mp.Gn[ig2+n-3] * std::conj(mp.Gn[ig3+n-1]);
                    std::complex<double> gam3 = wg1 * std::conj(mp.Gn[ig2-n-1]) * mp.Gn[ig3-n-3];
                    zeta.gam0r[iz23+n] += gam0.real();
                    zeta.gam0i[iz23+n] += gam0.imag();
                    zeta.gam1r[iz23+n] += gam1.real();
                    zeta.gam1i[iz23+n] += gam1.imag();
                    zeta.gam2r[iz23+n] += gam2.real();
                    zeta.gam2i[iz23+n] += gam2.imag();
                    zeta.gam3r[iz23+n] += gam3.real();
                    zeta.gam3i[iz23+n] += gam3.imag();

                    // This time there is nothing symmetrical to exploit between the
                    // 2,3 and 3,2 terms.  Just recaulculate Gamma values for k3, k2.
                    gam0 = wg1 * mp.Gn[ig3+n-3] * mp.Gn[ig2-n-3];
                    gam1 = std::conj(wg1) * mp.Gn[ig3+n-1] * mp.Gn[ig2-n-1];
                    gam2 = wg1 * mp.Gn[ig3+n-3] * std::conj(mp.Gn[ig2+n-1]);
                    gam3 = wg1 * std::conj(mp.Gn[ig3-n-1]) * mp.Gn[ig2-n-3];
                    zeta.gam0r[iz32+n] += gam0.real();
                    zeta.gam0i[iz32+n] += gam0.imag();
                    zeta.gam1r[iz32+n] += gam1.real();
                    zeta.gam1i[iz32+n] += gam1.imag();
                    zeta.gam2r[iz32+n] += gam2.real();
                    zeta.gam2i[iz32+n] += gam2.imag();
                    zeta.gam3r[iz32+n] += gam3.real();
                    zeta.gam3i[iz32+n] += gam3.imag();
                }
            }
        }
    }

    template <int C>
    static void CalculateZeta(const Cell<GData,C>& c1, int ordered,
                              MultipoleScratch<GData, GData>& mp2,
                              MultipoleScratch<GData, GData>& mp3,
                              double* weight, double* weight_im,
                              ZetaData<GData,GData,GData>& zeta, int nbins, int maxn)
    {
        const double w1 = c1.getW();
        const std::complex<double> wg1 = c1.getData().getWG();
        const int step = 2*maxn+1;
        int iz = maxn;
        // If ordered == 1, then also count contribution from swapping cats 2,3
        const bool swap23 = (ordered == 1);
        for (int k2=0; k2<nbins; ++k2) {
            const int iw2 = k2*(maxn+1);
            const int ig2 = k2*(2*maxn+3) + maxn+3;
            for (int k3=0; k3<nbins; ++k3, iz+=step) {
                const int iw3 = k3*(maxn+1);
                const int ig3 = k3*(2*maxn+3) + maxn+3;
                for (int n=0; n<=maxn; ++n) {
                    std::complex<double> www = w1 * mp3.Wn[iw2+n] * std::conj(mp2.Wn[iw3+n]);
                    if (swap23) {
                        www += w1 * mp2.Wn[iw2+n] * std::conj(mp3.Wn[iw3+n]);
                    }
                    weight[iz+n] += www.real();
                    weight_im[iz+n] += www.imag();
                    if (n > 0) {
                        weight[iz-n] += www.real();
                        weight_im[iz-n] -= www.imag();
                    }
                }
                for (int n=-maxn; n<=maxn; ++n) {
                    std::complex<double> gam0 = wg1 * mp3.Gn[ig2+n-3] * mp2.Gn[ig3-n-3];
                    std::complex<double> gam1 = std::conj(wg1) * mp3.Gn[ig2+n-1] * mp2.Gn[ig3-n-1];
                    std::complex<double> gam2 = wg1 * mp3.Gn[ig2+n-3] * std::conj(mp2.Gn[ig3+n-1]);
                    std::complex<double> gam3 = wg1 * std::conj(mp3.Gn[ig2-n-1]) * mp2.Gn[ig3-n-3];
                    if (swap23) {
                        gam0 += wg1 * mp2.Gn[ig2+n-3] * mp3.Gn[ig3-n-3];
                        gam1 += std::conj(wg1) * mp2.Gn[ig2+n-1] * mp3.Gn[ig3-n-1];
                        gam2 += wg1 * mp2.Gn[ig2+n-3] * std::conj(mp3.Gn[ig3+n-1]);
                        gam3 += wg1 * std::conj(mp2.Gn[ig2-n-1]) * mp3.Gn[ig3-n-3];
                    }
                    zeta.gam0r[iz+n] += gam0.real();
                    zeta.gam0i[iz+n] += gam0.imag();
                    zeta.gam1r[iz+n] += gam1.real();
                    zeta.gam1i[iz+n] += gam1.imag();
                    zeta.gam2r[iz+n] += gam2.real();
                    zeta.gam2i[iz+n] += gam2.imag();
                    zeta.gam3r[iz+n] += gam3.real();
                    zeta.gam3i[iz+n] += gam3.imag();
                }
            }
        }
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
    // Also, the index is just the index for n=0.  The finalize function in python
    // will copy this to the other 2*maxn locations for each row.

    double nnn = double(c1.getN()) * c2.getN() * c3.getN();
    _ntri[index] += nnn;
    xdbg<<ws()<<"MP Index = "<<index<<", nnn = "<<nnn<<" => "<<_ntri[index]<<std::endl;

    double www = c1.getW() * c2.getW() * c3.getW();
    _meand1[index] += www * d1;
    _meanlogd1[index] += www * logd1;
    _meand2[index] += www * d2;
    _meanlogd2[index] += www * logd2;
    _meand3[index] += www * d3;
    _meanlogd3[index] += www * logd3;

    DirectHelper<D1,D2,D3>::ProcessMultipole(
        static_cast<const Cell<D1,C>&>(c1),
        static_cast<const Cell<D2,C>&>(c2),
        static_cast<const Cell<D3,C>&>(c3), d1, d2, d3,
        sinphi, cosphi, www, _weight, _weight_im, _zeta, index, _nubins);
}

// We don't currently have any code with D2 != D3, but if we eventually do,
// this distinction might be relevant.
template <int D1, int D2, int D3>
std::unique_ptr<BaseMultipoleScratch> Corr3<D1,D2,D3>::getMP2(bool use_ww)
{ return make_unique<MultipoleScratch<D1,D2> >(_nbins, _nubins, use_ww); }

template <int D1, int D2, int D3>
std::unique_ptr<BaseMultipoleScratch> Corr3<D1,D2,D3>::getMP3(bool use_ww)
{ return make_unique<MultipoleScratch<D1,D3> >(_nbins, _nubins, use_ww); }

template <int D1, int D2, int D3> template <int C>
void Corr3<D1,D2,D3>::calculateGn(
    const BaseCell<C>& c1, const BaseCell<C>& c2,
    double rsq, double r, double logr, int k,
    BaseMultipoleScratch& mp)
{
    xdbg<<ws()<<"Gn Index = "<<k<<std::endl;
    // For now we only include the counts and weight from c2.
    // We'll include c1 as part of the triple in calculateZeta.
    double n = c2.getN();
    double w = c2.getW();
    mp.npairs[k] += n;
    mp.sumw[k] += w;
    mp.sumwr[k] += w * r;
    mp.sumwlogr[k] += w * logr;
    if (mp.ww) {
        double ww = w * w;
        mp.sumww[k] += ww;
        mp.sumwwr[k] += ww * r;
        mp.sumwwlogr[k] += ww * logr;
    }

    DirectHelper<D1,D2,D3>::CalculateGn(
        static_cast<const Cell<D1,C>&>(c1),
        static_cast<const Cell<D2,C>&>(c2),
        rsq, r, k, _nubins, w,
        static_cast<MultipoleScratch<D1, D2>&>(mp));
}

template <int D1, int D2, int D3> template <int C>
void Corr3<D1,D2,D3>::calculateZeta(const BaseCell<C>& c1, BaseMultipoleScratch& mp)
{
    xdbg<<ws()<<"Zeta c1 = "<<c1.getPos()<<"  "<<c1.getSize()<<"  "<<c1.getW()<<std::endl;
    // First finish the computation of meand2, etc. based on the 2pt accumulations.
    double n1 = c1.getN();
    double w1 = c1.getW();
    const int maxn = _nubins;
    const int nnbins = 2*maxn+1;
    for (int k2=0; k2<_nbins; ++k2) {
        int i22 = (k2 * _nbins + k2) * nnbins + maxn;
        // Do the k2=k3 bins.
        // We need to be careful not to count cases where c2=c3.
        // This means we need to subtract off sums of w^2 for instance.
        _ntri[i22] += n1 * mp.npairs[k2] * (mp.npairs[k2]-1);
        _meand2[i22] += w1 * (mp.sumw[k2] * mp.sumwr[k2] - mp.sumwwr[k2]);
        _meanlogd2[i22] += w1 * (mp.sumw[k2] * mp.sumwlogr[k2] - mp.sumwwlogr[k2]);
        _meand3[i22] += w1 * (mp.sumw[k2] * mp.sumwr[k2] - mp.sumwwr[k2]);
        _meanlogd3[i22] += w1 * (mp.sumw[k2] * mp.sumwlogr[k2] - mp.sumwwlogr[k2]);
        for (int k3=k2+1; k3<_nbins; ++k3) {
            int i23 = (k2 * _nbins + k3) * nnbins + maxn;
            int i32 = (k3 * _nbins + k2) * nnbins + maxn;
            double nnn = n1 * mp.npairs[k2] * mp.npairs[k3];
            _ntri[i23] += nnn;
            _ntri[i32] += nnn;
            double ww12 = w1 * mp.sumw[k2];
            double ww13 = w1 * mp.sumw[k3];
            double wwwd2 = ww13 * mp.sumwr[k2];
            _meand2[i23] += wwwd2;
            _meand3[i32] += wwwd2;
            double wwwlogd2 = ww13 * mp.sumwlogr[k2];
            _meanlogd2[i23] += wwwlogd2;
            _meanlogd3[i32] += wwwlogd2;
            double wwwd3 = ww12 * mp.sumwr[k3];
            _meand3[i23] += wwwd3;
            _meand2[i32] += wwwd3;
            double wwwlogd3 = ww12 * mp.sumwlogr[k3];
            _meanlogd3[i23] += wwwlogd3;
            _meanlogd2[i32] += wwwlogd3;
        }
    }
    // Finish the calculation for Zeta_n(d1,d2) using G_n(d).
    // In Porth et al, this is eqs. 21, 23, 24, 25 for GGG, and eqn 27 for NNN.
    // The version for KKK is obvious from these.
    DirectHelper<D1,D2,D3>::CalculateZeta(
        static_cast<const Cell<D1,C>&>(c1),
        static_cast<MultipoleScratch<D1, D2>&>(mp),
        _weight, _weight_im, _zeta, _nbins, _nubins);
}

template <int D1, int D2, int D3> template <int C>
void Corr3<D1,D2,D3>::calculateZeta(
    const BaseCell<C>& c1, int ordered,
    BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3)
{
    xdbg<<ws()<<"Zeta c1 = "<<c1.getPos()<<"  "<<c1.getSize()<<"  "<<c1.getW()<<std::endl;
    // First finish the computation of meand2, etc. based on the 2pt accumulations.
    double n1 = c1.getN();
    double w1 = c1.getW();
    const int maxn = _nubins;
    const int nnbins = 2*maxn+1;
    int i=maxn;
    if (ordered == 3) {
        // Keep track of locations p2 and p3 separately.
        for (int k2=0; k2<_nbins; ++k2) {
            for (int k3=0; k3<_nbins; ++k3, i+=nnbins) {
                // This is a bit confusing.  In sumw2 (and similar paramters), the "2"
                // refers to these being computed for cat2, which is at p2, opposite d2.
                // So the distances that have been accumulated in them are actually d3 values.
                // k3 is our index for d3 bins, so we use mp2.sumw[k3] and mp3.sumw[k2], etc.
                _ntri[i] += n1 * mp3.npairs[k2] * mp2.npairs[k3];
                double ww12 = w1 * mp3.sumw[k2];
                double ww13 = w1 * mp2.sumw[k3];
                _meand2[i] += ww13 * mp3.sumwr[k2];
                _meanlogd2[i] += ww13 * mp3.sumwlogr[k2];
                _meand3[i] += ww12 * mp2.sumwr[k3];
                _meanlogd3[i] += ww12 * mp2.sumwlogr[k3];
            }
        }
    } else {
        XAssert(ordered == 1);
        // ordered == 1 means points 2 and 3 can swap freely.
        // So add up the results where k2 and k3 take both spots.
        for (int k2=0; k2<_nbins; ++k2) {
            for (int k3=0; k3<_nbins; ++k3, i+=nnbins) {
                _ntri[i] += n1 * (mp2.npairs[k2] * mp3.npairs[k3] +
                                  mp3.npairs[k2] * mp2.npairs[k3]);
                _meand2[i] += w1 * (mp2.sumw[k3] * mp3.sumwr[k2] +
                                    mp3.sumw[k3] * mp2.sumwr[k2]);
                _meanlogd2[i] += w1 * (mp2.sumw[k3] * mp3.sumwlogr[k2] +
                                       mp3.sumw[k3] * mp2.sumwlogr[k2]);
                _meand3[i] += w1 * (mp2.sumw[k2] * mp3.sumwr[k3] +
                                    mp3.sumw[k2] * mp2.sumwr[k3]);
                _meanlogd3[i] += w1 * (mp2.sumw[k2] * mp3.sumwlogr[k3] +
                                       mp3.sumw[k3] * mp2.sumwlogr[k3]);
            }
        }
    }
    // Finish the calculation for Zeta_n(d1,d2) using G_n(d).
    DirectHelper<D1,D2,D3>::CalculateZeta(
        static_cast<const Cell<D1,C>&>(c1), ordered,
        static_cast<MultipoleScratch<D1,D2>&>(mp2),
        static_cast<MultipoleScratch<D1,D3>&>(mp3),
        _weight, _weight_im, _zeta, _nbins, _nubins);
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
    if (_is_multipole) {
        for (int i=0; i<_ntot; ++i) _weight_im[i] = rhs._weight_im[i];
    }
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
    if (_is_multipole) {
        for (int i=0; i<_ntot; ++i) _weight_im[i] += rhs._weight_im[i];
    }
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
    dbg<<"Start multipole auto\n";
    reset_ws();
    Assert(_coords == -1 || _coords == C);
    _coords = C;
    const long n1 = field.getNTopLevel();
    dbg<<"field has "<<n1<<" top level nodes\n";
    Assert(n1 > 0);

    MetricHelper<M,0> metric(0, 0, _xp, _yp, _zp);
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

        std::unique_ptr<BaseMultipoleScratch> mp = getMP2(true);

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
            const BaseCell<C>& c1 = *cells[i];
            corr.template multipoleSplit1<B>(c1, cells, metric, *mp);
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
void BaseCorr3::multipole(const BaseField<C>& field1, const BaseField<C>& field2, bool dots)
{
    dbg<<"Start multipole cross12\n";
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

        std::unique_ptr<BaseMultipoleScratch> mp = getMP2(true);

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
            corr.template multipoleSplit1<B>(c1, c2list, metric, *mp);
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
void BaseCorr3::multipole(const BaseField<C>& field1, const BaseField<C>& field2,
                          const BaseField<C>& field3, bool dots, int ordered)
{
    dbg<<"Start multipole cross full\n";
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

        // Note: We don't need to subtract off w^2 for the k2=k3 bins, so don't accumulate
        // sumww and related arrays.  (That's what false here means.)
        std::unique_ptr<BaseMultipoleScratch> mp2 = getMP2(false);
        std::unique_ptr<BaseMultipoleScratch> mp3 = getMP3(false);

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
            corr.template multipoleSplit1<B>(c1, c2list, c3list, metric, ordered, *mp2, *mp3);
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
void BaseCorr3::splitC2Cells(
    const BaseCell<C>& c1, const std::vector<const BaseCell<C>*>& c2list,
    const MetricHelper<M,0>& metric, std::vector<const BaseCell<C>*>& newc2list)
{
    const Position<C>& p1 = c1.getPos();
    double s1 = c1.getSize();
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
            XAssert(c2->getLeft());
            Assert(c2->getRight());
            newc2list.push_back(c2->getLeft());
            newc2list.push_back(c2->getRight());
        } else {
            newc2list.push_back(c2);
        }
    }
}

template <int B, int M, int C>
void BaseCorr3::multipoleSplit1(
    const BaseCell<C>& c1, const std::vector<const BaseCell<C>*>& c2list,
    const MetricHelper<M,0>& metric, BaseMultipoleScratch& mp)
{
    xdbg<<ws()<<"MultipoleSplit1: c1 = "<<c1.getPos()<<"  "<<c1.getSize()<<"  "<<c1.getW()<<"  len c2 = "<<c2list.size()<<std::endl;

    double s1 = c1.getSize();
    xdbg<<"B,M,C = "<<B<<"  "<<M<<"  "<<C<<std::endl;

    // Split cells in c2list if they will definitely need to be split.
    // And remove any that cannot contribute to the sums for this c1.
    std::vector<const BaseCell<C>*> newc2list;
    splitC2Cells<B>(c1, c2list, metric, newc2list);

    // See if we can stop splitting c1
    // The criterion is that for the largest separation we will be caring about,
    // c1 is at least possibly small enough to use as is without futher splitting.
    // i.e. s1 < maxsep * b
    inc_ws();
    double maxbsq_eff = BinTypeHelper<B>::getEffectiveBSq(_maxsepsq, _bsq);
    if (SQR(s1) > maxbsq_eff) {
        multipoleSplit1<B>(*c1.getLeft(), newc2list, metric, mp);
        multipoleSplit1<B>(*c1.getRight(), newc2list, metric, mp);
    } else {
        // Zero out scratch arrays
        mp.clear();
        multipoleFinish<B>(c1, newc2list, metric, mp);
    }
    dec_ws();
}
template <int B, int M, int C>
void BaseCorr3::multipoleSplit1(
    const BaseCell<C>& c1,
    const std::vector<const BaseCell<C>*>& c2list,
    const std::vector<const BaseCell<C>*>& c3list,
    const MetricHelper<M,0>& metric, int ordered,
    BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3)
{
    xdbg<<ws()<<"MultipoleSplit1: c1 = "<<c1.getPos()<<"  "<<c1.getSize()<<"  "<<c1.getW()<<"  len c2 = "<<c2list.size()<<" len c3 = "<<c3list.size()<<std::endl;
    xdbg<<"B,M,C = "<<B<<"  "<<M<<"  "<<C<<std::endl;

    double s1 = c1.getSize();

    // Split cells in both lists if appropriate
    std::vector<const BaseCell<C>*> newc2list;
    std::vector<const BaseCell<C>*> newc3list;
    splitC2Cells<B>(c1, c2list, metric, newc2list);
    xdbg<<"c2 split "<<c2list.size()<<" => "<<newc2list.size()<<std::endl;
    splitC2Cells<B>(c1, c3list, metric, newc3list);
    xdbg<<"c3 split "<<c3list.size()<<" => "<<newc3list.size()<<std::endl;

    // See if we can stop splitting c1
    inc_ws();
    double maxbsq_eff = BinTypeHelper<B>::getEffectiveBSq(_maxsepsq, _bsq);
    if (SQR(s1) > maxbsq_eff) {
        multipoleSplit1<B>(*c1.getLeft(), newc2list, newc3list, metric, ordered, mp2, mp3);
        multipoleSplit1<B>(*c1.getRight(), newc2list, newc3list, metric, ordered, mp2, mp3);
    } else {
        // Zero out scratch arrays
        mp2.clear();
        mp3.clear();
        multipoleFinish<B>(c1, newc2list, newc3list, metric, ordered, mp2, mp3);
    }
    dec_ws();
}

template <int B, int M, int C>
void BaseCorr3::splitC2CellsOrCalculateGn(
    const BaseCell<C>& c1, const std::vector<const BaseCell<C>*>& c2list,
    const MetricHelper<M,0>& metric, std::vector<const BaseCell<C>*>& newc2list, bool& anysplit1,
    BaseMultipoleScratch& mp)
{
    const Position<C>& p1 = c1.getPos();
    double s1 = c1.getSize();
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
                    calculateGn(c1, *c2, rsq, r, logr, k, mp);
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
            XAssert(c2->getLeft());
            XAssert(c2->getRight());
            newc2list.push_back(c2->getLeft());
            newc2list.push_back(c2->getRight());
        } else {
            newc2list.push_back(c2);
        }
    }
}

template <int B, int M, int C>
void BaseCorr3::multipoleFinish(
    const BaseCell<C>& c1, const std::vector<const BaseCell<C>*>& c2list,
    const MetricHelper<M,0>& metric, BaseMultipoleScratch& mp)
{
    // This is structured a lot like the previous function.
    // However, in this one, we will actually be filling the Gn array.
    // We fill what we can before splitting further.
    // Note: if we decide to split c1, we need to make a copy of Gn before
    // recursing.  This is why we split up the calculation into two functions like
    // this.  We want to minimize the number of copies of Gn (et al) we need to make.
    xdbg<<ws()<<"MultipoleFinish1: c1 = "<<c1.getPos()<<"  "<<c1.getSize()<<"  "<<c1.getW()<<"  len c2 = "<<c2list.size()<<std::endl;

    xdbg<<"B,M,C = "<<B<<"  "<<M<<"  "<<C<<std::endl;
    bool anysplit1=false;

    // As before, split any cells in c2list that need to be split given the current c1.
    // And remove any that cannot contribute to the sums.
    // This time we also accumulate the Gn array for any that can be done.
    // Those also don't get added to newc2list.
    std::vector<const BaseCell<C>*> newc2list;
    splitC2CellsOrCalculateGn<B>(
        c1, c2list, metric, newc2list, anysplit1, mp);

    xdbg<<"newsize = "<<newc2list.size()<<", anysplit1 = "<<anysplit1<<std::endl;
    if (newc2list.size() > 0) {
        inc_ws();
        if (anysplit1) {
            XAssert(c1.getLeft());
            XAssert(c1.getRight());
            // Then we need to split c1 further.  This means we need a copy of Gn and the
            // other scratch arrays, so we can pass what we have now to each child cell.
            std::unique_ptr<BaseMultipoleScratch> mp_copy = mp.duplicate();
            multipoleFinish<B>(*c1.getLeft(), newc2list, metric, mp);
            multipoleFinish<B>(*c1.getRight(), newc2list, metric, *mp_copy);
        } else {
            // If we still have c2 items to process, but don't have to split c1,
            // we don't need to make copies.
            multipoleFinish<B>(c1, newc2list, metric, mp);
        }
        dec_ws();
    } else {
        // We finished all the calculations for Gn.
        // Turn this into Zeta_n.
        calculateZeta(c1, mp);
    }
}

template <int B, int M, int C>
void BaseCorr3::multipoleFinish(
    const BaseCell<C>& c1, const std::vector<const BaseCell<C>*>& c2list,
    const std::vector<const BaseCell<C>*>& c3list, const MetricHelper<M,0>& metric, int ordered,
    BaseMultipoleScratch& mp2, BaseMultipoleScratch& mp3)
{
    xdbg<<ws()<<"MultipoleFinish1: c1 = "<<c1.getPos()<<"  "<<c1.getSize()<<"  "<<c1.getW()<<"  len c2 = "<<c2list.size()<<"  len c3 = "<<c3list.size()<<std::endl;

    xdbg<<"B,M,C = "<<B<<"  "<<M<<"  "<<C<<std::endl;
    bool anysplit1=false;

    std::vector<const BaseCell<C>*> newc2list;
    splitC2CellsOrCalculateGn<B>(c1, c2list, metric, newc2list, anysplit1, mp2);
    std::vector<const BaseCell<C>*> newc3list;
    splitC2CellsOrCalculateGn<B>(c1, c3list, metric, newc3list, anysplit1, mp3);
    xdbg<<"newsize = "<<newc2list.size()<<","<<newc3list.size()<<", anysplit1 = "<<anysplit1<<std::endl;

    if (newc2list.size() > 0 || newc3list.size() > 0) {
        inc_ws();
        if (anysplit1) {
            XAssert(c1.getLeft());
            XAssert(c1.getRight());
            // Then we need to split c1 further.  This means we need a copy of Gn and the
            // other scratch arrays, so we can pass what we have now to each child cell.
            std::unique_ptr<BaseMultipoleScratch> mp2_copy = mp2.duplicate();
            std::unique_ptr<BaseMultipoleScratch> mp3_copy = mp3.duplicate();
            multipoleFinish<B>(
                *c1.getLeft(), newc2list, newc3list, metric, ordered, mp2, mp3);
            multipoleFinish<B>(
                *c1.getRight(), newc2list, newc3list, metric, ordered, *mp2_copy, *mp3_copy);
        } else {
            // If we still have c2 items to process, but don't have to split c1,
            // we don't need to make copies.
            multipoleFinish<B>(c1, newc2list, newc3list, metric, ordered, mp2, mp3);
        }
        dec_ws();
    } else {
        // We finished all the calculations for Gn.
        // Turn this into Zeta_n.
        calculateZeta(c1, ordered, mp2, mp3);
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
    py::array_t<double>& weightp, py::array_t<double>& weight_imp,
    py::array_t<double>& ntrip)
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
    double* weight_im = static_cast<double*>(weight_imp.mutable_data());
    double* ntri = static_cast<double*>(ntrip.mutable_data());

    dbg<<"Start BuildCorr3 "<<D1<<" "<<D2<<" "<<D3<<" "<<bin_type<<std::endl;
    Assert(D2 == D1);
    Assert(D3 == D1);

    return new Corr3<D1,D2,D3>(
            bin_type, minsep, maxsep, nbins, binsize, b,
            minu, maxu, nubins, ubinsize, bu, minv, maxv, nvbins, vbinsize, bv, xp, yp, zp,
            zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7,
            meand1, meanlogd1, meand2, meanlogd2, meand3, meanlogd3,
            meanu, meanv, weight, weight_im, ntri);
}

// A quick helper struct so we only instantiate multipole for Multipole BinTypes.
// _B is actually the corresponding 2pt BinType since the algorithm will use 2pt tests.
// For now, we only have one such 3pt type, LogMultipole, so _B is always Log.
template <int B>
struct ValidMPB
{ enum { _B = Log }; };

template <int B, int M, int C>
void ProcessAutob(BaseCorr3& corr, BaseField<C>& field, bool dots)
{
    Assert((ValidMC<M,C>::_M == M));
#ifndef DIRECT_MULTIPOLE
    if (B == LogMultipole) {
        corr.template multipole<ValidMPB<B>::_B,ValidMC<M,C>::_M>(field, dots);
    } else {
#endif
        corr.template process<B,ValidMC<M,C>::_M>(field, dots);
#ifndef DIRECT_MULTIPOLE
    }
#endif
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
#ifndef DIRECT_MULTIPOLE
    if (B == LogMultipole) {
        switch(ordered) {
          case 0:
               corr.template multipole<ValidMPB<B>::_B,ValidMC<M,C>::_M>(
                   field2, field1, field2, dots, 1);
               // Drop through.
          case 1:
               corr.template multipole<ValidMPB<B>::_B,ValidMC<M,C>::_M>(
                   field1, field2, dots);
               break;
          default:
               Assert(false);
        }
    } else {
#endif
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
#ifndef DIRECT_MULTIPOLE
    }
#endif
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
#ifndef DIRECT_MULTIPOLE
    if (B == LogMultipole) {
        switch(ordered) {
          case 0:
               corr.template multipole<ValidMPB<B>::_B,ValidMC<M,C>::_M>(
                   field2, field1, field3, dots, 1);
               corr.template multipole<ValidMPB<B>::_B,ValidMC<M,C>::_M>(
                   field3, field1, field2, dots, 1);
               // Drop through.
          case 1:
               corr.template multipole<ValidMPB<B>::_B,ValidMC<M,C>::_M>(
                   field1, field2, field3, dots, 1);
               break;
          case 3:
               corr.template multipole<ValidMPB<B>::_B,ValidMC<M,C>::_M>(
                   field1, field2, field3, dots, 3);
               break;
          default:
               Assert(false);
        }
    } else {
#endif
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
#ifndef DIRECT_MULTIPOLE
    }
#endif
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
        py::array_t<double>& weightp, py::array_t<double>& weight_imp,
        py::array_t<double>& ntrip);

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
