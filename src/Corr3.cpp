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

#include "PyBind11Helper.h"

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
    _zeta(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7),
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
    _zeta(0,0,0,0,0,0,0,0), _weight(0)
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
        std::shared_ptr<BaseCorr3> bc3p = duplicate();
        BaseCorr3& bc3 = *bc3p;
#else
        BaseCorr3& bc3 = *this;
#endif

        MetricHelper<M,0> metric(0, 0, _xp, _yp, _zp);

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
            bc3.template process3<B>(c1, metric);
            for (long j=i+1;j<n1;++j) {
                const BaseCell<C>& c2 = *field.getCells()[j];
                bc3.template process12<B>(bc3, bc3, c1, c2, metric);
                bc3.template process12<B>(bc3, bc3, c2, c1, metric);
                if (!BinTypeHelper<B>::sort_d123) {
                    bc3.process111<B>(bc3, bc3, bc3, bc3, bc3,
                                      c1, c1, c2, metric);
                    bc3.process111<B>(bc3, bc3, bc3, bc3, bc3,
                                      c2, c1, c2, metric);
                }
                for (long k=j+1;k<n1;++k) {
                    const BaseCell<C>& c3 = *field.getCells()[k];
                    bc3.template process111<B>(bc3, bc3, bc3, bc3, bc3, c1, c2, c3, metric);
                    if (!BinTypeHelper<B>::sort_d123) {
                        // If not sorting later, then here we need to include all triples
                        bc3.template process111<B>(bc3, bc3, bc3, bc3, bc3, c2, c1, c3, metric);
                        bc3.template process111<B>(bc3, bc3, bc3, bc3, bc3, c3, c1, c2, metric);
                    }
                }
            }
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            addData(bc3);
        }
    }
#endif
    if (dots) std::cout<<std::endl;
}

template <int B, int M, int C>
void BaseCorr3::process(BaseCorr3& corr212, BaseCorr3& corr221,
                        const BaseField<C>& field1, const BaseField<C>& field2, bool dots)
{
    reset_ws();
    xdbg<<"_coords = "<<_coords<<std::endl;
    xdbg<<"C = "<<C<<std::endl;
    Assert(_coords == -1 || _coords == C);
    _coords = C;
    const long n1 = field1.getNTopLevel();
    const long n2 = field2.getNTopLevel();
    xdbg<<"field1 has "<<n1<<" top level nodes\n";
    xdbg<<"field2 has "<<n2<<" top level nodes\n";
    Assert(n1 > 0);
    Assert(n2 > 0);

    MetricHelper<M,0> metric(0, 0, _xp, _yp, _zp);

#ifdef DEBUGLOGGING
    if (verbose_level >= 2) {
        xdbg<<"field1: \n";
        for (long i=0;i<n1;++i) {
            xdbg<<"node "<<i<<std::endl;
            const BaseCell<C>& c1 = *field1.getCells()[i];
            c1.WriteTree(get_dbgout());
        }
        xdbg<<"field2: \n";
        for (long i=0;i<n2;++i) {
            xdbg<<"node "<<i<<std::endl;
            const BaseCell<C>& c2 = *field2.getCells()[i];
            c2.WriteTree(get_dbgout());
        }
    }
#endif

#ifdef _OPENMP
#pragma omp parallel
    {
        // Give each thread their own copy of the data vector to fill in.
        std::shared_ptr<BaseCorr3> bc122p = duplicate();
        std::shared_ptr<BaseCorr3> bc212p = corr212.duplicate();
        std::shared_ptr<BaseCorr3> bc221p = corr221.duplicate();
        BaseCorr3& bc122 = *bc122p;
        BaseCorr3& bc212 = *bc212p;
        BaseCorr3& bc221 = *bc221p;
#else
        BaseCorr3& bc122 = *this;
        BaseCorr3& bc212 = corr212;
        BaseCorr3& bc221 = corr221;
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
            const BaseCell<C>& c1 = *field1.getCells()[i];
            for (long j=0;j<n2;++j) {
                const BaseCell<C>& c2 = *field2.getCells()[j];
                bc122.template process12<B>(bc212, bc221, c1, c2, metric);
                for (long k=j+1;k<n2;++k) {
                    const BaseCell<C>& c3 = *field2.getCells()[k];
                    bc122.template process111<B>(bc122, bc212, bc221, bc212, bc221,
                                                 c1, c2, c3, metric);
                }
            }
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            addData(bc122);
            corr212.addData(bc212);
            corr221.addData(bc221);
        }
    }
#endif
    if (dots) std::cout<<std::endl;
}

template <int B, int M, int C>
void BaseCorr3::process(BaseCorr3& corr132,
                        BaseCorr3& corr213, BaseCorr3& corr231,
                        BaseCorr3& corr312, BaseCorr3& corr321,
                        const BaseField<C>& field1, const BaseField<C>& field2,
                        const BaseField<C>& field3, bool dots)
{
    reset_ws();
    xdbg<<"_coords = "<<_coords<<std::endl;
    xdbg<<"C = "<<C<<std::endl;
    Assert(_coords == -1 || _coords == C);
    _coords = C;
    const long n1 = field1.getNTopLevel();
    const long n2 = field2.getNTopLevel();
    const long n3 = field3.getNTopLevel();
    xdbg<<"field1 has "<<n1<<" top level nodes\n";
    xdbg<<"field2 has "<<n2<<" top level nodes\n";
    xdbg<<"field3 has "<<n3<<" top level nodes\n";
    Assert(n1 > 0);
    Assert(n2 > 0);
    Assert(n3 > 0);

    MetricHelper<M,0> metric(0, 0, _xp, _yp, _zp);

#ifdef DEBUGLOGGING
    if (verbose_level >= 2) {
        xdbg<<"field1: \n";
        for (long i=0;i<n1;++i) {
            xdbg<<"node "<<i<<std::endl;
            const BaseCell<C>& c1 = *field1.getCells()[i];
            c1.WriteTree(get_dbgout());
        }
        xdbg<<"field2: \n";
        for (long i=0;i<n2;++i) {
            xdbg<<"node "<<i<<std::endl;
            const BaseCell<C>& c2 = *field2.getCells()[i];
            c2.WriteTree(get_dbgout());
        }
        xdbg<<"field3: \n";
        for (long i=0;i<n3;++i) {
            xdbg<<"node "<<i<<std::endl;
            const BaseCell<C>& c3 = *field3.getCells()[i];
            c3.WriteTree(get_dbgout());
        }
    }
#endif

#ifdef _OPENMP
#pragma omp parallel
    {
        // Give each thread their own copy of the data vector to fill in.
        std::shared_ptr<BaseCorr3> bc123p = duplicate();
        std::shared_ptr<BaseCorr3> bc132p = corr132.duplicate();
        std::shared_ptr<BaseCorr3> bc213p = corr213.duplicate();
        std::shared_ptr<BaseCorr3> bc231p = corr231.duplicate();
        std::shared_ptr<BaseCorr3> bc312p = corr312.duplicate();
        std::shared_ptr<BaseCorr3> bc321p = corr321.duplicate();
        BaseCorr3& bc123 = *bc123p;
        BaseCorr3& bc132 = *bc132p;
        BaseCorr3& bc213 = *bc213p;
        BaseCorr3& bc231 = *bc231p;
        BaseCorr3& bc312 = *bc312p;
        BaseCorr3& bc321 = *bc321p;
#else
        BaseCorr3& bc123 = *this;
        BaseCorr3& bc132 = corr132;
        BaseCorr3& bc213 = corr213;
        BaseCorr3& bc231 = corr231;
        BaseCorr3& bc312 = corr312;
        BaseCorr3& bc321 = corr321;
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
            const BaseCell<C>& c1 = *field1.getCells()[i];
            for (long j=0;j<n2;++j) {
                const BaseCell<C>& c2 = *field2.getCells()[j];
                for (long k=0;k<n3;++k) {
                    const BaseCell<C>& c3 = *field3.getCells()[k];
                    bc123.template process111<B>(
                        bc132, bc213, bc231, bc312, bc321,
                        c1, c2, c3, metric);
                }
            }
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            addData(bc123);
            corr132.addData(bc132);
            corr213.addData(bc213);
            corr231.addData(bc231);
            corr312.addData(bc312);
            corr321.addData(bc321);
        }
    }
#endif
    if (dots) std::cout<<std::endl;
}

template <int B, int M, int C>
void BaseCorr3::process3(const BaseCell<C>& c1, const MetricHelper<M,0>& metric)
{
    // Does all triangles with 3 points in c1
    xdbg<<"Process3: c1 = "<<c1.getData().getPos()<<"  "<<"  "<<c1.getSize()<<"  "<<c1.getData().getN()<<std::endl;
    dbg<<ws()<<"Process3: c1 = "<<indices(c1)<<"\n";

    if (c1.getW() == 0) {
        dbg<<ws()<<"    w == 0.  return\n";
        return;
    }
    if (c1.getSize() < _halfminsep) {
        dbg<<ws()<<"    size < halfminsep.  return\n";
        return;
    }

    inc_ws();
    Assert(c1.getLeft());
    Assert(c1.getRight());
    process3<B>(*c1.getLeft(), metric);
    process3<B>(*c1.getRight(), metric);
    process12<B>(*this, *this, *c1.getLeft(), *c1.getRight(), metric);
    process12<B>(*this, *this, *c1.getRight(), *c1.getLeft(), metric);
    if (!BinTypeHelper<B>::sort_d123) {
        process111<B>(*this, *this, *this, *this, *this,
                      *c1.getLeft(), *c1.getLeft(), *c1.getRight(), metric);
        process111<B>(*this, *this, *this, *this, *this,
                      *c1.getRight(), *c1.getLeft(), *c1.getRight(), metric);
    }
    dec_ws();
}

template <int B, int M, int C>
void BaseCorr3::process12(BaseCorr3& bc212, BaseCorr3& bc221,
                          const BaseCell<C>& c1, const BaseCell<C>& c2,
                          const MetricHelper<M,0>& metric)
{
    // Does all triangles with one point in c1 and the other two points in c2
    xdbg<<"Process12: c1 = "<<c1.getData().getPos()<<"  "<<"  "<<c1.getSize()<<"  "<<c1.getData().getN()<<std::endl;
    xdbg<<"           c2  = "<<c2.getData().getPos()<<"  "<<"  "<<c2.getSize()<<"  "<<c2.getData().getN()<<std::endl;
    dbg<<ws()<<"Process12: c1 = "<<indices(c1)<<"  c2 = "<<indices(c2)<<"\n";

    // Some trivial stoppers:
    if (c1.getW() == 0) {
        dbg<<ws()<<"    w1 == 0.  return\n";
        return;
    }
    if (c2.getW() == 0) {
        dbg<<ws()<<"    w2 == 0.  return\n";
        return;
    }
    double s2 = c2.getSize();
    if (BinTypeHelper<B>::tooSmallS2(s2, _halfminsep, _minu, _minv))
    {
        dbg<<ws()<<"    s2 smaller than minimum triangle side.  return\n";
        return;
    }

    double s1 = c1.getSize();
    double rsq = metric.DistSq(c1.getData().getPos(), c2.getData().getPos(), s1, s2);
    double s1ps2 = s1 + s2;

    // If all possible triangles will have d2 < minsep, then abort the recursion here.
    // i.e. if d + s1 + s2 < minsep
    if (BinTypeHelper<B>::tooSmallDist(rsq, s1ps2, _minsep, _minsepsq)) {
        dbg<<ws()<<"    d2 cannot be as large as minsep\n";
        return;
    }

    // Similarly, we can abort if all possible triangles will have d > maxsep.
    // i.e. if  d - s1 - s2 >= maxsep
    if (BinTypeHelper<B>::tooLargeDist(rsq, s1ps2, _maxsep, _maxsepsq)) {
        dbg<<ws()<<"    d cannot be as small as maxsep\n";
        return;
    }

    // Depending on the binning, we may be able to stop due to allowed angles.
    if (BinTypeHelper<B>::noAllowedAngles(rsq, s1ps2, s1, s2,
                                         _minu, _minusq, _maxu, _maxusq,
                                         _minv, _minvsq, _maxv, _maxvsq)) {
        dbg<<ws()<<"    No possible triangles with allowed angles\n";
        return;
    }

    inc_ws();
    Assert(c2.getLeft());
    Assert(c2.getRight());
    process12<B>(bc212, bc221, c1, *c2.getLeft(), metric);
    process12<B>(bc212, bc221, c1, *c2.getRight(), metric);
    // 111 order is 123, 132, 213, 231, 312, 321   Here 3->2.
    BaseCorr3& bc122 = *this;  // alias for clarity.
    bc122.process111<B>(bc122, bc212, bc221, bc212, bc221,
                        c1, *c2.getLeft(), *c2.getRight(), metric);
    dec_ws();
}

template <int B, int M, int C>
void BaseCorr3::process111(
    BaseCorr3& bc132, BaseCorr3& bc213, BaseCorr3& bc231, BaseCorr3& bc312, BaseCorr3& bc321,
    const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
    const MetricHelper<M,0>& metric, double d1sq, double d2sq, double d3sq)
{
    dbg<<ws()<<"Process111: c1 = "<<indices(c1)<<"  c2 = "<<indices(c2)<<"  c3 = "<<indices(c3)<<"\n";
    // Does all triangles with 1 point each in c1, c2, c3
    if (c1.getW() == 0) {
        dbg<<ws()<<"    w1 == 0.  return\n";
        return;
    }
    if (c2.getW() == 0) {
        dbg<<ws()<<"    w2 == 0.  return\n";
        return;
    }
    if (c3.getW() == 0) {
        dbg<<ws()<<"    w3 == 0.  return\n";
        return;
    }

    // Calculate the distances if they aren't known yet
    double s=0.;
    if (d1sq == 0.)
        d1sq = metric.DistSq(c2.getData().getPos(), c3.getData().getPos(), s, s);
    if (d2sq == 0.)
        d2sq = metric.DistSq(c1.getData().getPos(), c3.getData().getPos(), s, s);
    if (d3sq == 0.)
        d3sq = metric.DistSq(c1.getData().getPos(), c2.getData().getPos(), s, s);

    BaseCorr3& bc123 = *this;  // alias for clarity.

    inc_ws();
    if (BinTypeHelper<B>::sort_d123) {
        xdbg<<"Before sort: d123 = "<<sqrt(d1sq)<<"  "<<sqrt(d2sq)<<"  "<<sqrt(d3sq)<<std::endl;

        // Need to end up with d1 > d2 > d3
        if (d1sq > d2sq) {
            if (d2sq > d3sq) {
                xdbg<<"123\n";
                // 123 -> 123
                bc123.template process111Sorted<B>(bc132, bc213, bc231, bc312, bc321,
                                                   c1, c2, c3, metric, d1sq, d2sq, d3sq);
            } else if (d1sq > d3sq) {
                xdbg<<"132\n";
                // 132 -> 123
                bc132.template process111Sorted<B>(bc123, bc312, bc321, bc213, bc231,
                                                   c1, c3, c2, metric, d1sq, d3sq, d2sq);
            } else {
                xdbg<<"312\n";
                // 312 -> 123
                bc312.template process111Sorted<B>(bc321, bc132, bc123, bc231, bc213,
                                                   c3, c1, c2, metric, d3sq, d1sq, d2sq);
            }
        } else {
            if (d1sq > d3sq) {
                xdbg<<"213\n";
                // 213 -> 123
                bc213.template process111Sorted<B>(bc231, bc123, bc132, bc321, bc312,
                                                   c2, c1, c3, metric, d2sq, d1sq, d3sq);
            } else if (d2sq > d3sq) {
                xdbg<<"231\n";
                // 231 -> 123
                bc231.template process111Sorted<B>(bc213, bc321, bc312, bc123, bc132,
                                                   c2, c3, c1, metric, d2sq, d3sq, d1sq);
            } else {
                xdbg<<"321\n";
                // 321 -> 123
                bc321.template process111Sorted<B>(bc312, bc231, bc213, bc132, bc123,
                                                   c3, c2, c1, metric, d3sq, d2sq, d1sq);
            }
        }
    } else {
        // For the non-sorting BinTypes (i.e. LogSAS so far), we just need to make sure
        // 1-3-2 is counter-clockwise
        if (!metric.CCW(c1.getData().getPos(), c3.getData().getPos(),
                        c2.getData().getPos())) {
            // Swap 2,3
            bc132.template process111Sorted<B>(bc123, bc312, bc321, bc213, bc231,
                                               c1, c3, c2, metric, d1sq, d3sq, d2sq);
        } else {
            bc123.template process111Sorted<B>(bc132, bc213, bc231, bc312, bc321,
                                               c1, c2, c3, metric, d1sq, d2sq, d3sq);
        }
    }
    dec_ws();
}

template <int B, int M, int C>
void BaseCorr3::process111Sorted(
    BaseCorr3& bc132, BaseCorr3& bc213, BaseCorr3& bc231, BaseCorr3& bc312, BaseCorr3& bc321,
    const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
    const MetricHelper<M,0>& metric, double d1sq, double d2sq, double d3sq)
{
    const double s1 = c1.getSize();
    const double s2 = c2.getSize();
    const double s3 = c3.getSize();

    xdbg<<"Process111Sorted: c1 = "<<c1.getData().getPos()<<"  "<<"  "<<c1.getSize()<<"  "<<c1.getData().getN()<<std::endl;
    xdbg<<"                  c2 = "<<c2.getData().getPos()<<"  "<<"  "<<c2.getSize()<<"  "<<c2.getData().getN()<<std::endl;
    xdbg<<"                  c3 = "<<c3.getData().getPos()<<"  "<<"  "<<c3.getSize()<<"  "<<c3.getData().getN()<<std::endl;
    xdbg<<"                  d123 = "<<sqrt(d1sq)<<"  "<<sqrt(d2sq)<<"  "<<sqrt(d3sq)<<std::endl;
    if (BinTypeHelper<B>::sort_d123) {
        Assert(d1sq >= d2sq);
        Assert(d2sq >= d3sq);
    }

    if ((s1 == 0. && (&c1 == &c2 || &c1 == &c3)) ||
        (s2 == 0. && (&c2 == &c1 || &c2 == &c3)) ||
        (s3 == 0. && (&c3 == &c1 || &c3 == &c2))) {
        dbg<<ws()<<"Stopping early -- two identical leaf cells in triangle\n";
        xdbg<<"s = "<<s1<<" "<<s2<<" "<<s3<<std::endl;
        xdbg<<"dsq = "<<d1sq<<" "<<d2sq<<" "<<d3sq<<std::endl;
        return;
    }

    // Various quanities that we'll set along the way if we need them.
    // At the end, if singleBin is true, then all these will be set correctly.
    double d1=-1., d2=-1., d3=-1., u=-1., v=-1.;
    if (BinTypeHelper<B>::stop111(d1sq, d2sq, d3sq, s1, s2, s3, d1, d2, d3, u, v,
                                  _minsep, _minsepsq, _maxsep, _maxsepsq,
                                  _minu, _minusq, _maxu, _maxusq,
                                  _minv, _minvsq, _maxv, _maxvsq))
    {
        dbg<<ws()<<"Stopping early -- no possible triangles in range\n";
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
        if (BinTypeHelper<B>::isTriangleInRange(c1, c2, c3, metric,
                                                d1, d2, d3, u, v,
                                                _logminsep, _minsep, _maxsep, _binsize, _nbins,
                                                _minu, _maxu, _ubinsize, _nubins,
                                                _minv, _maxv, _vbinsize, _nvbins,
                                                logd1, logd2, logd3,
                                                _ntot, index))
        {
            directProcess111<B>(c1, c2, c3, d1, d2, d3, u, v, logd1, logd2, logd3, index);
        } else {
            dbg<<ws()<<"Triangle not in range\n";
        }
    } else {
        xdbg<<"Need to split.\n";

        Assert(split1 == false || s1 > 0);
        Assert(split2 == false || s2 > 0);
        Assert(split3 == false || s3 > 0);

        if (split3) {
            if (split2) {
                if (split1) {
                    // split 1,2,3
                    Assert(c1.getLeft());
                    Assert(c1.getRight());
                    Assert(c2.getLeft());
                    Assert(c2.getRight());
                    Assert(c3.getLeft());
                    Assert(c3.getRight());
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getLeft(), *c2.getLeft(), *c3.getLeft(), metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getLeft(), *c2.getLeft(), *c3.getRight(), metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getLeft(), *c2.getRight(), *c3.getLeft(), metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getLeft(), *c2.getRight(), *c3.getRight(), metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getRight(), *c2.getLeft(), *c3.getLeft(), metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getRight(), *c2.getLeft(), *c3.getRight(), metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getRight(), *c2.getRight(), *c3.getLeft(), metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getRight(), *c2.getRight(), *c3.getRight(), metric);
                } else {
                    // split 2,3
                    Assert(c2.getLeft());
                    Assert(c2.getRight());
                    Assert(c3.getLeft());
                    Assert(c3.getRight());
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  c1, *c2.getLeft(), *c3.getLeft(), metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  c1, *c2.getLeft(), *c3.getRight(), metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  c1, *c2.getRight(), *c3.getLeft(), metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  c1, *c2.getRight(), *c3.getRight(), metric);
                }
            } else {
                if (split1) {
                    // split 1,3
                    Assert(c1.getLeft());
                    Assert(c1.getRight());
                    Assert(c3.getLeft());
                    Assert(c3.getRight());
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getLeft(), c2, *c3.getLeft(), metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getLeft(), c2, *c3.getRight(), metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getRight(), c2, *c3.getLeft(), metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getRight(), c2, *c3.getRight(), metric);
                } else {
                    // split 3 only
                    Assert(c3.getLeft());
                    Assert(c3.getRight());
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  c1, c2, *c3.getLeft(), metric, 0., 0., d3sq);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  c1, c2, *c3.getRight(), metric, 0., 0., d3sq);
                }
            }
        } else {
            if (split2) {
                if (split1) {
                    // split 1,2
                    Assert(c1.getLeft());
                    Assert(c1.getRight());
                    Assert(c2.getLeft());
                    Assert(c2.getRight());
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getLeft(), *c2.getLeft(), c3, metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getLeft(), *c2.getRight(), c3, metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getRight(), *c2.getLeft(), c3, metric);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  *c1.getRight(), *c2.getRight(), c3, metric);
                } else {
                    // split 2 only
                    Assert(c2.getLeft());
                    Assert(c2.getRight());
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  c1, *c2.getLeft(), c3, metric, 0., d2sq);
                    process111<B>(bc132, bc213, bc231, bc312, bc321,
                                  c1, *c2.getRight(), c3, metric, 0., d2sq);
                }
            } else {
                // split 1 only
                Assert(c1.getLeft());
                Assert(c1.getRight());
                process111<B>(bc132, bc213, bc231, bc312, bc321,
                              *c1.getLeft(), c2, c3, metric, d1sq);
                process111<B>(bc132, bc213, bc231, bc312, bc321,
                              *c1.getRight(), c2, c3, metric, d1sq);
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
        const double , const double , const double ,
        ZetaData<NData,NData,NData>& , int )
    {}
};

template <>
struct DirectHelper<KData,KData,KData>
{
    template <int C>
    static void ProcessZeta(
        const Cell<KData,C>& c1, const Cell<KData,C>& c2, const Cell<KData,C>& c3,
        const double , const double , const double ,
        ZetaData<KData,KData,KData>& zeta, int index)
    {
        zeta.zeta[index] += c1.getData().getWK() * c2.getData().getWK() * c3.getData().getWK();
    }
};

template <>
struct DirectHelper<GData,GData,GData>
{
    template <int C>
    static void ProcessZeta(
        const Cell<GData,C>& c1, const Cell<GData,C>& c2, const Cell<GData,C>& c3,
        const double d1, const double d2, const double d3,
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
};


template <int B, int C>
void BaseCorr3::directProcess111(
    const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
    const double d1, const double d2, const double d3, const double u, const double v,
    const double logd1, const double logd2, const double logd3, const int index)
{
    finishProcess(c1, c2, c3, d1, d2, d3, u, v, logd1, logd2, logd3, index);
}

template <int D1, int D2, int D3> template <int C>
void Corr3<D1,D2,D3>::finishProcess(
    const BaseCell<C>& c1, const BaseCell<C>& c2, const BaseCell<C>& c3,
    const double d1, const double d2, const double d3, const double u, const double v,
    const double logd1, const double logd2, const double logd3, const int index)
{
    double nnn = double(c1.getData().getN()) * c2.getData().getN() * c3.getData().getN();
    _ntri[index] += nnn;
    dbg<<ws()<<"Index = "<<index<<", nnn = "<<nnn<<" => "<<_ntri[index]<<std::endl;

    double www = c1.getData().getW() * c2.getData().getW() * c3.getData().getW();
    _meand1[index] += www * d1;
    _meanlogd1[index] += www * logd1;
    _meand2[index] += www * d2;
    _meanlogd2[index] += www * logd2;
    _meand3[index] += www * d3;
    _meanlogd3[index] += www * logd3;
    _meanu[index] += www * u;
    _meanv[index] += www * v;
    _weight[index] += www;

    DirectHelper<D1,D2,D3>::template ProcessZeta<C>(
        static_cast<const Cell<D1,C>&>(c1),
        static_cast<const Cell<D2,C>&>(c2),
        static_cast<const Cell<D3,C>&>(c3),
        d1, d2, d3, _zeta, index);
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

template <int B, int M, int C>
void ProcessAutob(BaseCorr3& corr, BaseField<C>& field, bool dots)
{
    Assert((ValidMC<M,C>::_M == M));
    corr.template process<B,ValidMC<M,C>::_M>(field, dots);
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
      default:
           Assert(false);
    }
}

template <int B, int M, int C>
void ProcessCross12b(BaseCorr3& corr122, BaseCorr3& corr212, BaseCorr3& corr221,
                     BaseField<C>& field1, BaseField<C>& field2, bool dots)
{
    Assert((ValidMC<M,C>::_M == M));
    corr122.template process<B,ValidMC<M,C>::_M>(corr212, corr221, field1, field2, dots);
}

template <int B, int C>
void ProcessCross12a(BaseCorr3& corr122, BaseCorr3& corr212, BaseCorr3& corr221,
                     BaseField<C>& field1, BaseField<C>& field2,
                     bool dots, Metric metric)
{
    switch(metric) {
      case Euclidean:
           ProcessCross12b<B,Euclidean>(corr122, corr212, corr221, field1, field2, dots);
           break;
      case Arc:
           ProcessCross12b<B,Arc>(corr122, corr212, corr221, field1, field2, dots);
           break;
      case Periodic:
           ProcessCross12b<B,Periodic>(corr122, corr212, corr221, field1, field2, dots);
           break;
      default:
           Assert(false);
    }
}

template <int C>
void ProcessCross12(BaseCorr3& corr122, BaseCorr3& corr212, BaseCorr3& corr221,
                    BaseField<C>& field1, BaseField<C>& field2,
                    bool dots, BinType bin_type, Metric metric)
{
    switch(bin_type) {
      case LogRUV:
           ProcessCross12a<LogRUV>(corr122, corr212, corr221, field1, field2, dots, metric);
           break;
      case LogSAS:
           ProcessCross12a<LogSAS>(corr122, corr212, corr221, field1, field2, dots, metric);
           break;
      default:
           Assert(false);
    }
}

template <int B, int M, int C>
void ProcessCrossb(BaseCorr3& corr123, BaseCorr3& corr132, BaseCorr3& corr213,
                   BaseCorr3& corr231, BaseCorr3& corr312, BaseCorr3& corr321,
                   BaseField<C>& field1, BaseField<C>& field2, BaseField<C>& field3,
                   bool dots)
{
    Assert((ValidMC<M,C>::_M == M));
    corr123.template process<B,ValidMC<M,C>::_M>(
               corr132, corr213, corr231, corr312, corr321,
               field1, field2, field3, dots);
}

template <int B, int C>
void ProcessCrossa(BaseCorr3& corr123, BaseCorr3& corr132, BaseCorr3& corr213,
                   BaseCorr3& corr231, BaseCorr3& corr312, BaseCorr3& corr321,
                   BaseField<C>& field1, BaseField<C>& field2, BaseField<C>& field3,
                   bool dots, Metric metric)
{
    switch(metric) {
      case Euclidean:
           ProcessCrossb<B,Euclidean>(corr123, corr132, corr213, corr231, corr312, corr321,
                                      field1, field2, field3, dots);
           break;
      case Arc:
           ProcessCrossb<B,Arc>(corr123, corr132, corr213, corr231, corr312, corr321,
                                field1, field2, field3, dots);
           break;
      case Periodic:
           ProcessCrossb<B,Periodic>(corr123, corr132, corr213, corr231, corr312, corr321,
                                     field1, field2, field3, dots);
           break;
      default:
           Assert(false);
    }
}

template <int C>
void ProcessCross(BaseCorr3& corr123, BaseCorr3& corr132, BaseCorr3& corr213,
                  BaseCorr3& corr231, BaseCorr3& corr312, BaseCorr3& corr321,
                  BaseField<C>& field1, BaseField<C>& field2, BaseField<C>& field3,
                  bool dots, BinType bin_type, Metric metric)
{
    dbg<<"Start ProcessCross3 "<<bin_type<<" "<<metric<<std::endl;
    switch(bin_type) {
      case LogRUV:
           ProcessCrossa<LogRUV>(corr123, corr132, corr213, corr231, corr312, corr321,
                                 field1, field2, field3, dots, metric);
           break;
      case LogSAS:
           ProcessCrossa<LogSAS>(corr123, corr132, corr213, corr231, corr312, corr321,
                                 field1, field2, field3, dots, metric);
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
    typedef void (*cross12_type)(BaseCorr3& corr122, BaseCorr3& corr212,
                                 BaseCorr3& corr221,
                                 BaseField<C>& field1, BaseField<C>& field2,
                                 bool dots, BinType bin_type, Metric metric);
    typedef void (*cross_type)(BaseCorr3& corr123, BaseCorr3& corr132,
                               BaseCorr3& corr213, BaseCorr3& corr231,
                               BaseCorr3& corr312, BaseCorr3& corr321,
                               BaseField<C>& field1, BaseField<C>& field2, BaseField<C>& field3,
                               bool dots, BinType bin_type, Metric metric);

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
