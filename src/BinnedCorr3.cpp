/* Copyright (c) 2003-2015 by Mike Jarvis
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

#include "dbg.h"
#include "BinnedCorr3.h"
#include "Split.h"
#include "ProjectHelper.h"

#ifdef _OPENMP
#include "omp.h"
#endif

// Switch these for more time-consuming Assert statements
//#define XAssert(x) Assert(x)
#define XAssert(x)

template <int D1, int D2, int D3>
BinnedCorr3<D1,D2,D3>::BinnedCorr3(
    double minsep, double maxsep, int nbins, double binsize, double b,
    double minu, double maxu, int nubins, double ubinsize, double bu,
    double minv, double maxv, int nvbins, double vbinsize, double bv,
    double minrpar, double maxrpar,
    double* zeta0, double* zeta1, double* zeta2, double* zeta3,
    double* zeta4, double* zeta5, double* zeta6, double* zeta7,
    double* meand1, double* meanlogd1, double* meand2, double* meanlogd2,
    double* meand3, double* meanlogd3, double* meanu, double* meanv,
    double* weight, double* ntri) :
    _minsep(minsep), _maxsep(maxsep), _nbins(nbins), _binsize(binsize), _b(b),
    _minu(minu), _maxu(maxu), _nubins(nubins), _ubinsize(ubinsize), _bu(bu),
    _minv(minv), _maxv(maxv), _nvbins(nvbins), _vbinsize(vbinsize), _bv(bv),
    _minrpar(minrpar), _maxrpar(maxrpar),
    _coords(-1), _owns_data(false),
    _zeta(zeta0,zeta1,zeta2,zeta3,zeta4,zeta5,zeta6,zeta7),
    _meand1(meand1), _meanlogd1(meanlogd1), _meand2(meand2), _meanlogd2(meanlogd2),
    _meand3(meand3), _meanlogd3(meanlogd3), _meanu(meanu), _meanv(meanv),
    _weight(weight), _ntri(ntri)
{
    // Some helpful variables we can calculate once here.
    _logminsep = log(_minsep);
    _halfminsep = 0.5*_minsep;
    _halfmind3 = 0.5*_minsep*_minu;
    _minsepsq = _minsep*_minsep;
    _maxsepsq = _maxsep*_maxsep;
    _minusq = _minu*_minu;
    _maxusq = _maxu*_maxu;
    _minabsv = _maxv * _minv < 0. ? 0. : std::min(std::abs(_maxv),std::abs(_minv));
    _maxabsv = std::max(std::abs(_maxv),std::abs(_minv));
    _minabsvsq = _minabsv*_minabsv;
    _maxabsvsq = _maxabsv*_maxabsv;
    _bsq = _b * _b;
    _busq = _bu * _bu;
    _bvsq = _bv * _bv;
    _sqrttwobv = sqrt(2. * _bv);
    _nuv = _nubins * _nvbins;
    _ntot = _nbins * _nuv;
}

template <int D1, int D2, int D3>
BinnedCorr3<D1,D2,D3>::BinnedCorr3(const BinnedCorr3<D1,D2,D3>& rhs, bool copy_data) :
    _minsep(rhs._minsep), _maxsep(rhs._maxsep), _nbins(rhs._nbins),
    _binsize(rhs._binsize), _b(rhs._b),
    _minu(rhs._minu), _maxu(rhs._maxu), _nubins(rhs._nubins),
    _ubinsize(rhs._ubinsize), _bu(rhs._bu),
    _minv(rhs._minv), _maxv(rhs._maxv), _nvbins(rhs._nvbins),
    _vbinsize(rhs._vbinsize), _bv(rhs._bv),
    _logminsep(rhs._logminsep), _halfminsep(rhs._halfminsep), _halfmind3(rhs._halfmind3),
    _minsepsq(rhs._minsepsq), _maxsepsq(rhs._maxsepsq),
    _minusq(rhs._minusq), _maxusq(rhs._maxusq),
    _minabsv(rhs._minabsv), _maxabsv(rhs._maxabsv),
    _minabsvsq(rhs._minabsvsq), _maxabsvsq(rhs._maxabsvsq),
    _bsq(rhs._bsq), _busq(rhs._busq), _bvsq(rhs._bvsq), _sqrttwobv(rhs._sqrttwobv),
    _coords(rhs._coords), _nuv(rhs._nuv), _ntot(rhs._ntot),
    _owns_data(true), _zeta(0,0,0,0,0,0,0,0), _weight(0)
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
BinnedCorr3<D1,D2,D3>::~BinnedCorr3()
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
void BinnedCorr3<D1,D2,D3>::clear()
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

// BinnedCorr3::process3 is invalid if D1 != D2 or D3, so this helper struct lets us only call
// process3, process21 and process111 when D1 == D2 == D3
template <int D1, int D2, int D3, int C, int M>
struct ProcessHelper
{
    static void process3(BinnedCorr3<D1,D2,D3>& , const Cell<D1,C>* ) {}
    static void process21(BinnedCorr3<D1,D2,D3>& , const Cell<D1,C>*, const Cell<D3,C>* ) {}
    static void process111(BinnedCorr3<D1,D2,D3>& , const Cell<D1,C>*, const Cell<D2,C>*,
                           const Cell<D3,C>* ) {}
};

template <int D1, int D3, int C, int M>
struct ProcessHelper<D1,D1,D3,C,M>
{
    static void process3(BinnedCorr3<D1,D1,D3>& b, const Cell<D1,C>* ) {}
    static void process21(BinnedCorr3<D1,D1,D3>& b, const Cell<D1,C>* , const Cell<D3,C>* ) {}
    static void process111(BinnedCorr3<D1,D1,D3>& b, const Cell<D1,C>* , const Cell<D1,C>*,
                           const Cell<D3,C>* ) {}
};

template <int D, int C, int M>
struct ProcessHelper<D,D,D,C,M>
{
    static void process3(BinnedCorr3<D,D,D>& b, const Cell<D,C>* c123)
    { b.template process3<C,M>(c123); }
    static void process21(BinnedCorr3<D,D,D>& b, const Cell<D,C>* c12, const Cell<D,C>* c3)
    { b.template process21<true,C,M>(c12,c3); }
    static void process111(BinnedCorr3<D,D,D>& b, const Cell<D,C>* c1, const Cell<D,C>* c2,
                           const Cell<D,C>* c3)
    { b.template process111<true,C,M>(c1,c2,c3); }
};

template <int D1, int D2, int D3> template <int C, int M>
void BinnedCorr3<D1,D2,D3>::process(const Field<D1,C>& field, bool dots)
{
    Assert(D1 == D2);
    Assert(_coords == -1 || _coords == C);
    _coords = C;
    const long n1 = field.getNTopLevel();
    xdbg<<"field has "<<n1<<" top level nodes\n";
    dbg<<"zeta[0] = "<<_zeta<<std::endl;
    Assert(n1 > 0);
#ifdef _OPENMP
#pragma omp parallel
    {
        // Give each thread their own copy of the data vector to fill in.
        BinnedCorr3<D1,D2,D3> bc3(*this,false);
#else
        BinnedCorr3<D1,D2,D3>& bc3 = *this;
#endif

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int i=0;i<n1;++i) {
            const Cell<D1,C>* c1 = field.getCells()[i];
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                if (dots) std::cout<<'.'<<std::flush;
#ifdef _OPENMP
                dbg<<omp_get_thread_num()<<" "<<i<<std::endl;
#endif
                xdbg<<"field = \n";
#ifndef NDEBUG
                if (dbgout && XDEBUG) c1->WriteTree(*dbgout);
#endif
            }
            ProcessHelper<D1,D2,D3,C,M>::process3(bc3,c1);
            for (int j=i+1;j<n1;++j) {
                const Cell<D1,C>* c2 = field.getCells()[j];
                ProcessHelper<D1,D2,D3,C,M>::process21(bc3,c1,c2);
                ProcessHelper<D1,D2,D3,C,M>::process21(bc3,c2,c1);
                for (int k=j+1;k<n1;++k) {
                    const Cell<D1,C>* c3 = field.getCells()[k];
                    ProcessHelper<D1,D2,D3,C,M>::process111(bc3,c1,c2,c3);
                }
            }
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            *this += bc3;
        }
    }
#endif
    if (dots) std::cout<<std::endl;
    dbg<<"zeta[0] -> "<<_zeta<<std::endl;
}

template <int D1, int D2, int D3> template <int C, int M>
void BinnedCorr3<D1,D2,D3>::process(const Field<D1,C>& field1, const Field<D2,C>& field2,
                                    const Field<D3,C>& field3, bool dots)
{
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
#ifndef NDEBUG
    if (dbgout && XDEBUG) {
        xdbg<<"field1: \n";
        for (int i=0;i<n1;++i) {
            xdbg<<"node "<<i<<std::endl;
            const Cell<D1,C>* c1 = field1.getCells()[i];
            c1->WriteTree(*dbgout);
        }
        xdbg<<"field2: \n";
        for (int i=0;i<n2;++i) {
            xdbg<<"node "<<i<<std::endl;
            const Cell<D2,C>* c2 = field2.getCells()[i];
            c2->WriteTree(*dbgout);
        }
        xdbg<<"field3: \n";
        for (int i=0;i<n3;++i) {
            xdbg<<"node "<<i<<std::endl;
            const Cell<D3,C>* c3 = field3.getCells()[i];
            c3->WriteTree(*dbgout);
        }
    }
#endif

#ifdef _OPENMP
#pragma omp parallel
    {
        // Give each thread their own copy of the data vector to fill in.
        BinnedCorr3<D1,D2,D3> bc3(*this,false);
#else
        BinnedCorr3<D1,D2,D3>& bc3 = *this;
#endif

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int i=0;i<n1;++i) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                if (dots) std::cout<<'.'<<std::flush;
#ifdef _OPENMP
                dbg<<omp_get_thread_num()<<" "<<i<<std::endl;
#endif
            }
            const Cell<D1,C>* c1 = field1.getCells()[i];
            for (int j=0;j<n2;++j) {
                const Cell<D2,C>* c2 = field2.getCells()[j];
                for (int k=0;k<n3;++k) {
                    const Cell<D3,C>* c3 = field3.getCells()[k];
                    bc3.template process111<false,C,M>(c1,c2,c3);
                }
            }
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            *this += bc3;
        }
    }
#endif
    if (dots) std::cout<<std::endl;
}

// Does all triangles with 3 points in c123
    template <int D1, int D2, int D3> template <int C, int M>
    void BinnedCorr3<D1,D2,D3>::process3(const Cell<D1,C>* c123)
{
    xdbg<<"Process3: c123 = "<<c123->getData().getPos()<<"  "<<"  "<<c123->getSize()<<"  "<<c123->getData().getN()<<std::endl;
    if (c123->getW() == 0) {
        xdbg<<"    w == 0.  return\n";
        return;
    }
    if (c123->getSize() < _halfminsep) {
        xdbg<<"    size < halfminsep.  return\n";
        return;
    }

    Assert(c123->getLeft());
    Assert(c123->getRight());
    process3<C,M>(c123->getLeft());
    process3<C,M>(c123->getRight());
    process21<true,C,M>(c123->getLeft(),c123->getRight());
    process21<true,C,M>(c123->getRight(),c123->getLeft());
}

// Does all triangles with two points in c12 and 3rd point in c3
// This version is allowed to swap the positions of points 1,2,3
    template <int D1, int D2, int D3> template <bool sort, int C, int M>
    void BinnedCorr3<D1,D2,D3>::process21(const Cell<D1,C>* c12, const Cell<D3,C>* c3)
{
    xdbg<<"Process21: c12 = "<<c12->getData().getPos()<<"  "<<"  "<<c12->getSize()<<"  "<<c12->getData().getN()<<std::endl;
    xdbg<<"           c3  = "<<c3->getData().getPos()<<"  "<<"  "<<c3->getSize()<<"  "<<c3->getData().getN()<<std::endl;

    // Some trivial stoppers:
    if (c12->getW() == 0) {
        xdbg<<"    w12 == 0.  return\n";
        return;
    }
    if (c3->getW() == 0) {
        xdbg<<"    w3 == 0.  return\n";
        return;
    }
    if (c12->getSize() == 0.) {
        xdbg<<"    size12 == 0.  return\n";
        return;
    }
    if (c12->getSize() < _halfmind3) {
        xdbg<<"    size12 < halfminsep * umin.  return\n";
        return;
    }

    double s12 = c12->getSize();
    double s3 = c3->getSize();
    double d2sq = MetricHelper<M>::DistSq(c12->getData().getPos(), c3->getData().getPos(),
                                          s12, s3);
    double s12ps3 = s12 + s3;

    // If all possible triangles will have d2 < minsep, then abort the recursion here.
    // i.e. if  d2 + s1 + s3 < minsep
    // Since we aren't sorting, we only need to check the actual d2 value.
    if (d2sq < _minsepsq && s12ps3 < _minsep && d2sq < SQR(_minsep - s12ps3)) {
        xdbg<<"    d2 cannot be as large as minsep\n";
        return;
    }

    // Similarly, we can abort if all possible triangles will have d2 > maxsep.
    // i.e. if  d2 - s1 - s3 >= maxsep
    if (d2sq >= _maxsepsq && d2sq >= SQR(_maxsep + s12ps3)) {
        xdbg<<"    d2 cannot be as small as maxsep\n";
        return;
    }

    // If the user has set a minu > 0, then we may be able to stop here for that.
    // The maximum possible u value at this point is 2s12 / (d2 - s12 - s3)
    // If this is less than minu, we can stop.
    // 2s12 < minu * (d2 - s12 - s3)
    // minu * d2 > 2s12 + minu * (s12 + s3)
    if (d2sq > SQR(s12 + s3) && _minusq * d2sq > SQR(2.*s12 + _minu * (s12 + s3))) {
        xdbg<<"    u cannot be as large as minu\n";
        return;
    }

    Assert(c12->getLeft());
    Assert(c12->getRight());
    process21<true,C,M>(c12->getLeft(),c3);
    process21<true,C,M>(c12->getRight(),c3);
    process111<true,C,M>(c12->getLeft(),c12->getRight(),c3);
}

// A helper to calculate the distances and possibly sort the points.
// First the sort = false case.
template <int D1, int D2, int D3, bool sort, int C, int M>
struct SortHelper
{
    static void sort3(
        const Cell<D1,C>*& c1, const Cell<D2,C>*& c2, const Cell<D3,C>*& c3,
        double& d1sq, double& d2sq, double& d3sq)
    {
        // TODO: Think about what the right thing to do with s1,s2,s3 is when the metric
        //       DistSq function wants to adjust these values.  The current code isn't right
        //       for non-Euclidean metrics.
        double s=0.;
        if (d1sq == 0.)
            d1sq = MetricHelper<M>::DistSq(c2->getData().getPos(), c3->getData().getPos(), s,s);
        if (d2sq == 0.)
            d2sq = MetricHelper<M>::DistSq(c1->getData().getPos(), c3->getData().getPos(), s,s);
        if (d3sq == 0.)
            d3sq = MetricHelper<M>::DistSq(c1->getData().getPos(), c2->getData().getPos(), s,s);
    }
    static bool stop111(
        double d1sq, double d2sq, double d3sq, double d2,
        double s1, double s2, double s3,
        double minsep, double minsepsq, double maxsep, double maxsepsq,
        double minu, double minusq, double maxu, double maxusq,
        double minabsv, double minabsvsq, double maxabsv, double maxabsvsq)
    {
        double sums = s1+s2+s3;

        // Since we aren't sorting here, we may not have d1 > d2 > d3.
        // We want to abort the recursion if there are no triangles in the given positions
        // where d1 will be the largest, d2 the middle, and d3 the smallest.

        // First, if the smallest d3 is larger than either the largest d1 or the largest d2,
        // then it can't be the smallest side.
        // i.e. if d3 - s1-s2 > d1 + s2+s3
        // d3 > d1 + s1+s2 + s2+s3
        // d3^2 > (d1 + s1+2s2+s3))^2
        // Lemma: (x+y)^2 < 2x^2 + 2y^2
        // d3^2 > 2d1^2 + 2(s1+2s2+s3)^2  (We only need that here, since we don't have d1,d3.)
        if (d3sq > d1sq && d3sq > 2.*d1sq + 2.*SQR(sums + s2)) {
            xdbg<<"d1 cannot be as large as d3\n";
            return true;
        }

        // Likewise for d2.
        if (d3sq > d2sq && d3sq > SQR(d2 + sums + s1)) {
            xdbg<<"d2 cannot be as large as d3\n";
            return true;
        }

        // Similar for d1 being the largest.
        // if d1 + s2+s3 < d2 - s1-s3
        // d2 > d1 + s2+s3 + s1+s3
        if (d2sq > d1sq && d2 > sums + s3 && d1sq < SQR(d2 - sums - s3)) {
            xdbg<<"d1 cannot be as large as d2\n";
            return true;
        }

        // If all possible triangles will have d2 < minsep, then abort the recursion here.
        // i.e. if  d2 + s1 + s3 < minsep
        // Since we aren't sorting, we only need to check the actual d2 value.
        if (d2sq < minsepsq && s1+s3 < minsep && d2sq < SQR(minsep - s1 - s3)) {
            xdbg<<"d2 cannot be as large as minsep\n";
            return true;
        }

        // Similarly, we can abort if all possible triangles will have d2 > maxsep.
        // i.e. if  d2 - s1 - s3 >= maxsep
        if (d2sq >= maxsepsq && d2sq >= SQR(maxsep + s1 + s3)) {
            xdbg<<"d2 cannot be as small as maxsep\n";
            return true;
        }

        // If the user sets minu > 0, then we can abort if no possible triangle can have
        // u = d3/d2 as large as this.
        // The maximum possible u from our triangle is (d3+s1+s2) / (d2-s1-s3).
        // Abort if (d3+s1+s2) / (d2-s1-s3) < minu
        // (d3+s1+s2) < minu * (d2-s1-s3)
        // d3 < minu * (d2-s1-s3) - (s1+s2)
        if (minu > 0. && d3sq < minusq*d2sq && d2 > s1 +s3) {
            double temp = minu * (d2-s1-s3);
            if (temp > s1 + s2 && d3sq < SQR(temp - s1 - s2)) {
                xdbg<<"u cannot be as large as minu\n";
                return true;
            }
        }

        // If the user sets a maxu < 1, then we can abort if no possible triangle can have
        // u as small as this.
        // The minimum possible u from our triangle is (d3-s1-s2) / (d2+s1+s3).
        // Abort if (d3-s1-s2) / (d2+s1+s3) > maxu
        // (d3-s1-s2) > maxu * (d2+s1+s3)
        // d3 > maxu * (d2+s1+s3) + (s1+s2)
        if (maxu < 1. && d3sq >= maxusq*d2sq && d3sq >= SQR(maxu * (d2 + s1 + s3) + s1 + s2)) {
            xdbg<<"u cannot be as small as maxu\n";
            return true;
        }

        // If the user sets minv, maxv to be near 0, then we can abort if no possible triangle
        // can have v = (d1-d2)/d3 as small in absolute value as either of these.
        // |v| is |d1-d2|/d3.  The minimum numerator is a bit non-obvious.
        // The easy part is from c1, c2.  These can shrink |d1-d2| by s1+s2.
        // The edge of c3 can shrink |d1-d2| by at most another s3, assuming d3 < d2, so the
        // angle at c3 is acute.  i.e. it's not 2s3 as one might naively assume.
        // Thus, the minimum possible |v| from our triangle is (d1-d2-(s1+s2+s3)) / (d3+s1+s2).
        // Abort if (d1-d2-s1-s2-s3) / (d3+s1+s2) > maxabsv
        // (d1-d2-s1-s2-s3) > maxabsv * (d3+s1+s2)
        // d1 > maxabsv d3 + d2+s1+s2+s3 + maxabsv*(s1+s2)
        // Here, rather than using the lemma that (x+y)^2 < 2x^2 + 2y^2,
        // we can instead realize that d3 < d2, so just check if
        // d1 > maxabsv d2 + d2+s1+s2+s3 + maxabsv*(s1+s2)
        // The main advantage of this check is when d3 ~= d2 anyway, so this is effective.
        if (maxabsv < 1. && d1sq > SQR((1.+maxabsv)*d2 + sums + maxabsv * (s1+s2))) {
            xdbg<<"|v| cannot be as small as maxabsv\n";
            return true;
        }

        // It will unusual, but if minabsv > 0, then we can also potentially stop if no triangle
        // can have |v| as large as minabsv.
        // The maximum possible |v| from our triangle is (d1-d2+(s1+s2+s3)) / (d3-s1-s2).
        // Abort if (d1-d2+s1+s2+s3) / (d3-s1-s2) < minabsv
        // (d1-d2+s1+s2+s3) < minabsv * (d3-s1-s2)
        // d1-d2 < minabsv d3 - (s1+s2+s3) - minabsv*(s1+s2)
        // d1^2-d2^2 < (minabsv d3 - (s1+s2+s3) - minabsv*(s1+s2)) (d1+d2)
        // This is most relevant when d1 ~= d2, so make this more restrictive with d1->d2 on rhs.
        // d1^2-d2^2 < (minabsv d3 - (s1+s2+s3) - minabsv*(s1+s2)) 2d2
        // minabsv d3 > (d1^2-d2^2)/(2d2) + (s1+s2+s3) + minabsv*(s1+s2)
        if (minabsv > 0. && d3sq > SQR(s1+s2) &&
            minabsvsq*d3sq > SQR((d1sq-d2sq)/(2.*d2) + sums + minabsv*(s1+s2))) {
            xdbg<<"|v| cannot be as large as minabsv\n";
            return true;
        }

        return false;
    }
};

// This one has sort = true, so the points get sorted, and always returns true.
template <int D, int C, int M>
struct SortHelper<D,D,D,true,C,M>
{
    static void sort3(
        const Cell<D,C>*& c1, const Cell<D,C>*& c2, const Cell<D,C>*& c3,
        double& d1sq, double& d2sq, double& d3sq)
    {
        double s=0.;
        if (d1sq == 0.)
            d1sq = MetricHelper<M>::DistSq(c2->getData().getPos(), c3->getData().getPos(), s,s);
        if (d2sq == 0.)
            d2sq = MetricHelper<M>::DistSq(c1->getData().getPos(), c3->getData().getPos(), s,s);
        if (d3sq == 0.)
            d3sq = MetricHelper<M>::DistSq(c1->getData().getPos(), c2->getData().getPos(), s,s);

        // Need to end up with d3 < d2 < d1
        if (d1sq < d2sq) {
            if (d2sq < d3sq) {
                // 123 -> 321
                std::swap(c1,c3);
                std::swap(d1sq,d3sq);
            } else if (d1sq < d3sq) {
                // 132 -> 321
                std::swap(c1,c3);
                std::swap(d1sq,d3sq);
                std::swap(c1,c2);
                std::swap(d1sq,d2sq);
            } else {
                // 312 -> 321
                std::swap(c1,c2);
                std::swap(d1sq,d2sq);
            }
        } else {
            if (d1sq < d3sq) {
                // 213 -> 321
                std::swap(c2,c3);
                std::swap(d2sq,d3sq);
                std::swap(c1,c2);
                std::swap(d1sq,d2sq);
            } else if (d2sq < d3sq) {
                // 231 -> 321
                std::swap(c2,c3);
                std::swap(d2sq,d3sq);
            } else {
                // 321 -> 321
            }
        }
    }
    static bool stop111(
        double d1sq, double d2sq, double d3sq, double d2,
        double s1, double s2, double s3,
        double minsep, double minsepsq, double maxsep, double maxsepsq,
        double minu, double minusq, double maxu, double maxusq,
        double minabsv, double minabsvsq, double maxabsv, double maxabsvsq)
    {
        // If all possible triangles will have d2 < minsep, then abort the recursion here.
        // This means at least two sides must have d + (s+s) < minsep.
        // Probably if d2 + s1+s3 < minsep, we can stop, but also check d3.
        // If one of these don't pass, then it's pretty unlikely that d1 will, so don't bother
        // checking that one.
        if (d2sq < minsepsq && s1+s3 < minsep && s1+s2 < minsep &&
            (s1+s3 == 0. || d2sq < SQR(minsep - s1-s3)) &&
            (s1+s2 == 0. || d3sq < SQR(minsep - s1-s2)) ) {
            xdbg<<"d2 cannot be as large as minsep\n";
            return true;
        }

        // Similarly, we can abort if all possible triangles will have d2 > maxsep.
        // This means at least two sides must have d - (s+s) > maxsep.
        // Again, d2 - s1 - s3 >= maxsep is not sufficient.  Also check d1.
        // And again, it's pretty unlikely that d3 needs to be checked if one of the first
        // two don't pass.
        if (d2sq >= maxsepsq &&
            (s1+s3 == 0. || d2sq >= SQR(maxsep + s1+s3)) &&
            (s2+s3 == 0. || d1sq >= SQR(maxsep + s2+s3))) {
            xdbg<<"d2 cannot be as small as maxsep\n";
            return true;
        }

        // If the user sets minu > 0, then we can abort if no possible triangle can have
        // u = d3/d2 as large as this.
        // The maximum possible u from our triangle is (d3+s1+s2) / (d2-s1-s3).
        // Abort if (d3+s1+s2) / (d2-s1-s3) < minu
        // (d3+s1+s2) < minu * (d2-s1-s3)
        // d3 < minu * (d2-s1-s3) - (s1+s2)
        if (minu > 0. && d3sq < minusq*d2sq && d2 > s1+s3) {
            double temp = minu * (d2-s1-s3);
            if (temp > s1+s2 && d3sq < SQR(temp - s1-s2)) {
                // However, d2 might not really be the middle leg.  So check d1 as well.
                double minusq_d1sq = minusq * d1sq;
                if (d3sq < minusq_d1sq && d1sq > 2.*SQR(s2+s3) &&
                    minusq_d1sq > 2.*d3sq + 2.*SQR(s1+s2 + minu * (s2+s3))) {
                    xdbg<<"u cannot be as large as minu\n";
                    return true;
                }
            }
        }

        // If the user sets a maxu < 1, then we can abort if no possible triangle can have
        // u as small as this.
        // The minimum possible u from our triangle is (d3-s1-s2) / (d2+s1+s3).
        // Abort if (d3-s1-s2) / (d2+s1+s3) > maxu
        // (d3-s1-s2) > maxu * (d2+s1+s3)
        // d3 > maxu * (d2+s1+s3) + (s1+s2)
        if (maxu < 1. && d3sq >= maxusq*d2sq && d3sq >= SQR(maxu * (d2+s1+s3) + s1+s2)) {
            // This time, just make sure no other side could become the smallest side.
            // d3 - s1-s2 < d2 - s1-s3
            // d3 - s1-s2 < d1 - s2-s3
            if ( d2sq > SQR(s1+s3) && d1sq > SQR(s2+s3) &&
                 (s2 > s3 || d3sq <= SQR(d2 - s3 + s2)) &&
                 (s1 > s3 || d1sq >= 2.*d3sq + 2.*SQR(s3 - s1)) ) {
                xdbg<<"u cannot be as small as maxu\n";
                return true;
            }
        }

        // If the user sets minv, maxv to be near 0, then we can abort if no possible triangle
        // can have v = (d1-d2)/d3 as small in absolute value as either of these.
        // d1 > maxabsv d3 + d2+s1+s2+s3 + maxabsv*(s1+s2)
        // As before, use the fact that d3 < d2, so check
        // d1 > maxabsv d2 + d2+s1+s2+s3 + maxabsv*(s1+s2)
        double sums = s1+s2+s3;
        if (maxabsv < 1. && d1sq > SQR((1.+maxabsv)*d2 + sums + maxabsv * (s1+s2))) {
            // We don't need any extra checks here related to the possibility of the sides
            // switching roles, since if this condition is true, than d1 has to be the largest
            // side no matter what.  d1-s2 > d2+s1
            xdbg<<"v cannot be as small as maxabsv\n";
            return true;
        }

        // It will unusual, but if minabsv > 0, then we can also potentially stop if no triangle
        // can have |v| as large as minabsv.
        // d1-d2 < minabsv d3 - (s1+s2+s3) - minabsv*(s1+s2)
        // d1^2-d2^2 < (minabsv d3 - (s1+s2+s3) - minabsv*(s1+s2)) (d1+d2)
        // This is most relevant when d1 ~= d2, so make this more restrictive with d1->d2 on rhs.
        // d1^2-d2^2 < (minabsv d3 - (s1+s2+s3) - minabsv*(s1+s2)) 2d2
        // minabsv d3 > (d1^2-d2^2)/(2d2) + (s1+s2+s3) + minabsv*(s1+s2)
        if (minabsv > 0. && d3sq > SQR(s1+s2) &&
            minabsvsq*d3sq > SQR((d1sq-d2sq)/(2.*d2) + sums + minabsv*(s1+s2))) {
            // And again, we don't need anything else here, since it's fine if d1,d2 swap or
            // even if d2,d3 swap.
            xdbg<<"|v| cannot be as large as minabsv\n";
            return true;
        }

        return false;
    }
};

// Does all triangles with 1 point each in c1, c2, c3
template <int D1, int D2, int D3> template <bool sort, int C, int M>
void BinnedCorr3<D1,D2,D3>::process111(
    const Cell<D1,C>* c1, const Cell<D2,C>* c2, const Cell<D3,C>* c3,
    double d1sq, double d2sq, double d3sq)
{
    if (c1->getW() == 0) {
        xdbg<<"    w1 == 0.  return\n";
        return;
    }
    if (c2->getW() == 0) {
        xdbg<<"    w2 == 0.  return\n";
        return;
    }
    if (c3->getW() == 0) {
        xdbg<<"    w3 == 0.  return\n";
        return;
    }

    // Calculate the distances if they aren't known yet, and sort so that d3 < d2 < d1
    SortHelper<D1,D2,D3,sort,C,M>::sort3(c1,c2,c3,d1sq,d2sq,d3sq);

    xdbg<<"Process111: c1 = "<<c1->getData().getPos()<<"  "<<"  "<<c1->getSize()<<"  "<<c1->getData().getN()<<std::endl;
    xdbg<<"            c2 = "<<c2->getData().getPos()<<"  "<<"  "<<c2->getSize()<<"  "<<c2->getData().getN()<<std::endl;
    xdbg<<"            c3 = "<<c3->getData().getPos()<<"  "<<"  "<<c3->getSize()<<"  "<<c3->getData().getN()<<std::endl;
    xdbg<<"            d123 = "<<sqrt(d1sq)<<"  "<<sqrt(d2sq)<<"  "<<sqrt(d3sq)<<std::endl;
    Assert(!sort || d1sq >= d2sq);
    Assert(!sort || d2sq >= d3sq);

    const double s1 = c1->getAllSize();
    const double s2 = c2->getAllSize();
    const double s3 = c3->getAllSize();
    const double d2 = sqrt(d2sq);

    if (SortHelper<D1,D2,D3,sort,C,M>::stop111(d1sq,d2sq,d3sq,d2,s1,s2,s3,
                                               _minsep,_minsepsq,_maxsep,_maxsepsq,
                                               _minu,_minusq,_maxu,_maxusq,
                                               _minabsv,_minabsvsq,_maxabsv,_maxabsvsq))
        return;

    // For 1,3 decide whether to split on the noraml criteria with s1+s3/d2 < b
    bool split1 = false, split3 = false;
    CalcSplitSq(split1,split3,d2sq,s1,s3,_bsq);

    // For 2, split if it's possible for d3 to become larger than the largest possible d2 or
    // if d1 could become smaller than the current smallest possible d2.
    // i.e. if d3 + s1 + s2 > d2 + s1 + s3 => d3 > d2 - s2 + s3
    //      or d1 - s2 - s3 < d2 - s1 - s3 => d1 < d2 + s2 - s1
    const double s2ms1 = s2 - s1;
    const double s2ms3 = s2 - s3;
    bool split2 = ( (s2ms3 > 0. && d3sq > SQR(d2 - s2ms3)) ||
                    (s2ms1 > 0. && d1sq < SQR(d2 + s2ms1)) );

    xdbg<<"r: split = "<<split1<<" "<<split2<<" "<<split3<<std::endl;

    // Now check for splits related to the u value.
    if (!sort && (d3sq > d2sq || d2sq > d1sq)) {
        // If we aren't sorting the sides, then d3 is not necessarily less than d2
        // (nor d2 less than d1).  If this is the case, we always want to split something,
        // since we don't actually have a valid u here.
        // Split the largest one at least.
        if (s1 > s2) {
            if (s1 > s3)
                split1 = true;
            else if (s3 > 0)
                split3 = true;
        } else {
            if (s2 > s3)
                split2 = true;
            else if (s3 > 0)
                split3 = true;
        }
        // Also split any that can directly lead to a swap of two that are out of order
        // d2 + s1 < d3 - s1
        if (d3sq > d2sq && s1 > 0 && d3sq > SQR(d2 + 2.*s1)) split1 = true;
        // d1 + s3 < d2 - s3
        if (d1sq < d2sq && s3 > 0 && (d2 < 2.*s3 ||  d1sq < SQR(d2 - 2.*s3))) split3 = true;
    }

    // We don't need to split c1,c3 for d2 but we might need to split c1,c2 for d3.
    // u = d3 / d2
    // du = d(d3) / d2 = (s1+s2)/d2
    // du < bu -> same split calculation as before, but with d2, not d3, and _bu instead of _b.
    CalcSplitSq(split1,split2,d2sq,s1,s2,_busq);

    xdbg<<"u: split = "<<split1<<" "<<split2<<" "<<split3<<std::endl;


    // Finally the splits related to v.
    // I don't currently do any checks related to _minv, _maxv.
    // Not sure how important they are.
    // But there could be some gain to checking that here.

    // v is a bit complicated.
    // Consider the angle bisector of d1,d2.  Call this line z.
    // Then let phi = the angle between d1 (or d2) and z
    // And let theta = the (acute) angle between d3 and z
    // Then projecting d1,d2,d3 onto z, one finds:
    // d1 cos phi - d2 cos phi = d3 cos theta
    // v = (d1-d2)/d3 = cos theta / cos phi
    //
    // 1. How does v change for different points within cell c3?
    //
    // Note that phi < 30 degrees, so cos phi won't make much
    // of a difference here.
    // The biggest change in v from moving c3 is in theta:
    // dv = |dv/dtheta| dtheta = |sin(theta)|/cos(phi) (s3/z)
    // dv < b -> s3/z |sin(theta)|/cos(phi) < b
    //
    // v >= cos(theta), so sqrt(1-v^2) <= sin(theta)
    // Also, z cos(phi) >= 3/4 d2  (where the 3/4 is the case where theta=90, phi=30 deg.)
    //
    // So s3 * sqrt(1-v^2) / (0.75 d2) < b
    // s3/d2 < 0.75 b / sqrt(1-v^2)
    //
    // In the limit as v -> 1, the triangle collapses, and the differential doesn't
    // really work (theta == 0).  So here we calculate what triangle could happen from
    // c3 moving by up to a distance of s3:
    // dv = 1-cos(dtheta) = 1-cos(s3/z) ~= 1/2(s3/z)^2 < 1/2(s3/d2)^2
    // So in this case, s3/d2 < sqrt(2b)
    // In general we require both to be true.

    // These may be set here and then used below, but only if we aren't splitting already.
    // Initialize them to zero to avoid compiler warnings.
    double d1=0.,d3=0.,v=0.,onemvsq=0.;

    if (!(split1 && split2 && split3)) {
        d1 = sqrt(d1sq);
        d3 = sqrt(d3sq);
        if (d3 == 0.) {
            // Very unusual!  But possible, so make sure we don't get nans
            v = 0.;
        } else {
            v = (d1-d2)/d3;
        }
        onemvsq = 1.-SQR(v);
    }

    if (!split3) {
        split3 = s3 > _sqrttwobv * d2 || SQR(s3) * onemvsq > 0.5625 * _bvsq * d2sq;
    }

    // 2. How does v change for different pairs of points within c1,c2?
    //
    // These two cells mostly serve to twist the line d3.
    // We make the approximation that the angle bisector hits d3 near the middle.
    // Then dtheta = (s1+s2)/(d3/2).
    // Then from the same kind of derivation as above, we get
    // |sin(theta)|/cos(phi) 2(s1+s2)/d3 < b
    // (s1+s2)/d3 < sqrt(3)/4 b / sqrt(1-v^2)
    //
    // And again, in the limit where v -> 1, the approximations fail, so we need to look
    // directly at how the collapsed triangle opens up.
    // For each one, sin(dtheta) ~= s/(d3/2)
    // Need to split if 2sin^2(dtheta) > b
    if (!split1) split1 = s1 > 0.5*_sqrttwobv * d3;
    if (!split2) split2 = s2 > 0.5*_sqrttwobv * d3;

    if (!(split1 && split2) && onemvsq > 1.e-2) {
        CalcSplitSq(split1,split2,d3sq,s1,s2,_bvsq * 3./16. / onemvsq);
    }

    xdbg<<"v: split = "<<split1<<" "<<split2<<" "<<split3<<std::endl;

    if (split1) {
        if (split2) {
            if (split3) {
                // split 1,2,3
                Assert(c1->getLeft());
                Assert(c1->getRight());
                Assert(c2->getLeft());
                Assert(c2->getRight());
                Assert(c3->getLeft());
                Assert(c3->getRight());
                process111<sort,C,M>(c1->getLeft(),c2->getLeft(),c3->getLeft());
                process111<sort,C,M>(c1->getLeft(),c2->getLeft(),c3->getRight());
                process111<sort,C,M>(c1->getLeft(),c2->getRight(),c3->getLeft());
                process111<sort,C,M>(c1->getLeft(),c2->getRight(),c3->getRight());
                process111<sort,C,M>(c1->getRight(),c2->getLeft(),c3->getLeft());
                process111<sort,C,M>(c1->getRight(),c2->getLeft(),c3->getRight());
                process111<sort,C,M>(c1->getRight(),c2->getRight(),c3->getLeft());
                process111<sort,C,M>(c1->getRight(),c2->getRight(),c3->getRight());
            } else {
                // split 1,2
                Assert(c1->getLeft());
                Assert(c1->getRight());
                Assert(c2->getLeft());
                Assert(c2->getRight());
                process111<sort,C,M>(c1->getLeft(),c2->getLeft(),c3);
                process111<sort,C,M>(c1->getLeft(),c2->getRight(),c3);
                process111<sort,C,M>(c1->getRight(),c2->getLeft(),c3);
                process111<sort,C,M>(c1->getRight(),c2->getRight(),c3);
            }
        } else {
            if (split3) {
                // split 1,3
                Assert(c1->getLeft());
                Assert(c1->getRight());
                Assert(c3->getLeft());
                Assert(c3->getRight());
                process111<sort,C,M>(c1->getLeft(),c2,c3->getLeft());
                process111<sort,C,M>(c1->getLeft(),c2,c3->getRight());
                process111<sort,C,M>(c1->getRight(),c2,c3->getLeft());
                process111<sort,C,M>(c1->getRight(),c2,c3->getRight());
            } else {
                // split 1 only
                Assert(c1->getLeft());
                Assert(c1->getRight());
                process111<sort,C,M>(c1->getLeft(),c2,c3,d1sq);
                process111<sort,C,M>(c1->getRight(),c2,c3,d1sq);
            }
        }
    } else {
        if (split2) {
            if (split3) {
                // split 2,3
                Assert(c2->getLeft());
                Assert(c2->getRight());
                Assert(c3->getLeft());
                Assert(c3->getRight());
                process111<sort,C,M>(c1,c2->getLeft(),c3->getLeft());
                process111<sort,C,M>(c1,c2->getLeft(),c3->getRight());
                process111<sort,C,M>(c1,c2->getRight(),c3->getLeft());
                process111<sort,C,M>(c1,c2->getRight(),c3->getRight());
            } else {
                // split 2 only
                Assert(c2->getLeft());
                Assert(c2->getRight());
                process111<sort,C,M>(c1,c2->getLeft(),c3,0.,d2sq);
                process111<sort,C,M>(c1,c2->getRight(),c3,0.,d2sq);
            }
        } else {
            if (split3) {
                // split 3 only
                Assert(c3->getLeft());
                Assert(c3->getRight());
                process111<sort,C,M>(c1,c2,c3->getLeft(),0.,0.,d3sq);
                process111<sort,C,M>(c1,c2,c3->getRight(),0.,0.,d3sq);
            } else {
                // No splits required.
                // Now we can check to make sure the final d2, u, v are in the right ranges.
                if (d2 < _minsep || d2 >= _maxsep) {
                    xdbg<<"d2 not in minsep .. maxsep\n";
                    return;
                }

                double u = d3/d2;
                if (u < _minu || u >= _maxu) {
                    xdbg<<"u not in minu .. maxu\n";
                    return;
                }

                if (!MetricHelper<M>::CCW(c1->getData().getPos(), c2->getData().getPos(),
                                          c3->getData().getPos()))
                    v = -v;
                if (v < _minv || v >= _maxv) {
                    xdbg<<"v not in minv .. maxv\n";
                    return;
                }

                double logr = log(d2);
                xdbg<<"            logr = "<<logr<<std::endl;
                xdbg<<"            u = "<<u<<std::endl;
                xdbg<<"            v = "<<v<<std::endl;

                const int kr = int(floor((logr-_logminsep)/_binsize));
                Assert(kr >= 0);
                Assert(kr < _nbins);

                int ku = int(floor((u-_minu)/_ubinsize));
                if (ku >= _nubins) {
                    // Rounding error can allow this.
                    XAssert((u-_minu)/_ubinsize - ku < 1.e-10);
                    Assert(ku==_nubins);
                    --ku;
                }
                Assert(ku >= 0);
                Assert(ku < _nubins);

                int kv = int(floor((v-_minv)/_vbinsize));
                if (kv >= _nvbins) {
                    // Rounding error can allow this.
                    XAssert((v-_minv)/_vbinsize - kv < 1.e-10);
                    Assert(kv==_nvbins);
                    --kv;
                }
                Assert(kv >= 0);
                Assert(kv < _nvbins);

                int index = kr * _nuv + ku * _nvbins + kv;
                Assert(index >= 0);
                Assert(index < _ntot);
                // Just to make extra sure we don't get seg faults (since the above
                // asserts aren't active in normal operations), do a real check that
                // index is in the allowed range.
                if (index < 0 || index >= _ntot) return;
                directProcess111<C,M>(*c1,*c2,*c3,d1,d2,d3,logr,u,v,index);
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
    template <int C, int M>
    static void ProcessZeta(
        const Cell<NData,C>& , const Cell<NData,C>& , const Cell<NData,C>&,
        const double , const double , const double ,
        ZetaData<NData,NData,NData>& , int )
    {}
};

template <>
struct DirectHelper<KData,KData,KData>
{
    template <int C, int M>
    static void ProcessZeta(
        const Cell<KData,C>& c1, const Cell<KData,C>& c2, const Cell<KData,C>& c3,
        const double , const double , const double ,
        ZetaData<KData,KData,KData>& zeta, int index)
    {
        zeta.zeta[index] += c1.getData().getWK() * c2.getData().getWK() * c3.getData().getWK();
        xdbg<<"            zeta -> "<<zeta.zeta[index]<<std::endl;
    }
};

template <>
struct DirectHelper<GData,GData,GData>
{
    template <int C, int M>
    static void ProcessZeta(
        const Cell<GData,C>& c1, const Cell<GData,C>& c2, const Cell<GData,C>& c3,
        const double d1, const double d2, const double d3,
        ZetaData<GData,GData,GData>& zeta, int index)
    {
        std::complex<double> g1, g2, g3;
        ProjectHelper<C>::ProjectShears(c1,c2,c3,g1,g2,g3);

        // The complex products g1 g2 and g1 g2* share most of the calculations,
        // so faster to do this manually.
        //double g1rg2r = g1.real() * g2.real();
        //double g1rg2i = g1.real() * g2.imag();
        //double g1ig2r = g1.imag() * g2.real();
        //double g1ig2i = g1.imag() * g2.imag();
        std::complex<double> gam0 = g1 * g2 * g3;
        std::complex<double> gam1 = std::conj(g1) * g2 * g3;
        std::complex<double> gam2 = g1 * std::conj(g2) * g3;
        std::complex<double> gam3 = g1 * g2 * std::conj(g3);

        zeta.gam0r[index] += gam0.real();
        zeta.gam0i[index] += gam0.imag();
        zeta.gam1r[index] += gam1.real();
        zeta.gam1i[index] += gam1.imag();
        zeta.gam2r[index] += gam2.real();
        zeta.gam2i[index] += gam2.imag();
        zeta.gam3r[index] += gam3.real();
        zeta.gam3i[index] += gam3.imag();
    }
};

#if 0
template <>
struct DirectHelper<NData,NData,KData>
{
    template <int C, int M>
    static void ProcessZeta(
        const Cell<NData,C>& c1, const Cell<KData,C>& c2,
        const double d1, const double d2, const double d3,
        ZetaData<NData,NData,KData>& zeta, int index)
    { zeta.zeta[index] += c1.getData().getW() * c2.getData().getWK(); }
};

template <>
struct DirectHelper<NData,NData,GData>
{
    template <int C, int M>
    static void ProcessZeta(
        const Cell<NData,C>& c1, const Cell<GData,C>& c2,
        const double d1, const double d2, const double d3,
        ZetaData<NData,NData,GData>& zeta, int index)
    {
        std::complex<double> g2;
        ProjectHelper<C>::ProjectShear(c1,c2,g2);
        // The minus sign here is to make it accumulate tangential shear, rather than radial.
        // g2 from the above ProjectShear is measured along the connecting line, not tangent.
        g2 *= -c1.getData().getW();
        zeta.zeta[index] += real(g2);
        zeta.zeta_im[index] += imag(g2);

    }
};


template <>
struct DirectHelper<KData,KData,GData>
{
    template <int C, int M>
    static void ProcessZeta(
        const Cell<KData,C>& c1, const Cell<GData,C>& c2,
        const double d1, const double d2, const double d3,
        ZetaData<KData,KData,GData>& zeta, int index)
    {
        std::complex<double> g2;
        ProjectHelper<C>::ProjectShear(c1,c2,g2);
        // The minus sign here is to make it accumulate tangential shear, rather than radial.
        // g2 from the above ProjectShear is measured along the connecting line, not tangent.
        g2 *= -c1.getData().getWK();
        zeta.zeta[index] += real(g2);
        zeta.zeta_im[index] += imag(g2);
    }
};
#endif

template <int D1, int D2, int D3> template <int C, int M>
void BinnedCorr3<D1,D2,D3>::directProcess111(
    const Cell<D1,C>& c1, const Cell<D2,C>& c2, const Cell<D3,C>& c3,
    const double d1, const double d2, const double d3,
    const double logr, const double u, const double v, const int index)
{
    double nnn = double(c1.getData().getN()) * double(c2.getData().getN()) *
        double(c3.getData().getN());
    _ntri[index] += nnn;
    xdbg<<"            index = "<<index<<std::endl;
    xdbg<<"            nnn = "<<nnn<<std::endl;

    double www = double(c1.getData().getW()) * double(c2.getData().getW()) *
        double(c3.getData().getW());
    _meand1[index] += www * d1;
    _meanlogd1[index] += www * log(d1);
    _meand2[index] += www * d2;
    _meanlogd2[index] += www * logr;
    _meand3[index] += www * d3;
    _meanlogd3[index] += www * log(d3);
    _meanu[index] += www * u;
    _meanv[index] += www * v;
    _weight[index] += www;

    DirectHelper<D1,D2,D3>::template ProcessZeta<C,M>(c1,c2,c3,d1,d2,d3,_zeta,index);
}

template <int D1, int D2, int D3>
void BinnedCorr3<D1,D2,D3>::operator=(const BinnedCorr3<D1,D2,D3>& rhs)
{
    Assert(rhs._ntot == _ntot);
    _zeta.copy(rhs._zeta,_ntot);
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
void BinnedCorr3<D1,D2,D3>::operator+=(const BinnedCorr3<D1,D2,D3>& rhs)
{
    Assert(rhs._ntot == _ntot);
    _zeta.add(rhs._zeta,_ntot);
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
// The C interface for python
//
//

extern "C" {
#include "BinnedCorr3_C.h"
}

void* BuildNNNCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                   double minu, double maxu, int nubins, double ubinsize, double bu,
                   double minv, double maxv, int nvbins, double vbinsize, double bv,
                   double minrpar, double maxrpar,
                   double* meand1, double* meanlogd1, double* meand2, double* meanlogd2,
                   double* meand3, double* meanlogd3, double* meanu, double* meanv,
                   double* weight, double* ntri)
{
    dbg<<"Start BuildNNNCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr3<NData,NData,NData>(
            minsep, maxsep, nbins, binsize, b,
            minu, maxu, nubins, ubinsize, bu,
            minv, maxv, nvbins, vbinsize, bv,
            minrpar, maxrpar,
            0, 0, 0, 0, 0, 0, 0, 0,
            meand1, meanlogd1, meand2, meanlogd2, meand3, meanlogd3, meanu, meanv,
            weight, ntri));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildKKKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                   double minu, double maxu, int nubins, double ubinsize, double bu,
                   double minv, double maxv, int nvbins, double vbinsize, double bv,
                   double minrpar, double maxrpar,
                   double* zeta,
                   double* meand1, double* meanlogd1, double* meand2, double* meanlogd2,
                   double* meand3, double* meanlogd3, double* meanu, double* meanv,
                   double* weight, double* ntri)
{
    dbg<<"Start BuildKKKCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr3<KData,KData,KData>(
            minsep, maxsep, nbins, binsize, b,
            minu, maxu, nubins, ubinsize, bu,
            minv, maxv, nvbins, vbinsize, bv,
            minrpar, maxrpar,
            zeta, 0, 0, 0, 0, 0, 0, 0,
            meand1, meanlogd1, meand2, meanlogd2, meand3, meanlogd3, meanu, meanv,
            weight, ntri));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}


void* BuildGGGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                   double minu, double maxu, int nubins, double ubinsize, double bu,
                   double minv, double maxv, int nvbins, double vbinsize, double bv,
                   double minrpar, double maxrpar,
                   double* gam0r, double* gam0i, double* gam1r, double* gam1i,
                   double* gam2r, double* gam2i, double* gam3r, double* gam3i,
                   double* meand1, double* meanlogd1, double* meand2, double* meanlogd2,
                   double* meand3, double* meanlogd3, double* meanu, double* meanv,
                   double* weight, double* ntri)
{
    dbg<<"Start BuildGGGCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr3<GData,GData,GData>(
            minsep, maxsep, nbins, binsize, b,
            minu, maxu, nubins, ubinsize, bu,
            minv, maxv, nvbins, vbinsize, bv,
            minrpar, maxrpar,
            gam0r, gam0i, gam1r, gam1i, gam2r, gam2i, gam3r, gam3i,
            meand1, meanlogd1, meand2, meanlogd2, meand3, meanlogd3, meanu, meanv,
            weight, ntri));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

#if 0
void* BuildNNKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                   double minu, double maxu, int nubins, double ubinsize, double bu,
                   double minv, double maxv, int nvbins, double vbinsize, double bv,
                   double minrpar, double maxrpar,
                   double* zeta,
                   double* meand1, double* meanlogd1, double* meand2, double* meanlogd2,
                   double* meand3, double* meanlogd3, double* meanu, double* meanv,
                   double* weight, double* ntri)
{
    dbg<<"Start BuildNNKCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr3<NData,NData,KData>(
            minsep, maxsep, nbins, binsize, b,
            minu, maxu, nubins, ubinsize, bu,
            minv, maxv, nvbins, vbinsize, bv,
            minrpar, maxrpar,
            zeta, 0, 0, 0, 0, 0, 0, 0,
            meand1, meanlogd1, meand2, meanlogd2, meand3, meanlogd3, meanu, meanv,
            weight, ntri));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildNNGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                   double minu, double maxu, int nubins, double ubinsize, double bu,
                   double minv, double maxv, int nvbins, double vbinsize, double bv,
                   double minrpar, double maxrpar,
                   double* zeta, double* zeta_im,
                   double* meand1, double* meanlogd1, double* meand2, double* meanlogd2,
                   double* meand3, double* meanlogd3, double* meanu, double* meanv,
                   double* weight, double* ntri)
{
    dbg<<"Start BuildNNGCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr3<NData,NData,GData>(
            minsep, maxsep, nbins, binsize, b,
            minu, maxu, nubins, ubinsize, bu,
            minv, maxv, nvbins, vbinsize, bv,
            minrpar, maxrpar,
            zeta, zeta_im, 0, 0, 0, 0, 0, 0,
            meand1, meanlogd1, meand2, meanlogd2, meand3, meanlogd3, meanu, meanv,
            weight, ntri));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildKKGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                   double minu, double maxu, int nubins, double ubinsize, double bu,
                   double minv, double maxv, int nvbins, double vbinsize, double bv,
                   double minrpar, double maxrpar,
                   double* zeta, double* zeta_im,
                   double* meand1, double* meanlogd1, double* meand2, double* meanlogd2,
                   double* meand3, double* meanlogd3, double* meanu, double* meanv,
                   double* weight, double* ntri)
{
    dbg<<"Start BuildKKGCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr3<KData,KData,GData>(
            minsep, maxsep, nbins, binsize, b,
            minu, maxu, nubins, ubinsize, bu,
            minv, maxv, nvbins, vbinsize, bv,
            minrpar, maxrpar,
            zeta, zeta_im, 0, 0, 0, 0, 0, 0,
            meand1, meanlogd1, meand2, meanlogd2, meand3, meanlogd3, meanu, meanv,
            weight, ntri));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}
#endif

void DestroyNNNCorr(void* corr)
{
    dbg<<"Start DestroyNNNCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr3<NData,NData,NData>*>(corr);
}

void DestroyKKKCorr(void* corr)
{
    dbg<<"Start DestroyKKKCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr3<KData,KData,KData>*>(corr);
}


void DestroyGGGCorr(void* corr)
{
    dbg<<"Start DestroyGGGCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr3<GData,GData,GData>*>(corr);
}

#if 0
void DestroyNNKCorr(void* corr)
{
    dbg<<"Start DestroyNNKCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr3<NData,NData,KData>*>(corr);
}

void DestroyNNGCorr(void* corr)
{
    dbg<<"Start DestroyNNGCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr3<NData,NData,GData>*>(corr);
}

void DestroyKKGCorr(void* corr)
{
    dbg<<"Start DestroyKKGCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr3<KData,KData,GData>*>(corr);
}
#endif


void ProcessAutoNNN(void* corr, void* field, int dots, int coord, int metric)
{
    dbg<<"Start ProcessAutoNNN\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<NData,NData,NData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<NData,Flat>*>(field),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<NData,NData,NData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<NData,ThreeD>*>(field),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr3<NData,NData,NData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<NData,ThreeD>*>(field),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr3<NData,NData,NData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<NData,ThreeD>*>(field),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr3<NData,NData,NData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<NData,Sphere>*>(field),dots);
        else
            Assert(false);
    }
}

void ProcessAutoKKK(void* corr, void* field, int dots, int coord, int metric)
{
    dbg<<"Start ProcessAutoKKK\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<KData,KData,KData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<KData,Flat>*>(field),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<KData,KData,KData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<KData,ThreeD>*>(field),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr3<KData,KData,KData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<KData,ThreeD>*>(field),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr3<KData,KData,KData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<KData,ThreeD>*>(field),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr3<KData,KData,KData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<KData,Sphere>*>(field),dots);
        else
            Assert(false);
    }
}

void ProcessAutoGGG(void* corr, void* field, int dots, int coord, int metric)
{
    dbg<<"Start ProcessAutoGGG\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<GData,GData,GData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<GData,Flat>*>(field),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<GData,GData,GData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<GData,ThreeD>*>(field),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr3<GData,GData,GData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<GData,ThreeD>*>(field),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr3<GData,GData,GData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<GData,ThreeD>*>(field),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr3<GData,GData,GData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<GData,Sphere>*>(field),dots);
        else
            Assert(false);
    }
}

void ProcessCrossNNN(void* corr, void* field1, void* field2, void* field3, int dots,
                     int coord, int metric)
{
    dbg<<"Start ProcessCrossNNN\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<NData,NData,NData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<NData,Flat>*>(field1),
                *static_cast<Field<NData,Flat>*>(field2),
                *static_cast<Field<NData,Flat>*>(field3),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<NData,NData,NData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<NData,ThreeD>*>(field2),
                *static_cast<Field<NData,ThreeD>*>(field3),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr3<NData,NData,NData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<NData,ThreeD>*>(field2),
                *static_cast<Field<NData,ThreeD>*>(field3),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr3<NData,NData,NData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<NData,ThreeD>*>(field2),
                *static_cast<Field<NData,ThreeD>*>(field3),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr3<NData,NData,NData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<NData,Sphere>*>(field1),
                *static_cast<Field<NData,Sphere>*>(field2),
                *static_cast<Field<NData,Sphere>*>(field3),dots);
        else
            Assert(false);
    }
}

void ProcessCrossKKK(void* corr, void* field1, void* field2, void* field3, int dots,
                     int coord, int metric)
{
    dbg<<"Start ProcessCrossKKK\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<KData,KData,KData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<KData,Flat>*>(field1),
                *static_cast<Field<KData,Flat>*>(field2),
                *static_cast<Field<KData,Flat>*>(field3),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<KData,KData,KData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<KData,ThreeD>*>(field1),
                *static_cast<Field<KData,ThreeD>*>(field2),
                *static_cast<Field<KData,ThreeD>*>(field3),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr3<KData,KData,KData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<KData,ThreeD>*>(field1),
                *static_cast<Field<KData,ThreeD>*>(field2),
                *static_cast<Field<KData,ThreeD>*>(field3),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr3<KData,KData,KData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<KData,ThreeD>*>(field1),
                *static_cast<Field<KData,ThreeD>*>(field2),
                *static_cast<Field<KData,ThreeD>*>(field3),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr3<KData,KData,KData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<KData,Sphere>*>(field1),
                *static_cast<Field<KData,Sphere>*>(field2),
                *static_cast<Field<KData,Sphere>*>(field3),dots);
        else
            Assert(false);
    }
}

void ProcessCrossGGG(void* corr, void* field1, void* field2, void* field3, int dots,
                     int coord, int metric)
{
    dbg<<"Start ProcessCrossGGG\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<GData,GData,GData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<GData,Flat>*>(field1),
                *static_cast<Field<GData,Flat>*>(field2),
                *static_cast<Field<GData,Flat>*>(field3),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<GData,GData,GData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<GData,ThreeD>*>(field1),
                *static_cast<Field<GData,ThreeD>*>(field2),
                *static_cast<Field<GData,ThreeD>*>(field3),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr3<GData,GData,GData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<GData,ThreeD>*>(field1),
                *static_cast<Field<GData,ThreeD>*>(field2),
                *static_cast<Field<GData,ThreeD>*>(field3),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr3<GData,GData,GData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<GData,ThreeD>*>(field1),
                *static_cast<Field<GData,ThreeD>*>(field2),
                *static_cast<Field<GData,ThreeD>*>(field3),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr3<GData,GData,GData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<GData,Sphere>*>(field1),
                *static_cast<Field<GData,Sphere>*>(field2),
                *static_cast<Field<GData,Sphere>*>(field3),dots);
        else
            Assert(false);
    }
}

#if 0
void ProcessCrossNNK(void* corr, void* field1, void* field2, void* field3, int dots,
                     int coord, int metric)
{
    dbg<<"Start ProcessCrossNNK\n";
    if (coord == Flat) {
    if (metric == Euclidean)
        static_cast<BinnedCorr3<NData,NData,KData>*>(corr)->process<Flat,Euclidean>(
            *static_cast<Field<NData,Flat>*>(field1),
            *static_cast<Field<NData,Flat>*>(field2),
            *static_cast<Field<KData,Flat>*>(field3),dots);
    else
        Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<NData,NData,KData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<NData,ThreeD>*>(field2),
                *static_cast<Field<KData,ThreeD>*>(field3),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr3<NData,NData,KData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<NData,ThreeD>*>(field2),
                *static_cast<Field<KData,ThreeD>*>(field3),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr3<NData,NData,KData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<NData,ThreeD>*>(field2),
                *static_cast<Field<KData,ThreeD>*>(field3),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr3<NData,NData,KData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<NData,Sphere>*>(field1),
                *static_cast<Field<NData,Sphere>*>(field2),
                *static_cast<Field<KData,Sphere>*>(field3),dots);
        else
            Assert(false);
    }
}

void ProcessCrossNNG(void* corr, void* field1, void* field2, void* field3, int dots,
                     int coord, int metric)
{
    dbg<<"Start ProcessCrossNNG\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<NData,NData,GData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<NData,Flat>*>(field1),
                *static_cast<Field<NData,Flat>*>(field2),
                *static_cast<Field<GData,Flat>*>(field3),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<NData,NData,GData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<NData,ThreeD>*>(field2),
                *static_cast<Field<GData,ThreeD>*>(field3),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr3<NData,NData,GData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<NData,ThreeD>*>(field2),
                *static_cast<Field<GData,ThreeD>*>(field3),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr3<NData,NData,GData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<NData,ThreeD>*>(field1),
                *static_cast<Field<NData,ThreeD>*>(field2),
                *static_cast<Field<GData,ThreeD>*>(field3),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr3<NData,NData,GData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<NData,Sphere>*>(field1),
                *static_cast<Field<NData,Sphere>*>(field2),
                *static_cast<Field<GData,Sphere>*>(field3),dots);
        else
            Assert(false);
    }
}

void ProcessCrossKKG(void* corr, void* field1, void* field2, void* field3, int dots,
                     int coord, int metric)
{
    dbg<<"Start ProcessCrossKKG\n";
    if (coord == Flat) {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<KData,KData,GData>*>(corr)->process<Flat,Euclidean>(
                *static_cast<Field<KData,Flat>*>(field1),
                *static_cast<Field<KData,Flat>*>(field2),
                *static_cast<Field<GData,Flat>*>(field3),dots);
        else
            Assert(false);
    } else {
        if (metric == Euclidean)
            static_cast<BinnedCorr3<KData,KData,GData>*>(corr)->process<ThreeD,Euclidean>(
                *static_cast<Field<KData,ThreeD>*>(field1),
                *static_cast<Field<KData,ThreeD>*>(field2),
                *static_cast<Field<GData,ThreeD>*>(field3),dots);
        else if (metric == Perp)
            static_cast<BinnedCorr3<KData,KData,GData>*>(corr)->process<ThreeD,Perp>(
                *static_cast<Field<KData,ThreeD>*>(field1),
                *static_cast<Field<KData,ThreeD>*>(field2),
                *static_cast<Field<GData,ThreeD>*>(field3),dots);
        else if (metric == Lens)
            static_cast<BinnedCorr3<KData,KData,GData>*>(corr)->process<ThreeD,Lens>(
                *static_cast<Field<KData,ThreeD>*>(field1),
                *static_cast<Field<KData,ThreeD>*>(field2),
                *static_cast<Field<GData,ThreeD>*>(field3),dots);
        else if (metric == Arc)
            static_cast<BinnedCorr3<KData,KData,GData>*>(corr)->process<Sphere,Arc>(
                *static_cast<Field<KData,Sphere>*>(field1),
                *static_cast<Field<KData,Sphere>*>(field2),
                *static_cast<Field<GData,Sphere>*>(field3),dots);
        else
            Assert(false);
    }
}
#endif

