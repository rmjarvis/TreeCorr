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

#include "dbg.h"
#include "BinnedCorr3.h"
#include "Split.h"

#ifdef _OPENMP
#include "omp.h"
#endif

// Switch these for more time-consuming Assert statements
//#define XAssert(x) Assert(x)
#define XAssert(x)

template <typename T>
inline T SQR(T x) { return x * x; }

template <int DC1, int DC2, int DC3>
BinnedCorr3<DC1,DC2,DC3>::BinnedCorr3(
    double minsep, double maxsep, int nbins, double binsize, double b,
    double minu, double maxu, int nubins, double ubinsize, double bu,
    double minv, double maxv, int nvbins, double vbinsize, double bv,
    double* zeta0, double* zeta1, double* zeta2, double* zeta3,
    double* zeta4, double* zeta5, double* zeta6, double* zeta7,
    double* meanlogr, double* meanu, double* meanv, double* weight, double* ntri) :
    _minsep(minsep), _maxsep(maxsep), _nbins(nbins), _binsize(binsize), _b(b),
    _minu(minu), _maxu(maxu), _nubins(nubins), _ubinsize(ubinsize), _bu(bu),
    _minv(minv), _maxv(maxv), _nvbins(nvbins), _vbinsize(vbinsize), _bv(bv),
    _metric(-1), _owns_data(false),
    _zeta(zeta0,zeta1,zeta2,zeta3,zeta4,zeta5,zeta6,zeta7),
    _meanlogr(meanlogr), _meanu(meanu), _meanv(meanv), _weight(weight), _ntri(ntri)
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
    _bsq = _b * _b;
    _busq = _bu * _bu;
    _bvsq = _bv * _bv;
    _sqrttwobv = sqrt(2. * _bv);
    _nuv = _nubins * _nvbins;
    _ntot = _nbins * _nuv;
}

template <int DC1, int DC2, int DC3>
BinnedCorr3<DC1,DC2,DC3>::BinnedCorr3(const BinnedCorr3<DC1,DC2,DC3>& rhs, bool copy_data) :
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
    _bsq(rhs._bsq), _busq(rhs._busq), _bvsq(rhs._bvsq), _sqrttwobv(rhs._sqrttwobv),
    _metric(rhs._metric), _nuv(rhs._nuv), _ntot(rhs._ntot),
    _owns_data(true), _zeta(0,0,0,0,0,0,0,0), _weight(0)
{
    _zeta.new_data(_ntot);
    _meanlogr = new double[_ntot];
    _meanu = new double[_ntot];
    _meanv = new double[_ntot];
    if (rhs._weight) _weight = new double[_ntot];
    _ntri = new double[_ntot];

    if (copy_data) *this = rhs;
    else clear();
}

template <int DC1, int DC2, int DC3>
BinnedCorr3<DC1,DC2,DC3>::~BinnedCorr3()
{
    if (_owns_data) {
        _zeta.delete_data(_nbins);
        delete [] _meanlogr; _meanlogr = 0;
        delete [] _meanu; _meanu = 0;
        delete [] _meanv; _meanv = 0;
        if (_weight) delete [] _weight; _weight = 0;
        delete [] _ntri; _ntri = 0;
    }
}

template <int DC1, int DC2, int DC3>
void BinnedCorr3<DC1,DC2,DC3>::clear()
{
    _zeta.clear(_nbins);
    for (int i=0; i<_nbins; ++i) _meanlogr[i] = 0.;
    for (int i=0; i<_nbins; ++i) _meanu[i] = 0.;
    for (int i=0; i<_nbins; ++i) _meanv[i] = 0.;
    if (_weight) for (int i=0; i<_nbins; ++i) _weight[i] = 0.;
    for (int i=0; i<_nbins; ++i) _ntri[i] = 0.;
    _metric = -1;
}

// BinnedCorr3::process3 is invalid if DC1 != DC2 or DC3, so this helper struct lets us only call 
// process3, process21 and process111 when DC1 == DC2 == DC3
template <int DC1, int DC2, int DC3, int M>
struct ProcessHelper
{
    static void process3(BinnedCorr3<DC1,DC2,DC3>& , const Cell<DC1,M>* ) {}
    static void process21(BinnedCorr3<DC1,DC2,DC3>& , const Cell<DC1,M>*, const Cell<DC3,M>* ) {}
    static void process111(BinnedCorr3<DC1,DC2,DC3>& , const Cell<DC1,M>*, const Cell<DC2,M>*,
                           const Cell<DC3,M>* ) {}
};

template <int DC1, int DC3, int M>
struct ProcessHelper<DC1,DC1,DC3,M>
{
    static void process3(BinnedCorr3<DC1,DC1,DC3>& b, const Cell<DC1,M>* ) {}
    static void process21(BinnedCorr3<DC1,DC1,DC3>& b, const Cell<DC1,M>* , const Cell<DC3,M>* ) {}
    static void process111(BinnedCorr3<DC1,DC1,DC3>& b, const Cell<DC1,M>* , const Cell<DC1,M>*,
                           const Cell<DC3,M>* ) {}
};

template <int DC, int M>
struct ProcessHelper<DC,DC,DC,M>
{
    static void process3(BinnedCorr3<DC,DC,DC>& b, const Cell<DC,M>* c123) { b.process3(c123); }
    static void process21(BinnedCorr3<DC,DC,DC>& b, const Cell<DC,M>* c12, const Cell<DC,M>* c3) 
    { b.template process21<true>(c12,c3); }
    static void process111(BinnedCorr3<DC,DC,DC>& b, const Cell<DC,M>* c1, const Cell<DC,M>* c2,
                            const Cell<DC,M>* c3) 
    { b.template process111<true>(c1,c2,c3); }
};

template <int DC1, int DC2, int DC3> template <int M>
void BinnedCorr3<DC1,DC2,DC3>::process(const Field<DC1,M>& field, bool dots)
{
    Assert(DC1 == DC2);
    Assert(_metric == -1 || _metric == M);
    _metric = M;
    const long n1 = field.getNTopLevel();
    xdbg<<"field has "<<n1<<" top level nodes\n";
    if (dots) std::cout<<"Starting "<<n1<<" jobs:\n"<<std::flush;
    Assert(n1 > 0);
#ifdef _OPENMP
#pragma omp parallel 
    {
        // Give each thread their own copy of the data vector to fill in.
        BinnedCorr3<DC1,DC2,DC3> bc3(*this,false);
#else
        BinnedCorr3<DC1,DC2,DC3>& bc3 = *this;
#endif

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int i=0;i<n1;++i) {
#ifdef _OPENMP
#pragma omp critical
#endif
            const Cell<DC1,M>* c1 = field.getCells()[i];
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
            ProcessHelper<DC1,DC2,DC3,M>::process3(bc3,c1);
            for (int j=i+1;j<n1;++j) {
                const Cell<DC1,M>* c2 = field.getCells()[j];
                ProcessHelper<DC1,DC2,DC3,M>::process21(bc3,c1,c2);
                ProcessHelper<DC1,DC2,DC3,M>::process21(bc3,c2,c1);
                for (int k=j+1;k<n1;++k) {
                    const Cell<DC1,M>* c3 = field.getCells()[k];
                    ProcessHelper<DC1,DC2,DC3,M>::process111(bc3,c1,c2,c3);
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

template <int DC1, int DC2, int DC3> template <int M>
void BinnedCorr3<DC1,DC2,DC3>::process(const Field<DC1,M>& field1, const Field<DC2,M>& field2,
                                       const Field<DC3,M>& field3, bool dots)
{
    xdbg<<"_metric = "<<_metric<<std::endl;
    xdbg<<"M = "<<M<<std::endl;
    Assert(_metric == -1 || _metric == M);
    _metric = M;
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
            const Cell<DC1,M>* c1 = field1.getCells()[i];
            c1->WriteTree(*dbgout);
        }
        xdbg<<"field2: \n";
        for (int i=0;i<n2;++i) {
            xdbg<<"node "<<i<<std::endl;
            const Cell<DC2,M>* c2 = field2.getCells()[i];
            c2->WriteTree(*dbgout);
        }
        xdbg<<"field3: \n";
        for (int i=0;i<n3;++i) {
            xdbg<<"node "<<i<<std::endl;
            const Cell<DC3,M>* c3 = field3.getCells()[i];
            c3->WriteTree(*dbgout);
        }
    }
#endif

    if (dots) std::cout<<"Starting "<<n1<<" jobs:\n"<<std::flush;
#ifdef _OPENMP
#pragma omp parallel 
    {
        // Give each thread their own copy of the data vector to fill in.
        BinnedCorr3<DC1,DC2,DC3> bc3(*this,false);
#else
        BinnedCorr3<DC1,DC2,DC3>& bc3 = *this;
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
            const Cell<DC1,M>* c1 = field1.getCells()[i];
            for (int j=0;j<n2;++j) {
                const Cell<DC2,M>* c2 = field2.getCells()[j];
                for (int k=0;k<n3;++k) {
                    const Cell<DC3,M>* c3 = field3.getCells()[k];
                    bc3.template process111<false>(c1,c2,c3);
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
template <int DC1, int DC2, int DC3> template <int M>
void BinnedCorr3<DC1,DC2,DC3>::process3(const Cell<DC1,M>* c123)
{
    xdbg<<"Process3: c123 = "<<c123->getData().getPos()<<"  "<<"  "<<c123->getSize()<<"  "<<c123->getData().getN()<<std::endl;
    if (c123->getSize() < _halfminsep) {
        xdbg<<"    size < halfminsep.  return\n";
        return;
    }

    Assert(c123->getLeft());
    Assert(c123->getRight());
    process3(c123->getLeft()); 
    process3(c123->getRight()); 
    process21<true>(c123->getLeft(),c123->getRight()); 
    process21<true>(c123->getRight(),c123->getLeft()); 
}

// Does all triangles with two points in c12 and 3rd point in c3
// This version is allowed to swap the positions of points 1,2,3
template <int DC1, int DC2, int DC3> template <bool sort, int M>
void BinnedCorr3<DC1,DC2,DC3>::process21(const Cell<DC1,M>* c12, const Cell<DC3,M>* c3)
{
    xdbg<<"Process21: c12 = "<<c12->getData().getPos()<<"  "<<"  "<<c12->getSize()<<"  "<<c12->getData().getN()<<std::endl;
    xdbg<<"           c3  = "<<c3->getData().getPos()<<"  "<<"  "<<c3->getSize()<<"  "<<c3->getData().getN()<<std::endl;

    // Some trivial stoppers:
    if (c12->getSize() == 0.) {
        xdbg<<"    size == 0.  return\n";
        return;
    }
    if (c12->getSize() < _halfmind3) {
        xdbg<<"    size < halfminsep * umin.  return\n";
        return;
    }

    double d2sq = DistSq(c12->getData().getPos(), c3->getData().getPos());
    double s12 = c12->getSize();
    double s3 = c3->getSize();
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
    process21<true>(c12->getLeft(),c3);
    process21<true>(c12->getRight(),c3);
    process111<true>(c12->getLeft(),c12->getRight(),c3);
}

// A helper to calculate the distances and possibly sort the points.
// First the sort = false case.
template <int DC1, int DC2, int DC3, bool sort, int M>
struct SortHelper
{
    static void sort3(
        const Cell<DC1,M>*& c1, const Cell<DC2,M>*& c2, const Cell<DC3,M>*& c3,
        double& d1sq, double& d2sq, double& d3sq)
    {
        if (d1sq == 0.) d1sq = DistSq(c2->getData().getPos(), c3->getData().getPos());
        if (d2sq == 0.) d2sq = DistSq(c1->getData().getPos(), c3->getData().getPos());
        if (d3sq == 0.) d3sq = DistSq(c1->getData().getPos(), c2->getData().getPos());
    }
    static bool stop111(
        double d1sq, double d2sq, double d3sq, double d2,
        double s1, double s2, double s3,
        double minsep, double minsepsq, double maxsep, double maxsepsq,
        double minu, double minusq, double maxu, double maxusq,
        double minabsv, double maxabsv)
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
        // Thus, the minimum possible |v| from our triangle is (|d1-d2|-(s1+s2+s3)) / (d3+s1+s2).
        // Abort if (d1-d2-s1-s2-s3) / (d3+s1+s2) > maxabsv
        // (d1-d2-s1-s2-s3) > maxabsv * (d3+s1+s2)
        // d1 > maxabsv d3 + d2+s1+s2+s3 + maxabsv*(s1+s2)
        // Here, rather than using the lemma that (x+y)^2 < 2x^2 + 2y^2,
        // we can instead realize that d3 < d2, so just check if
        // d1 > maxabsv d2 + d2+s1+s2+s3 + maxabsv*(s1+s2)
        // The main advantage of this check is when d3 ~= d2 anyway, so this is effective.
        if (maxabsv < 1. && d1sq > SQR((1.+maxabsv)*d2 + sums + maxabsv * (s1+s2))) {
            xdbg<<"v cannot be as small as maxabsv\n";
            return true;
        }

        return false;
    }
};

// This one has sort = true, so the points get sorted, and always returns true.
template <int DC, int M>
struct SortHelper<DC,DC,DC,true,M>
{
    static void sort3(
        const Cell<DC,M>*& c1, const Cell<DC,M>*& c2, const Cell<DC,M>*& c3,
        double& d1sq, double& d2sq, double& d3sq)
    {
        if (d1sq == 0.) d1sq = DistSq(c2->getData().getPos(), c3->getData().getPos());
        if (d2sq == 0.) d2sq = DistSq(c1->getData().getPos(), c3->getData().getPos());
        if (d3sq == 0.) d3sq = DistSq(c1->getData().getPos(), c2->getData().getPos());

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
        double minabsv, double maxabsv)
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

        return false;
    }
};

// Does all triangles with 1 point each in c1, c2, c3
template <int DC1, int DC2, int DC3> template <bool sort, int M>
void BinnedCorr3<DC1,DC2,DC3>::process111(
    const Cell<DC1,M>* c1, const Cell<DC2,M>* c2, const Cell<DC3,M>* c3,
    double d1sq, double d2sq, double d3sq)
{
    // Calculate the distances if they aren't known yet, and sort so that d3 < d2 < d1
    SortHelper<DC1,DC2,DC3,sort,M>::sort3(c1,c2,c3,d1sq,d2sq,d3sq);

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

    if (SortHelper<DC1,DC2,DC3,sort,M>::stop111(d1sq,d2sq,d3sq,d2,s1,s2,s3,
                                                _minsep,_minsepsq,_maxsep,_maxsepsq,
                                                _minu,_minusq,_maxu,_maxusq,_minabsv,_maxabsv)) 
        return;

    // For 1,3 decide whether to split on the noraml criteria with s1+s3/d2 < b
    bool split1 = false, split3 = false;
    CalcSplitSq(split1,split3,*c1,*c3,d2sq,s1+s3,_bsq);

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
    CalcSplitSq(split1,split2,*c1,*c2,d2sq,s1+s2,_busq);

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
    double d1,d3,v,onemvsq; 

    if (!(split1 && split2 && split3)) {
        d1 = sqrt(d1sq);
        d3 = sqrt(d3sq);
        v = (d1-d2)/d3;
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
        CalcSplitSq(split1,split2,*c1,*c2,d3sq,s1+s2,_bvsq * 3./16. / onemvsq);
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
                process111<sort>(c1->getLeft(),c2->getLeft(),c3->getLeft());
                process111<sort>(c1->getLeft(),c2->getLeft(),c3->getRight());
                process111<sort>(c1->getLeft(),c2->getRight(),c3->getLeft());
                process111<sort>(c1->getLeft(),c2->getRight(),c3->getRight());
                process111<sort>(c1->getRight(),c2->getLeft(),c3->getLeft());
                process111<sort>(c1->getRight(),c2->getLeft(),c3->getRight());
                process111<sort>(c1->getRight(),c2->getRight(),c3->getLeft());
                process111<sort>(c1->getRight(),c2->getRight(),c3->getRight());
             } else {
                // split 1,2
                Assert(c1->getLeft());
                Assert(c1->getRight());
                Assert(c2->getLeft());
                Assert(c2->getRight());
                process111<sort>(c1->getLeft(),c2->getLeft(),c3);
                process111<sort>(c1->getLeft(),c2->getRight(),c3);
                process111<sort>(c1->getRight(),c2->getLeft(),c3);
                process111<sort>(c1->getRight(),c2->getRight(),c3);
            }
        } else {
            if (split3) {
                // split 1,3
                Assert(c1->getLeft());
                Assert(c1->getRight());
                Assert(c3->getLeft());
                Assert(c3->getRight());
                process111<sort>(c1->getLeft(),c2,c3->getLeft());
                process111<sort>(c1->getLeft(),c2,c3->getRight());
                process111<sort>(c1->getRight(),c2,c3->getLeft());
                process111<sort>(c1->getRight(),c2,c3->getRight());
            } else {
                // split 1 only
                Assert(c1->getLeft());
                Assert(c1->getRight());
                process111<sort>(c1->getLeft(),c2,c3,d1sq);
                process111<sort>(c1->getRight(),c2,c3,d1sq);
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
                process111<sort>(c1,c2->getLeft(),c3->getLeft());
                process111<sort>(c1,c2->getLeft(),c3->getRight());
                process111<sort>(c1,c2->getRight(),c3->getLeft());
                process111<sort>(c1,c2->getRight(),c3->getRight());
            } else {
                // split 2 only
                Assert(c2->getLeft());
                Assert(c2->getRight());
                process111<sort>(c1,c2->getLeft(),c3,0.,d2sq);
                process111<sort>(c1,c2->getRight(),c3,0.,d2sq);
            }
        } else {
            if (split3) {
                // split 3 only
                Assert(c3->getLeft());
                Assert(c3->getRight());
                process111<sort>(c1,c2,c3->getLeft(),0.,0.,d3sq);
                process111<sort>(c1,c2,c3->getRight(),0.,0.,d3sq);
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

                if (!CCW(c1->getData().getPos(), c2->getData().getPos(), c3->getData().getPos()))
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
                directProcess111(*c1,*c2,*c3,d1,d2,d3,logr,u,v,index);
            }
        }
    }
}

// We also set up a helper class for doing the direct processing
template <int DC1, int DC2, int DC3>
struct DirectHelper;

template <>
struct DirectHelper<NData,NData,NData>
{
    template <int M>
    static void ProcessZeta(
        const Cell<NData,M>& , const Cell<NData,M>& , const Cell<NData,M>&, 
        const double , const double , const double ,
        ZetaData<NData,NData,NData>& , int )
    {}
};
 
#if 0
template <>
struct DirectHelper<NData,KData>
{
    template <int M>
    static void ProcessZeta(
        const Cell<NData,M>& c1, const Cell<KData,M>& c2,
        const double d1, const double d2, const double d3,
        ZetaData<NData,KData>& zeta, int index)
    { zeta.zeta[index] += c1.getData().getW() * c2.getData().getWK(); }
};
 
template <>
struct DirectHelper<NData,GData>
{
    template <int M>
    static void ProcessZeta(
        const Cell<NData,M>& c1, const Cell<GData,M>& c2,
        const double d1, const double d2, const double d3,
        ZetaData<NData,GData>& zeta, int index)
    {
        std::complex<double> g2;
        MetricHelper<M>::ProjectShear(c1,c2,dsq,g2);
        // The minus sign here is to make it accumulate tangential shear, rather than radial.
        // g2 from the above ProjectShear is measured along the connecting line, not tangent.
        g2 *= -c1.getData().getW();
        zeta.zeta[index] += real(g2);
        zeta.zeta_im[index] += imag(g2);

    }
};

template <>
struct DirectHelper<KData,KData>
{
    template <int M>
    static void ProcessZeta(
        const Cell<KData,M>& c1, const Cell<KData,M>& c2,
        const double d1, const double d2, const double d3,
        ZetaData<KData,KData>& zeta, int index)
    { zeta.zeta[index] += c1.getData().getWK() * c2.getData().getWK(); }
};
 
template <>
struct DirectHelper<KData,GData>
{
    template <int M>
    static void ProcessZeta(
        const Cell<KData,M>& c1, const Cell<GData,M>& c2,
        const double d1, const double d2, const double d3,
        ZetaData<KData,GData>& zeta, int index)
    {
        std::complex<double> g2;
        MetricHelper<M>::ProjectShear(c1,c2,dsq,g2);
        // The minus sign here is to make it accumulate tangential shear, rather than radial.
        // g2 from the above ProjectShear is measured along the connecting line, not tangent.
        g2 *= -c1.getData().getWK();
        zeta.zeta[index] += real(g2);
        zeta.zeta_im[index] += imag(g2);
    }
};
 
template <>
struct DirectHelper<GData,GData>
{
    template <int M>
    static void ProcessZeta(
        const Cell<GData,M>& c1, const Cell<GData,M>& c2,
        const double d1, const double d2, const double d3,
        ZetaData<GData,GData>& zeta, int index)
    {
        std::complex<double> g1, g2;
        MetricHelper<M>::ProjectShears(c1,c2,dsq,g1,g2);

        // The complex products g1 g2 and g1 g2* share most of the calculations,
        // so faster to do this manually.
        double g1rg2r = g1.real() * g2.real();
        double g1rg2i = g1.real() * g2.imag();
        double g1ig2r = g1.imag() * g2.real();
        double g1ig2i = g1.imag() * g2.imag();

        zeta.zetap[index] += g1rg2r + g1ig2i;       // g1 * conj(g2)
        zeta.zetap_im[index] += g1ig2r - g1rg2i;
        zeta.zetam[index] += g1rg2r - g1ig2i;       // g1 * g2
        zeta.zetam_im[index] += g1ig2r + g1rg2i;
    }
};
#endif

// The way meanlogr and weight are processed is the same for everything except NN.
// So do this as a separate template specialization:
template <int DC1, int DC2, int DC3>
struct DirectHelper2
{
    template <int M>
    static void ProcessWeight(
        const Cell<DC1,M>& c1, const Cell<DC2,M>& c2, const Cell<DC3,M>& c3,
        const double logr, const double u, const double v, const double ,
        double* meanlogr, double* meanu, double* meanv, double* weight, int index)
    {
        double www = double(c1.getData().getW()) * double(c2.getData().getW()) *
            double(c3.getData().getW());
        meanlogr[index] += www * logr;
        meanu[index] += www * u;
        meanv[index] += www * v;
        weight[index] += www;
    }
};
            
template <>
struct DirectHelper2<NData, NData, NData>
{
    template <int M>
    static void ProcessWeight(
        const Cell<NData,M>& , const Cell<NData,M>& , const Cell<NData,M>& ,
        const double logr, const double u, const double v, const double nnn,
        double* meanlogr, double* meanu, double* meanv, double* , int index)
    { 
        meanlogr[index] += nnn * logr; 
        meanu[index] += nnn * u; 
        meanv[index] += nnn * v; 
    }
};

template <int DC1, int DC2, int DC3> template <int M>
void BinnedCorr3<DC1,DC2,DC3>::directProcess111(
    const Cell<DC1,M>& c1, const Cell<DC2,M>& c2, const Cell<DC3,M>& c3,
    const double d1, const double d2, const double d3,
    const double logr, const double u, const double v, const int index)
{
    double nnn = double(c1.getData().getN()) * double(c2.getData().getN()) *
        double(c3.getData().getN());
    _ntri[index] += nnn;
    xdbg<<"            index = "<<index<<std::endl;
    xdbg<<"            nnn = "<<nnn<<std::endl;

    DirectHelper<DC1,DC2,DC3>::ProcessZeta(c1,c2,c3,d1,d2,d3,_zeta,index);

    DirectHelper2<DC1,DC2,DC3>::ProcessWeight(c1,c2,c3,logr,u,v,nnn,
                                              _meanlogr,_meanu,_meanv,_weight,index);
}

template <int DC1, int DC2, int DC3>
void BinnedCorr3<DC1,DC2,DC3>::operator=(const BinnedCorr3<DC1,DC2,DC3>& rhs)
{
    Assert(rhs._ntot == _ntot);
    _zeta.copy(rhs._zeta,_ntot);
    for (int i=0; i<_ntot; ++i) _meanlogr[i] = rhs._meanlogr[i];
    for (int i=0; i<_ntot; ++i) _meanu[i] = rhs._meanu[i];
    for (int i=0; i<_ntot; ++i) _meanv[i] = rhs._meanv[i];
    if (_weight) for (int i=0; i<_ntot; ++i) _weight[i] = rhs._weight[i];
    for (int i=0; i<_ntot; ++i) _ntri[i] = rhs._ntri[i];
}

template <int DC1, int DC2, int DC3>
void BinnedCorr3<DC1,DC2,DC3>::operator+=(const BinnedCorr3<DC1,DC2,DC3>& rhs)
{
    Assert(rhs._ntot == _ntot);
    _zeta.add(rhs._zeta,_ntot);
    for (int i=0; i<_ntot; ++i) _meanlogr[i] += rhs._meanlogr[i];
    for (int i=0; i<_ntot; ++i) _meanu[i] += rhs._meanu[i];
    for (int i=0; i<_ntot; ++i) _meanv[i] += rhs._meanv[i];
    if (_weight) for (int i=0; i<_ntot; ++i) _weight[i] += rhs._weight[i];
    for (int i=0; i<_ntot; ++i) _ntri[i] += rhs._ntri[i];
}

//
//
// The C interface for python
//
//

void* BuildNNNCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                   double minu, double maxu, int nubins, double ubinsize, double bu,
                   double minv, double maxv, int nvbins, double vbinsize, double bv,
                   double* meanlogr, double* meanu, double* meanv, double* ntri)
{
    dbg<<"Start BuildNNCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr3<NData,NData,NData>(
            minsep, maxsep, nbins, binsize, b,
            minu, maxu, nubins, ubinsize, bu,
            minv, maxv, nvbins, vbinsize, bv,
            0, 0, 0, 0, 0, 0, 0, 0,
            meanlogr, meanu, meanv, 0, ntri));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

#if 0
void* BuildNKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* zeta,
                  double* meanlogr, double* weight, double* ntri)
{
    dbg<<"Start BuildNKCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr3<NData,KData>(
            minsep, maxsep, nbins, binsize, b,
            zeta, 0, 0, 0,
            meanlogr, weight, ntri));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildNGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* zeta, double* zeta_im,
                  double* meanlogr, double* weight, double* ntri)
{
    dbg<<"Start BuildNGCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr3<NData,GData>(
            minsep, maxsep, nbins, binsize, b,
            zeta, zeta_im, 0, 0,
            meanlogr, weight, ntri));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildKKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* zeta,
                  double* meanlogr, double* weight, double* ntri)
{
    dbg<<"Start BuildKKCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr3<KData,KData>(
            minsep, maxsep, nbins, binsize, b,
            zeta, 0, 0, 0,
            meanlogr, weight, ntri));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildKGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* zeta, double* zeta_im,
                  double* meanlogr, double* weight, double* ntri)
{
    dbg<<"Start BuildKGCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr3<KData,GData>(
            minsep, maxsep, nbins, binsize, b,
            zeta, zeta_im, 0, 0,
            meanlogr, weight, ntri));
    xdbg<<"corr = "<<corr<<std::endl;
    return corr;
}

void* BuildGGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                  double* zetap, double* zetap_im, double* zetam, double* zetam_im,
                  double* meanlogr, double* weight, double* ntri)
{
    dbg<<"Start BuildGGCorr\n";
    void* corr = static_cast<void*>(new BinnedCorr3<GData,GData>(
            minsep, maxsep, nbins, binsize, b,
            zetap, zetap_im, zetam, zetam_im,
            meanlogr, weight, ntri));
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

#if 0
void DestroyNKCorr(void* corr)
{
    dbg<<"Start DestroyNKCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr3<NData,KData>*>(corr);
}

void DestroyNGCorr(void* corr)
{
    dbg<<"Start DestroyNGCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr3<NData,GData>*>(corr);
}

void DestroyKKCorr(void* corr)
{
    dbg<<"Start DestroyKKCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr3<KData,KData>*>(corr);
}

void DestroyKGCorr(void* corr)
{
    dbg<<"Start DestroyKGCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr3<KData,GData>*>(corr);
}

void DestroyGGCorr(void* corr)
{
    dbg<<"Start DestroyGGCorr\n";
    xdbg<<"corr = "<<corr<<std::endl;
    delete static_cast<BinnedCorr3<GData,GData>*>(corr);
}
#endif


void ProcessAutoNNNFlat(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoNNNFlat\n";
    static_cast<BinnedCorr3<NData,NData,NData>*>(corr)->process(
        *static_cast<Field<NData,Flat>*>(field),dots);
}

void ProcessAutoNNNSphere(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoNNNSphere\n";
    static_cast<BinnedCorr3<NData,NData,NData>*>(corr)->process(
        *static_cast<Field<NData,Sphere>*>(field),dots);
}

#if 0
void ProcessAutoKKFlat(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoKKFlat\n";
    static_cast<BinnedCorr3<KData,KData>*>(corr)->process(
        *static_cast<Field<KData,Flat>*>(field),dots);
}

void ProcessAutoKKSphere(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoKKSphere\n";
    static_cast<BinnedCorr3<KData,KData>*>(corr)->process(
        *static_cast<Field<KData,Sphere>*>(field),dots);
}

void ProcessAutoGGFlat(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoGGFlat\n";
    static_cast<BinnedCorr3<GData,GData>*>(corr)->process(
        *static_cast<Field<GData,Flat>*>(field),dots);
}

void ProcessAutoGGSphere(void* corr, void* field, int dots)
{
    dbg<<"Start ProcessAutoGGSphere\n";
    static_cast<BinnedCorr3<GData,GData>*>(corr)->process(
        *static_cast<Field<GData,Sphere>*>(field),dots);
}
#endif

void ProcessCrossNNNFlat(void* corr, void* field1, void* field2, void* field3, int dots)
{
    dbg<<"Start ProcessCrossNNNFlat\n";
    static_cast<BinnedCorr3<NData,NData,NData>*>(corr)->process(
        *static_cast<Field<NData,Flat>*>(field1),
        *static_cast<Field<NData,Flat>*>(field2),
        *static_cast<Field<NData,Flat>*>(field3),dots);
}

void ProcessCrossNNNSphere(void* corr, void* field1, void* field2, void* field3, int dots)
{
    dbg<<"Start ProcessCrossNNNSphere\n";
    static_cast<BinnedCorr3<NData,NData,NData>*>(corr)->process(
        *static_cast<Field<NData,Sphere>*>(field1),
        *static_cast<Field<NData,Sphere>*>(field2),
        *static_cast<Field<NData,Sphere>*>(field3),dots);
}

#if 0
void ProcessCrossNKFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNKFlat\n";
    static_cast<BinnedCorr3<NData,KData>*>(corr)->process(
        *static_cast<Field<NData,Flat>*>(field1),
        *static_cast<Field<KData,Flat>*>(field2),dots);
}

void ProcessCrossNKSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNKSphere\n";
    static_cast<BinnedCorr3<NData,KData>*>(corr)->process(
        *static_cast<Field<NData,Sphere>*>(field1),
        *static_cast<Field<KData,Sphere>*>(field2),dots);
}

void ProcessCrossNGFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNGFlat\n";
    static_cast<BinnedCorr3<NData,GData>*>(corr)->process(
        *static_cast<Field<NData,Flat>*>(field1),
        *static_cast<Field<GData,Flat>*>(field2),dots);
}

void ProcessCrossNGSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossNGSphere\n";
    static_cast<BinnedCorr3<NData,GData>*>(corr)->process(
        *static_cast<Field<NData,Sphere>*>(field1),
        *static_cast<Field<GData,Sphere>*>(field2),dots);
}

void ProcessCrossKKFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossKKFlat\n";
    static_cast<BinnedCorr3<KData,KData>*>(corr)->process(
        *static_cast<Field<KData,Flat>*>(field1),
        *static_cast<Field<KData,Flat>*>(field2),dots);
}

void ProcessCrossKKSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossKKSphere\n";
    static_cast<BinnedCorr3<KData,KData>*>(corr)->process(
        *static_cast<Field<KData,Sphere>*>(field1),
        *static_cast<Field<KData,Sphere>*>(field2),dots);
}

void ProcessCrossKGFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossKGFlat\n";
    static_cast<BinnedCorr3<KData,GData>*>(corr)->process(
        *static_cast<Field<KData,Flat>*>(field1),
        *static_cast<Field<GData,Flat>*>(field2),dots);
}

void ProcessCrossKGSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossKGSphere\n";
    static_cast<BinnedCorr3<KData,GData>*>(corr)->process(
        *static_cast<Field<KData,Sphere>*>(field1),
        *static_cast<Field<GData,Sphere>*>(field2),dots);
}

void ProcessCrossGGFlat(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossGGFlat\n";
    static_cast<BinnedCorr3<GData,GData>*>(corr)->process(
        *static_cast<Field<GData,Flat>*>(field1),
        *static_cast<Field<GData,Flat>*>(field2),dots);
}

void ProcessCrossGGSphere(void* corr, void* field1, void* field2, int dots)
{
    dbg<<"Start ProcessCrossGGSphere\n";
    static_cast<BinnedCorr3<GData,GData>*>(corr)->process(
        *static_cast<Field<GData,Sphere>*>(field1),
        *static_cast<Field<GData,Sphere>*>(field2),dots);
}
#endif

