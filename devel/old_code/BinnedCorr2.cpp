
#include "dbg.h"
#include "BinnedCorr2.h"
#include "CorrIO.h"

#ifdef _OPENMP
#include "omp.h"
#endif

// Switch these for more time-consuming Assert statements
//#define XAssert(x) Assert(x)
#define XAssert(x)

template <typename T>
inline T SQR(T x) { return x * x; }

template <int DC1, int DC2>
BinnedCorr2<DC1,DC2>::BinnedCorr2(const ConfigFile& params) : _params(params), _metric(-1)
{
    // Read sep_units as a string.  GetUnits is defined in InputFile.cpp
    _sepunits = GetUnits(_params,"sep_units",0,
                         params.keyExists("ra_col") ? "" : "arcsec");
    xdbg<<"sepunits = "<<_sepunits<<std::endl;

    // 3 of min_sep, max_sep, nbins, bin_size are required.
    int nbins;
    if (!params.keyExists("nbins")) {
        // Then min_sep, max_sep, and bin_size are all required.
        if (!params.keyExists("min_sep")) myerror("Missing required parameter 'min_sep'");
        if (!params.keyExists("max_sep")) myerror("Missing required parameter 'max_sep'");
        if (!params.keyExists("bin_size")) myerror("Missing required parameter 'bin_size'");
        _minsep = double(params["min_sep"]) * _sepunits;
        _maxsep = double(params["max_sep"]) * _sepunits;
        _binsize = params["bin_size"];
        Assert(_minsep != 0.);
        Assert(_binsize != 0.);
        nbins = int(ceil(log(_maxsep/_minsep)/_binsize));
    } else if (!params.keyExists("bin_size")) {
        // Then min_sep, max_sep, and nbins are all required.
        if (!params.keyExists("min_sep")) myerror("Missing required parameter 'min_sep'");
        if (!params.keyExists("max_sep")) myerror("Missing required parameter 'max_sep'");
        _minsep = double(params["min_sep"]) * _sepunits;
        _maxsep = double(params["max_sep"]) * _sepunits;
        nbins = params["nbins"];
        Assert(_minsep != 0.);
        Assert(nbins != 0);
        _binsize = log(_maxsep/_minsep)/nbins;
    } else if (!params.keyExists("max_sep")) {
        // Then min_sep, nbins, and bin_size are all required.
        if (!params.keyExists("min_sep")) myerror("Missing required parameter 'min_sep'");
        _minsep = double(params["min_sep"]) * _sepunits;
        nbins = params["nbins"];
        _binsize = params["bin_size"];
        _maxsep = exp(nbins*_binsize) * _minsep;
    } else {
        // Then min_sep should not be specified.
        if (params.keyExists("min_sep")) 
            myerror("Only 3 of 'min_sep', 'max_sep', 'nbins', and 'bin_size should be "
                    "provided.");
        _maxsep = double(params["max_sep"]) * _sepunits;
        nbins = params["nbins"];
        _binsize = params["bin_size"];
        _minsep = _maxsep / exp(nbins*_binsize);
    }
    dbg<<"nbins = "<<nbins<<": min,maxsep = "<<_minsep<<','<<_maxsep<<
        "  binsize = "<<_binsize<<std::endl;

    // Some helpful variables we can calculate once here.
    _logminsep = log(_minsep);
    _halfminsep = 0.5*_minsep;
    _minsepsq = _minsep*_minsep;
    _maxsepsq = _maxsep*_maxsep;
    //xxdbg<<"logminsep = "<<_logminsep<<std::endl;
    //xxdbg<<"halfminsep = "<<_halfminsep<<std::endl;
    //xxdbg<<"minsepsq = "<<_minsepsq<<std::endl;
    //xxdbg<<"maxsepsq = "<<_maxsepsq<<std::endl;

    // Calculate b based on bin_slop parameter.  (Default = 1)
    double binslop = params.read("bin_slop",1.0);
    _b = binslop * _binsize;
    _bsq = _b * _b;
    xdbg<<"binslop = "<<binslop<<", b = "<<_b<<", bsq = "<<_bsq<<std::endl;

    // Make _data the right size.
    _data.resize(nbins);
}

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::clearData()
{
    for (size_t i=0;i<_data.size();++i) _data[i].clear();
}

// BinnedCorr2::process2 is invalid if DC1 != DC2, so this helper struct lets us only call 
// process2 when DC1 == DC2.
template <int DC1, int DC2>
struct ProcessHelper
{
    template <int M>
    static void process2(BinnedCorr2<DC1,DC2>& , const Cell<DC1,M>& ) {}
    static void callProcess2(BinnedCorr2<DC1,DC2>&, const Field<DC1>& , int ) {}
    static void callProcess11(BinnedCorr2<DC1,DC2>&,
                              const Field<DC1>& , const Field<DC1>& , int , int ) {}
};

    
template <int DC>
struct ProcessHelper<DC,DC>
{
    template <int M>
    static void process2(BinnedCorr2<DC,DC>& b, const Cell<DC,M>& c12) { b.process2(c12); }
    static void callProcess2(BinnedCorr2<DC,DC>& b, const Field<DC>& field, int i) 
    { b.callProcess2(field,i); }
    static void callProcess11(BinnedCorr2<DC,DC>& b,
                              const Field<DC>& field1, const Field<DC>& field2, int i, int j)
    { b.callProcess11(field1,field2,i,j); }
};

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::process(const Field<DC1>& field)
{
    Assert(DC1 == DC2);
    dbg<<"Starting process for 1 field: "<<field.getFileName()<<std::endl;
    const int n1 = field.getN();
    xdbg<<"field has "<<n1<<" top level nodes\n";
    Assert(n1 > 0);
#ifdef _OPENMP
#pragma omp parallel 
    {
        // Give each thread their own copy of the data vector to fill in.
        BinnedCorr2<DC1,DC2> bc2(*this);
        bc2.clearData();
#else
        BinnedCorr2<DC1,DC2>& bc2 = *this;
#endif

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int i=0;i<n1;++i) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                //dbg<<omp_get_thread_num()<<" "<<i<<std::endl;
                dbg<<'.';
            }
            ProcessHelper<DC1,DC2>::callProcess2(bc2, field, i);
            for (int j=i+1;j<n1;++j) 
                ProcessHelper<DC1,DC2>::callProcess11(bc2, field, field, i, j);
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            Assert(_metric == -1 || bc2._metric == -1 || _metric == bc2._metric);
            if (bc2._metric != -1) _metric = bc2._metric;
            *this += bc2;
        }
    }
#endif
    dbg<<std::endl;
}

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::process(const Field<DC1>& field1, const Field<DC2>& field2)
{
    dbg<<"Starting process for 2 fields: "<<
        field1.getFileName()<<"  "<<field2.getFileName()<<std::endl;
    const int n1 = field1.getN();
    const int n2 = field2.getN();
    xdbg<<"field1 has "<<n1<<" top level nodes\n";
    xdbg<<"field2 has "<<n2<<" top level nodes\n";
    Assert(n1 > 0);
    Assert(n2 > 0);

#ifdef _OPENMP
#pragma omp parallel 
    {
        // Give each thread their own copy of the data vector to fill in.
        BinnedCorr2<DC1,DC2> bc2(*this);
        bc2.clearData();
#else
        BinnedCorr2<DC1,DC2>& bc2 = *this;
#endif

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int i=0;i<n1;++i) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                //dbg<<omp_get_thread_num()<<" "<<i<<std::endl;
                dbg<<'.';
            }
            for (int j=0;j<n2;++j) bc2.callProcess11(field1, field2, i, j);
        }
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            Assert(_metric == -1 || bc2._metric == -1 || _metric == bc2._metric);
            if (bc2._metric != -1) _metric = bc2._metric;
            *this += bc2;
        }
    }
#endif
    dbg<<std::endl;
}

template <int DC1, int DC2> template <int M>
void BinnedCorr2<DC1,DC2>::doProcessPairwise(const InputFile& file1, const InputFile& file2)
{
    std::vector<CellData<DC1,M>*> celldata1;
    std::vector<CellData<DC2,M>*> celldata2;
    Field<DC1>::BuildCellData(file1,celldata1);
    Field<DC2>::BuildCellData(file2,celldata2);

    _metric = M;

    const int n = celldata1.size();
    const int sqrtn = int(sqrt(double(n)));

#ifdef _OPENMP
#pragma omp parallel 
    {
        // Give each thread their own copy of the data vector to fill in.
        BinnedCorr2<DC1,DC2> bc2(*this);
        bc2.clearData();
#else
        BinnedCorr2<DC1,DC2>& bc2 = *this;
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int i=0;i<n;++i) {
            // Let the progress dots happen every sqrt(n) iterations.
            if (i % sqrtn == 0) {
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    //xdbg<<omp_get_thread_num()<<" "<<i<<std::endl;
                    dbg<<'.';
                }
            }
            // Note: This transfers ownership of the pointer to the Cell,
            // and the data is deleted when the Cell goes out of scope.
            // TODO: I really should switch to using shared_ptr, so these
            // memory issues are more seamless...
            Cell<DC1,M> c1(celldata1[i]);
            Cell<DC2,M> c2(celldata2[i]);
            const double dsq = DistSq(c1.getData().getPos(),c2.getData().getPos());
            if (dsq >= _minsepsq && dsq < _maxsepsq) {
                bc2.directProcess11(c1,c2,dsq);
            }
        }
        dbg<<std::endl;
#ifdef _OPENMP
        // Accumulate the results
#pragma omp critical
        {
            *this += bc2;
        }
    }
#endif
}

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::processPairwise(const InputFile& file1, const InputFile& file2)
{
    dbg<<"Starting processPairwise for 2 files: "<<
        file1.getFileName()<<"  "<<file2.getFileName()<<std::endl;
    const int n = file1.getNTot();
    const int n2 = file2.getNTot();
    Assert(n > 0);
    Assert(n == n2);
    xdbg<<"files have "<<n<<" objects\n";

    if (file1.useRaDec()) {
        Assert(file2.useRaDec());
        doProcessPairwise<Sphere>(file1,file2);
    } else {
        Assert(!file2.useRaDec());
        doProcessPairwise<Flat>(file1,file2);
    }
}

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::callProcess2(const Field<DC1>& field, int i)
{
    if (field.getMetric() == Sphere) {
        Assert(_metric == -1 || _metric == Sphere);
        _metric = Sphere;
        const Cell<DC1,Sphere>& c1 = *field.getCells_Sphere()[i];
        if (XDEBUG) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
#ifdef _OPENMP
                xxdbg<<"Thread: "<<omp_get_thread_num()<<" ";
#endif
                xxdbg<<"Start P2: size = "<<c1.getSize();
                xxdbg<<", center = "<<c1.getData().getPos()<<"   N = "<<c1.getData().getN()<<std::endl;
            }
        }
        ProcessHelper<DC1,DC2>::process2(*this,c1);
    } else {
        Assert(_metric == -1 || _metric == Flat);
        _metric = Flat;
        const Cell<DC1,Flat>& c1 = *field.getCells_Flat()[i];
        if (XDEBUG) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
#ifdef _OPENMP
                xxdbg<<"Thread: "<<omp_get_thread_num()<<" ";
#endif
                xxdbg<<"Start P2: size = "<<c1.getSize();
                xxdbg<<", center = "<<c1.getData().getPos()<<"   N = "<<c1.getData().getN()<<std::endl;
            }
        }
        ProcessHelper<DC1,DC2>::process2(*this,c1);
    }
}

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::callProcess11(
    const Field<DC1>& field1, const Field<DC2>& field2, int i, int j)
{
    Assert(field1.getMetric() == field2.getMetric());
    if (field1.getMetric() == Sphere) {
        Assert(_metric == -1 || _metric == Sphere);
        _metric = Sphere;
        const Cell<DC1,Sphere>& c1 = *field1.getCells_Sphere()[i];
        const Cell<DC2,Sphere>& c2 = *field2.getCells_Sphere()[j];
        if (XDEBUG) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
#ifdef _OPENMP
                xxdbg<<"Thread: "<<omp_get_thread_num()<<" ";
#endif
                xxdbg<< "Start P11: "<<c1.getData().getPos()<<" -- "<<c2.getData().getPos();
                xxdbg<< "   N = "<<c1.getData().getN()<<","<<c2.getData().getN()<<std::endl;
            }
        }
        process11(c1,c2);
    } else {
        Assert(_metric == -1 || _metric == Flat);
        _metric = Flat;
        const Cell<DC1,Flat>& c1 = *field1.getCells_Flat()[i];
        const Cell<DC2,Flat>& c2 = *field2.getCells_Flat()[j];
        if (XDEBUG) {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
#ifdef _OPENMP
                xxdbg<<"Thread: "<<omp_get_thread_num()<<" ";
#endif
                xxdbg<< "Start P11: "<<c1.getData().getPos()<<" -- "<<c2.getData().getPos();
                xxdbg<< "   N = "<<c1.getData().getN()<<","<<c2.getData().getN()<<std::endl;
            }
        }
        process11(c1,c2);
    }
}

template <int DC1, int DC2> template <int M>
void BinnedCorr2<DC1,DC2>::process2(const Cell<DC1,M>& c12)
{
    if (c12.getSize() < _halfminsep) return;

    Assert(c12.getLeft());
    Assert(c12.getRight());
    process2(*c12.getLeft());
    process2(*c12.getRight());
    process11(*c12.getLeft(),*c12.getRight());
}

template <int DC1, int DC2> template <int M>
void BinnedCorr2<DC1,DC2>::process11(const Cell<DC1,M>& c1, const Cell<DC2,M>& c2)
{
    const double dsq = DistSq(c1.getData().getPos(),c2.getData().getPos());
    const double s1ps2 = c1.getAllSize()+c2.getAllSize();

    if (dsq < _minsepsq && s1ps2 < _minsep && dsq < SQR(_minsep - s1ps2)) return;
    if (dsq >= _maxsepsq && dsq >= SQR(_maxsep + s1ps2)) return;

    // See if need to split:
    bool split1=false, split2=false;
    CalcSplitSq(split1,split2,c1,c2,dsq,_bsq);

    if (split1) {
        if (split2) {
            if (!c1.getLeft()) {
                std::cerr<<"minsep = "<<_minsep<<", maxsep = "<<_maxsep<<std::endl;
                std::cerr<<"minsepsq = "<<_minsepsq<<", maxsepsq = "<<_maxsepsq<<std::endl;
                std::cerr<<"c1.Size = "<<c1.getSize()<<", c2.Size = "<<c2.getSize()<<std::endl;
                std::cerr<<"c1.SizeSq = "<<c1.getSizeSq()<<
                    ", c2.SizeSq = "<<c2.getSizeSq()<<std::endl;
                std::cerr<<"c1.N = "<<c1.getData().getN()<<", c2.N = "<<c2.getData().getN()<<std::endl;
                std::cerr<<"c1.Pos = "<<c1.getData().getPos();
                std::cerr<<", c2.Pos = "<<c2.getData().getPos()<<std::endl;
                std::cerr<<"dsq = "<<dsq<<", s1ps2 = "<<s1ps2<<std::endl;
            }
            Assert(c1.getLeft());
            Assert(c1.getRight());
            Assert(c2.getLeft());
            Assert(c2.getRight());
            process11(*c1.getLeft(),*c2.getLeft());
            process11(*c1.getLeft(),*c2.getRight());
            process11(*c1.getRight(),*c2.getLeft());
            process11(*c1.getRight(),*c2.getRight());
        } else {
            Assert(c1.getLeft());
            Assert(c1.getRight());
            process11(*c1.getLeft(),c2);
            process11(*c1.getRight(),c2);
        }
    } else {
        if (split2) {
            Assert(c2.getLeft());
            Assert(c2.getRight());
            process11(c1,*c2.getLeft());
            process11(c1,*c2.getRight());
        } else if (dsq >= _minsepsq && dsq < _maxsepsq) {
            XAssert(NoSplit(c1,c2,sqrt(dsq),_b));
            directProcess11(c1,c2,dsq);
        }
    }
}

template <int DC1, int DC2> template <int M>
void BinnedCorr2<DC1,DC2>::directProcess11(
    const Cell<DC1,M>& c1, const Cell<DC2,M>& c2, const double dsq)
{
    XAssert(dsq >= _minsepsq);
    XAssert(dsq < _maxsepsq);
    XAssert(c1.getSize()+c2.getSize() < sqrt(dsq)*_b + 0.0001);

    const double logr = log(dsq)/2.;
    XAssert(logr >= _logminsep);

    XAssert(_binsize != 0.);
    const int k = int((logr - _logminsep)/_binsize);
    XAssert(k >= 0); 
    XAssert(k<int(_data.size()));

    BinData2<DC1,DC2>& bindata = _data[k];

    bindata.directProcess11(c1,c2,dsq,logr);
}

template <int DC1, int DC2> 
void BinnedCorr2<DC1,DC2>::finalize(double var1, double var2)
{
    dbg<<"Finalize processing\n";
    if (DC1 != NData) { Assert(var1 != 0.); }
    if (DC2 != NData) { Assert(var2 != 0.); }
    Assert(_metric != -1);

    long n = _data.size();
    double log_units = log(_sepunits);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (long i=0;i<n;++i) {
        _data[i].finalize(var1,var2);

        // For Sphere, all the measured distances were really secants.
        // Convert them to great-circle distances.
        if (_metric == Sphere) {
            double r = exp(_data[i].meanlogr);
            r = 2.*asin(0.5*r);
            _data[i].meanlogr = log(r);
        }

        // Apply the units to the meanlogr values.
        if (_data[i].npair == 0.) _data[i].meanlogr = _logminsep+(i+0.5)*_binsize; 
        _data[i].meanlogr -= log_units;
    }
}

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::operator=(const BinnedCorr2<DC1,DC2>& rhs)
{
    Assert(rhs._data.size() == _data.size());
    for (size_t i=0; i<_data.size(); ++i) _data[i] = rhs._data[i];
}

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::operator+=(const BinnedCorr2<DC1,DC2>& rhs)
{
    Assert(rhs._data.size() == _data.size());
    for (size_t i=0; i<_data.size(); ++i) _data[i] += rhs._data[i];
}

template <int DC1, int DC2>
void BinnedCorr2<DC1,DC2>::rescaleNPair(double scale)
{
    for (size_t i=0; i<_data.size(); ++i) _data[i].npair *= scale;
}

template class BinnedCorr2<NData,NData>;
template class BinnedCorr2<NData,GData>;
template class BinnedCorr2<GData,GData>;
template class BinnedCorr2<KData,KData>;
template class BinnedCorr2<NData,KData>;
template class BinnedCorr2<KData,GData>;
