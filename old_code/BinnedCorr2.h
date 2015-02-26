#ifndef Corr2_H
#define Corr2_H

#include <vector>
#include <string>

#include "ConfigFile.h"
#include "InputFile.h"
#include "Cell.h"
#include "Field.h"
#include "Split.h"
#include "BinData2.h"

// BinnedCorr2 encapsulates a binned correlation function.
// It can be constructed from either 1 or 2 Field objects, along with a ConfigFile.
// The constructor with only 1 field is only valid if DC1 == DC2.
template <int DC1, int DC2>
class BinnedCorr2
{

public:

    BinnedCorr2(const ConfigFile& params);

    void process(const Field<DC1>& field);
    void process(const Field<DC1>& field1, const Field<DC2>& field2);
    void processPairwise(const InputFile& file1, const InputFile& file2);

    // Main worker functions for calculating the result
    template <int M>
    void process2(const Cell<DC1,M>& c12);

    template <int M>
    void process11(const Cell<DC1,M>& c1, const Cell<DC2,M>& c2);

    template <int M>
    void directProcess11(const Cell<DC1,M>& c1, const Cell<DC2,M>& c2, const double dsq);

    void finalize(double var1, double var2);

    void clearData();

    // Call process2 with the i-th Cell in field1.
    void callProcess2(const Field<DC1>& field, int i);

    // Call process11 with the i-th Cell in field1 and the j-th Cell in field2.
    void callProcess11(const Field<DC1>& field1, const Field<DC2>& field2, int i, int j);

    double getMinSep() const { return _minsep; }
    double getMaxSep() const { return _maxsep; }
    double getBinSize() const { return _binsize; }
    double getB() const { return _b; }
    double getSepUnits() const { return _sepunits; }
    Metric getMetric() const { Assert(_metric != -1); return static_cast<Metric>(_metric); }

    const std::vector<BinData2<DC1,DC2> >& getData() const { return _data; }

    // Note: op= only copies _data.  Not all the params.
    void operator=(const BinnedCorr2<DC1,DC2>& rhs);
    void operator+=(const BinnedCorr2<DC1,DC2>& rhs);

    // Rescale npair by a scale factor.  e.g. 1./the number of file pairs that were processed.
    // Only usually relevant for NN correlations.
    void rescaleNPair(double nproc);

protected:

    template <int M>
    void doProcessPairwise(const InputFile& file1, const InputFile& file2);

    const ConfigFile& _params;
    double _minsep;
    double _maxsep;
    double _binsize;
    double _logminsep;
    double _halfminsep;
    double _minsepsq;
    double _maxsepsq;
    double _b;
    double _bsq;
    double _sepunits;
    int _metric; // Stores which Metric is being used for the analysis.

    std::vector<BinData2<DC1,DC2> > _data;
};


#endif
