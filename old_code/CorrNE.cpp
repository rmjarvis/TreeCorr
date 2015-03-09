
#include <fstream>
#include <string>

#include "dbg.h"
#include "OldCell.h"
#include "Form.h"
#include "OldCorr.h"
#include "OldCorrIO.h"

double outputsize = 1.e3;
//#define XAssert(s) Assert(s)
#define XAssert(s)

#ifdef MEMDEBUG
AllocList* allocList;
#endif

int recursen=-1;
#include "Process2.h"

//#define DOCROSSRAND

std::ostream* dbgout = 0;
bool XDEBUG = false;

// Constants to set:
const double minsep = 1.;        // (arcsec) Minimum separation to consider
const double maxsep = 200.*60.;  // (arcsec) Maximum separation to consider
const double binsize = 0.05;    // Ratio of distances for consecutive bins
const double binslop = 1.0;      // Allowed slop on getting right distance bin
const double smoothscale = 2.;   // Smooth from r/smoothscale to r*smoothscale

// Derived Constants:
const double logminsep = log(minsep);
const double halfminsep = 0.5*minsep;
const int nbins = int(ceil(log(maxsep/minsep)/binsize));
const double b = 2.*binslop*binsize;
const double minsepsq = minsep*minsep;
const double maxsepsq = maxsep*maxsep;
const double bsq = b*b;

#if 0
const int nthetabins = 32; // nbins for theta binning test
const double rmintheta = 40.*60.;  // min sep (arcsec) for theta binning test
const int ktheta = int(floor((log(rmintheta)-logminsep)/binsize));
#endif

template <class T> inline T SQR(const T& x) { return x*x; }

void DirectProcess11(
    std::vector<BinData2<1,3> >& data, const NCell& c1, const Cell& c2,
    const double dsq, const Position2D& r)
{
    xdbg<<std::string(recursen+1,'-')<<"Direct: d = "<<sqrt(dsq)<<std::endl;
    XAssert(c1.getSize()+c2.getSize() < sqrt(dsq)*b + 0.0001);
    XAssert(std::abs(c2.getMeanPos() - c1.getMeanPos() - r) < 0.0001);
    XAssert(std::abs(dsq - std::norm(r)) < 0.0001);

    Assert(dsq >= minsepsq);
    Assert(dsq < maxsepsq);

    std::complex<double> cr(r.getX(),r.getY());
    const std::complex<double> expm2iarg = SQR(conj(cr))/dsq;

    const double logr = log(dsq)/2.;
    Assert(logr >= logminsep);

    const int k = int(floor((logr - logminsep)/binsize));
    Assert(k >= 0); Assert(k<int(data.size()));

    const double nw = c1.getN()*c2.getWeight();
    const std::complex<double> net = -double(c1.getN())*c2.getWE()*expm2iarg;
    const double npairs = c1.getN()*c2.getN();

    BinData2<1,3>& crossbin = data[k];
    crossbin.meangammat += net;
    crossbin.weight += nw;
    crossbin.meanlogr += nw*logr;
    crossbin.npair += npairs;
}

void FinalizeProcess(std::vector<BinData2<1,3> >& data, double vare)
{
    for(int i=0;i<nbins;++i) {
        BinData2<1,3>& crossbin = data[i];
        double wt = crossbin.weight;
        if (wt == 0.) 
            crossbin.meangammat = crossbin.meanlogr = crossbin.vargammat = 0.;
        else {
            crossbin.meangammat /= wt;
            crossbin.meanlogr /= wt;
            crossbin.vargammat = vare/crossbin.npair;
        }
        if (crossbin.npair<100.) crossbin.meanlogr = logminsep+(i+0.5)*binsize;
    }
}

int main(int argc, char* argv[])
{
#ifdef MEMDEBUG
    atexit(&DumpUnfreed);
#endif

    if (argc < 3) myerror("Usage: corrne brightfile faintfile");

    dbgout = new std::ofstream("ne.debug");

    std::ifstream brightfin(argv[1]);
    std::ifstream faintfin(argv[2]);

    double vare;
    std::vector<NCellData> brightdata;
    std::vector<CellData> faintdata;

    dbg << "Read bright gals\n";
    Read(brightfin,minsep,binsize,brightdata);
    NCell brightfield(brightdata);

    dbg << "Read faint gals\n";
    Read(faintfin,minsep,binsize,faintdata,vare);

    Cell faintfield(faintdata);

    dbg << "bright size = "<<brightdata.size();
    dbg <<", faintsize = "<<faintdata.size()<<std::endl;

    dbg<<"vare = "<<vare<<": sig_sn (per component) = "<<sqrt(vare)<<std::endl;
    dbg<<"nbins = "<<nbins<<": min,maxsep = "<<minsep<<','<<maxsep<<std::endl;

    std::vector<BinData2<1,3> > crossdata(nbins);
    Process11(crossdata,minsep,maxsep,minsepsq,maxsepsq,brightfield,faintfield);
    FinalizeProcess(crossdata,vare);

    if (dbgout) {
        dbg<<"done process crossdata:\n";
    }

    std::ofstream x2out("x2.out");
    WriteNE(x2out,minsep,binsize,smoothscale,crossdata);

    if (dbgout && dbgout != &std::cout) {delete dbgout; dbgout=0;}
    return 0;
}

