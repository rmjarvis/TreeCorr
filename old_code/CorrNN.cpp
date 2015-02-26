
#include <fstream>
#include <string>
#include <algorithm>

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
const double binsize = 0.05;     // Ratio of distances for consecutive bins
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

template <class T> inline T SQR(const T& x) { return x*x; }


#if 0
std::vector<int> bincount(nbins,0);
const int nthetabins = 32; // nbins for theta binning test
const double rmintheta = 40.*60.;  // min sep (arcsec) for theta binning test
const int ktheta = int(floor((log(rmintheta)-logminsep)/binsize));
std::vector<std::complex<double> > testthetanet(nthetabins,0.);
std::vector<double> testthetanw(nthetabins,0.);
std::vector<int> testthetan(nthetabins,0.);
#endif

void DirectProcess11(
    std::vector<BinData2<1,1> >& data, const NCell& c1, const NCell& c2,
    const double dsq, const Position2D& r)
{
    xdbg<<std::string(recursen+1,'-')<<"Direct: d = "<<sqrt(dsq)<<std::endl;
    XAssert(c1.getSize()+c2.getSize() < sqrt(dsq)*b + 0.0001);
    XAssert(Dist(c2.getMeanPos() - c1.getMeanPos(),r) < 0.0001);
    XAssert(std::abs(dsq - std::norm(r)) < 0.0001);

    Assert(dsq >= minsepsq);
    Assert(dsq < maxsepsq);

    const double logr = log(dsq)/2.;
    Assert(logr >= logminsep);

    const int k = int(floor((logr - logminsep)/binsize));
    Assert(k >= 0); 
    Assert(k<int(data.size()));

    const double npairs = c1.getN()*c2.getN();

    data[k].npair += npairs;
}

int main(int argc, char* argv[])
{
#ifdef MEMDEBUG
    atexit(&DumpUnfreed);
#endif

    if (argc < 3) myerror("Usage: corrnn datafile randfiles");

    dbgout = new std::ofstream("nn.debug");

    std::ifstream fin(argv[1]);
    std::ifstream randlistfin(argv[2]);

    std::vector<NCellData> data;

    dbg << "Read gals\n";
    Read(fin,minsep,binsize,data);
    if (!fin) myerror("reading file ",argv[1]);
    NCell wholefield(data);

    dbg << "ngals = "<<data.size();

    dbg << "Read rand fields\n";
    int nrandfields;
    randlistfin >> nrandfields;
    if (!randlistfin) myerror("reading randlistfile ",argv[2]);
    if (!(nrandfields > 0)) myerror("no random fields");
    std::vector<std::vector<NCellData> > randdata(nrandfields);
    std::vector<NCell*> randfield(nrandfields);
    for(int i=0;i<nrandfields;++i) {
        std::string randfieldname;
        randlistfin >> randfieldname;
        if (!randlistfin) myerror("reading randlistfile ",argv[2]);
        std::ifstream randfin(randfieldname.c_str());
        Read(randfin,minsep,binsize,randdata[i]);
        if (!randfin) myerror("reading randfile ",randfieldname.c_str());
        randfield[i] = new NCell(randdata[i]);
    }

    dbg<<"nbins = "<<nbins<<": min,maxsep = "<<minsep<<','<<maxsep<<std::endl;

    std::vector<BinData2<1,1> > DD(nbins), RR(nbins), DR(nbins);

    Process2(DD,minsep,maxsep,minsepsq,maxsepsq,wholefield);
    for(int k=0;k<nbins;++k) DD[k].npair *= 2;

    for(int i=0;i<nrandfields;++i) {
        dbg<<"rand: i = "<<i<<std::endl;
        Process2(RR,minsep,maxsep,minsepsq,maxsepsq,*randfield[i]);
        Process11(DR,minsep,maxsep,minsepsq,maxsepsq,wholefield,*randfield[i]);
    }
    for(int k=0;k<nbins;++k) RR[k].npair *= 2;
    int ndr = nrandfields;
    int nrr = nrandfields;

#ifdef DOCROSSRAND
    for(int i=0;i<nrandfields;++i) {
        for(int j=i+1;j<nrandfields;++j) {
            dbg<<"j = "<<j<<std::endl;
            Process11(RR,minsep,maxsep,minsepsq,maxsepsq,
                      *randfield[i],*randfield[j]);
            ++nrr;
        }
    }
#endif
    dbg<<"done process DD,DR,RR data\n";

    for(int i=0;i<nbins;++i) {
        dbg<<"RR["<<i<<"] = "<<RR[i].npair<<", DD = "<<DD[i].npair<<", DR = "<<DR[i].npair<<std::endl;
        RR[i].npair /= nrr;
        DR[i].npair /= ndr;
    }
    dbg<<"done divide by number of realizations\n";

    std::ofstream fout("nn.out");
    WriteNN(fout,minsep,binsize,DD,DR,RR,nrr);
    dbg<<"done write"<<std::endl;

    if (dbgout && dbgout != &std::cout) 
    { delete dbgout; dbgout=0; }
    return 0;
}

