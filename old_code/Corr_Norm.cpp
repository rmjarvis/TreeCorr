
#include <fstream>
#include <string>
#include <algorithm>

#include "dbg.h"
#include "OldCell.h"
#include "Form.h"
#include "OldCorr.h"
#include "OldCorrIO.h"

const double outputsize = 1.e3;
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
const double bsq = b*b;
const double minsepsq = minsep*minsep;
const double maxsepsq = maxsep*maxsep;

#ifdef TEST
const int nthetabins = 32; // nbins for theta binning test
const double rmintheta = 40.*60.;  // min sep (arcsec) for theta binning test
const int ktheta = int(floor((log(rmintheta)-logminsep)/binsize));
#endif

template <class T> inline T SQR(const T& x) { return x*x; }

void DirectProcess11(
    std::vector<BinData2<3,3> >& data, const Cell& c1, const Cell& c2,
    const double d, const Position2D& r)
{
    XAssert(c1.getSize()+c2.getSize() < d*b + 0.0001);
    XAssert(std::abs(c2.getMeanPos() - c1.getMeanPos() - r) < 0.0001);
    XAssert(std::abs(d - std::abs(r)) < 0.0001);

    Assert(d >= minsep);
    Assert(d < maxsep);

    std::complex<double> cr(r.getX(),r.getY());
    const std::complex<double> expm4iarg = SQR(SQR(conj(cr)/d));

    const double logr = log(d);
    Assert(logr >= logminsep);

    const int k = int(floor((logr - logminsep)/binsize));
    Assert(k >= 0); Assert(k<int(data.size()));

    const double ww = c1.getWeight()*c2.getWeight();
    const std::complex<double> e1 = c1.getWE();
    const std::complex<double> e2 = c2.getWE();

    const std::complex<double> ee = e1*e2*expm4iarg;
    const std::complex<double> eec = e1*conj(e2);

    const double npairs = c1.getN()*c2.getN();

    BinData2<3,3>& twoptbin = data[k];
    twoptbin.xiplus += eec;
    twoptbin.ximinus += ee;
    twoptbin.weight += ww;
    twoptbin.meanlogr += ww*logr;
    twoptbin.npair += npairs;
}

#ifdef TEST
std::vector<std::complex<double> > testthetanet(nthetabins,0.);
std::vector<double> testthetanw(nthetabins,0.);
std::vector<int> testthetan(nthetabins,0.);
#endif

void DirectProcess11(
    std::vector<BinData2<1,1> >& data, const NCell& c1, const NCell& c2,
    const double d, const Position2D& r)
{
    xdbg<<std::string(recursen+1,'-')<<"Direct: d = "<<d<<std::endl;
    XAssert(c1.getSize()+c2.getSize() < d*b + 0.0001);
    XAssert(Dist(c2.getMeanPos() - c1.getMeanPos(),r) < 0.0001);
    XAssert(std::abs(d - std::abs(r)) < 0.0001);

    Assert(d >= minsep);
    Assert(d < maxsep);

    const double logr = log(d);
    Assert(logr >= logminsep);

    const int k = int(floor((logr - logminsep)/binsize));
    Assert(k >= 0); Assert(k<int(data.size()));

    const double npairs = c1.getN()*c2.getN();

    data[k].npair += npairs;
}

void DirectProcess11(
    std::vector<BinData2<1,3> >& data, const NCell& c1, const Cell& c2,
    const double d, const Position2D& r)
{
    xdbg<<std::string(recursen+1,'-')<<"Direct: d = "<<d<<std::endl;
    XAssert(c1.getSize()+c2.getSize() < d*b + 0.0001);
    XAssert(std::abs(c2.getMeanPos() - c1.getMeanPos() - r) < 0.0001);
    XAssert(std::abs(d - std::abs(r)) < 0.0001);

    Assert(d >= minsep);
    Assert(d < maxsep);

    std::complex<double> cr(r.getX(),r.getY());
    const std::complex<double> expm2iarg = SQR(conj(cr)/d);

    const double logr = log(d);
    Assert(logr >= logminsep);

    const int k = int(floor((logr - logminsep)/binsize));
    Assert(k >= 0); Assert(k<int(data.size()));

    const double nw = c1.getN()*c2.getWeight();
    const std::complex<double> net = -double(c1.getN())*c2.getWE()*expm2iarg;
    const double npairs = c1.getN()*c2.getN();

#ifdef TEST
    if (k >= ktheta) {
        int j = int(floor( (arg(r)/TWOPI + 0.5) * testthetanet.size() ));
        Assert(j>=0);
        if (j == testthetanet.size()) {
            Assert(std::abs(arg(r) - PI) < 0.0001);
            j--;
        }
        Assert(j<testthetanet.size());
        testthetanet[j] += net;
        testthetanw[j] += nw;
        testthetan[j] += npairs;
    }
#endif

    BinData2<1,3>& crossbin = data[k];
    crossbin.meangammat += net;
    crossbin.weight += nw;
    crossbin.meanlogr += nw*logr;
    crossbin.npair += npairs;
}

void FinalizeProcess(std::vector<BinData2<3,3> >& data, double vare)
{
    for(int i=0;i<nbins;++i) {
        BinData2<3,3>& twoptbin = data[i];
        double wt = twoptbin.weight;
        if (wt == 0.) twoptbin.xiplus = twoptbin.ximinus =
            twoptbin.meanlogr = twoptbin.varxi = 0.;
        else {
            twoptbin.xiplus /= wt;
            twoptbin.ximinus /= wt;
            twoptbin.meanlogr /= wt;
            twoptbin.varxi = vare*vare/twoptbin.npair;
        }
        if (twoptbin.npair<100.) twoptbin.meanlogr = logminsep+(i+0.5)*binsize;
    }
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
            crossbin.meanlogr /= 2.*wt; // since accumulate sum(w 2 logr)
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

    if (argc < 5) 
        myerror("Usage: corr_norm brightfile faintfile starfile randfiles");

    dbgout = new std::ofstream("corr_norm.debug");

    std::ifstream brightfin(argv[1]);
    std::ifstream faintfin(argv[2]);
    std::ifstream starfin(argv[3]);
    std::ifstream randlistfin(argv[4]);

    double vare;
    std::vector<NCellData> brightdata, stardata;
    std::vector<CellData> faintdata;

    dbg << "Read bright gals\n";
    Read(brightfin,minsep,binsize,brightdata);
    NCell brightfield(brightdata);

    dbg << "Read faint gals\n";
    Read(faintfin,minsep,binsize,faintdata,vare);
    Cell faintfield(faintdata);

    dbg << "Read stars\n";
    Read(starfin,minsep,binsize,stardata);
    NCell starfield(stardata);

    dbg << "bright size = "<<brightdata.size();
    dbg <<", faintsize = "<<faintdata.size();
    dbg <<", starsize = "<<stardata.size()<<std::endl;

    dbg << "Read rand fields\n";
    int nrandfields;
    randlistfin >> nrandfields;
    if (!randlistfin) myerror("reading randlistfile ",argv[4]);
    if (!(nrandfields > 0)) myerror("no random fields");
    std::vector<std::vector<NCellData> > randdata(nrandfields);
    std::vector<NCell*> randfield(nrandfields);
    for(int i=0;i<nrandfields;++i) {
        std::string randfieldname;
        randlistfin >> randfieldname;
        if (!randlistfin) myerror("reading randlistfile ",argv[4]);
        std::ifstream randfin(randfieldname.c_str());
        Read(randfin,minsep,binsize,randdata[i]);
        randfield[i] = new NCell(randdata[i]);
    }

    dbg<<"vare = "<<vare<<": sig_sn (per component) = "<<sqrt(vare)<<std::endl;
    dbg<<"nbins = "<<nbins<<": min,maxsep = "<<minsep<<','<<maxsep<<std::endl;

    std::vector<BinData2<3,3> > twoptdata(nbins);
    Process2(twoptdata,minsep,maxsep,minsepsq,maxsepsq,faintfield);
    FinalizeProcess(twoptdata,vare);
    if (dbgout) {
        dbg<<"done process twoptdata:\n";
    }

    std::vector<BinData2<1,3> > crossdata(nbins),randcrossdata(nbins);
    Process11(crossdata,minsep,maxsep,minsepsq,maxsepsq,brightfield,faintfield);
    FinalizeProcess(crossdata,vare);

    if (dbgout) {
        dbg<<"done process crossdata:\n";
    }
    std::vector<BinData2<1,3> > starcrossdata(nbins);
    Process11(starcrossdata,minsep,maxsep,minsepsq,maxsepsq,starfield,faintfield);
    FinalizeProcess(starcrossdata,vare);
    if (dbgout) {
        dbg<<"done process starcrossdata:\n";
    }

    std::vector<BinData2<1,1> > DD(nbins), RR(nbins), DR(nbins);
    Process2(DD,minsep,maxsep,minsepsq,maxsepsq,brightfield);
    for(int k=0;k<nbins;++k) DD[k].npair *= 2;

    for(int i=0;i<nrandfields;++i) {
        dbg<<"rand: i = "<<i<<std::endl;

        Process11(randcrossdata,minsep,maxsep,minsepsq,maxsepsq,*randfield[i],faintfield);
        Process2(RR,minsep,maxsep,minsepsq,maxsepsq,*randfield[i]);
        Process11(DR,minsep,maxsep,minsepsq,maxsepsq,brightfield,*randfield[i]);
    }
    for(int k=0;k<nbins;++k) RR[k].npair *= 2;
    int nrr = nrandfields;
    int ndr = nrandfields;

#ifdef DOCROSSRAND
    for(int i=0;i<nrandfields;++i) {
        for(int j=i+1;j<nrandfields;++j) {
            dbg<<"j = "<<j<<std::endl;
            Process11(RR,minsep,maxsep,minsepsq,maxsepsq,*randfield[i],*randfield[j]);
            ++nrr;
        }
    }
#endif

    for(int k=0;k<nbins;++k) { RR[k].npair /= nrr; DR[k].npair /= ndr; }

    FinalizeProcess(randcrossdata,vare);
    dbg<<"done process DD,DR,RR data:\n";

    std::ofstream e2out("e2.out");
    std::ofstream m2out("m2.out");
    std::ofstream x2out("x2.out");
    std::ofstream normout("norm.out");
    std::ofstream rx2out("rx2.out");
    std::ofstream sx2out("sx2.out");
    std::ofstream omout("omega.out");

    WriteEE(e2out,minsep,binsize,smoothscale,twoptdata);
    WriteM2(m2out,minsep,binsize,twoptdata);
    WriteNE(x2out,minsep,binsize,smoothscale,crossdata);
    WriteNorm(normout,minsep,binsize,smoothscale,crossdata,twoptdata,DD,DR,RR);
    WriteNE(rx2out,minsep,binsize,smoothscale,randcrossdata);
    WriteNE(sx2out,minsep,binsize,smoothscale,starcrossdata);
    WriteNN(omout,minsep,binsize,DD,DR,RR,nrr);

#ifdef TEST
    std::ofstream tout("theta.out");
    for(int j = 0;j<testthetanet.size();++j) {
        double theta = (j + 0.5)/testthetanet.size() * TWOPI;
        testthetanet[j] /= testthetanw[j];
        double sigsq = vare/testthetan[j];
        tout << theta << "  "<<testthetanet[j].real()<<"  "<<sigsq;
        tout << "  "<<testthetanet[j].imag()<<"  "<<sigsq<<std::endl;
    }
#endif

    if (dbgout && dbgout != &std::cout) 
    { delete dbgout; dbgout=0; }
    return 0;
}

