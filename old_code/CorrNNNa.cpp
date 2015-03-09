
#include <string>
#include <fstream>
#include <vector>

#include "dbg.h"
#include "OldCell.h"
#include "OldCorr.h"
#include "OldCorrIO.h"

#ifdef MEMDEBUG
AllocList* allocList;
#endif

// Constants to set:
// Three parameters are r,u,v
// if three sides are d1 >= d2 >= d3, then 
// r = d3
// u = d3/d2 (this is reciprocal of usual definition)
// v = (d1-d2)/d3
// r ranges from rmin to rmax in logarithmic bins
// u ranges from 0..1
// v ranges from 0..1
// This prescription ignores sense of the three points, 
// so I take v range from -1..1 with 
// negative v corresponding to 1,2,3 in clockwise order
// positive v corresponding to counterclockwise order
const double minsep = 5.;       // (arcsec) Minimum separation to consider
const double maxsep = 150.*60.;  // (arcsec) Maximum separation to consider
const double binsize = 0.05;     // Size of bins in logr, u, v
const double binslop = 1.0;      // Allowed slop on getting right bin (in number of bins)
// Note binslop on u,v are ~2*binslop

// Derived Constants:
const int nrbins = int(ceil(log(maxsep/minsep)/binsize));
const int nubins = int(ceil(1./binsize));
const double du = 1./nubins;
const int nvbins = 2*nubins;
const double dv = du;
const double b = binslop*binsize;
const double bsq = b*b;
const double minsepsq = minsep*minsep;
const double maxsepsq = maxsep*maxsep;

template<class T> inline T SQR(T x) { return x*x; }

const double outputsize = 4.e3; 
// only ouput progress lines for cells larger than outputsize

std::ostream* dbgout;
bool XDEBUG = false;

//#define XAssert(s) assert(s)
#define XAssert(s) 
// Switch commented line of these two to get more 
// error checking during the execution 

int recursen=-1;
#include "Process3.h"

void DirectProcessV(
    BinData3<1,1,1>& bin, const NCell& c1, const NCell& c2, const NCell& c3,
    const double d1, const double d2, const double d3,
    const bool swap12, const double u, const double v)
{
    bin.ntri += c3.getN();
}

void Add(
    std::vector<std::vector<BinData3<1,1,1> > >& data,
    const std::vector<std::vector<BinData3<1,1,1> > >& temp, 
    const NCell& c1, const NCell& c2, const Position2D& r3, double d3)
{
    const double n1n2 = c1.getN()*c2.getN();

    for(int ku = 0; ku < int(data.size()); ++ku) {
        std::vector<BinData3<1,1,1> >& dataku = data[ku];
        const std::vector<BinData3<1,1,1> >& tempku = temp[ku];
        for(int kv = 0; kv < int(dataku.size()); ++kv) {
            double tempbin = tempku[kv].ntri;
            if (tempbin == 0.0) continue;
            dataku[kv].ntri += n1n2*tempbin;
        }
    }
}

int main(int argc, char *argv[])
{
#ifdef MEMDEBUG
    atexit(&DumpUnfreed);
#endif

    if (argc < 3) myerror("Usage: corrnnn datafile randfiles");
    std::ifstream fin(argv[1]);
    std::ifstream randlistfin(argv[2]);

    //dbgout = &std::cout;
    dbgout = new std::ofstream("nnn.debug");

    std::vector<NCellData> data;

    dbg<<"Read gals\n";
    Read(fin,minsep,binsize,data);
    NCell field(data);

    dbg<<"ngals = "<<data.size()<<std::endl;
    dbg << "Read rand fields\n";
    int nrandfields;
    randlistfin >> nrandfields;
    if (!(nrandfields > 0)) myerror("no random fields");
    std::vector<std::vector<NCellData> > randdata(nrandfields);
    std::vector<NCell*> randfield(nrandfields);
    for(int n=0;n<nrandfields;++n) {
        std::string randfieldname;
        randlistfin >> randfieldname;
        if (!randlistfin) myerror(std::string("reading randlistfile ")+argv[2]);
        std::ifstream randfin(randfieldname.c_str());
        Read(randfin,minsep,binsize,randdata[n]);
        randfield[n] = new NCell(randdata[n]);
    }

    dbg<<"nrbins = "<<nrbins<<": min,maxsep = "<<minsep<<','<<maxsep<<std::endl;
    dbg<<"nubins = "<<nubins<<",  nvbins = "<<nvbins<<std::endl;

    dbg<<"field size = "<<field.getSize()<<", meanpos = "<<field.getMeanPos()<<std::endl;

    std::vector<std::vector<std::vector<BinData3<1,1,1> > > > DDD(
        nrbins, std::vector<std::vector<BinData3<1,1,1> > >(
            nubins,std::vector<BinData3<1,1,1> >(nvbins)));
    std::vector<std::vector<std::vector<BinData3<1,1,1> > > > DDR(
        nrbins, std::vector<std::vector<BinData3<1,1,1> > >(
            nubins,std::vector<BinData3<1,1,1> >(nvbins)));
    std::vector<std::vector<std::vector<BinData3<1,1,1> > > > DRD(
        nrbins, std::vector<std::vector<BinData3<1,1,1> > >(
            nubins,std::vector<BinData3<1,1,1> >(nvbins)));
    std::vector<std::vector<std::vector<BinData3<1,1,1> > > > RDD(
        nrbins, std::vector<std::vector<BinData3<1,1,1> > >(
            nubins,std::vector<BinData3<1,1,1> >(nvbins)));
    std::vector<std::vector<std::vector<BinData3<1,1,1> > > > DRR(
        nrbins, std::vector<std::vector<BinData3<1,1,1> > >(
            nubins,std::vector<BinData3<1,1,1> >(nvbins)));
    std::vector<std::vector<std::vector<BinData3<1,1,1> > > > RDR(
        nrbins, std::vector<std::vector<BinData3<1,1,1> > >(
            nubins,std::vector<BinData3<1,1,1> >(nvbins)));
    std::vector<std::vector<std::vector<BinData3<1,1,1> > > > RRD(
        nrbins, std::vector<std::vector<BinData3<1,1,1> > >(
            nubins,std::vector<BinData3<1,1,1> >(nvbins)));
    std::vector<std::vector<std::vector<BinData3<1,1,1> > > > RRR(
        nrbins, std::vector<std::vector<BinData3<1,1,1> > >(
            nubins,std::vector<BinData3<1,1,1> >(nvbins)));
    std::vector<std::vector<std::vector<BinData3<1,1,1> > > > RRRa(
        nrbins, std::vector<std::vector<BinData3<1,1,1> > >(
            nubins,std::vector<BinData3<1,1,1> >(nvbins)));
    std::vector<std::vector<std::vector<BinData3<1,1,1> > > > RRRb(
        nrbins, std::vector<std::vector<BinData3<1,1,1> > >(
            nubins,std::vector<BinData3<1,1,1> >(nvbins)));

    Process3(DDD,minsep,maxsep,field);
    dbg<<"Done DDD\n";

    for(int n=0;n<nrandfields;++n) {
        dbg<<"rand: n = "<<n<<std::endl;
        Process3(RRR,minsep,maxsep,*randfield[n]);
        dbg<<"Done RRR\n";
        //Process21(RRRa,minsep,maxsep,*randfield[n],*randfield[n]);
        //dbg<<"Done RRRa\n";
        //Process111(RRRb,minsep,maxsep,*randfield[n],*randfield[n],*randfield[n]);
        //dbg<<"Done RRRb\n";
        Process21(RRD,minsep,maxsep,*randfield[n],field);
        dbg<<"Done RRD\n";
        Process111(RDR,minsep,maxsep,*randfield[n],field,*randfield[n]);
        dbg<<"Done RDR\n";
        Process111(DRR,minsep,maxsep,field,*randfield[n],*randfield[n]);
        dbg<<"Done DRR\n";
        Process111(RDD,minsep,maxsep,*randfield[n],field,field);
        dbg<<"Done RDD\n";
        Process111(DRD,minsep,maxsep,field,*randfield[n],field);
        dbg<<"Done DRD\n";
        Process21(DDR,minsep,maxsep,field,*randfield[n]);
        dbg<<"Done DDR\n";
    }
    dbg<<"Done processing\n";

    //double rrrsum=0.;
    //double rrrasum=0.;
    //double rrrbsum=0.;
    for(int i=0;i<nrbins;++i) {
        //double rrrsum_r=0.;
        //double rrrasum_r=0.;
        //double rrrbsum_r=0.;
        for(int j=0;j<nubins;++j) for(int k=0;k<nvbins;++k) {
            DDR[i][j][k].ntri /= nrandfields;
            DRD[i][j][k].ntri /= 2.*nrandfields;
            RDD[i][j][k].ntri /= 2.*nrandfields;
            DRR[i][j][k].ntri /= 2.*nrandfields;
            RDR[i][j][k].ntri /= 2.*nrandfields;
            RRD[i][j][k].ntri /= nrandfields;
            //dbg<<"RRR = "<<RRR[i][j][k]<<"  "<<RRRa[i][j][k]<<"  "<<RRRb[i][j][k]<<std::endl;
            RRR[i][j][k].ntri /= nrandfields;
            //RRRa[i][j][k] /= nrandfields;
            //RRRb[i][j][k] /= 2.*nrandfields;
            //dbg<<"-> "<<RRR[i][j][k]<<"  "<<RRRa[i][j][k]<<"  "<<RRRb[i][j][k]<<std::endl;
            //rrrsum += RRR[i][j][k];
            //rrrasum += RRRa[i][j][k];
            //rrrbsum += RRRb[i][j][k];
            //rrrsum_r += RRR[i][j][k];
            //rrrasum_r += RRRa[i][j][k];
            //rrrbsum_r += RRRb[i][j][k];
        }
        //dbg<<"Rbin sum: rrr = "<<rrrsum_r<<", rrrA = "<<rrrasum_r<<", rrrB = "<<rrrbsum_r<<std::endl;
    }
    //dbg<<"Total sum: rrr = "<<rrrsum<<", rrrA = "<<rrrasum<<", rrrB = "<<rrrbsum<<std::endl;

    std::ofstream fout("nnn.out");
    WriteNNN(fout,minsep,binsize,DDD,DDR,DRD,RDD,DRR,RDR,RRD,RRR);

    if (dbgout && dbgout != &std::cout) {delete dbgout; dbgout=0;}
    return 0;
}
