
#include <vector>
#include <fstream>
#include <string>
#include <complex>
#include "dbg.h"
#include "OldCorr.h"
#include "OldCorrIO.h"

const double outputsize = 3.e3;
// Only output progress lines when size > outputsize.

#define XAssert(s) Assert(s)  
// XAssert adds extra (time-consuming) checks
//#define XAssert(s)

#ifdef MEMDEBUG
AllocList* allocList;
#endif

int recursen=-1;
#include "Process2.h"

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
const double b = binslop*binsize;
const double minsepsq = minsep*minsep;
const double maxsepsq = maxsep*maxsep;
const double bsq = b*b;

std::ostream* dbgout = 0;
bool XDEBUG = false;

// Switch commented line of these two to get more
// error checking during the execution

std::vector<int> bincount(nbins,0);

void DirectProcess11(
    std::vector<BinData2<3,3> >& data, const Cell& c1, const Cell& c2,
    const double dsq, const Position2D& r)
{
    XAssert(c1.getSize()+c2.getSize() < sqrt(dsq)*b + 0.0001);
    XAssert(Dist(c2.getMeanPos() - c1.getMeanPos(),r) < 0.0001);
    XAssert(std::abs(dsq - DistSq(c1.getMeanPos(),c2.getMeanPos())) < 0.0001);

    Assert(dsq > minsepsq);
    Assert(dsq < maxsepsq);

    std::complex<double> cr(r.getX(),r.getY());
    std::complex<double> expm4iarg = conj(cr*cr)/dsq; // now expm2iarg
    expm4iarg *= expm4iarg; // now expm4iarg

    const double logr = log(dsq)/2.;
    Assert(logr >= logminsep);

    const int k = int(floor((logr - logminsep)/binsize));
    Assert(k >= 0); Assert(k<int(data.size()));

    const double ww = c1.getWeight()*c2.getWeight();
    const std::complex<double> e1 = c1.getWE();
    const std::complex<double> e2 = c2.getWE();

    const std::complex<double> ee = e1*e2*expm4iarg;
    const std::complex<double> eec = e1*conj(e2);

    const double npairs = c1.getN()*c2.getN();

    BinData2<3,3>& bindata = data[k];
    bindata.xiplus += eec;
    bindata.ximinus += ee;
    bindata.weight += ww;
    bindata.meanlogr += ww*logr;
    bindata.npair += npairs;

    ++bincount[k];
}

void FinalizeProcess(std::vector<BinData2<3,3> >& data, double vare)
{
    for(int i=0;i<nbins;++i) {
        BinData2<3,3>& bindata = data[i];
        double wt = data[i].weight;
        if (wt == 0.) 
            bindata.xiplus = bindata.ximinus =
                bindata.meanlogr = bindata.varxi = 0.;
        else {
            bindata.xiplus /= wt;
            bindata.ximinus /= wt;
            bindata.meanlogr /= wt;
            bindata.varxi = vare*vare/bindata.npair;
        }
        if (bindata.npair < 100.) bindata.meanlogr = logminsep+(i+0.5)*binsize; 
    }
}

void Read(
    std::istream& fin, double minsep, double binsize,
    std::vector<CellData>& celldata, 
    std::vector<std::vector<CellData> >& expcelldata, double& vare)
{
    double sumw=0.;
    vare=0.;
    int j;
    double x,y,e1,e2,w;
    while(fin >> j>>x>>y>>e1>>e2>>w) {
        // Note: e1,e2 are really gamma_1, gamma_2
        // _NOT_ observed ellipticities

        Position2D pos(x,y);
        std::complex<double> e(e1,e2);
        sumw += w;
        vare += w*w*norm(e);

        celldata.push_back(CellData(pos,e,w));

        while(j >= int(expcelldata.size())) 
            expcelldata.push_back(std::vector<CellData>());
        expcelldata[j].push_back(CellData(pos,e,w));
    }
    int ngal = celldata.size();
    vare /= 2*sumw*sumw/ngal; // 2 because want var per component
    dbg<<"ngal = "<<ngal<<", totw = "<<sumw<<std::endl;
}

int main(int argc, char* argv[])
{
#ifdef MEMDEBUG
    atexit(&DumpUnfreed);
#endif

    if (argc < 2 || argc > 3) myerror("Usage: corree_multi filename");

    dbgout = &std::cout;

    std::ifstream fin(argv[1]);

    double vare=0.;
    std::vector<CellData> celldata;
    std::vector<std::vector<CellData> > expcelldata;
    Read(fin,minsep,binsize,celldata,expcelldata,vare);

    dbg<<"ngal = "<<celldata.size()<<std::endl;
    dbg<<"nexp = "<<expcelldata.size()<<std::endl;
    dbg<<"vare = "<<vare<<": sig_sn (per component) = "<<sqrt(vare)<<std::endl;
    dbg<<"nbins = "<<nbins<<": min,maxsep = "<<minsep<<','<<maxsep<<std::endl;

    Cell wholefield(celldata);
    std::vector<Cell*> exposures;
    double maxexpsize = 0.;
    for(int i=0;i<int(expcelldata.size());++i) if(expcelldata[i].size()>0) {
        exposures.push_back(new Cell(expcelldata[i]));
        double size = exposures.back()->getSize();
        dbg<<"size("<<i<<") = "<<size<<"  "<<maxexpsize<<std::endl;
        if (size > maxexpsize) maxexpsize = size;
    }
    // *2 to get diameter, rather than radius
    dbg<<"max size = "<<2*maxexpsize<<std::endl;
    // round up to the next bin size
    const int k1 = int(floor((log(2*maxexpsize) - logminsep)/binsize))+1;
    maxexpsize =  exp(k1*binsize+logminsep);
    dbg<<"max size = "<<maxexpsize<<std::endl;
    double maxexpsizesq = maxexpsize*maxexpsize;

    std::vector<BinData2<3,3> > data(nbins);

    for(int i=0;i<int(exposures.size());++i) 
        for(int j=0;j<int(exposures.size());++j) if (i!=j) {
            dbg<<"i,j = "<<i<<','<<j<<std::endl;
            Process11(data,minsep,maxexpsize,minsepsq,maxexpsizesq,
                      *exposures[i],*exposures[j]);
        }
    Process2(data,maxexpsize,maxsep,maxexpsizesq,maxsepsq,wholefield);

    //  Process2(data,minsep,maxsep,minsepsq,maxsepsq,wholefield);

    FinalizeProcess(data,vare);
    dbg<<"done processing\n";

    std::ofstream e2out("e2.out");
    std::ofstream m2out("m2.out");

    WriteEE(e2out,minsep,binsize,smoothscale,data);
    WriteM2(m2out,minsep,binsize,data);

    for(int i=0;i<int(exposures.size());++i) delete exposures[i];
    if (dbgout && dbgout != &std::cout) 
    { delete dbgout; dbgout=0; }
    return 0;
}

