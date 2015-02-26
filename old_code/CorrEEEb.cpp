// This program is an implementation of the "Faster Algorithm" in
// Jarvis et al (astroph/0307393) for calculating the three point
// correlation function.  
//
// The general idea (see the paper for more specifics and a spiffy
// graphic) is to break the problem into separate units doing one bin 
// of log r (r = d3 = the smallest side of the triangle) at a time.
// 
// First we find all pairs of Cells which have a separation 
// appropriate for the log r bin we are considering and whose component
// points will not be more than 1 bin off (technically the parameter
// binslop below).  So, if one Cell has a size (defined as the maximum
// deviation of any point from the centroid) s1 and the other has
// a size s2, and their centroids are separated by d3, then we 
// split the Cells iff (s1+s2)/d3 > b, where b is the size of the log r
// bins.
//
// Once we have all pairs which pass this test, we want to finish the 
// correlation with all appropriate third points.  We choose the 
// following two parameters to characterize the shape of the triangle 
// (d3 sets the overall size of the triangle):
//
// u = d3/d2
// v = +- (d1-d2)/d3
// where v>0 if 1,2,3 are counter-clockwise and v<0 if 1,2,3 are clockwise.
//
// The requirements for the third point in the triangle are that the 
// component points form triangles that belong at most 1 bin away from
// the bin for the centroids.  Also, d3 must be the smallest side. 
//
// Mathematically, the requirements are:
//
// d2 > d3
// d1 > d3
// s3 u/d2 < b
// s3 sqrt(1-v^2)/d2 < b
//
// So we start with the top Cell and split down the tree until all of these
// conditions are satisfied and find the correlation with the pairs 
// we have already found.
//
// The basic version of the algorithm (not this code) does this separately
// for every pair 1,2.  This code instead takes the pairs of 1,2 Cells
// and makes another tree according to the midpoints of the d3 line.
// These pairs then make up a PairCell.  The pairs contained within a
// PairCell are binned according to their orientation.  Then we can
// correlate these PairCells with the final third point of the triangle
// calculating a different v for each orientation saved in the PairCell.
// (If d3 is not much less than d1 and d2, u may be different vary
// somewhat as well.)
//
//

#include "dbg.h"
#include "OldCorr.h"
#include "PairCell.h"
#include "OldCorrIO.h"
#include <fstream>

#define WRITEM3

#ifdef MEMDEBUG
AllocList* allocList;
#endif

// Constants to set:
// Three parameters are r,u,v
// if three sides are d1 >= d2 >= d3, define:
//
// r = d3
// u = d3/d2 
// v = (d1-d2)/d3 if x1,x2,x3 are counterclockwise
// v = (d2-d1)/d3 if x1,x2,x3 are clockwise
// 
// r ranges from rmin to rmax in logarithmic bins
// u ranges from 0..1
// v ranges from -1..1

const double minsep = 1.;       // (arcsec) Minimum separation to consider
const double maxsep = 300.*60.;  // (arcsec) Maximum separation to consider
const double binsize = 0.10;     // Size of bins in logr, u, v
const double binslop = 1.0;      // Allowed slop on getting right bin (in number of bins)
// The b parameter from Jarvis et al is really binsize*binslop

// The number of bins needed given the binsize
const int nrbins = int(ceil(log(maxsep/minsep)/binsize));
const int nubins = int(ceil(1./binsize));
const double du = 1./nubins;
const int nvbins = 2*nubins;
const double dv = du;

// b is the same as the b parameter in Jarvis, et al
const double b = binslop*binsize;
const double bsq = b*b;
const double maxsepsq = maxsep*maxsep;

std::ostream* dbgout=0;
bool XDEBUG = false;

double outputsize = 1.e100;

//#define XAssert(s) Assert(s)
#define XAssert(s)
// Switch commented line of these two to get more
// error checking during the execution

int recursen=-1;
#include "Process2.h"

void DirectProcess11(
    std::vector<PairCellData>& pairdata, const Cell& c1, const Cell& c2,
    double d, const std::complex<double>& r)
{
    Assert(&c1 != &c2);
    pairdata.push_back(PairCellData(c1,c2,d,r));
}

void AddData(
    BinData3<3,3,3>& bindata, const PairCellData& p12, const Cell& c3, 
    bool swap12, double u, double v)
{
    // Given a particular PairCell orientation, and a third Cell c3,
    // calculate the appropriate correlation products and place
    // them into the given bin.

    double www = c3.getWeight()*p12.w1w2;
    std::complex<double> e3 = c3.getWE() * p12.expm2ialpha3;
    bindata.gam0 += p12.e1e2*e3;
    if (swap12) {
        bindata.gam1 += p12.e1e2c*e3;
        bindata.gam2 += conj(p12.e1e2c)*e3;
    } else {
        bindata.gam1 += conj(p12.e1e2c)*e3;
        bindata.gam2 += p12.e1e2c*e3;
    }
    bindata.gam3 += p12.e1e2*conj(e3);
    bindata.meanr += p12.d3*www;
    bindata.meanu += u*www;
    bindata.meanv += v*www;
    bindata.weight += www;
    bindata.ntri += c3.getN() * p12.n1n2;
}

void DirectProcess11SingleU(
    std::vector<std::vector<BinData3<3,3,3> > >& data,
    const PairCell& p12, const Cell& c3,
    const std::complex<double>& rm, const double dm, const double u) 
{
    // This function takes a PairCell (p12) and a Cell (c3) and 
    // computes the correlation function data for every orientation
    // in p12.
    // The Direct prefix on this and the next couple of functions
    // indicate that c3 does not need to be split any further.
    // The SingleU suffix here indicates that all the orientations
    // saven in p12 will have the same value of u given the third
    // point c3.
    int j = int(floor(u/du));
    Assert(j>=0 && j < nubins);
    std::vector<BinData3<3,3,3> >& dataj = data[j];

    ++recursen;
    xdbg<<std::string(recursen,'.')<<
        "Start DirectP12: "<<c3.getMeanPos()<<" -- "<<p12.getMeanPos()<<
        "   D3 = "<<p12.getMeanD3()<<"  DM = "<<dm<<std::endl;

    double usqover8 = u*u/8.;
    for(int i=0;i<p12.getNBins();++i) {
        const PairCellData& pairdatai = p12.getData(i);
        if (pairdatai.w1w2 == 0.0) continue;

        // theta = angle from x3--M--x2
        // phi = x2--x3--M (~= x1--x3--M)
        // where M is the intersection of angle bisector with d3
        // For small u (ie. here), this is roughly = p12's M (median intersection)
        // v = -cos(theta)/cos(phi)
        // sin(phi) ~= (d3/2 sin(theta))/dm = u*sin(theta)/2
        // v ~= -cos(theta)/sqrt(1-u^2*sin^2(theta)/4)
        //   ~= -cos(theta)*(1 + u^2 sin^2(theta) / 8)
        std::complex<double> temp = rm * conj(pairdatai.sover2);
        double costheta = real(temp)/std::abs(temp);
        double secphi = 1.+usqover8 * (1.-costheta*costheta);
        double v = std::abs(costheta)*secphi;
        if (v > 0.999999) v = 0.999999;
        Assert(v>=0. && v<1.);

        bool swap12 = (costheta > 0.);
        bool clockwise = (imag(temp) < 0.);
        if (swap12 != clockwise) v = -v;

        int k = int(floor((v+1.)/dv));
        Assert (k >= 0 && k < nvbins);

        AddData(dataj[k],pairdatai,c3,swap12,u,v);

    }

    xdbg<<std::string(recursen,'.')<<"Done DirectP11 Pair - Cell\n";
    recursen--;
}

void DirectProcess11MultiU(
    std::vector<std::vector<BinData3<3,3,3> > >& data,
    const PairCell& p12, const Cell& c3)
{
    // The same as the above function, except that the value of u
    // is not expected to be the same for every orientation in p12, so
    // we need to calculate it for each.
    ++recursen;
    xdbg<<std::string(recursen,'.')<<
        "Start DirectP11Multi: "<<c3.getMeanPos()<<" -- "<<p12.getMeanPos()<<
        "   D3 = "<<p12.getMeanD3()<<"  DM = "<<Dist(c3.getMeanPos(),p12.getMeanPos())<<std::endl;

    for(int i=0;i<p12.getNBins();++i) {
        const PairCellData& pairdatai = p12.getData(i);
        if (pairdatai.w1w2 == 0.0) continue;
        std::complex<double> rm = c3.getMeanPos() - pairdatai.pos;
        double d3 = pairdatai.d3;

        double d1 = std::abs(rm-pairdatai.sover2);
        if (d1 < d3) continue;
        double d2 = std::abs(rm+pairdatai.sover2);
        if (d2 < d3) continue;
        bool swap12 = (d2 > d1);
        if (swap12) std::swap(d1,d2);
        double u = d3/d2;
        Assert(d1>=d2);
        XAssert(std::abs(std::abs(2.*pairdatai.sover2)-d3) < binsize*binsize);
        double v = (d1-d2)/d3;
        if (v >= 0.99999) v = 0.99999;
        bool clockwise = (imag(rm*conj(pairdatai.sover2)) < 0.);

        // if (swap12) clockwise = !clockwise;
        // if (clockwise) v = -v;
        if (clockwise != swap12) v = -v; 

        int j = int(floor(u/du));
        int k = int(floor((v+1.)/dv));
        //Assert(j >= 0 && j < nubins);
        //Assert(k >= 0 && k < nvbins);

        AddData(data[j][k],pairdatai,c3,swap12,u,v);

    }

    xdbg<<std::string(recursen,'.')<<"Done DirectP11Multi\n";
    recursen--;
}

void DirectProcess11(
    std::vector<std::vector<BinData3<3,3,3> > >& data,
    const PairCell& p12, const Cell& c3, 
    const double dm, const std::complex<double>& rm) 
{
    // This function takes a PairCell (p12) and a Cell (c3) and 
    // computes the correlation function data for every orientation
    // in p12.  Basically, this function just figures out whether
    // u might vary among the orientations in p12 or not and 
    // calls the appropriate function above.

    // Min value of d2 = dm - d3/2
    // Umax = d3/d2min = d3/dm (1 + d3/2dm)
    // Max value of d2 = sqrt(dm^2 + (d3/2)^2) ~= dm + d3^2/8dm
    // Umin = d3/d2max = d3/dm (1 - d3^2/8dm^2)
    // Delta = d3^2/2dm^2(1+d3/4dm)
    // Want Delta < binsize
    // This is true for u^2(1+u/4) < 2b
    // Or u ~< sqrt(2b/(1+sqrt(b/8))) = maxsingleu (constant for b)
    //
    // Also need to make sure that the center doesn't move around too 
    // much from bin to bin (rather than within each bin, which we
    // already tested) causing bins to have different u's.

    const double maxsingleusq = 2.*b/(1.+sqrt(b/8));
    const double maxsingleu = sqrt(maxsingleusq);

    double u = p12.getMeanD3()/dm;
    if (u < maxsingleu && (c3.getSize() + p12.getAllSize())/dm < b) 
        DirectProcess11SingleU(data,p12,c3,rm,dm,u);
    else
        DirectProcess11MultiU(data,p12,c3);
}

void FinalizeProcess(std::vector<std::vector<BinData3<3,3,3> > >& data, double vare)
{
    // The accumulation of three point data has been a sum.
    // This function basically just renormalizes by the weight in each
    // bin to end up with averages instead.

    for(int j=0;j<int(data.size());++j) {
        for(int k=0;k<int(data[j].size());++k) {
            BinData3<3,3,3>& bindata=data[j][k];
            //dbg<<"j,k = "<<j<<','<<k<<" w = "<<bindata.weight<<", ntri = "<<bindata.ntri<<std::endl;
            const double w = bindata.weight;
            if (w == 0.) continue;
            Assert(bindata.ntri > 0.);
            bindata.gam0 /= w;
            bindata.gam1 /= w;
            bindata.gam2 /= w;
            bindata.gam3 /= w;
            bindata.meanr /= w; 
            bindata.meanu /= w;
            bindata.meanv /= w;
            bindata.vargam = vare*vare*vare/bindata.ntri;
            //dbg<<"done: "<<j<<','<<k<<": meanr = "<<bindata.meanr
            // <<"    weight = "<<bindata.weight<<std::endl;
        }
    }
}

int main(int argc, char *argv[])
{

#ifdef MEMDEBUG
    atexit(&DumpUnfreed);
#endif
    if (argc<2) myerror("Usage: correeeb infilename");

    //dbgout = new std::ofstream("e3b.debug");
    dbgout = &std::cout;

    // First read in all the data
    std::ifstream infile(argv[1]);
    double vare;
    std::vector<CellData> celldata;
    Read(infile,minsep,binsize,celldata,vare);

    // vare is an estimate of the variance of e per component
    dbg<<"ngal = "<<celldata.size()<<std::endl;
    Assert(vare >= 0.);
    dbg<<"vare = "<<vare<<": sig_sn (per component) = "<<sqrt(vare)<<std::endl;
    dbg<<"nrbins = "<<nrbins<<": min,maxsep = "<<minsep<<','<<maxsep<<std::endl;
    dbg<<"nubins = "<<nubins<<", nvbins = "<<nvbins<<std::endl;

    // Make the tree of Cells from the data
    Cell wholefield(celldata,MEAN);

    std::vector<PairCellData> pairdata; pairdata.reserve(celldata.size());
    std::vector<std::vector<std::vector<BinData3<3,3,3> > > > threeptdata(
        nrbins,
        std::vector<std::vector<BinData3<3,3,3> > >(
            nubins,std::vector<BinData3<3,3,3> >(nvbins)));

    const double sin60 = sqrt(3.)/2.;

    for(int i=0; i<nrbins; ++i) {
        // For each r bin, find all the pairs which qualify as the smallest side
        pairdata.clear();
        double minr = minsep*exp(i*binsize);
        double maxr = minsep*exp((i+1)*binsize);
        double minrsq = minr*minr;
        double maxrsq = maxr*maxr;
        if (maxr > maxsep) maxr = maxsep;
        Process2(pairdata,minr,maxr,minrsq,maxrsq,wholefield); 
        // With the DirectProcess11 defined above for vector<PairCellData>,
        // this Process2 just adds the pair data to the vector.

        dbg<<"for i = "<<i<<", size = "<<pairdata.size()<<std::endl;
        if (pairdata.size() == 0) continue;

        // Make a tree of these pairs
        double minsize = minr*b*sin60/2.;
        // any smaller and don't bother subdividing Cells
        //int nthetabins = int(ceil(PI/2.*nvbins/binslop));
        int nthetabins = int(ceil(2.*nvbins/binslop));
        // when triangle is roughly isosceles, dv ~= dtheta
        // so theta bins should be no larger than 2binslop/nvbins
        // nthetabins = PI / (2binslop/nvbins) = PI/2 * nvbins/binslop

        PairCell pairs(nthetabins,minsize,pairdata,MEAN);
        double minr1 = minr*sin60;
        double minr1sq = minr1*minr1;
        Process11(threeptdata[i],minr1,maxsep,minr1sq,maxsepsq,pairs,wholefield);
        FinalizeProcess(threeptdata[i],vare);
    }
    dbg<<"done rbins\n";

    std::ofstream e3out("e3.out");
    WriteEEE(e3out,minsep,binsize,threeptdata);

#ifdef WRITEM3
    std::ofstream m3out("m3.out");
    WriteM3(m3out,minsep,binsize,threeptdata);
#endif

    if (dbgout && dbgout != &std::cout) 
    { delete dbgout; dbgout=0; }

    return 0;
}

