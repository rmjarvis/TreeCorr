// This program calculates the <M^3> statistic from the 
// correlation function.
//

#include <fstream>
#include <algorithm>

#include "dbg.h"
#include "OldCorr.h"
#include "OldCorrIO.h"

const double minsep = 1.;       // (arcsec) Minimum separation to consider
const double maxsep = 200.*60.;  // (arcsec) Maximum separation to consider
const double binsize = 0.05;     // Size of bins in logr, u, v

// Derived Constants:

// The number of bins needed given the binsize
const int nrbins = int(ceil(log(maxsep/minsep)/binsize));
const int nubins = int(ceil(1./binsize));
const int nvbins = 2*nubins;

bool XDEBUG = false;
std::ostream* dbgout=0;

int main()
{
    dbgout = &std::cout;
    //dbgout = new std::ofstream("e3_m3.debug");

    std::vector<std::vector<std::vector<BinData3<3,3,3> > > > threeptdata(
        nrbins, std::vector<std::vector<BinData3<3,3,3> > >(
            nubins,std::vector<BinData3<3,3,3> >(nvbins)));
    std::ifstream e3in("e3.out");
    Read(e3in,minsep,binsize,threeptdata);

    std::ofstream m3out("m3.out");
    WriteM3(m3out,minsep,binsize,threeptdata);

#if 0
    std::ofstream m3out112("e3.m3112.out");
    WriteM3(m3out112,minsep,binsize,threeptdata,1.,1.,2.);

    std::ofstream m3out121("e3.m3121.out");
    WriteM3(m3out121,minsep,binsize,threeptdata,1.,2.,1.);

    std::ofstream m3out211("e3.m3211.out");
    WriteM3(m3out211,minsep,binsize,threeptdata,2.,1.,1.);

    std::ofstream m3out122("e3.m3122.out");
    WriteM3(m3out122,minsep,binsize,threeptdata,1.,2.,2.);

    std::ofstream m3out124("e3.m3124.out");
    WriteM3(m3out124,minsep,binsize,threeptdata,1.,2.,4.);

    std::ofstream m3out114("e3.m3114.out");
    WriteM3(m3out114,minsep,binsize,threeptdata,1.,1.,4.);

    std::ofstream m3out144("e3.m3144.out");
    WriteM3(m3out144,minsep,binsize,threeptdata,1.,4.,4.);
#endif

    if (dbgout && dbgout != &std::cout) 
    { delete dbgout; dbgout=0; }
    return 0;
}

