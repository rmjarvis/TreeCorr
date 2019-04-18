#ifndef CORRIO_H
#define CORRIO_H

#include "OldCorr.h"
#include "Form.h"
#include "OldCell.h"
#include "Cell3D.h"

void Read(std::istream& fin, double minsep, double binsize,
          std::vector<CellData>& celldata, double& vare);
void Read(std::istream& fin, double minsep, double binsize,
          std::vector<NCellData>& celldata);
void Read(std::istream& fin, double minsep, double binsize,
          std::vector<TCell3DData>& celldata);
void Read(std::istream& fin, double minsep, double binsize,
          std::vector<NCell3DData>& celldata);

void WriteEEE(
    std::ostream& fout, double minsep, double binsize,
    const std::vector<std::vector<std::vector<BinData3<3,3,3> > > >& data);
void Read(
    std::istream& fin, double minsep, double binsize,
    std::vector<std::vector<std::vector<BinData3<3,3,3> > > >& data);
void WriteM3(
    std::ostream& fout, double minsep, double binsize,
    const std::vector<std::vector<std::vector<BinData3<3,3,3> > > >& data,
    double k1=1., double k2=1., double k3=1.);

void WriteEE(std::ostream& fout, double minsep, double binsize, double smoothscale,
             std::vector<BinData2<3,3> >& data);
void WriteM2(std::ostream& fout, double minsep, double binsize,
             std::vector<BinData2<3,3> >& data);

void WriteNorm(
    std::ostream& fout, double minsep, double binsize, double smoothscale,
    const std::vector<BinData2<1,3> >& crossdata,
    const std::vector<BinData2<3,3> >& twoptdata,
    const std::vector<BinData2<1,1> >& dd, const std::vector<BinData2<1,1> >& dr,
    const std::vector<BinData2<1,1> >& rr);
void WriteNE(std::ostream& fout, double minsep, double binsize, double smoothscale,
             const std::vector<BinData2<1,3> >& crossdata);
void WriteNN(
    std::ostream& fout, double minsep, double binsize,
    const std::vector<BinData2<1,1> >& dd, const std::vector<BinData2<1,1> >& dr, 
    const std::vector<BinData2<1,1> >& rr, double nrr);

void WriteNNN(
    std::ostream& fout, double minsep, double binsize,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& ddd,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& ddr,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& drd,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& rdd,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& drr,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& rdr,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& rrd,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& rrr);
void WriteNNN(
    std::ostream& fout, double minsep, double binsize,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& ddd,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& ddr,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& drr,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& rrr);
void WriteNNN(
    std::ostream& fout, double minsep, double binsize,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& ddd,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& rrr);

#endif
