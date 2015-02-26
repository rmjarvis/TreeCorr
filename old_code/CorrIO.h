#ifndef CorrIO_H
#define CorrIO_H

#include "BinData2.h"
#include "CalcT.h"

void WriteGG(std::ostream& fout, double minsep, double binsize, 
             double smoothscale, int prec,
             const std::vector<BinData2<GData,GData> >& data);

void WriteM2(std::ostream& fout, double minsep, double binsize, UForm uform, int prec,
             const std::vector<BinData2<GData,GData> >& data);

void WriteNG(std::ostream& fout, double minsep, double binsize, 
             double smoothscale, const std::string& ne_stat, int prec,
             const std::vector<BinData2<NData,GData> >& data,
             const std::vector<BinData2<NData,GData> >& rand);

void WriteNM(std::ostream& fout, double minsep, double binsize,
             const std::string& ne_stat, UForm uform, int prec,
             const std::vector<BinData2<NData,GData> >& data,
             const std::vector<BinData2<NData,GData> >& rand);

void WriteNorm(std::ostream& fout, double minsep, double binsize,
               const std::string& ne_stat, const std::string& nn_stat, 
               UForm uform, int prec,
               const std::vector<BinData2<NData,GData> >& ne,
               const std::vector<BinData2<NData,GData> >& re,
               const std::vector<BinData2<GData,GData> >& ee,
               const std::vector<BinData2<NData,NData> >& dd,
               const std::vector<BinData2<NData,NData> >& dr,
               const std::vector<BinData2<NData,NData> >& rr);

void WriteNN(std::ostream& fout, double minsep, double binsize,
             const std::string& nn_stat, int prec,
             const std::vector<BinData2<NData,NData> >& dd,
             const std::vector<BinData2<NData,NData> >& dr, 
             const std::vector<BinData2<NData,NData> >& rd, 
             const std::vector<BinData2<NData,NData> >& rr);

void WriteKK(std::ostream& fout, double minsep, double binsize, 
             double smoothscale, int prec,
             const std::vector<BinData2<KData,KData> >& data);

void WriteNK(std::ostream& fout, double minsep, double binsize, 
             double smoothscale, const std::string& ne_stat, int prec,
             const std::vector<BinData2<NData,KData> >& data,
             const std::vector<BinData2<NData,KData> >& rand);

void WriteKG(std::ostream& fout, double minsep, double binsize, 
             double smoothscale, int prec,
             const std::vector<BinData2<KData,GData> >& data);

#endif
