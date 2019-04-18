
#include "dbg.h"
#include "Form.h"
#include "CalcT.h"
#include "BinData2.h"

template <class T> inline T SQR(T x) { return x*x; }

std::string str(char prefix, std::string text, int width)
{
    if (int(text.size()) > width) {
        return prefix + std::string(text,0,width);
    } else {
        // Prefer extra space on left, so n2 should round down.
        int n2 = (width - text.size()) / 2;
        int n1 = width - text.size() - n2;
        return prefix + std::string(n1,' ') + text + std::string(n2,' ');
    }
}

void WriteGG(std::ostream& fout, double minsep, double binsize, double smoothscale, int prec,
             const std::vector<BinData2<GData,GData> >& data)
{
    int width = prec + 8;
    Form sci;
    sci.sci().prec(prec).width(width).right().trail(1); 

    fout << str('#',"R_nominal",width) << str('.',"<R>",width);
    fout << str('.',"xi+",width) << str('.',"xi-",width);
    fout << str('.',"xi+_im",width) << str('.',"xi-_im",width);
    fout << str('.',"sig_xi",width);
    fout << str('.',"weight",width) << str('.',"npairs",width);
    if (smoothscale > 0.) {
        fout << str('.',"R_sm",width);
        fout << str('.',"xi+_sm",width) << str('.',"xi-_sm",width);
        fout << str('.',"sig_sm",width);
    }
    fout << '.' << std::endl;

    for(int i=0;i<int(data.size());++i) {

        double R = exp(data[i].meanlogr);
        fout << sci(minsep*exp((i+0.5)*binsize)) << sci(R);
        fout << sci(real(data[i].xiplus));
        fout << sci(real(data[i].ximinus));
        fout << sci(imag(data[i].xiplus));
        fout << sci(imag(data[i].ximinus));
        Assert(data[i].varxi >= 0.);
        fout << sci(sqrt(data[i].varxi));
        fout << sci(data[i].weight);
        fout << sci(data[i].npair);

        if (smoothscale > 0.) {
            std::complex<double> xiplussm=0.,ximinussm=0.;
            double varxism=0.,weightsm=0.,meanlogrsm=0.;

            for(int j=0;j<int(data.size());++j) {

                double r = exp(data[j].meanlogr);
                double xip = real(data[j].xiplus);
                double xim = real(data[j].ximinus);
                double var = data[j].varxi;
                double wj = data[j].weight;
                double s = r/R;

                if (s>1/smoothscale && s<smoothscale) {
                    meanlogrsm += data[j].meanlogr*wj;
                    xiplussm += xip*wj;
                    ximinussm += xim*wj;
                    varxism += var*SQR(wj);
                    weightsm += wj;
                }
            }

            if (weightsm > 0) {
                meanlogrsm /= weightsm;
                xiplussm /= weightsm;
                ximinussm /= weightsm;
                varxism /= SQR(weightsm);
            }

            fout << sci((weightsm==0.) ? R : exp(meanlogrsm));
            fout << sci(real(xiplussm));
            fout << sci(real(ximinussm));
            Assert(varxism >= 0.);
            fout << sci(sqrt(varxism));
        }
        fout << std::endl;
    }
}

void WriteM2(std::ostream& fout, double minsep, double binsize, UForm uform, int prec,
             const std::vector<BinData2<GData,GData> >& data)
{
    // Use s = r/R:
    // <Map^2>(R) = int_r=0..2R [s^2 dlogr (T+(s) xi+(r) + T-(s) xi-(r))/2]
    // <Mx^2>(R)  =     "       [ "    "   (T+(s) xi+(r) - T-(s) xi-(r))/2]
    // <Gam^2>(R) = int_r=0..2R [s^2 dlogr S+(s) xi+(r)]
    // <Gam^2>(E/B)(R) = int_r=0..2R [s^2 dlogr (S+(s) xi+(r) +- S+(s) xi-(r))/2]

    int width = prec + 8;
    Form sci;
    sci.sci().prec(prec).width(width).right().trail(1); 

    fout << str('#',"R",width);
    fout << str('.',"<Map^2>",width) << str('.',"<Mx^2>",width);
    fout << str('.',"<MMx>(a)",width) << str('.',"<MMx>(b)",width);
    fout << str('.',"sig_map",width);
    fout << str('.',"<Gam^2>",width) << str('.',"sig_gam",width);
    fout << '.' << std::endl;

    const double dlogr = binsize;

    for(int i=0;i<int(data.size());++i) {
        double mapsq=0.,mxsq=0.,varmap=0.,mmx_1=0.,mmx_2=0.;
        double gamsq=0.,vargam=0.;
        //double gamEsq=0.,gamBsq=0.,varEBgam=0.;

        double R = minsep*exp((i+0.5)*binsize);

        for(int j=0;j<int(data.size());++j) {

            double r = exp(data[j].meanlogr);
            double xip = real(data[j].xiplus);
            double xim = real(data[j].ximinus);
            double xipi = imag(data[j].xiplus);
            double ximi = imag(data[j].ximinus);
            double var = data[j].varxi;
            double s = r/R;
            double ssqdlogr = s*s*dlogr;
            double tp = Tplus(uform,s), tm = Tminus(uform,s);
            double sp = Splus(s);
            //double sm = Sminus(s);

            mapsq += ssqdlogr*(tp*xip + tm*xim)/2.;
            mxsq  += ssqdlogr*(tp*xip - tm*xim)/2.;
            mmx_1 += ssqdlogr*(tp*xipi + tm*ximi)/2.;
            mmx_2 += -ssqdlogr*(tp*xipi - tm*ximi)/2.;
            varmap += SQR(ssqdlogr)*(SQR(tp)+SQR(tm))*var/4.;

            gamsq += ssqdlogr * sp * xip;
            vargam += SQR(ssqdlogr*sp)*var;
            //gamEsq += ssqdlogr * (sp*xip + sm*xim)/2.;
            //gamBsq += ssqdlogr * (sp*xip - sm*xim)/2.;
            //varEBgam += SQR(ssqdlogr)*(SQR(sp)+SQR(sm))*var/4.;
        }

        fout << sci(R);
        Assert(varmap >= 0.);
        fout << sci(mapsq) << sci(mxsq);
        fout << sci(mmx_1) << sci(mmx_2) << sci(sqrt(varmap));
        Assert(vargam >= 0.);
        fout << sci(gamsq) << sci(sqrt(vargam));
        fout << std::endl;
    }
}

void WriteNG(std::ostream& fout, double minsep, double binsize, double smoothscale,
             const std::string& ne_stat, int prec,
             const std::vector<BinData2<NData,GData> >& data,
             const std::vector<BinData2<NData,GData> >& rand)
{

    Assert(ne_stat == "compensated" || ne_stat == "simple");
    bool compensate = (ne_stat == "compensated");

    int width = prec + 8;
    Form sci;
    sci.sci().prec(prec).width(width).right().trail(1); 

    fout << str('#',"R_nominal",width) << str('.',"<R>",width);
    fout << str('.',"<gamT>",width) << str('.',"<gamX>",width) << str('.',"sig",width);

    if (compensate) {
        fout << str('.',"gamT_d",width) << str('.',"gamX_d",width);
        fout << str('.',"weight_d",width) << str('.',"npairs_d",width);
        fout << str('.',"gamT_r",width) << str('.',"gamX_r",width);
        fout << str('.',"weight_r",width) << str('.',"npairs_r",width);
    } else {
        fout << str('.',"weight",width) << str('.',"npairs",width);
    }
    if (smoothscale > 0.)
        fout << str('.',"R_sm",width) << str('.',"gamT_sm",width) << str('.',"sig_sm",width);
    fout << '.' << std::endl;

    for(int i=0;i<int(data.size());++i) {

        double R = exp(data[i].meanlogr);

        std::complex<double> g = data[i].meangammat;
        double vargam = data[i].vargammat;
        if (compensate) {
            // The formula derived in Rozo et al is:
            // g^ = ( lam Sum g_d - Sum g_r ) / Npairs_r
            // lam = Npairs_r / Npairs_d
            //
            // When we allow weighted averages, this becomes
            // g^ = ( (Sum w_r)/(Sum w_d) Sum w_d g_d - Sum w_r g_r ) / Sum w_r
            //    = Sum w_d g_d / Sum w_d - Sum w_r g_r / Sum w_r
            //
            // These ratios are what are already calculated in each histogram.
            // i.e. we just need to subtract the rand value to get the 
            // compensated estimator.
            g -= rand[i].meangammat;

            // The variance is just the sum of the two variances that have 
            // already been calculated.
            vargam += rand[i].vargammat;
        }

        fout << sci(minsep*exp((i+0.5)*binsize)) << sci(R);
        fout << sci(real(g)) << sci(imag(g));
        Assert(vargam >= 0.);
        fout << sci(sqrt(vargam));
        if (compensate) {
            fout << sci(real(data[i].meangammat));
            fout << sci(imag(data[i].meangammat));
            fout << sci(data[i].weight);
            fout << sci(data[i].npair);
            fout << sci(real(rand[i].meangammat));
            fout << sci(imag(rand[i].meangammat));
            fout << sci(rand[i].weight);
            fout << sci(rand[i].npair);
        } else {
            fout << sci(data[i].weight);
            fout << sci(data[i].npair);
        }

        if (smoothscale > 0.) {
            std::complex<double> gamsm=0.;
            double vargamsm=0.,weightsm=0.,meanlogrsm=0.;

            for(int j=0;j<int(data.size());++j) {

                double r = exp(data[j].meanlogr);
                double gam = real(data[j].meangammat);
                double vargam = data[j].vargammat;
                if (compensate) { 
                    gam -= real(rand[j].meangammat);
                    vargam += rand[j].vargammat;
                }
                double wj = data[j].weight;
                double s = r/R;

                if (s>1/smoothscale && s<smoothscale) {
                    meanlogrsm += data[j].meanlogr*wj;
                    gamsm += gam*wj;
                    vargamsm += vargam*SQR(wj);
                    weightsm += wj;
                }
            }

            if (weightsm > 0) {
                meanlogrsm /= weightsm;
                gamsm /= weightsm;
                vargamsm /= SQR(weightsm);
            }

            fout << sci((weightsm==0.) ? R : exp(meanlogrsm));
            fout << sci(real(gamsm));
            Assert(vargamsm >= 0.);
            fout << sci(sqrt(vargamsm));
        }
        fout << std::endl;
    }
}

void WriteNM(std::ostream& fout, double minsep, double binsize,
             const std::string& ne_stat, UForm uform, int prec,
             const std::vector<BinData2<NData,GData> >& data,
             const std::vector<BinData2<NData,GData> >& rand)
{
    Assert(ne_stat == "compensated" || ne_stat == "simple");
    bool compensate = (ne_stat == "compensated");

    int width = prec + 8;
    Form sci;
    sci.sci().prec(prec).width(width).right().trail(1); 

    fout << str('#',"R",width);
    fout << str('.',"<NMap>",width) << str('.',"<NMx>",width) << str('.',"sig_nmap",width);
    fout << '.' << std::endl;

    double dlogr = binsize;

    for(int i=0;i<int(data.size());++i) {

        double R = minsep*exp((i+0.5)*binsize);
        double nmap=0., nmx=0., varnm=0.;

        for(int j=0;j<int(data.size());++j) {

            double r = exp(data[j].meanlogr);
            double s = r/R;
            double ssqdlogr = s*s*dlogr;
            std::complex<double> gam = data[j].meangammat;
            double vargam = data[j].vargammat;
            if (compensate) { 
                gam -= rand[j].meangammat;
                vargam += rand[j].vargammat;
            }
            double tc = Tcross(uform,s);
            nmap += ssqdlogr * tc * real(gam);
            nmx += ssqdlogr * tc * imag(gam);
            varnm += SQR(ssqdlogr*tc)*vargam;
        }
        fout << sci(R);
        Assert(varnm >= 0.);
        fout << sci(nmap) << sci(nmx) << sci(sqrt(varnm));
        fout << std::endl;
    }
}

void WriteNorm(std::ostream& fout, double minsep, double binsize,
               const std::string& ne_stat, const std::string& nn_stat, 
               UForm uform, int prec,
               const std::vector<BinData2<NData,GData> >& ne,
               const std::vector<BinData2<NData,GData> >& re,
               const std::vector<BinData2<GData,GData> >& ee,
               const std::vector<BinData2<NData,NData> >& dd,
               const std::vector<BinData2<NData,NData> >& dr,
               const std::vector<BinData2<NData,NData> >& rr)
{
    Assert(ne_stat == "compensated" || ne_stat == "simple");
    bool ne_compensate = (ne_stat == "compensated");

    Assert(nn_stat == "compensated" || nn_stat == "simple");
    bool nn_compensate = (nn_stat == "compensated");

    Assert(ne.size() ==  ee.size());
    Assert(ne.size() ==  dd.size());
    Assert(ne.size() ==  rr.size());

    if (ne_compensate) { Assert(ne.size() == re.size()); }
    if (nn_compensate) { Assert(ne.size() == dr.size()); }

    int width = prec + 8;
    Form sci;
    sci.sci().prec(prec).width(width).right().trail(1); 

    fout << str('#',"R_nominal",width) << str('.',"<R>",width);
    fout << str('.',"<NMap>",width) << str('.',"<NMx>",width) << str('.',"sig_nm",width);
    fout << str('.',"<N^2>",width) << str('.',"sig_nn",width);
    fout << str('.',"<Map^2>",width) << str('.',"sig_mm",width);
    fout << str('.',"nmnorm",width) << str('.',"signmnorm",width);
    fout << str('.',"nnnorm",width) << str('.',"signnnorm",width);
    fout << '.' << std::endl;

    std::vector<double> omega(dd.size());
    std::vector<double> varomega(dd.size());

    for(int i=0;i<int(dd.size());++i) {
        if (rr[i].npair > 0) {
            if (nn_compensate) 
                omega[i] = (dd[i].npair-2*dr[i].npair+rr[i].npair)/rr[i].npair;
            else
                omega[i] = (dd[i].npair-rr[i].npair)/rr[i].npair;
            varomega[i] = 1./ rr[i].npair;
        } else {
            omega[i] = varomega[i] = 0.;
        }
    }

    double dlogr = binsize;

    for(int i=0;i<int(ne.size());++i) {
        double mapsq=0.,varmap=0.;
        double nmap=0., nmx=0., varnm=0., nn=0., varnn=0.;

        std::complex<double> gammatsm=0.;
        double vargammatsm=0.,crosswtsm=0.;

        double R = minsep*exp((i+0.5)*binsize);

        for(int j=0;j<int(ne.size());++j) {

            double r = exp(ee[j].meanlogr);
            double xip = real(ee[j].xiplus);
            double xim = real(ee[j].ximinus);
            double var = ee[j].varxi;
            double s = r/R;
            double ssqdlogr = s*s*dlogr;
            double tp = Tplus(uform,s), tm = Tminus(uform,s);

            mapsq += ssqdlogr*(tp*xip + tm*xim)/2.;
            varmap += SQR(ssqdlogr)*(SQR(tp)+SQR(tm))*var/4.;

            std::complex<double> gam = ne[j].meangammat;
            double vargam = ne[j].vargammat;
            if (ne_compensate) { 
                gam -= re[j].meangammat;
                vargam += re[j].vargammat;
            }
            double tc = Tcross(uform,s);
            nmap += ssqdlogr * tc * real(gam);
            nmx += ssqdlogr * tc * imag(gam);
            varnm += SQR(ssqdlogr*tc)*vargam;

            nn += ssqdlogr * tp * omega[j];
            varnn += SQR(ssqdlogr*tp)*varomega[j];
        }

        if (crosswtsm > 0) {
            gammatsm /= crosswtsm;
            vargammatsm /= SQR(crosswtsm);
        }

        double nmnorm = mapsq*nn == 0. ? 0. : nmap*nmap / (mapsq*nn);
        double varnmnorm = mapsq*nn == 0. ? 0. : SQR(nmap/mapsq/nn)*(4.*varnm) +
            SQR(nmnorm)*(varnn/SQR(nn) + varmap/SQR(mapsq));
        std::cout<<"nmnorm, varnmnorm = "<<nmnorm<<','<<varnmnorm<<std::endl;
        std::cout<<"varnmnorm = "<<SQR(nmap/mapsq/nn)*(4.*varnm)<<" + "<<
            SQR(nmnorm)*varnn/SQR(nn)<<" + "<<SQR(nmnorm)*varmap/SQR(mapsq)<<std::endl;

        double nnnorm = mapsq == 0. ? 0. : nn/mapsq;
        double varnnnorm = 
            mapsq == 0. ? 0. :
            varnn/SQR(mapsq) + varmap*SQR(nnnorm/mapsq);
        std::cout<<"nnnorm, varnnnorm = "<<nnnorm<<','<<varnnnorm<<std::endl;
        std::cout<<"varnnnorm = "<<varnn/SQR(mapsq)<<" + "<<varmap*SQR(nnnorm/mapsq)<<std::endl;

        fout << sci(minsep*exp((i+0.5)*binsize)) << sci(R);
        Assert(varnm >= 0.);
        fout << sci(nmap) << sci(nmx) << sci(sqrt(varnm));
        Assert(varnn >= 0.);
        fout << sci(nn) << sci(sqrt(varnn));
        Assert(varmap >= 0.);
        fout << sci(mapsq) << sci(sqrt(varmap));
        Assert(varnmnorm >= 0.);
        fout << sci(nmnorm) << sci(sqrt(varnmnorm));
        Assert(varnnnorm >= 0.);
        fout << sci(nnnorm) << sci(sqrt(varnnnorm));
        fout << std::endl;
    }
}

void WriteNN(std::ostream& fout, double minsep, double binsize,
             const std::string& nn_stat, int prec,
             const std::vector<BinData2<NData,NData> >& dd,
             const std::vector<BinData2<NData,NData> >& dr, 
             const std::vector<BinData2<NData,NData> >& rd, 
             const std::vector<BinData2<NData,NData> >& rr)
{
    Assert(nn_stat == "compensated" || nn_stat == "simple");
    bool compensate = (nn_stat == "compensated");

    Assert(dd.size() == rr.size());
    if (compensate) {
        Assert(dd.size() == dr.size());
        Assert(dd.size() == rd.size());
    }

    int width = prec + 8;
    Form sci;
    sci.sci().prec(prec).width(width).right().trail(1); 

    fout << str('#',"R_nominal",width) << str('.',"<R>",width);
    fout << str('.',"omega",width) << str('.',"sig_omega",width);
    fout << str('.',"DD",width) << str('.',"RR",width);
    if (compensate)
        fout << str('.',"DR",width) << str('.',"RD",width);
    fout << '.' << std::endl;

    std::vector<double> omega(dd.size());
    std::vector<double> varomega(dd.size());

    for(int i=0;i<int(dd.size());++i) {
        if (rr[i].npair > 0) {
            if (compensate) 
                omega[i] = (dd[i].npair-dr[i].npair-rd[i].npair+rr[i].npair)/rr[i].npair;
            else
                omega[i] = (dd[i].npair-rr[i].npair)/rr[i].npair;
            varomega[i] = 1./rr[i].npair;
        } else {
            omega[i] = varomega[i] = 0.;
        }
    }

    for(int i=0;i<int(omega.size());++i) {
        double R = exp(dd[i].meanlogr);
        fout << sci(minsep*exp((i+0.5)*binsize)) << sci(R);
        Assert(varomega[i] >= 0.);
        fout << sci(omega[i]) << sci(sqrt(varomega[i]));
        fout << sci(dd[i].npair) << sci(rr[i].npair);
        if (compensate)
            fout << sci(dr[i].npair) << sci(rd[i].npair);
        fout << std::endl;
    }
}


void WriteKK(std::ostream& fout, double minsep, double binsize, double smoothscale, int prec,
             const std::vector<BinData2<KData,KData> >& data)
{ 
    int width = prec + 8;
    Form sci;
    sci.sci().prec(prec).width(width).right().trail(1); 

    fout << str('#',"R_nominal",width) << str('.',"<R>",width);
    fout << str('.',"xi",width) << str('.',"sig_xi",width);
    fout << str('.',"weight",width) << str('.',"npairs",width);
    if (smoothscale > 0.)
        fout << str('.',"R_sm",width) << str('.',"xi_sm",width) << str('.',"sig_sm",width);
    fout << '.' << std::endl;

    for(int i=0;i<int(data.size());++i) {

        double R = exp(data[i].meanlogr);
        fout << sci(minsep*exp((i+0.5)*binsize)) << sci(R);
        fout << sci(data[i].xi);
        Assert(data[i].varxi >= 0.);
        fout << sci(sqrt(data[i].varxi));
        fout << sci(data[i].weight);
        fout << sci(data[i].npair);

        if (smoothscale > 0.) {
            double xism=0.;
            double varxism=0.,weightsm=0.,meanlogrsm=0.;

            for(int j=0;j<int(data.size());++j) {

                double r = exp(data[j].meanlogr);
                double xi = data[j].xi;
                double var = data[j].varxi;
                double wj = data[j].weight;
                double s = r/R;

                if (s>1/smoothscale && s<smoothscale) {
                    meanlogrsm += data[j].meanlogr*wj;
                    xism += xi*wj;
                    varxism += var*SQR(wj);
                    weightsm += wj;
                }
            }

            if (weightsm > 0) {
                meanlogrsm /= weightsm;
                xism /= weightsm;
                varxism /= SQR(weightsm);
            }

            fout << sci((weightsm==0.) ? R : exp(meanlogrsm));
            fout << sci(xism);
            Assert(varxism >= 0.);
            fout << sci(sqrt(varxism));
        }
        fout << std::endl;
    }
}

void WriteNK(std::ostream& fout, double minsep, double binsize, double smoothscale,
             const std::string& nk_stat, int prec,
             const std::vector<BinData2<NData,KData> >& data,
             const std::vector<BinData2<NData,KData> >& rand)
{
    Assert(nk_stat == "compensated" || nk_stat == "simple");
    bool compensate = (nk_stat == "compensated");

    int width = prec + 8;
    Form sci;
    sci.sci().prec(prec).width(width).right().trail(1); 

    fout << str('#',"R_nominal",width) << str('.',"<R>",width);
    fout << str('.',"<kappa>",width) << str('.',"sig",width);
    if (compensate) {
        fout << str('.',"kappa_d",width) << str('.',"weight_d",width) << str('.',"npairs_d",width);
        fout << str('.',"kappa_r",width) << str('.',"weight_r",width) << str('.',"npairs_r",width);
    } else {
        fout << str('.',"weight",width) << str('.',"npairs",width);
    }
    if (smoothscale > 0.)
        fout << str('.',"R_sm",width) << str('.',"kappa_sm",width) << str('.',"sig_sm",width);
    fout << '.' << std::endl;

    for(int i=0;i<int(data.size());++i) {

        double R = exp(data[i].meanlogr);

        double k = data[i].meankappa;
        double vark = data[i].varkappa;
        if (compensate) {
            k -= rand[i].meankappa;
            vark += rand[i].varkappa;
        }

        fout << sci(minsep*exp((i+0.5)*binsize)) << sci(R);
        fout << sci(k);
        Assert(vark >= 0.);
        fout << sci(sqrt(vark));
        if (compensate) {
            fout << sci(data[i].meankappa);
            fout << sci(data[i].weight);
            fout << sci(data[i].npair);
            fout << sci(rand[i].meankappa);
            fout << sci(rand[i].weight);
            fout << sci(rand[i].npair);
        } else {
            fout << sci(data[i].weight);
            fout << sci(data[i].npair);
        }

        if (smoothscale > 0.) {
            double ksm=0.;
            double varksm=0.,weightsm=0.,meanlogrsm=0.;

            for(int j=0;j<int(data.size());++j) {

                double r = exp(data[j].meanlogr);
                double k = data[j].meankappa;
                double vark = data[j].varkappa;
                if (compensate) { 
                    k -= rand[j].meankappa;
                    vark += rand[j].varkappa;
                }
                double wj = data[j].weight;
                double s = r/R;

                if (s>1/smoothscale && s<smoothscale) {
                    meanlogrsm += data[j].meanlogr*wj;
                    ksm += k*wj;
                    varksm += vark*SQR(wj);
                    weightsm += wj;
                }
            }

            if (weightsm > 0) {
                meanlogrsm /= weightsm;
                ksm /= weightsm;
                varksm /= SQR(weightsm);
            }

            fout << sci((weightsm==0.) ? R : exp(meanlogrsm));
            fout << sci(ksm);
            Assert(varksm >= 0.);
            fout << sci(sqrt(varksm));
        }
        fout << std::endl;
    }
}

void WriteKG(std::ostream& fout, double minsep, double binsize, double smoothscale, int prec,
             const std::vector<BinData2<KData,GData> >& data)
{
    int width = prec + 8;
    Form sci;
    sci.sci().prec(prec).width(width).right().trail(1); 

    fout << str('#',"R_nominal",width) << str('.',"<R>",width);
    fout << str('.',"<kgamT>",width) << str('.',"<kgamX>",width) << str('.',"sig",width);
    fout << str('.',"weight",width) << str('.',"npairs",width);
    if (smoothscale > 0.)
        fout << str('.',"R_sm",width) << str('.',"kgamT_sm",width) << str('.',"sig_sm",width);
    fout << '.' << std::endl;

    for(int i=0;i<int(data.size());++i) {

        double R = exp(data[i].meanlogr);

        std::complex<double> kg = data[i].meankgammat;
        double varkg = data[i].varkgammat;

        fout << sci(minsep*exp((i+0.5)*binsize)) << sci(R);
        fout << sci(real(kg)) << sci(imag(kg));
        Assert(varkg >= 0.);
        fout << sci(sqrt(varkg));
        fout << sci(data[i].weight);
        fout << sci(data[i].npair);

        if (smoothscale > 0.) {
            std::complex<double> kgsm=0.;
            double varkgsm=0.,weightsm=0.,meanlogrsm=0.;

            for(int j=0;j<int(data.size());++j) {

                double r = exp(data[j].meanlogr);
                double kg = real(data[j].meankgammat);
                double varkg = data[j].varkgammat;
                double wj = data[j].weight;
                double s = r/R;

                if (s>1/smoothscale && s<smoothscale) {
                    meanlogrsm += data[j].meanlogr*wj;
                    kgsm += kg*wj;
                    varkgsm += varkg*SQR(wj);
                    weightsm += wj;
                }
            }

            if (weightsm > 0) {
                meanlogrsm /= weightsm;
                kgsm /= weightsm;
                varkgsm /= SQR(weightsm);
            }

            fout << sci((weightsm==0.) ? R : exp(meanlogrsm));
            fout << sci(real(kgsm));
            Assert(varkgsm >= 0.);
            fout << sci(sqrt(varkgsm));
        }
        fout << std::endl;
    }
}

