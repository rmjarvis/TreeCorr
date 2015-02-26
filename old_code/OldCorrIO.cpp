
#include "dbg.h"
#include "OldCorr.h"
#include "Form.h"
#include "CalcT.h"
#include "OldCell.h"
#include "Cell3D.h"

template <class T> inline T SQR(T x) { return x*x; }

const UForm uform = Crittenden;

void WriteEEE(
    std::ostream& fout, double minsep, double binsize,
    const std::vector<std::vector<std::vector<BinData3<3,3,3> > > >& data)
{
    const double du = 1./int(ceil(1./binsize));
    const double dv = du;
    // du (= dv) = 1/nubins (= 2/nvbins)

    Form sci; sci.sci().prec(10).width(15).trail(1); 
    Form fix; fix.fix().prec(10).width(15).trail(1);

    fout <<"#meanr  actual   meanu    actual      meanv     actual      gam0r    sigma      gam0i    sigma     gam1r    sigma      gam1i     sigma     gam2r    sigma       gam2i    sigma     gam3r     sigma      gam3i    sigma    weight     ntri\n";

    for(int i=0;i<int(data.size());++i) {
        for(int j=0;j<int(data[i].size());++j) {
            for(int k=0;k<int(data[i][j].size());++k) {
                const BinData3<3,3,3>& bindata = data[i][j][k];
                if (bindata.weight > 0.) {
                    fout << fix(minsep*exp((i+0.5)*binsize));
                    fout << fix(bindata.meanr);
                    fout << fix((j+0.5)*du);
                    fout << fix(bindata.meanu);
                    fout << fix((k+0.5)*dv-1.);
                    fout << fix(bindata.meanv);
                    Assert(bindata.vargam >= 0.);
                    double sig = sqrt(bindata.vargam);
                    fout << sci(real(bindata.gam0)) << sci(sig);
                    fout << sci(imag(bindata.gam0)) << sci(sig);
                    fout << sci(real(bindata.gam1)) << sci(sig);
                    fout << sci(imag(bindata.gam1)) << sci(sig);
                    fout << sci(real(bindata.gam2)) << sci(sig);
                    fout << sci(imag(bindata.gam2)) << sci(sig);
                    fout << sci(real(bindata.gam3)) << sci(sig);
                    fout << sci(imag(bindata.gam3)) << sci(sig);
                    fout << sci(bindata.weight) << sci(bindata.ntri);
                    fout << std::endl;
                }
            }
        }
    }
}

void Read(
    std::istream& fin, double minsep, double binsize,
    std::vector<std::vector<std::vector<BinData3<3,3,3> > > >& data) 
{
    const double du = 1./int(ceil(1./binsize));
    const double dv = du;
    // du (= dv) = 1/nubins (= 2/nvbins)

    std::string line;

    if (!getline(fin,line)) myerror("reading first line"+line);

    xdbg<<"first line = "<<line<<std::endl;

    while (getline(fin,line)) {
        //xdbg<<"line = "<<line<<std::endl;
        std::istringstream linein(line);

        double r,r2,u,u2,v,v2;
        linein >> r >> r2 >> u >> u2 >> v >> v2;
        int i = int(floor(log(r/minsep)/binsize));
        int j = int(floor(u/du));
        int k = int(floor((v+1)/dv));
        BinData3<3,3,3>& bindata = data[i][j][k];
        bindata.meanr = r2;
        bindata.meanu = u2;
        bindata.meanv = v2;

        double g0r,g0i,g1r,g1i,g2r,g2i,g3r,g3i,s;
        linein >> g0r >> s >> g0i >> s >> g1r >> s >> g1i >> s;
        linein >> g2r >> s >> g2i >> s >> g3r >> s >> g3i >> s;

        bindata.gam0 = std::complex<double>(g0r,g0i);
        bindata.gam1 = std::complex<double>(g1r,g1i);
        bindata.gam2 = std::complex<double>(g2r,g2i);
        bindata.gam3 = std::complex<double>(g3r,g3i);
        bindata.vargam = s*s;

        linein >> bindata.weight >> bindata.ntri;
        if (!linein) myerror("reading line: "+line);
    }
}

void WriteM3(
    std::ostream& fout, double minsep, double binsize,
    const std::vector<std::vector<std::vector<BinData3<3,3,3> > > >& data, 
    double k1, double k2, double k3) 
{
    Form sci; sci.sci().prec(2).width(9).trail(1); 
    Form fix; fix.fix().prec(3).width(9).trail(1);

    fout <<"#r   map^3  sigma  map^2mx sigma  mapmx^2  sigma  mx^3  sigma    <M^3>     <M^2M*>\n";

    for(int n=0;n<int(data.size());++n) {
        double R = minsep*exp((n+0.5)*binsize);
        double Rsq = R*R;
        std::complex<double> mmm=0,mmmc=0.,mmcm=0.,mcmm=0.;
        double varmmm=0.,varmmmc=0.,varmmcm=0.,varmcmm=0.;
        for(int i=0;i<int(data.size());++i) {
            for(int j=0;j<int(data[i].size());++j) {
                for(int k=0;k<int(data[i][j].size());++k) {
                    const BinData3<3,3,3>& bindata = data[i][j][k];
                    if (bindata.weight == 0.) continue;
                    if (bindata.meanu == 0.) continue;

                    double s = bindata.meanr;
                    double sds = s*s*binsize/Rsq; // binsize = dlogs
                    // this is really sds/R^2

                    double d2 = s/bindata.meanu;
                    double d1 = std::abs(bindata.meanv)*s+d2;
                    double tx = (s*s+d2*d2-d1*d1)/(2*s);
                    double d2sqmtxsq = d2*d2 - tx*tx;
                    Assert(d2sqmtxsq >= 0.);
                    double ty = sqrt(d2sqmtxsq);
                    std::complex<double> t(tx,ty);
                    if (bindata.meanv<0) t = conj(t);

                    double twoarea = s*ty;
                    double jac = d2*d2*d2*d1/twoarea;
                    // jac = |J(tx,ty;u,v)|
                    double d2t = binsize*binsize*jac/Rsq/TWOPI;
                    // binsize = du = dv

                    std::complex<double> G0 = bindata.gam0;
                    std::complex<double> G1 = bindata.gam1;
                    std::complex<double> G2 = bindata.gam2;
                    std::complex<double> G3 = bindata.gam3;
                    double varG = bindata.vargam;

                    std::complex<double> T0,T1,T2,T3;

                    CalcT(uform,s/R,t/R,&T0,&T1,&T2,&T3,k1,k2,k3);
                    mmm += sds*d2t*G0*T0;
                    mcmm += sds*d2t*(G1*T1+G2*T2+G3*T3);
                    varmmm += SQR(sds*d2t)*norm(T0)*varG;
                    varmcmm += SQR(sds*d2t)*(norm(T1)+norm(T2)+norm(T3))*varG;

                    CalcT(uform,s/R,t/R,&T0,&T1,&T2,&T3,k1,k3,k2);
                    mmm += sds*d2t*G0*T0;
                    mcmm += sds*d2t*(G1*T1+G2*T2+G3*T3);
                    varmmm += SQR(sds*d2t)*norm(T0)*varG;
                    varmcmm += SQR(sds*d2t)*(norm(T1)+norm(T2)+norm(T3))*varG;

                    CalcT(uform,s/R,t/R,&T0,&T1,&T2,&T3,k2,k1,k3);
                    mmm += sds*d2t*G0*T0;
                    mmcm += sds*d2t*(G1*T1+G2*T2+G3*T3);
                    varmmm += SQR(sds*d2t)*norm(T0)*varG;
                    varmmcm += SQR(sds*d2t)*(norm(T1)+norm(T2)+norm(T3))*varG;

                    CalcT(uform,s/R,t/R,&T0,&T1,&T2,&T3,k2,k3,k1);
                    mmm += sds*d2t*G0*T0;
                    mmcm += sds*d2t*(G1*T1+G2*T2+G3*T3);
                    varmmm += SQR(sds*d2t)*norm(T0)*varG;
                    varmmcm += SQR(sds*d2t)*(norm(T1)+norm(T2)+norm(T3))*varG;

                    CalcT(uform,s/R,t/R,&T0,&T1,&T2,&T3,k3,k1,k2);
                    mmm += sds*d2t*G0*T0;
                    mmmc += sds*d2t*(G1*T1+G2*T2+G3*T3);
                    varmmm += SQR(sds*d2t)*norm(T0)*varG;
                    varmmmc += SQR(sds*d2t)*(norm(T1)+norm(T2)+norm(T3))*varG;

                    CalcT(uform,s/R,t/R,&T0,&T1,&T2,&T3,k3,k2,k1);
                    mmm += sds*d2t*G0*T0;
                    mmmc += sds*d2t*(G1*T1+G2*T2+G3*T3);
                    varmmm += SQR(sds*d2t)*norm(T0)*varG;
                    varmmmc += SQR(sds*d2t)*(norm(T1)+norm(T2)+norm(T3))*varG;
                }
            }
        }

        double map3 = 0.25*real(mcmm+mmcm+mmmc+mmm);
        double mapmapmx = 0.25*imag(mcmm+mmcm-mmmc+mmm);
        double mapmxmap = 0.25*imag(mcmm-mmcm+mmmc+mmm);
        double mxmapmap = 0.25*imag(-mcmm+mmcm+mmmc+mmm);
        double mxmxmap = 0.25*real(mcmm+mmcm-mmmc-mmm);
        double mxmapmx = 0.25*real(mcmm-mmcm+mmmc-mmm);
        double mapmxmx = 0.25*real(-mcmm+mmcm+mmmc-mmm);
        double mx3 = 0.25*imag(mcmm+mmcm+mmmc-mmm);
        double var = (varmmmc + varmmcm + varmcmm + varmmm)/32.;
        Assert(var >= 0.);
        double sig = sqrt(var); 

        fout << fix(R);
        fout << sci(map3) << sci(sig);
        fout << sci(mapmapmx) << sci(sig);
        fout << sci(mapmxmap) << sci(sig);
        fout << sci(mxmapmap) << sci(sig);
        fout << sci(mxmxmap) << sci(sig);
        fout << sci(mxmapmx) << sci(sig);
        fout << sci(mapmxmx) << sci(sig);
        fout << sci(mx3) << sci(sig);
        fout << sci(mmm) << sci(mcmm) << sci(mmcm) << sci(mmmc);
        fout << std::endl;
    }
}

void WriteEE(std::ostream& fout, double minsep, double binsize, double smoothscale,
             std::vector<BinData2<3,3> >& data)
{
    static Form sci;
    static Form fix; 
    static bool first = true;
    if (first) {
        sci.sci().prec(2).width(9).trail(1); 
        fix.fix().prec(3).width(9).trail(1);
        first = false;
    }

    fout <<"#  R(')  .   xi+   . sig_xi+ .   xi-   . sig_xi- .  xi+_im .sig_xi+im.  xi-_im .sig_xi-im. R_sm(') .  xi+_sm .sig_xi+sm.  xi-_sm .sig_xi-sm.  weight . npairs\n";
    for(int i=0;i<int(data.size());++i) {
        std::complex<double> xiplussm=0.,ximinussm=0.;
        double varxism=0.,weightsm=0.,meanlogrsm=0.;

        double R = exp(data[i].meanlogr);

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

        fout << fix(R);
        Assert(data[i].varxi >= 0.);
        fout << sci(real(data[i].xiplus)) << sci(sqrt(data[i].varxi));
        fout << sci(real(data[i].ximinus)) << sci(sqrt(data[i].varxi));
        fout << sci(imag(data[i].xiplus)) << sci(sqrt(data[i].varxi));
        fout << sci(imag(data[i].ximinus)) << sci(sqrt(data[i].varxi));
        fout << fix((weightsm==0.) ? R : exp(meanlogrsm));
        Assert(varxism >= 0.);
        fout << sci(real(xiplussm)) << sci(sqrt(varxism));
        fout << sci(real(ximinussm)) << sci(sqrt(varxism));
        fout << sci(data[i].weight) << sci(data[i].npair);
        fout << std::endl;
    }
}

void WriteM2(std::ostream& fout, double minsep, double binsize,
             std::vector<BinData2<3,3> >& data)
{
    static Form sci;
    static Form fix; 
    static bool first = true;
    if (first) {
        sci.sci().prec(2).width(9).trail(1); 
        fix.fix().prec(3).width(9).trail(1);
        first = false;
    }

    // Use s = r/R:
    // <Map^2>(R) = int_r=0..2R [s^2 dlogr (T+(s) xi+(r) + T-(s) xi-(r))/2]
    // <Mx^2>(R)  =     "       [ "    "   (T+(s) xi+(r) - T-(s) xi-(r))/2]
    // <Gam^2>(R) = int_r=0..2R [s^2 dlogr S+(s) xi+(r)]
    // <Gam^2>(E/B)(R) = int_r=0..2R [s^2 dlogr (S+(s) xi+(r) +- S+(s) xi-(r))/2]

    const double dlogr = binsize;

    fout <<"#  R(')  . <Map^2> . sig_map .  <Mx^2> .  sig_mx .  <MMx>  . sig_mmx .  <zero> .sig_zero . <Gam^2> . sig_gam\n";
    for(int i=0;i<int(data.size());++i) {
        double mapsq=0.,mxsq=0.,varmap=0.,mmx=0.,zero=0.;
        double gamsq=0.,vargam=0.;
        //double gamEsq=0.,gamBsq=0.,varEBgam=0.;

        double R = exp(data[i].meanlogr);

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
            mmx   += ssqdlogr*(tp*xipi + tm*ximi)/2.;
            zero  += ssqdlogr*(tp*xipi - tm*ximi)/2.;
            varmap += SQR(ssqdlogr)*(SQR(tp)+SQR(tm))*var/4.;

            gamsq += ssqdlogr * sp * xip;
            vargam += SQR(ssqdlogr*sp)*var;
            //gamEsq += ssqdlogr * (sp*xip + sm*xim)/2.;
            //gamBsq += ssqdlogr * (sp*xip - sm*xim)/2.;
            //varEBgam += SQR(ssqdlogr)*(SQR(sp)+SQR(sm))*var/4.;

#if 0
            if (R > 10.*60. && R < 10.5*60.) {
                xdbg<<"mapsq += "<<ssqdlogr*tp/2.<<"*"<<xip<<" + "<<ssqdlogr*tm/2.<<"*"<<xim<<" = "<<ssqdlogr*(tp*xip+tm*xim)/2.<<std::endl;
                xdbg<<"int(0.."<<r/60.<<") = "<<mapsq<<std::endl;
            }
#endif
        }

        fout << fix(R);
        Assert(varmap >= 0.);
        fout << sci(mapsq) << sci(sqrt(varmap));
        fout << sci(mxsq) << sci(sqrt(varmap));
        fout << sci(mmx) << sci(sqrt(varmap));
        fout << sci(zero) << sci(sqrt(varmap));
        fout << sci(gamsq) << sci(sqrt(vargam));
        fout << std::endl;
    }
}

void Read(std::istream& fin, double minsep, double binsize,
          std::vector<NCellData>& celldata)
{
    double x,y;
    while (fin >> x>>y) {
        Position2D pos(x,y);
        celldata.push_back(NCellData(pos));
    }
    dbg<<"ngal = "<<celldata.size()<<std::endl;
}

void Read(std::istream& fin, double minsep, double binsize,
          std::vector<CellData>& celldata, double& vare)
{
    double sumw=0.;
    vare=0.;
    double x,y,e1,e2,w;
    while (fin >> x>>y>>e1>>e2>>w) {
        // Note: e1,e2 are really gamma_1, gamma_2
        // _NOT_ observed ellipticities

        Position2D pos(x,y);
        std::complex<double> e(e1,e2);
        sumw += w;
        vare += w*w*norm(e);

        celldata.push_back(CellData(pos,e,w));
    }
    vare /= 2*sumw*sumw/celldata.size(); // 2 because want var per component
    dbg<<"ngal = "<<celldata.size()<<", totw = "<<sumw<<std::endl;
}

void Read(std::istream& fin, double minsep, double binsize,
          std::vector<NCell3DData>& celldata)
{
    double x,y,z;
    while (fin >> x>>y>>z) {
        Position3D pos(x,y,z);
        celldata.push_back(NCell3DData(pos));
    }
    dbg<<"ngal = "<<celldata.size()<<std::endl;
}

void Read(std::istream& fin, double minsep, double binsize,
          std::vector<TCell3DData>& celldata)
{
    double sumw=0.;
    double x,y,z,t,w;
    while (fin >> x>>y>>z>>t>>w) {
        Position3D pos(x,y,z);
        sumw += w;

        celldata.push_back(TCell3DData(pos,t,w));
    }
    dbg<<"ngal = "<<celldata.size()<<", totw = "<<sumw<<std::endl;
}

void WriteNE(std::ostream& fout, double minsep, double binsize, double smoothscale,
             const std::vector<BinData2<1,3> >& crossdata)
{
    static Form sci;
    static Form fix; 
    static bool first = true;
    if (first) {
        sci.sci().prec(2).width(9).trail(1); 
        fix.fix().prec(3).width(9).trail(1);
        first = false;
    }

    double dlogr = binsize;

    fout <<"#  R(')  . <NMap>  . sig_nmap.  <NMx>  . sig_nmx . <gamT>  .  sig_gT .  <gamX> .  sig_gX \n";
    for(int i=0;i<int(crossdata.size());++i) {
        double nmap=0., nmx=0., varnm=0.;

        std::complex<double> gammatsm=0.;
        double vargammatsm=0.,crosswtsm=0.;

        double R = exp(crossdata[i].meanlogr);

        for(int j=0;j<int(crossdata.size());++j) {

            double r = exp(crossdata[j].meanlogr);
            double s = r/R;
            double ssqdlogr = s*s*dlogr;

            if (s>1/smoothscale && s<smoothscale) {
                crosswtsm += crossdata[j].weight;
                gammatsm += crossdata[j].weight * crossdata[j].meangammat;
                vargammatsm += SQR(crossdata[j].weight) * crossdata[j].vargammat;
            }

            double gt = real(crossdata[j].meangammat);
            double gx = imag(crossdata[j].meangammat);
            double tc = Tcross(uform,s);
            nmap += ssqdlogr * tc * gt;
            nmx += ssqdlogr * tc * gx;
            varnm += SQR(ssqdlogr*tc)*crossdata[j].vargammat;
        }

        if (crosswtsm > 0) {
            gammatsm /= crosswtsm;
            vargammatsm /= SQR(crosswtsm);
        }

        fout << fix(R);
        Assert(varnm >= 0.);
        fout << sci(nmap) << sci(sqrt(varnm));
        fout << sci(nmx) << sci(sqrt(varnm));
        Assert(vargammatsm >= 0.);
        fout << sci(real(gammatsm)) << sci(sqrt(vargammatsm));
        fout << sci(imag(gammatsm)) << sci(sqrt(vargammatsm));
        fout << std::endl;
    }
}

void WriteNorm(
    std::ostream& fout, double minsep, double binsize, double smoothscale,
    const std::vector<BinData2<1,3> >& crossdata,
    const std::vector<BinData2<3,3> >& twoptdata,
    const std::vector<BinData2<1,1> >& dd, const std::vector<BinData2<1,1> >& dr,
    const std::vector<BinData2<1,1> >& rr)
{
    static Form sci;
    static Form fix; 
    static bool first = true;
    if (first) {
        sci.sci().prec(2).width(9).trail(1); 
        fix.fix().prec(3).width(9).trail(1);
        first = false;
    }

    Assert(crossdata.size() ==  twoptdata.size());
    Assert(crossdata.size() ==  dd.size());
    Assert(crossdata.size() ==  dr.size());
    Assert(crossdata.size() ==  rr.size());

    std::vector<double> omega(dd.size());
    std::vector<double> varomega(dd.size());

    for(int i=0;i<int(dd.size());++i) {
        if (rr[i].npair > 0) {
            omega[i] = (dd[i].npair-2*dr[i].npair+rr[i].npair)/rr[i].npair;
            varomega[i] = 1./ rr[i].npair;
        } else {
            omega[i] = varomega[i] = 0.;
        }
    }

    double dlogr = binsize;

    fout <<"#  R(')  . <NMap>  . sig_nmap.  <NMx>  . sig_nmx . <gamT>  .  sig_gT .  <gamX> .  sig_gX .   <NN>  .  sig_nn .  nmnorm .signmnorm.  nnnorm .signnnorm. <Map^2> .  sig_mm \n";
    for(int i=0;i<int(crossdata.size());++i) {
        double mapsq=0.,varmap=0.;
        double nmap=0., nmx=0., varnm=0., nn=0., varnn=0.;

        std::complex<double> gammatsm=0.;
        double vargammatsm=0.,crosswtsm=0.;

        double R = exp(twoptdata[i].meanlogr);

        for(int j=0;j<int(crossdata.size());++j) {

            double r = exp(twoptdata[j].meanlogr);
            double xip = real(twoptdata[j].xiplus);
            double xim = real(twoptdata[j].ximinus);
            double var = twoptdata[j].varxi;
            double s = r/R;
            double ssqdlogr = s*s*dlogr;
            double tp = Tplus(uform,s), tm = Tminus(uform,s);

            mapsq += ssqdlogr*(tp*xip + tm*xim)/2.;
            varmap += SQR(ssqdlogr)*(SQR(tp)+SQR(tm))*var/4.;

            if (s>1/smoothscale && s<smoothscale) {
                crosswtsm += crossdata[j].weight;
                gammatsm += crossdata[j].weight * crossdata[j].meangammat;
                vargammatsm += SQR(crossdata[j].weight) * crossdata[j].vargammat;
            }

            double gt = real(crossdata[j].meangammat);
            double gx = imag(crossdata[j].meangammat);
            double tc = Tcross(uform,s);
            nmap += ssqdlogr * tc * gt;
            nmx += ssqdlogr * tc * gx;
            varnm += SQR(ssqdlogr*tc)*crossdata[j].vargammat;

            nn += ssqdlogr * tp * omega[j];
            varnn += SQR(ssqdlogr*tp)*varomega[j];
        }

        if (crosswtsm > 0) {
            gammatsm /= crosswtsm;
            vargammatsm /= SQR(crosswtsm);
        }

        double nmnorm = mapsq*nn == 0. ? 0. : nmap*nmap / (mapsq*nn);
        double varnmnorm = mapsq*nn == 0. ? 0. : SQR(nmap/mapsq/nn)*(4.*varnm) +
            SQR(nmap)*(varnn/SQR(nn) + varmap/SQR(mapsq));

        double nnnorm = mapsq == 0. ? 0. : nn/mapsq;
        double varnnnorm = 
            mapsq == 0. ? 0. :
            varnn/SQR(mapsq) + varmap*SQR(nnnorm/mapsq);

        fout << fix(R);
        Assert(varnm >= 0.);
        fout << sci(nmap) << sci(sqrt(varnm));
        fout << sci(nmx) << sci(sqrt(varnm));
        Assert(vargammatsm >= 0.);
        fout << sci(real(gammatsm)) << sci(sqrt(vargammatsm));
        fout << sci(imag(gammatsm)) << sci(sqrt(vargammatsm));
        Assert(varnn >= 0.);
        fout << sci(nn) << sci(sqrt(varnn));
        Assert(varnmnorm >= 0.);
        Assert(varnnnorm >= 0.);
        fout << sci(nmnorm) << sci(sqrt(varnmnorm));
        fout << sci(nnnorm) << sci(sqrt(varnnnorm));
        Assert(varmap >= 0.);
        fout << sci(mapsq) << sci(sqrt(varmap));
        fout << std::endl;
    }
}

void WriteNN(
    std::ostream& fout, double minsep, double binsize,
    const std::vector<BinData2<1,1> >& dd, const std::vector<BinData2<1,1> >& dr, 
    const std::vector<BinData2<1,1> >& rr, const double ndd)
{
    Assert(dd.size() == dr.size());
    Assert(dd.size() == rr.size());

    std::vector<double> omega(dd.size());
    std::vector<double> varomega(dd.size());

    for(int i=0;i<int(dd.size());++i) {
        if (rr[i].npair > 0) {
            omega[i] = (dd[i].npair-2*dr[i].npair+rr[i].npair)/rr[i].npair;
            varomega[i] = 1./(rr[i].npair*ndd);
        } else {
            omega[i] = varomega[i] = 0.;
        }
    }

    static Form sci;
    static Form fix; 
    static bool first = true;
    if (first) {
        sci.sci().prec(2).width(9).trail(1); 
        fix.fix().prec(3).width(9).trail(1);
        first = false;
    }

    fout <<"#  R   omega   sig   dd   dr   rr\n";
    for(int i=0;i<int(omega.size());++i) {
        double R = minsep*exp((i+0.5)*binsize);

        fout << fix(R);
        Assert(varomega[i] >= 0.);
        fout << sci(omega[i]) << sci(sqrt(varomega[i]));
        fout << sci(dd[i].npair) << sci(dr[i].npair) << sci(rr[i].npair);
        fout << std::endl;
    }
}

void WriteNNN(
    std::ostream& fout, double minsep, double binsize,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& ddd, 
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& ddr,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& drr,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& rrr)
{
    const double du = 1./int(ceil(1./binsize));
    const double dv = du;
    // du (= dv) = 1/nubins (= 2/nvbins)

    Assert(ddd.size() == ddr.size());
    Assert(ddd.size() == drr.size());
    Assert(ddd.size() == rrr.size());

    std::vector<std::vector<std::vector<double> > > zeta(
        ddd.size(),
        std::vector<std::vector<double> >(
            ddd[0].size(),
            std::vector<double>(ddd[0][0].size(),0.)));
    std::vector<std::vector<std::vector<double> > > varzeta(
        ddd.size(),
        std::vector<std::vector<double> >(
            ddd[0].size(),
            std::vector<double>(ddd[0][0].size(),0.)));

    for(int i=0;i<int(ddd.size());++i) {
        for(int j=0;j<int(ddd[i].size());++j) {
            for(int k=0;k<int(ddd[i][j].size());++k) {
                if (rrr[i][j][k].ntri > 0) {
                    zeta[i][j][k] =
                        (ddd[i][j][k].ntri-3.*ddr[i][j][k].ntri+
                         3.*drr[i][j][k].ntri-rrr[i][j][k].ntri)/rrr[i][j][k].ntri;
                    varzeta[i][j][k] = 1./ rrr[i][j][k].ntri;
                } else {
                    zeta[i][j][k] = varzeta[i][j][k] = 0.;
                }
            }
        }
    }

    static Form sci;
    static Form fix; 
    static bool first = true;
    if (first) {
        sci.sci().prec(2).width(9).trail(1); 
        fix.fix().prec(3).width(9).trail(1);
        first = false;
    }

    fout <<"#  R(')   u    v     zeta   sig   ddd   ddr   drr    rrr\n";
    for(int i=0;i<int(zeta.size());++i) {
        double R = minsep*exp((i+0.5)*binsize);
        for(int j=0;j<int(zeta[i].size());++j) {
            double u = (j+0.5)*du;
            for(int k=0;k<int(zeta[i][j].size());++k) {
                double v = (k+0.5)*dv-1.;
                fout << fix(R) << fix(u) << fix(v);
                Assert(varzeta[i][j][k] >= 0.);
                double sig = sqrt(varzeta[i][j][k]);
                fout << sci(zeta[i][j][k]) << sci(sig);
                fout << sci(ddd[i][j][k].ntri) << sci(ddr[i][j][k].ntri) << 
                    sci(drr[i][j][k].ntri) << sci(rrr[i][j][k].ntri);
                fout << std::endl;
            }
        }
    }
}

void WriteNNN(
    std::ostream& fout, double minsep, double binsize,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& ddd,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& ddr,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& drd,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& rdd,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& drr,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& rdr,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& rrd,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& rrr)
{
    const double du = 1./int(ceil(1./binsize));
    const double dv = du;
    // du (= dv) = 1/nubins (= 2/nvbins)

    Assert(ddd.size() == ddr.size());
    Assert(ddd.size() == drr.size());
    Assert(ddd.size() == rrr.size());

    std::vector<std::vector<std::vector<double> > > zeta(
        ddd.size(),
        std::vector<std::vector<double> >(
            ddd[0].size(),
            std::vector<double>(ddd[0][0].size(),0.)));
    std::vector<std::vector<std::vector<double> > > varzeta(
        ddd.size(),
        std::vector<std::vector<double> >(
            ddd[0].size(),
            std::vector<double>(ddd[0][0].size(),0.)));

    for(int i=0;i<int(ddd.size());++i) {
        for(int j=0;j<int(ddd[i].size());++j) {
            for(int k=0;k<int(ddd[i][j].size());++k) {
                if (rrr[i][j][k].ntri > 0) {
                    zeta[i][j][k] = 
                        (ddd[i][j][k].ntri-ddr[i][j][k].ntri-
                         drd[i][j][k].ntri-rdd[i][j][k].ntri+
                         drr[i][j][k].ntri+rdr[i][j][k].ntri+
                         rrd[i][j][k].ntri-rrr[i][j][k].ntri)/rrr[i][j][k].ntri;
                    varzeta[i][j][k] = 1./ rrr[i][j][k].ntri;
                } else {
                    zeta[i][j][k] = varzeta[i][j][k] = 0.;
                }
            }
        }
    }

    static Form sci;
    static Form fix; 
    static bool first = true;
    if (first) {
        sci.sci().prec(2).width(9).trail(1); 
        fix.fix().prec(3).width(9).trail(1);
        first = false;
    }

    fout <<"#  R(')   u    v    zeta   sig   ddd   ddr   drd   rdd  drr  rdr  rrd  rrr\n";
    for(int i=0;i<int(zeta.size());++i) {
        double R = minsep*exp((i+0.5)*binsize);
        for(int j=0;j<int(zeta[i].size());++j) {
            double u = (j+0.5)*du;
            for(int k=0;k<int(zeta[i][j].size());++k) {
                double v = (k+0.5)*dv-1.;
                fout << fix(R) << fix(u) << fix(v);
                Assert(varzeta[i][j][k] >= 0.);
                double sig = sqrt(varzeta[i][j][k]);
                fout << sci(zeta[i][j][k]) << sci(sig);
                fout << sci(ddd[i][j][k].ntri) << sci(ddr[i][j][k].ntri) << 
                    sci(drd[i][j][k].ntri) << sci(rdd[i][j][k].ntri);
                fout << sci(drr[i][j][k].ntri) << sci(rdr[i][j][k].ntri) << 
                    sci(rrd[i][j][k].ntri) << sci(rrr[i][j][k].ntri);
                fout << std::endl;
            }
        }
    }
}

void WriteNNN(
    std::ostream& fout, double minsep, double binsize,
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& ddd, 
    const std::vector<std::vector<std::vector<BinData3<1,1,1> > > >& rrr)
{
    const double du = 1./int(ceil(1./binsize));
    const double dv = du;
    // du (= dv) = 1/nubins (= 2/nvbins)

    Assert(ddd.size() == rrr.size());

    std::vector<std::vector<std::vector<double> > > zeta(
        ddd.size(),
        std::vector<std::vector<double> >(
            ddd[0].size(),
            std::vector<double>(ddd[0][0].size(),0.)));
    std::vector<std::vector<std::vector<double> > > varzeta(
        ddd.size(),
        std::vector<std::vector<double> >(
            ddd[0].size(),
            std::vector<double>(ddd[0][0].size(),0.)));

    for(int i=0;i<int(ddd.size());++i) {
        for(int j=0;j<int(ddd[i].size());++j) {
            for(int k=0;k<int(ddd[i][j].size());++k) {
                if (rrr[i][j][k].ntri > 0) {
                    zeta[i][j][k] = (ddd[i][j][k].ntri-rrr[i][j][k].ntri)/rrr[i][j][k].ntri;
                    varzeta[i][j][k] = 1./ rrr[i][j][k].ntri;
                } else {
                    zeta[i][j][k] = varzeta[i][j][k] = 0.;
                }
            }
        }
    }

    static Form sci;
    static Form fix; 
    static bool first = true;
    if (first) {
        sci.sci().prec(2).width(9).trail(1); 
        fix.fix().prec(3).width(9).trail(1);
        first = false;
    }

    fout <<"#  R(')   u    v    zeta   sig   ddd    rrr\n";
    for(int i=0;i<int(zeta.size());++i) {
        double R = minsep*exp((i+0.5)*binsize);
        for(int j=0;j<int(zeta[i].size());++j) {
            double u = (j+0.5)*du;
            for(int k=0;k<int(zeta[i][j].size());++k) {
                double v = (k+0.5)*dv-1.;
                fout << fix(R) << fix(u) << fix(v);
                Assert(varzeta[i][j][k] >= 0.);
                double sig = sqrt(varzeta[i][j][k]);
                fout << sci(zeta[i][j][k]) << sci(sig);
                fout << sci(ddd[i][j][k].ntri) << sci(rrr[i][j][k].ntri);
                fout << std::endl;
            }
        }
    }
}

