
//#define XAssert(s) Assert(s)  
// The above XAssert adds extra (time-consuming) checks
// Using the following line disables these checks.
#define XAssert(s)

#include "dbg.h"
#include "BinnedCorr2.h"
#include "CorrIO.h"

#ifdef MEMDEBUG
AllocList* allocList;
#endif

#ifdef _OPENMP
#include "omp.h"
#endif

#include "valid_keys.h"

std::ostream* dbgout = 0;
bool XDEBUG = false;

template <int DC1, int DC2>
void RescaleNPair(BinnedCorr2<DC1,DC2>& , double ) {}

// Only rescale the counts if NN correlation.
void RescaleNPair(BinnedCorr2<NData,NData>& corr, double scale) { corr.rescaleNPair(scale); }


// The processing function for DC1 != DC2
template <int DC1, int DC2>
double DoAllProcessing(BinnedCorr2<DC1,DC2>& corr,
                       const std::vector<InputFile*>& files,
                       const std::vector<InputFile*>& files2,
                       const ConfigFile& params)
{
    Assert(files.size() > 0); // This should already have been checked.
    double var1=0.;
    double var2=0.;

    double minsep = corr.getMinSep();
    double maxsep = corr.getMaxSep();
    double b = corr.getB();
    double nproc = 0.;
    double nn = 0.;

    // There are two ways to specify the files for a cross correlation.
    // 1) files can have 2 items and files2 has 0
    // 2) files and files2 must each have at least 1 item.
    if (files.size() == 2 && files2.size() == 0) {
        if (params.read("pairwise",false)) {
            corr.processPairwise(*files[0],*files[1]);
            var1 = Field<DC1>::CalculateVar(*files[0]);
            var2 = Field<DC2>::CalculateVar(*files[1]);
            xdbg<<"direct var (pairwise) = "<<var1<<"  "<<var2<<std::endl;
            nproc += 1.0;
            Assert(files[0]->getNTot() == files[1]->getNTot());
            nn += files[0]->getNTot();
        } else {
            Field<DC1> field1(*files[0],params,minsep,maxsep,b);
            Field<DC2> field2(*files[1],params,minsep,maxsep,b);
            var1 = field1.getVar();
            var2 = field2.getVar();
            xdbg<<"direct var = "<<var1<<"  "<<var2<<std::endl;
            corr.process(field1,field2);
            nproc += 1.0;
            nn += field1.getNTot() * field2.getNTot();
        }
    } else if (files2.size() > 0) {
        const int n1 = files.size();
        const int n2 = files2.size();
        Assert(n1 != 0.);
        Assert(n2 != 0.);
        if (params.read("pairwise",false)) {
            Assert(n1 == n2);
            for (int i=0;i<n1;++i) {
                corr.processPairwise(*files[i],*files2[i]);
                nproc += 1.0;
                Assert(files[i]->getNTot() == files2[i]->getNTot());
                nn += files[i]->getNTot();
                double v1 = Field<DC1>::CalculateVar(*files[i]);
                double v2 = Field<DC2>::CalculateVar(*files2[i]);
                var1 += v1;
                var2 += v2;
                xdbg<<"var1 (pairwise) += "<<v1<<" = "<<var1<<std::endl;
                xdbg<<"var2 (pairwise) += "<<v2<<" = "<<var2<<std::endl;
            }
        } else {
            std::vector<Field<DC1>*> field1(n1);
            std::vector<Field<DC2>*> field2(n2);
            for (int i=0;i<n1;++i) {
                field1[i] = new Field<DC1>(*files[i],params,minsep,maxsep,b);
                var1 += field1[i]->getVar();
                xdbg<<"var1 += "<<field1[i]->getVar()<<" = "<<var1<<std::endl;
            }
            for (int j=0;j<n2;++j) {
                field2[j] = new Field<DC2>(*files2[j],params,minsep,maxsep,b);
                var2 += field2[j]->getVar();
                xdbg<<"var2 += "<<field2[j]->getVar()<<" = "<<var2<<std::endl;
            }
            for (int i=0;i<n1;++i) for (int j=0;j<n2;++j) {
                corr.process(*field1[i],*field2[j]);
                nproc += 1.0;
                nn += field1[i]->getNTot() * field2[j]->getNTot();
            }
            for (int i=0;i<n1;++i) delete field1[i];
            for (int j=0;j<n2;++j) delete field2[j];
        }
        var1 /= n1;
        var2 /= n2;
        xdbg<<"var1 /= "<<n1<<" = "<<var1<<std::endl;
        xdbg<<"var2 /= "<<n2<<" = "<<var2<<std::endl;
    } else {
        myerror("Invalid file_name inputs for a cross-correlation.\n"
                "Either file_name should have 2 items, or file_name2 should be specified.");
    }
    xdbg<<"Call finalize with var = "<<var1<<"  "<<var2<<std::endl;
    corr.finalize(var1,var2);
    xdbg<<"Nproc = "<<nproc<<std::endl;
    Assert(nproc > 0.);
    RescaleNPair(corr,1./nproc);
    xdbg<<"nn = "<<nn<<std::endl;
    nn /= nproc;
    xdbg<<"nn => "<<nn<<std::endl;
    return nn;
}

// A little different if DC1 == DC2
template <int DC>
double DoAllProcessing(BinnedCorr2<DC,DC>& corr,
                       const std::vector<InputFile*>& files,
                       const std::vector<InputFile*>& files2,
                       const ConfigFile& params)
{
    Assert(files.size() > 0); // This should already have been checked.
    double var1=0.;
    double var2=0.;

    double minsep = corr.getMinSep();
    double maxsep = corr.getMaxSep();
    double b = corr.getB();
    double nproc = 0.;
    double nn = 0.;

    // There are three ways to specify the files if DC1 == DC2
    // 1) files can have 1 item and files2 has 0
    // 2) files can have >1 items and files2 has 0
    // 3) files and files2 must each have at least 1 item.
    if (files.size() == 1 && files2.size() == 0) {
        Field<DC> field(*files[0],params,minsep,maxsep,b);
        var1 = var2 = field.getVar();
        xdbg<<"direct var = "<<var1<<std::endl;
        corr.process(field);
        nproc += 0.5;
        double n = field.getNTot();
        nn += 0.5*n*n;
    } else if (files2.size() == 0) {
        const int n = files.size();
        if (params.read("pairwise",false)) {
            Assert(n == 2);
            corr.processPairwise(*files[0],*files[1]);
            var1 = Field<DC>::CalculateVar(*files[0]);
            var2 = Field<DC>::CalculateVar(*files[1]);
            xdbg<<"direct var (pairwise) = "<<var1<<"  "<<var2<<std::endl;
            nproc += 1.0;
            Assert(files[0]->getNTot() == files[1]->getNTot());
            nn += files[0]->getNTot();
        } else {
            std::vector<Field<DC>*> field(n);
            for (int i=0;i<n;++i) {
                field[i] = new Field<DC>(*files[i],params,minsep,maxsep,b);
                var1 += field[i]->getVar();
                xdbg<<"var1 += "<<field[i]->getVar()<<" = "<<var1<<std::endl;
            }
            var1 /= n;
            xdbg<<"var1 /= "<<n<<" = "<<var1<<std::endl;
            var2 = var1;
            if (params.read<bool>("do_auto_corr",false)) {
                for (int i=0;i<n;++i) {
                    corr.process(*field[i]);
                    nproc += 0.5;
                    double n = field[i]->getNTot();
                    nn += 0.5*n*n;
                }
                // Auto-correlations have half the number of pair as cross correlations
                // per area.  So we need to multiply npair by 2 to be consistent with the
                // cross-correlations below.
                RescaleNPair(corr,2.);
                nproc *= 2.;
                nn *= 2;
            }
            if (params.read<bool>("do_cross_corr",true)) {
                for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) {
                    corr.process(*field[i],*field[j]);
                    nproc += 1.0;
                    nn += field[i]->getNTot() * field[j]->getNTot();
                }
            }
            for (int i=0;i<n;++i) delete field[i];
        }
    } else {
        Assert(files.size() > 0);
        const int n1 = files.size();
        const int n2 = files2.size();
        Assert(n1 != 0.);
        Assert(n2 != 0.);
        if (params.read("pairwise",false)) {
            Assert(n1 == n2);
            for (int i=0;i<n1;++i) {
                corr.processPairwise(*files[i],*files2[i]);
                nproc += 1.0;
                Assert(files[i]->getNTot() == files2[i]->getNTot());
                nn += files[i]->getNTot();
                double v1 = Field<DC>::CalculateVar(*files[i]);
                double v2 = Field<DC>::CalculateVar(*files2[i]);
                var1 += v1;
                var2 += v2;
                xdbg<<"var1 (pairwise) += "<<v1<<" = "<<var1<<std::endl;
                xdbg<<"var2 (pairwise) += "<<v2<<" = "<<var2<<std::endl;
            }
        } else {
            std::vector<Field<DC>*> field1(n1);
            std::vector<Field<DC>*> field2(n2);
            for (int i=0;i<n1;++i) {
                field1[i] = new Field<DC>(*files[i],params,minsep,maxsep,b);
                var1 += field1[i]->getVar();
                xdbg<<"var1 += "<<field1[i]->getVar()<<" = "<<var1<<std::endl;
            }
            for (int j=0;j<n2;++j) {
                field2[j] = new Field<DC>(*files2[j],params,minsep,maxsep,b);
                var2 += field2[j]->getVar();
                xdbg<<"var2 += "<<field2[j]->getVar()<<" = "<<var2<<std::endl;
            }
            for (int i=0;i<n1;++i) for (int j=0;j<n2;++j) {
                corr.process(*field1[i],*field2[j]);
                nproc += 1.0;
                nn += field1[i]->getNTot() * field2[j]->getNTot();
            }
            for(int i=0;i<n1;++i) delete field1[i];
            for(int j=0;j<n2;++j) delete field2[j];
        }
        var1 /= n1;
        var2 /= n2;
        xdbg<<"var1 /= "<<n1<<" = "<<var1<<std::endl;
        xdbg<<"var2 /= "<<n2<<" = "<<var2<<std::endl;
    }
    xdbg<<"Call finalize with var = "<<var1<<"  "<<var2<<std::endl;
    corr.finalize(var1,var2);
    xdbg<<"Nproc = "<<nproc<<std::endl;
    Assert(nproc > 0.);
    RescaleNPair(corr,1./nproc);
    xdbg<<"nn = "<<nn<<std::endl;
    nn /= nproc;
    xdbg<<"nn => "<<nn<<std::endl;
    return nn;
}

int main(int argc, const char* argv[])
try {
#ifdef MEMDEBUG
    atexit(&DumpUnfreed);
#endif

    if (argc < 2) {
        std::cerr<<
            "Usage: corr2 configfile [param=value ...]\n"
            "\tThe first parameter is the configuration file that has \n"
            "\tall the parameters for this run. \n"
            "\tThese values may be modified on the command line by \n"
            "\tentering param/value pais as param=value. \n";
            return 1;
    }

    // Read parameters
    ConfigFile params;
    // These are all the defaults, but might as well be explicit.
    params.setDelimiter("=");
    params.setInclude("+");
    params.setComment("#");
    params.load(argv[1]);
    for(int k=2;k<argc;k++) params.append(argv[k]);

    // Check that the ConfigFile doesn't have any extra parameters.
    // (Such values are probably typos.)
    std::set<std::string> valid_keys(ar_valid_keys, ar_valid_keys+n_valid_keys);
    std::set<std::string> invalid_keys = params.checkValid(valid_keys);
    if (invalid_keys.size() > 0) {
        std::cerr<<"Found invalid keys in the configuration file "<<argv[1]<<":\n";
        typedef std::set<std::string>::const_iterator SetIt;
        for (SetIt it=invalid_keys.begin(); it!=invalid_keys.end(); ++it) {
            std::cerr<<"  "<<*it<<std::endl;
            return 1;
        }
    }

    // Set number of openmp threads if necessary
#ifdef _OPENMP
    if (params.keyExists("num_threads")) {
        int num_threads = params["num_threads"];
        omp_set_num_threads(num_threads);
    }
#endif

    // Setup debugging
    if (params.read("verbose",0) > 0) {
        if (params.read<int>("verbose") > 1) XDEBUG = true;
        if (params.keyExists("debug_file")) {
            std::string debug_file = params["debug_file"];
            dbgout = new std::ofstream(debug_file.c_str());
        } else {
            dbgout = &std::cout;
        }
        dbgout->setf(std::ios_base::unitbuf);
    }
#ifdef _OPENMP
    if (omp_get_max_threads() > 1)
        dbg<<"Using "<<omp_get_max_threads()<<" threads.\n";
#endif

    // Check for deprecated backwards-compatible aliases
    if (params.keyExists("e1_col") || params.keyExists("e2_col")) {
        std::cerr<<"WARNING: The e1_col, e2_col specifications have been eliminated.\n";
        std::cerr<<"         Treating them as g1_col, g2_col instead.\n";
        std::cerr<<"         (You should change your parameter file...)\n";
        params["g1_col"] = params["e1_col"];
        if (params.keyExists("e1_col")) params["g1_col"] = params["e1_col"];
        if (params.keyExists("e2_col")) params["g2_col"] = params["e2_col"];
    }
    if (params.keyExists("e2_file_name")) {
        std::cerr<<"WARNING: The e2_file_name parameter has been changed to g2_file_name.\n";
        std::cerr<<"         (You should change your parameter file...)\n";
        params["g2_file_name"] = params["e2_file_name"];
    }
    if (params.keyExists("ne_file_name")) {
        std::cerr<<"WARNING: The ne_file_name parameter has been changed to ng_file_name.\n";
        std::cerr<<"         (You should change your parameter file...)\n";
        params["ng_file_name"] = params["ne_file_name"];
    }
    if (params.keyExists("ne_statistic")) {
        std::cerr<<"WARNING: The ne_statistic parameter has been changed to ng_statistic.\n";
        std::cerr<<"         (You should change your parameter file...)\n";
        params["ng_statistic"] = params["ne_statistic"];
    }
    if (params.keyExists("ke_file_name")) {
        std::cerr<<"WARNING: The ke_file_name parameter has been changed to kg_file_name.\n";
        std::cerr<<"         (You should change your parameter file...)\n";
        params["kg_file_name"] = params["ke_file_name"];
    }

    // Read in all the input files.
    std::vector<InputFile*> files;
    ReadInputFiles(files,params,"file_name",0);

    if (files.size() == 0) 
        myerror("No files read in.");

    std::vector<InputFile*> files2;
    ReadInputFiles(files2,params,"file_name2",1);

    std::vector<InputFile*> rand_files;
    ReadInputFiles(rand_files,params,"rand_file_name",0);

    std::vector<InputFile*> rand_files2;
    ReadInputFiles(rand_files2,params,"rand_file_name2",1);

    // Check for other invalid input parameters
    if (params.read("pairwise",false) && params.read("do_auto_corr",false)) 
        myerror("do_auto_corr is invalid when pairwise = true");

    if (params.read("pairwise",false) && !params.read("do_cross_corr",true)) 
        myerror("do_cross_corr is required when pairwise = true");

    if (params.read("pairwise",false) && files2.size() == 0 && files.size() != 2)
        myerror("pairwise requires exactly 2 input files in file_name field.");

    if (params.read("pairwise",false) && files2.size() != 0 && files.size() != files2.size())
        myerror("pairwise requires equal number of files for file_name and file_name2.");

    double smoothscale = params.read("smooth_scale",0.);

    // Default precision is 3 decimal places.
    int prec = params.read("precision",3);

    std::string m2_uform = params.read("m2_uform","Crittenden");
    if (m2_uform != "Schneider" && m2_uform != "Crittenden") 
        myerror("Invalid m2_uform: "+m2_uform);
    UForm uform = (m2_uform == "Schneider") ? Schneider : Crittenden;

    if (params.keyExists("g2_file_name") || params.keyExists("m2_file_name")) {
        dbg<<"Start e2 calculations...";
        BinnedCorr2<GData,GData> gg(params);
        DoAllProcessing(gg,files,files2,params);
        dbg<<"Done processing GG\n";
        if (params.keyExists("g2_file_name")) {
            std::ofstream os(params["g2_file_name"].c_str());
            WriteGG(os,gg.getMinSep()/gg.getSepUnits(),gg.getBinSize(),smoothscale,prec,
                    gg.getData()); 
        }
        if (params.keyExists("m2_file_name")) {
            std::ofstream os(params["m2_file_name"].c_str());
            WriteM2(os,gg.getMinSep()/gg.getSepUnits(),gg.getBinSize(),uform,prec,
                    gg.getData());
        }
    }
    if ( params.keyExists("ng_file_name") || params.keyExists("nm_file_name") 
         || params.keyExists("norm_file_name") ) {
        dbg<<"Start ng calculations...";

        std::string ng_stat = params.read(
            "ng_statistic", (rand_files.size() > 0) ? "compensated" : "simple");
        if (ng_stat != "compensated" && ng_stat != "simple") 
            myerror("Invalid ng_statistic: "+ng_stat);

        BinnedCorr2<NData,GData> ne(params);
        double nde = DoAllProcessing(ne,files,files2,params);
        dbg<<"Done processing NG\n";

        BinnedCorr2<NData,GData> re(params);
        if (ng_stat == "compensated") {
            if (rand_files.size() == 0)
                myerror("rand_files is required for ng_statistic = compensated");
            if (files2.size() > 0) {
                double nre = DoAllProcessing(re,rand_files,files2,params);
                dbg<<"Done processing RG\n";
                // Correct for any difference in the number of items in the rand files
                // compared to the number of items in the data files.
                RescaleNPair(re,nde/nre);
            } else {
                Assert(files.size() == 2);
                std::vector<InputFile*> f2(1,files[1]);
                double nre = DoAllProcessing(re,rand_files,f2,params);
                dbg<<"Done processing RG\n";
                RescaleNPair(re,nde/nre);
            }
        }
        if (params.keyExists("ng_file_name")) {
            std::ofstream os(params["ng_file_name"].c_str());
            WriteNG(os,ne.getMinSep()/ne.getSepUnits(),ne.getBinSize(),smoothscale,ng_stat,prec,
                    ne.getData(),re.getData()); 
        }
        if (params.keyExists("nm_file_name")) {
            std::ofstream os(params["nm_file_name"].c_str());
            WriteNM(os,ne.getMinSep()/ne.getSepUnits(),ne.getBinSize(),ng_stat,uform,prec,
                    ne.getData(),re.getData()); 
        }
        if (params.keyExists("norm_file_name")) {
            std::ofstream os(params["norm_file_name"].c_str());
            BinnedCorr2<GData,GData> gg(params);
            BinnedCorr2<NData,NData> dd(params);
            BinnedCorr2<NData,NData> dr(params);
            BinnedCorr2<NData,NData> rr(params);
            std::string n2_stat = params.read("n2_statistic","compensated");
            dbg<<"n2_statistic = "<<n2_stat<<std::endl;
            if (n2_stat != "compensated" && n2_stat != "simple") 
                myerror("Invalid n2_statistic: "+n2_stat);
            if (rand_files.size() == 0)
                myerror("rand_files is required for norm calculation");

            std::vector<InputFile*> dummy;
            if (files2.size() > 0) {
                DoAllProcessing(gg,files2,dummy,params);
                dbg<<"Done processing GG for Norm calculation\n";
                double ndd = DoAllProcessing(dd,files,dummy,params);
                dbg<<"Done processing DD for Norm calculation\n";
                if (n2_stat == "compensated") {
                    double ndr = DoAllProcessing(dr,files,rand_files,params);
                    dbg<<"Done processing DR for Norm calculation\n";
                    xdbg<<"Rescale dr by "<<ndd/ndr<<std::endl;
                    RescaleNPair(dr,ndd/ndr);
                }
                double nrr = DoAllProcessing(rr,rand_files,dummy,params);
                dbg<<"Done processing RR for Norm calculation\n";
                // Correct for any difference in the number of items in the rand files
                // compared to the number of items in the data files.
                xdbg<<"Rescale rr by "<<ndd/nrr<<std::endl;
                RescaleNPair(rr,ndd/nrr);
            } else {
                Assert(files.size() == 2);
                std::vector<InputFile*> f1(1,files[0]);
                std::vector<InputFile*> f2(1,files[1]);
                DoAllProcessing(gg,f2,dummy,params);
                dbg<<"Done processing GG\n";
                double ndd = DoAllProcessing(dd,f1,dummy,params);
                dbg<<"Done processing DD\n";
                if (n2_stat == "compensated") {
                    double ndr = DoAllProcessing(dr,f1,rand_files,params);
                    dbg<<"Done processing DR\n";
                    xdbg<<"Rescale dr by "<<ndd/ndr<<std::endl;
                    RescaleNPair(dr,ndd/ndr);
                }
                double nrr = DoAllProcessing(rr,rand_files,dummy,params);
                dbg<<"Done processing RR\n";
                xdbg<<"Rescale rr by "<<ndd/nrr<<std::endl;
                RescaleNPair(rr,ndd/nrr);
            }
            WriteNorm(os,ne.getMinSep()/ne.getSepUnits(),ne.getBinSize(),
                      ng_stat, n2_stat, uform, prec,
                      ne.getData(), re.getData(), gg.getData(),
                      dd.getData(), dr.getData(), rr.getData()); 
        }
    }
    if (params.keyExists("n2_file_name")) {
        dbg<<"Start n2 calculations...";

        std::string n2_stat = params.read("n2_statistic","compensated");
        dbg<<"n2_statistic = "<<n2_stat<<std::endl;
        if (n2_stat != "compensated" && n2_stat != "simple") 
            myerror("Invalid n2_statistic: "+n2_stat);
        if (rand_files.size() == 0) 
            myerror("rand_files is required for n2 calculation");

        BinnedCorr2<NData,NData> dd(params);
        double ndd = DoAllProcessing(dd,files,files2,params);
        dbg<<"Done processing DD\n";

        BinnedCorr2<NData,NData> dr(params);
        BinnedCorr2<NData,NData> rd(params);
        BinnedCorr2<NData,NData> rr(params);
        if (rand_files2.size() > 0) {
            if (files2.size() > 0) {
                if (n2_stat == "compensated") {
                    double ndr = DoAllProcessing(dr,files,rand_files2,params);
                    dbg<<"Done processing DR\n";
                    xdbg<<"Rescale dr by "<<ndd/ndr<<std::endl;
                    RescaleNPair(dr,ndd/ndr);
                    double nrd = DoAllProcessing(rd,rand_files,files2,params);
                    dbg<<"Done processing RD\n";
                    xdbg<<"Rescale rd by "<<ndd/nrd<<std::endl;
                    RescaleNPair(rd,ndd/nrd);
                }
                double nrr = DoAllProcessing(rr,rand_files,rand_files2,params);
                dbg<<"Done processing RR\n";
                xdbg<<"Rescale rr by "<<ndd/nrr<<std::endl;
                RescaleNPair(rr,ndd/nrr);
            } else {
                Assert(files.size() == 2);
                if (n2_stat == "compensated") {
                    std::vector<InputFile*> f1(1,files[0]);
                    std::vector<InputFile*> f2(1,files[1]);
                    double ndr = DoAllProcessing(dr,f1,rand_files2,params);
                    dbg<<"Done processing DR\n";
                    xdbg<<"Rescale dr by "<<ndd/ndr<<std::endl;
                    RescaleNPair(dr,ndd/ndr);
                    double nrd = DoAllProcessing(dr,rand_files,f2,params);
                    dbg<<"Done processing DR\n";
                    xdbg<<"Rescale dr by "<<ndd/nrd<<std::endl;
                    RescaleNPair(rd,ndd/nrd);
                }
                double nrr = DoAllProcessing(rr,rand_files,rand_files2,params);
                dbg<<"Done processing RR\n";
                xdbg<<"Rescale rr by "<<ndd/nrr<<std::endl;
                RescaleNPair(rr,ndd/nrr);
            }
        } else {
            if (files2.size() > 0) {
                myerror("No rand_files2 read in, and n2 calculation requested with files2.");
            } else {
                if (n2_stat == "compensated") {
                    double ndr = DoAllProcessing(dr,files,rand_files,params);
                    dbg<<"Done processing DR\n";
                    xdbg<<"Rescale dr by "<<ndd/ndr<<std::endl;
                    RescaleNPair(dr,ndd/ndr);
                    rd = dr;
                }
                std::vector<InputFile*> dummy;
                double nrr = DoAllProcessing(rr,rand_files,dummy,params);
                dbg<<"Done processing RR\n";
                xdbg<<"Rescale rr by "<<ndd/nrr<<std::endl;
                RescaleNPair(rr,ndd/nrr);
            }
        }
        std::ofstream os(params["n2_file_name"].c_str());
        WriteNN(os,dd.getMinSep()/dd.getSepUnits(),dd.getBinSize(), n2_stat, prec,
                dd.getData(), dr.getData(), rd.getData(), rr.getData()); 
    }
    if (params.keyExists("k2_file_name")) {
        dbg<<"Start k2 calculations...";
        BinnedCorr2<KData,KData> kk(params);
        DoAllProcessing(kk,files,files2,params);
        dbg<<"Done processing KK\n";
        if (params.keyExists("k2_file_name")) {
            std::ofstream os(params["k2_file_name"].c_str());
            WriteKK(os,kk.getMinSep()/kk.getSepUnits(),kk.getBinSize(),smoothscale,prec,
                    kk.getData()); 
        }
    }
    if (params.keyExists("nk_file_name")) {
        dbg<<"Start nk calculations...";

        std::string nk_stat = params.read(
            "nk_statistic", (rand_files.size() > 0) ? "compensated" : "simple");
        if (nk_stat != "compensated" && nk_stat != "simple") 
            myerror("Invalid nk_statistic: "+nk_stat);

        BinnedCorr2<NData,KData> nk(params);
        double ndk = DoAllProcessing(nk,files,files2,params);
        dbg<<"Done processing NK\n";

        BinnedCorr2<NData,KData> rk(params);
        if (nk_stat == "compensated") {
            if (rand_files.size() == 0)
                myerror("rand_files is required for nk_statistic = compensated");
            if (files2.size() > 0) {
                double nrk = DoAllProcessing(rk,rand_files,files2,params);
                dbg<<"Done processing RK\n";
                // Correct for any difference in the number of items in the rand files
                // compared to the number of items in the data files.
                RescaleNPair(rk,ndk/nrk);
            } else {
                Assert(files.size() == 2);
                std::vector<InputFile*> f2(1,files[1]);
                double nrk = DoAllProcessing(rk,rand_files,f2,params);
                dbg<<"Done processing RK\n";
                RescaleNPair(rk,ndk/nrk);
            }
        }
        std::ofstream os(params["nk_file_name"].c_str());
        WriteNK(os,nk.getMinSep()/nk.getSepUnits(),nk.getBinSize(),smoothscale,nk_stat,prec,
                nk.getData(),rk.getData()); 
    }
    if (params.keyExists("kg_file_name")) {
        dbg<<"Start kg calculations...";
        BinnedCorr2<KData,GData> kg(params);
        DoAllProcessing(kg,files,files2,params);
        dbg<<"Done processing KG\n";
        if (params.keyExists("kg_file_name")) {
            std::ofstream os(params["kg_file_name"].c_str());
            WriteKG(os,kg.getMinSep()/kg.getSepUnits(),kg.getBinSize(),smoothscale,prec,
                    kg.getData()); 
        }
    }
  
    if (dbgout && dbgout != &std::cout) 
    { delete dbgout; dbgout=0; }
    for (size_t i=0; i<files.size(); ++i) delete files[i];
    for (size_t i=0; i<files2.size(); ++i) delete files2[i];
    for (size_t i=0; i<rand_files.size(); ++i) delete rand_files[i];
    for (size_t i=0; i<rand_files2.size(); ++i) delete rand_files2[i];

    return 0;
} catch (std::exception& e) {
    myerror(std::string("Caught ") + e.what());
} catch (...) {
    myerror("Caught an unknown exception.");
}

