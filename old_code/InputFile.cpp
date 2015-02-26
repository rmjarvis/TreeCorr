
#include <sstream>

#include "dbg.h"
#include "InputFile.h"
#include "Angle.h"


template <typename T>
T ReadFileParam(const ConfigFile& params, const std::string& key, int num, T def)
{
    xdbg<<"Start ReadFileParam for key "<<key<<", num = "<<num<<", def = "<<def<<std::endl;
    if (params.keyExists(key)) {
        std::vector<ConvertibleString> v = params[key];
        if (v.size() == 0) {
            myerror("Unable to parse " + key);
        } else if (v.size() == 1) {
            xdbg<<"Single item: "<<v[0]<<std::endl;
            return T(v[0]);
        } else if (num >= int(v.size())) {
            myerror("List of "+key+" items is too short");
        } else {
            xdbg<<"Multiple items: v[num] =  "<<v[num]<<std::endl;
            return T(v[num]);
        }
    }
    xdbg<<"Use default "<<def<<std::endl;
    return def;
}

bool IsKColRequired(const ConfigFile& params, int num) 
{
    return (params.keyExists("k2_file_name") || 
            (num == 0 && params.keyExists("kg_file_name")) ||
            (num == 1 && params.keyExists("nk_file_name")));
}

bool IsGColRequired(const ConfigFile& params, int num) 
{
    return (params.keyExists("g2_file_name") || 
            params.keyExists("m2_file_name") ||
            params.keyExists("norm_file_name") ||
            (num == 1 && params.keyExists("kg_file_name")) ||
            (num == 1 && params.keyExists("nm_file_name")) ||
            (num == 1 && params.keyExists("ng_file_name")));
}

static angle::AngleUnit ConvertUnitStr(const std::string& str)
{
    if (str.find("rad") == 0) return angle::radians;
    else if (str.find("deg") == 0) return angle::degrees;
    else if (str.find("hour") == 0) return angle::hours;
    else if (str.find("arcmin") == 0) return angle::arcmin;
    else if (str.find("arcsec") == 0) return angle::arcsec;
    else myerror("Invalid angle unit: "+str);
    // Won't get here, but this avoids a warning.
    return angle::radians;
}

double GetUnits(const ConfigFile& params, const std::string& key, int num, 
                const std::string& def)
{
    std::string unit_str = ReadFileParam(params,key,num,def);
    if (def == "" && unit_str == "") 
        myerror("Required parameter " + key + " not found");
    return (1. * ConvertUnitStr(unit_str)) / angle::radians;
}

static bool IsNan(double x)
{ return !(x == x) || !(x * x >= 0); }

// Helper function to convert ASCII line into vector of values
static bool GetValues(std::istream& is, const std::string& file_name, std::vector<double>& values,
                      char delim, char comment_marker, long& row_num)
{
    // Check that the istream is valid to start with.
    if (!is) return false;

    // Skip if this is a comment.
    char first = is.peek();

    // Skip initial whitespace
    while (std::isspace(first)) { is.get(first); first = is.peek(); }
    if (first == comment_marker) {
        // If now starts with comment_marker, skip this line.
        is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        return GetValues(is,file_name,values,delim,comment_marker,row_num);
    }

    // Read values
    const int nval = values.size();
    if (delim == '\0') {
        double temp;
        for(int i=0;i<nval;++i) {
            is >> temp;
            values[i] = temp;
        }
    } else {
        ConvertibleString temp;
        for(int i=0;i<nval;++i) {
            getline(is,temp,delim);
            values[i] = temp;
        }
    }
    if (!is) return false;

    // Read to the end of the line
    is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Skip this line if there are any nan's in the input.
    bool has_nan = false;
    for(int i=0;i<nval;++i) if (IsNan(values[i])) has_nan = true;
    if (has_nan) {
        std::cerr<<"WARNING: Skipping row "<<row_num<<" in "<<file_name<<
            " because it has a NaN.\n";
        ++row_num;
        return GetValues(is,file_name,values,delim,comment_marker,row_num);
    }

    return true;
}

static void fitserror(int status)
{
    char msg[80];
    std::cerr<<"FITS error found: \n";
    fits_get_errstatus(status,msg);
    std::cerr<<"  "<<msg<<std::endl;
    while (fits_read_errmsg(msg))
        std::cerr<<msg<<std::endl;
    myerror("Fits error");
}

InputFile::InputFile(const std::string& file_name, const ConfigFile& params, int num) :
    _file_name(file_name), _use_ra_dec(false)
{
    dbg<<"Reading input file "<<num<<": "<<file_name<<std::endl;
    if (params.keyExists("file_type")) {
        // If file_type is specified, use that.
        std::string file_type = ReadFileParam(params,"file_type",num,std::string("ascii"));
        dbg<<"   file_type = "<<file_type<<std::endl;
        if (file_type == "ASCII" || file_type == "ascii")
            readAscii(file_name,params,num);
        else if (file_type == "FITS" || file_type == "fits")
            readFits(file_name,params,num);
        else 
            myerror("Unknown file_type "+file_type);
    } else {
        // Otherwise, try to get the type from the file extension.
        if (file_name.find(".fit") != std::string::npos) {
            dbg<<"   file_type assumed to be FITS from the file name.\n";
            readFits(file_name,params,num);
        } else {
            dbg<<"   file_type assumed to be ASCII from the file name.\n";
            readAscii(file_name,params,num);
        }
    }
    dbg<<"   ngal = "<<_pos.size()<<std::endl;
    xdbg<<"   kappa.size = "<<_kappa.size()<<std::endl;
    xdbg<<"   shear.size = "<<_shear.size()<<std::endl;
    xdbg<<"   weight.size = "<<_weight.size()<<std::endl;

    // Check if we need to flip g1 or g2
    bool flip1 = ReadFileParam(params,"flip_g1",num,false);
    bool flip2 = ReadFileParam(params,"flip_g2",num,false);
    if (flip1 && flip2) {
        dbg<<"   flipping sign of both g1 and g2\n";
        for(size_t i=0; i<_shear.size(); ++i) _shear[i] = -_shear[i];
    } else if (flip2) {
        dbg<<"   flipping sign of g2\n";
        for(size_t i=0; i<_shear.size(); ++i) _shear[i] = conj(_shear[i]);
    } else if (flip1) {
        dbg<<"   flipping sign of g1\n";
        for(size_t i=0; i<_shear.size(); ++i) _shear[i] = -conj(_shear[i]);
    }

    if (_use_ra_dec && params.read("project",false)) {
        // Project the position onto a tangent plane
        double ra_cen, dec_cen;
        if (params.keyExists("project_ra")) {
            ra_cen = params["project_ra"];
            ra_cen *= GetUnits(params,"ra_units",num);
        } else {
            ra_cen = 0.;
            for (size_t i=0;i<_pos.size();++i) ra_cen += _pos[i].getX();
            ra_cen /= _pos.size();
        }
        if (params.keyExists("project_dec")) {
            dec_cen = params["project_dec"];
            dec_cen *= GetUnits(params,"dec_units",num);
        } else {
            dec_cen = 0.;
            for (size_t i=0;i<_pos.size();++i) dec_cen += _pos[i].getY();
            dec_cen /= _pos.size();
        }
        project(ra_cen,dec_cen);
    }
}

void InputFile::readAscii(const std::string& file_name, const ConfigFile& params, int num)
{
    std::ifstream fin(file_name.c_str());
    if (!fin) myerror("Unable to open file " + file_name);

    int max_col = 0;

    int x_col = ReadFileParam(params,"x_col",num,0);
    int y_col = ReadFileParam(params,"y_col",num,0);
    int ra_col = ReadFileParam(params,"ra_col",num,0);
    int dec_col = ReadFileParam(params,"dec_col",num,0);
    double x_units=1., y_units=1., ra_units=1., dec_units=1.;
    std::ostringstream temp_oss;
    temp_oss << num+1;  // Use 1-based numbering for the error messages.
    std::string strnum = temp_oss.str();

    if (x_col > 0 || y_col > 0) {
        if (x_col <= 0) myerror("x_col missing or invalid for file "+strnum);
        if (y_col <= 0) myerror("y_col missing or invalid for file "+strnum);
        if (ra_col > 0 || dec_col > 0) 
            myerror("ra/dec cols are not allowed in conjunction with x/y cols");
        if (params.keyExists("x_units") && !params.keyExists("y_units")) 
            myerror("x_units specified without specifying y_units");
        if (params.keyExists("y_units") && !params.keyExists("x_units")) 
            myerror("x_units specified without specifying y_units");

        x_units = GetUnits(params,"x_units",num,"arcsec");
        y_units = GetUnits(params,"y_units",num,"arcsec");
        if (x_col > max_col) max_col = x_col;
        if (y_col > max_col) max_col = y_col;
    } else if (ra_col > 0 || dec_col > 0) {
        ra_units = GetUnits(params,"ra_units",num);
        dec_units = GetUnits(params,"dec_units",num);
        if (ra_col <= 0) myerror("ra_col missing or invalid for file "+strnum);
        if (dec_col <= 0) myerror("dec_col missing or invalid for file "+strnum);
        _use_ra_dec = true;

        if (ra_col > max_col) max_col = ra_col;
        if (dec_col > max_col) max_col = dec_col;
    } else {
        myerror("No valid position columns specified for file "+strnum);
    }

    int k_col = ReadFileParam(params,"k_col",num,0);
    if (k_col > max_col) max_col = k_col;
    if (!IsKColRequired(params,num)) k_col = 0;

    int g1_col = ReadFileParam(params,"g1_col",num,0);
    int g2_col = ReadFileParam(params,"g2_col",num,0);
    if (!IsGColRequired(params,num)) g1_col = g2_col = 0;

    if (g1_col > 0 || g2_col > 0) {
        if (g1_col <= 0) myerror("g1_col missing or invalid for file "+strnum);
        if (g2_col <= 0) myerror("g2_col missing or invalid for file "+strnum);
        if (g1_col > max_col) max_col = g1_col;
        if (g2_col > max_col) max_col = g2_col;
    }

    int w_col = ReadFileParam(params,"w_col",num,0);
    if (w_col > max_col) max_col = w_col;

    int flag_col = ReadFileParam(params,"flag_col",num,0);
    if (flag_col > max_col) max_col = flag_col;
    long ignore_flag;
    if (params.keyExists("ignore_flag")) {
        ignore_flag = params["ignore_flag"];
    } else {
        long ok_flag = params.read("ok_flag",0);
        ignore_flag = ~ok_flag;
    }

    // Set up delimiter and allowed comment markers
    char delim = ReadFileParam(params,"delimiter",num,'\0');
    char comment_marker = ReadFileParam(params,"comment_marker",num,'#');

    // Optionally not use all rows:
    long first_row = ReadFileParam(params,"first_row",num,1);
    long last_row = ReadFileParam(params,"last_row",num,-1);
    if (first_row < 1) myerror("first_row < 1");
    if (last_row > 0 && last_row < first_row) myerror("last_row < first_row");

    // Another advantage of using last_row is that you can reserve the vectors to avoid
    // gratuitous copying and extra allocations.
    if (last_row > 0) {
        xdbg<<"reserve "<<last_row-first_row+1<<std::endl;
        _pos.reserve(last_row-first_row+1);
        xdbg<<"reserved _pos\n";
        if (k_col > 0) {
            _kappa.reserve(last_row-first_row+1);
            xdbg<<"reserved _kappa\n";
        }
        if (g1_col > 0) {
            _shear.reserve(last_row-first_row+1);
            xdbg<<"reserved _shear\n";
        }
        if (w_col > 0 || flag_col > 0) {
            _weight.reserve(last_row-first_row+1);
            xdbg<<"reserved _weight\n";
        }
    }

    std::vector<double> values(max_col);
    for (long row_num=1; GetValues(fin,file_name,values,delim,comment_marker,row_num); ++row_num) {
        if (row_num < first_row) continue;
        if (last_row > 0 && row_num > last_row) break;
        //xxdbg<<"row "<<row_num<<std::endl;

        // Position
        if (x_col > 0) {
            Assert(x_col <= long(values.size()));
            Assert(y_col <= long(values.size()));
            double x = values[x_col-1] * x_units;
            double y = values[y_col-1] * y_units;
            _pos.push_back(Position<Flat>(x,y));
        } else {
            Assert(ra_col <= long(values.size()));
            Assert(dec_col <= long(values.size()));
            double ra = values[ra_col-1] * ra_units;
            double dec = values[dec_col-1] * dec_units;
            _pos.push_back(Position<Flat>(ra,dec));
        }

        // Kappa
        if (k_col > 0) {
            Assert(k_col <= long(values.size()));
            double k = values[k_col-1];
            _kappa.push_back(k);
        }

        // Shear
        if (g1_col > 0) {
            Assert(g1_col <= long(values.size()));
            Assert(g2_col <= long(values.size()));
            double g1 = values[g1_col-1];
            double g2 = values[g2_col-1];
            //xdbg<<"row = "<<row_num<<": g1,g2 = "<<g1<<','<<g2<<std::endl;
            _shear.push_back(std::complex<double>(g1,g2));
            //xdbg<<"shear["<<_shear.size()-1<<"] = "<<_shear.back()<<std::endl;
        }

        // Weight
        if (w_col > 0) {
            Assert(w_col <= long(values.size()));
            double w = values[w_col-1];
            _weight.push_back(w);
        }

        // Treat flags as a weight of 0 or 1 as appropriate
        if (flag_col > 0) {
            Assert(flag_col <= long(values.size()));
            long flag = values[flag_col-1];
            if (flag & ignore_flag) {
                xdbg<<"flag["<<row_num<<"] = "<<flag<<", so ignore.\n";
                if (w_col > 0) _weight.back() = 0;
                else _weight.push_back(0);
            } else if (w_col == 0) {
                _weight.push_back(1);
            }
        }
    }
    long nrows = int(_pos.size());
    xdbg<<"First few positions are ";
    for (int i=0;i<std::min(3L,nrows);++i) xdbg<<_pos[i]<<"  ";
    xdbg<<std::endl;
    if (k_col > 0) {
        xdbg<<"First few kappas are ";
        for (int i=0;i<std::min(3L,nrows);++i) xdbg<<_kappa[i]<<"  ";
        xdbg<<std::endl;
    }
    if (g1_col > 0) {
        xdbg<<"First few shears are ";
        for (int i=0;i<std::min(3L,nrows);++i) xdbg<<_shear[i]<<"  ";
        xdbg<<std::endl;
    }
    if (w_col > 0 || flag_col > 0) {
        xdbg<<"First few weights are ";
        for (int i=0;i<std::min(3L,nrows);++i) xdbg<<_weight[i]<<"  ";
        xdbg<<std::endl;
    }
}

template <typename T>
struct FitsType {};

template <>
struct FitsType<double> 
{ static int table_type() { return TDOUBLE; } };

template <>
struct FitsType<long> 
{ static int table_type() { return TLONG; } };

template <typename T> 
void ReadFitsColumn(fitsfile* fptr, const std::string& name, std::vector<T>& vec, long first_row,
                    const ConfigFile& params, int num)
{
    std::ostringstream temp_oss; 
    temp_oss << num+1;  // Use 1-based numbering for the error messages.
    std::string strnum = temp_oss.str();
    xdbg<<"ReadFitsColumn for "<<name<<std::endl;
    std::string col = ReadFileParam(params,name+"_col",num,std::string("0"));
    if (col == std::string("0")) myerror(name+"_col missing or invalid for file "+strnum);
    xdbg<<"col = "<<col<<std::endl;

    int hdu = params.read("hdu",1);
    if (params.keyExists(name+"_hdu")) {
        // column-specific hdu parameter supersedes default.
        hdu = params.get(name+"_hdu");
    }
    int nhdu;
    int status = 0;
    fits_get_num_hdus(fptr,&nhdu,&status);
    if (status) fitserror(status);
    xdbg<<"hdu = "<<hdu<<", nhdu = "<<nhdu<<std::endl;
    if (hdu >= nhdu) myerror("Specified hdu for "+name+" is more than the total number of hdus.");

    hdu += 1;  // Switch to 1-based convention used by cfitsio.
    int hdu_type;
    fits_movabs_hdu(fptr, hdu, &hdu_type, &status);
    if (status) fitserror(status);
    if (hdu_type != BINARY_TBL) myerror("HDU for "+name+" is not a binary table");
    xdbg<<"Moved to hdu "<<hdu<<" (in 1-based cfitsio definition)\n";

    int colnum, anynul;
    fits_get_colnum(fptr,CASEINSEN,(char*)col.c_str(),&colnum,&status);
    if (status) fitserror(status);
    fits_read_col(fptr,FitsType<T>::table_type(),colnum,first_row,1,long(vec.size()),0,
                  &vec[0],&anynul,&status);
    if (status) fitserror(status);
    xdbg<<"Read column "<<col<<std::endl;
}

void InputFile::readFits(const std::string& file_name, const ConfigFile& params, int num)
{
    fitsfile* fptr;
    int status=0;
    fits_open_table(&fptr, file_name.c_str(), READONLY, &status);
    if (status) fitserror(status);

    long nrows;
    fits_get_num_rows(fptr,&nrows,&status);
    if (status) fitserror(status);

    // Optionally don't use all rows:
    long first_row = ReadFileParam(params,"first_row",num,1);
    if (first_row < 1) myerror("first_row < 1");
    long last_row = ReadFileParam(params,"last_row",num,-1);
    if (last_row > 0) {
        if (last_row < first_row)
            myerror("last_row < first_row");
        else
            nrows = last_row - first_row + 1;
    }

    // Positions
    double x_units=1., y_units=1., ra_units=1., dec_units=1.;

    if (params.keyExists("x_col") || params.keyExists("y_col")) {
        if (params.keyExists("x_units") && !params.keyExists("y_units")) 
            myerror("x_units specified without specifying y_units");
        if (params.keyExists("y_units") && !params.keyExists("x_units")) 
            myerror("x_units specified without specifying y_units");
        if (!params.keyExists("x_col") || !params.keyExists("y_col")) 
            myerror("Both x and y cols must be specified");
        if (params.keyExists("ra_col") && params.keyExists("dec_col")) 
            myerror("ra/dec cols are not allowed in conjunction with x/y cols");

        x_units = GetUnits(params,"x_units",num,"arcsec");
        y_units = GetUnits(params,"y_units",num,"arcsec");

        std::vector<double> x_vec(nrows);
        ReadFitsColumn(fptr, "x", x_vec, first_row, params, num);
        std::vector<double> y_vec(nrows);
        ReadFitsColumn(fptr, "y", y_vec, first_row, params, num);

        _pos.resize(nrows);
        for(long i=0;i<nrows;++i) {
            _pos[i] = Position<Flat>(x_vec[i]*x_units,y_vec[i]*y_units);
        }
        xdbg<<"First few positions are ";
        for (int i=0;i<std::min(3L,nrows);++i) xdbg<<_pos[i]<<"  ";
        xdbg<<std::endl;

    } else if (params.keyExists("ra_col") || params.keyExists("dec_col")) {
        if (!params.keyExists("ra_col") || !params.keyExists("dec_col")) 
            myerror("Both ra and dec cols must be specified");
        _use_ra_dec = true;

        ra_units = GetUnits(params,"ra_units",num);
        dec_units = GetUnits(params,"dec_units",num);

        std::vector<double> ra_vec(nrows);
        ReadFitsColumn(fptr, "ra", ra_vec, first_row, params, num);
        std::vector<double> dec_vec(nrows);
        ReadFitsColumn(fptr, "dec", dec_vec, first_row, params, num);

        _pos.resize(nrows);
        for(long i=0;i<nrows;++i) {
            _pos[i] = Position<Flat>(ra_vec[i]*ra_units,dec_vec[i]*dec_units);
        }
        xdbg<<"First few positions are ";
        for (int i=0;i<std::min(3L,nrows);++i) xdbg<<_pos[i]<<"  ";
        xdbg<<std::endl;

    } else {
        myerror("No valid position columns specified");
    }

    // Kappa
    if (IsKColRequired(params,num) && params.keyExists("k_col")) {
        _kappa.resize(nrows);
        ReadFitsColumn(fptr, "k", _kappa, first_row, params, num);
        xdbg<<"First few kappas are ";
        for (int i=0;i<std::min(3L,nrows);++i) xdbg<<_kappa[i]<<"  ";
        xdbg<<std::endl;
    }

    // Shear
    if (IsGColRequired(params,num) && (params.keyExists("g1_col") || params.keyExists("g2_col"))) {
        std::vector<double> g1_vec(nrows);
        ReadFitsColumn(fptr, "g1", g1_vec, first_row, params, num);
        std::vector<double> g2_vec(nrows);
        ReadFitsColumn(fptr, "g2", g2_vec, first_row, params, num);

        _shear.resize(nrows);
        for(long i=0;i<nrows;++i) {
            //xdbg<<"row = "<<i<<": g1,g2 = "<<g1_vec[i]<<','<<g2_vec[i]<<std::endl;
            _shear[i] = std::complex<double>(g1_vec[i],g2_vec[i]);
            //xdbg<<"shear["<<i<<"] = "<<_shear[i]<<std::endl;
        }
        xdbg<<"First few shears are ";
        for (int i=0;i<std::min(3L,nrows);++i) xdbg<<_shear[i]<<"  ";
        xdbg<<std::endl;
    }

    // Weight
    if (params.keyExists("w_col")) {
        _weight.resize(nrows);
        ReadFitsColumn(fptr, "w", _weight, first_row, params, num);
        xdbg<<"First few weights are ";
        for (int i=0;i<std::min(3L,nrows);++i) xdbg<<_weight[i]<<"  ";
        xdbg<<std::endl;
    }

    // Flag
    if (params.keyExists("flag_col")) {
        long ignore_flag;
        if (params.keyExists("ignore_flag")) {
            ignore_flag = params["ignore_flag"];
        } else {
            long ok_flag = params.read("ok_flag",0);
            ignore_flag = ~ok_flag;
        }
        std::vector<long> flag_vec(nrows);
        ReadFitsColumn(fptr, "flag", flag_vec, first_row, params, num);
        if (_weight.size() == 0) _weight.resize(nrows,1.);
        for (int i=0;i<nrows;++i) {
            if (flag_vec[i] & ignore_flag) {
                xdbg<<"flag["<<i<<"] = "<<flag_vec[i]<<", so ignore.\n";
                _weight[i] = 0.;
            }
        }
        xdbg<<"First few weights are ";
        for (int i=0;i<std::min(3L,nrows);++i) xdbg<<_weight[i]<<"  ";
        xdbg<<std::endl;
    }

    // Check for Nans:
    //xdbg<<"Checking for NaN's:\n";
    for (long i=0, row_num=first_row; i<nrows; ++i, ++row_num) {
        //xdbg<<"Checking row "<<row_num<<" i = "<<i<<std::endl;
        bool has_nan = false;
        if (IsNan(_pos[i].getX() || IsNan(_pos[i].getY()))) has_nan = true;
        if (_shear.size() && (IsNan(real(_shear[i])) || IsNan(imag(_shear[i])))) has_nan = true;
        if (_kappa.size() && (IsNan(_kappa[i]))) has_nan = true;
        if (_weight.size() && (IsNan(_weight[i]))) has_nan = true;

        if (has_nan) {
            std::cerr<<"WARNING: Skipping row "<<row_num<<" in "<<file_name<<
                " because it has a NaN.\n";
            // Swap with the last row and decrement nrows
            if (i < nrows) {
                std::swap(_pos[i],_pos.back()); _pos.pop_back();
                if (_shear.size()) { std::swap(_shear[i],_shear.back()); _shear.pop_back(); }
                if (_kappa.size()) { std::swap(_kappa[i],_kappa.back()); _kappa.pop_back(); }
                if (_weight.size()) { std::swap(_weight[i],_weight.back()); _weight.pop_back(); }
            }
            --nrows;
            --i;  // Since incremented in for loop
        }
    }

}

void InputFile::project(double ra0, double dec0)
{
    dbg<<"Projecting points relative to (RA,Dec) = ("<<ra0<<','<<dec0<<") rad";
    dbg<<" = ("<<ra0*180./M_PI<<','<<dec0*180./M_PI<<") dec\n";

    // I use a stereographic projection, which preserves angles, but not distances
    // (you can't do both).  The distance error increases with distance from the 
    // projection point of course.
    // The equations are given at:
    //     http://mathworld.wolfram.com/StereographicProjection.html
    // x = k cos(phi) sin(lam-lam0)
    // y = k ( cos(phi0) sin(phi) - sin(phi0) cos(phi) cos(lam-lam0) )
    // k = 2 ( 1 + sin(phi0) sin(phi) + cos(phi0) cos(phi) cos(lam-lam0) )^-1
    //
    // In our case, lam = ra, phi = dec
    double cosdec0 = cos(dec0);
    double sindec0 = sin(dec0);
    const long n = _pos.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (long i=0; i<n; ++i) {
        double ra = _pos[i].getX();
        double dec = _pos[i].getY();
        xdbg<<"Project ("<<ra*180./M_PI<<','<<dec*180./M_PI<<") ";
        xdbg<<" relative to ("<<ra0*180./M_PI<<','<<dec0*180./M_PI<<")\n";

        double cosdec = cos(dec);
        double sindec = sin(dec);
        double cosdra = cos(ra - ra0);
        // Note: - sign here is to make +x correspond to -ra,
        //       so x increases for decreasing ra.
        //       East is left on the sky!
        double sindra = -sin(ra - ra0);
        //xxdbg<<"cosdec, sindec = "<<cosdec<<','<<sindec<<std::endl;
        //xxdbg<<"cosdra, sindra = "<<cosdra<<','<<sindra<<std::endl;

        double k = 2. / (1. + sindec0 * sindec + cosdec0 * cosdec * cosdra );
        //xxdbg<<"k = "<<k<<std::endl;
        double x = k * cosdec * sindra;
        double y = k * ( cosdec0 * sindec - sindec0 * cosdec * cosdra );
        xdbg<<"x,y -> "<<x<<','<<y<<std::endl;

        _pos[i] = Position<Flat>(x,y);

        // Now project the ellipticities if necessary:
        if (_shear.size() > 0) {
            Assert(i < long(_shear.size()));
            // For the projection, we use a spherical triangle with A = the point, 
            // B = the projection center, and C = the north pole.
            // The known values are:
            //   a = d(B,C) = Pi/2 - dec0
            //   b = d(A,C) = Pi/2 - dec
            //   C = ra - ra0
            // The value by which we need to rotate the shear is Pi - (A + B)
            // cos(Pi-(A+B)) = -cos(A+B) = -cosA cosB + sinA sinB
            // sin(Pi-(A+B)) = sin(A+B) = sinA cosB + cosA sinB
            //
            // cosc = cosa cosb + sina sinb cosC
            // cosc = sindec0 * sindec + cosdec0 * cosdec * cosdra;
            // cosa = cosb cosc + sinb sinc cosA
            // cosA = (sindec0 - sindec cosc) / (cosdec sinc)
            //      = (sindec0 - sindec (sindec0 sindec + cosdec0 cosdec cosdra) ) /
            //              (cosdec sinc)
            //      = (sindec0 cosdec - sindec cosdec0 cosdra ) / sinc
            // sinA / sina = sinC / sinc
            // sinA = cosdec0 * sindra / sinc
            //
            // Stable solution in case sinc (or cosdec) is small is to calculate denominators,
            // and then normalize using sinA^2 + cosA^2 = 1
            double cosA = sindec0 * cosdec - sindec * cosdec0 * cosdra;
            double sinA = cosdec0 * sindra;
            double normAsq = cosA*cosA + sinA*sinA;
            //xxdbg<<"A = "<<atan2(sinA,cosA)<<"  = "<<atan2(sinA,cosA)*180/M_PI<<" degrees\n";

            // cosb = cosa cosc + sina sinc cosB
            // cosB = (sindec - sindec0 cosc) / (cosdec0 sinc)
            //      = (sindec - sindec0 (sindec0 sindec + cosdec0 cosdec cosdra) ) / 
            //              (cosdec0 sinc)
            //      = (sindec cosdec0 - sindec0 cosdec cosdra ) / sinc
            // sinB / sinb = sinC / sinc
            // sinB = cosdec * sindra / sinc
            double cosB = sindec * cosdec0 - sindec0 * cosdec * cosdra;
            double sinB = cosdec * sindra;
            double normBsq = cosB*cosB + sinB*sinB;
            //xxdbg<<"B = "<<atan2(sinB,cosB)<<"  = "<<atan2(sinB,cosB)*180/M_PI<<" degrees\n";
            if (normAsq != 0. && normBsq != 0.) {
                // Otherwise this point is at the projection point, so no need for any projection.
                double cosbeta = -cosA * cosB + sinA * sinB;
                double sinbeta = sinA * cosB + cosA * sinB;
                //xxdbg<<"beta = "<<atan2(sinbeta,cosbeta)*180/M_PI<<std::endl;
                std::complex<double> expibeta(cosbeta,sinbeta);
                //xxdbg<<"expibeta = "<<expibeta/sqrt(normAsq*normBsq)<<std::endl;
                std::complex<double> exp2ibeta = (expibeta*expibeta)/(normAsq*normBsq);
                //xxdbg<<"exp2ibeta = "<<exp2ibeta<<std::endl;
                xdbg<<"shear = "<<_shear[i]<<" * "<<exp2ibeta<<" = "<<_shear[i]*exp2ibeta<<std::endl;
                _shear[i] *= exp2ibeta;
            }
        }
    }
    _use_ra_dec = false;
}

void ReadInputFiles(std::vector<InputFile*>& files, ConfigFile& params, 
                    const std::string& key, int num)
{
    xdbg<<"Start ReadInputFiles\n";

    // Check if this file_name is instead given as a file_list.
    std::string key2 = key;
    xdbg<<"key = "<<key<<std::endl;
    size_t pos = key2.find("name");
    Assert(pos != std::string::npos);  // npos indicates that substring wasn't found.
    key2.replace(pos, 4, "list");
    xdbg<<"key2 = "<<key2<<std::endl;
    if (params.keyExists(key2)) {
        if (params.keyExists(key)) myerror("Cannot specify both "+key+" and "+key2+".");
        std::string file_list = params[key2];
        xdbg<<"file_list = "<<file_list<<std::endl;

        std::vector<std::string> file_name;
        std::ifstream fin(file_list.c_str());
        if (!fin) myerror("Unable to open file " + file_list);
        dbg<<"Reading file_list "<<file_list<<std::endl;
        std::string temp;
        while (fin >> temp) {
            file_name.push_back(temp);
            dbg<<"   "<<temp<<std::endl;
        }
        xdbg<<"done: file_name.size = "<<file_name.size()<<std::endl;

        // Save this back to params.
        params[key] = file_name;
        xdbg<<"params["<<key<<"] = "<<params[key]<<std::endl;
    }

    // If this is file_name and there is no file_name2, then we need to increment num
    // for each file in the list.
    bool incr_num = num == 0 && key == "file_name" && !params.keyExists("file_name2");

    if (params.keyExists(key)) {
        std::vector<std::string> file_name = params[key];
        const int nfiles = file_name.size();
        for (int i=0; i<nfiles; ++i) {
            xdbg<<"Read file "<<i<<" = "<<file_name[i]<<std::endl;
            InputFile* file = new InputFile(file_name[i],params,num);
            files.push_back(file);
            if (incr_num) ++num;
        }
        if (files.size() > 1) {
            dbg<<"Done reading "<<files.size()<<" input files for "<<key<<".\n";
        } else {
            dbg<<"Done reading "<<key<<" = "<<files[0]->getFileName()<<std::endl;
        }
    }
}

