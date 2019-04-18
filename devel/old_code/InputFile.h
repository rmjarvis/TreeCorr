
#ifndef InputFile_H
#define InputFile_H

#include <vector>
#include <fstream>
#include <string>
#include <complex>
#include <cctype>

#include <fitsio.h>

#include "ConfigFile.h"
#include "Bounds.h"

class InputFile
{
public:
    InputFile(const std::string& file_name, const ConfigFile& params, int num);

    std::string getFileName() const { return _file_name; }
    bool useRaDec() const { return _use_ra_dec; }
    const std::vector<Position<Flat> >& getPos() const { return _pos; }
    const std::vector<double>& getKappa() const { return _kappa; }
    const std::vector<std::complex<double> >& getShear() const { return _shear; }
    const std::vector<double>& getWeight() const { return _weight; }
    size_t getNTot() const { return _pos.size(); }

private:

    void readAscii(const std::string& file_name, const ConfigFile& params, int num);
    void readFits(const std::string& file_name, const ConfigFile& params, int num);
    void project(double ra0, double dec0);

    std::string _file_name;
    bool _use_ra_dec;
    std::vector<Position<Flat> > _pos;
    std::vector<double> _kappa;
    std::vector<std::complex<double> > _shear;
    std::vector<double> _weight;
};

void ReadInputFiles(std::vector<InputFile*>& files, ConfigFile& params,
                    const std::string& key, int num);

double GetUnits(const ConfigFile& params, const std::string& key, int num, 
                const std::string& def="");

#endif
