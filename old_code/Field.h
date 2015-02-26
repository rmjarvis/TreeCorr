#ifndef Field_H
#define Field_H

#include "Cell.h"

template <int DC>
class Field
{
public:
    // A Field is a list of Cells that are relevant for a particular calculation.
    // It only constructs the Cells that will be relevant for the correlation function
    // calculation based on the provided min/max separation and the binsize.
    Field(const InputFile& file, const ConfigFile& params,
          double minsep, double maxsep, double b);
    ~Field() {}

    const std::vector<Cell<DC,Sphere>*>& getCells_Sphere() const
    { Assert(_metric == Sphere); return _cells_sphere; }

    const std::vector<Cell<DC,Flat>*>& getCells_Flat() const
    { Assert(_metric == Flat); return _cells_flat; }

    Metric getMetric() const { return _metric; }
    int getN() const { return _metric==Flat ? _cells_flat.size() : _cells_sphere.size(); }
    long getNTot() const 
    {
        long ntot = 0;
        if (_metric == Flat)
            for (size_t i=0;i<_cells_flat.size();++i) ntot += _cells_flat[i]->getData().getN(); 
        else
            for (size_t i=0;i<_cells_sphere.size();++i) ntot += _cells_sphere[i]->getData().getN(); 
        return ntot;
    }
    double getVar() const { return _var; }
    std::string getFileName() const { return _filename; }

    // These are static functions, so they can be used without constructing a Field object.
    // In the normal flow of things, they are called from within the Field constructor.
    static double CalculateVar(const InputFile& file);
    template <int M>
    static void BuildCellData(const InputFile& file, std::vector<CellData<DC,M>*>& celldata);

private:

    // A helper function for the constructor.
    // Needs to be its own function, since it is recursive.
    template <int M>
    void buildCells(std::vector<CellData<DC,M>*>& vdata,
                    std::vector<Cell<DC,M>*>& cells,
                    double minsizesq, double maxsizesq, SplitMethod sm);
    template <int M>
    void setupCells(std::vector<CellData<DC,M>*>& vdata,
                    double minsizesq, double maxsizesq,
                    SplitMethod sm, size_t start, size_t end,
                    std::vector<CellData<DC,M>*>& top_data,
                    std::vector<double>& top_sizesq,
                    std::vector<size_t>& top_start, std::vector<size_t>& top_end);

    std::string _filename;
    Metric _metric;
    double _var;
    std::vector<Cell<DC,Flat>*> _cells_flat;
    std::vector<Cell<DC,Sphere>*> _cells_sphere;

};

#endif
