
#include "dbg.h"
#include "Field.h"
#include "Cell.h"

template <int DC, int M>
struct BuildCellDataHelper;

template <int M>
struct BuildCellDataHelper<NData,M>
{
    static void call(const InputFile& file, std::vector<CellData<NData,M>*>& celldata)
    {
        const int n = file.getPos().size();
        celldata.reserve(n);
        for(int i=0;i<n;++i) {
            double w = file.getWeight().size() > 0 ? file.getWeight()[i] : 1.;
            if (w != 0) 
                celldata.push_back(new CellData<NData,M>(file.getPos()[i]));
        }
    }
};

template <int M>
struct BuildCellDataHelper<KData,M>
{
    static void call(const InputFile& file, std::vector<CellData<KData,M>*>& celldata)
    {
        Assert(file.getKappa().size() == file.getPos().size());
        Assert(file.getWeight().size() == 0 || 
               file.getWeight().size() == file.getPos().size());
        const int n = file.getPos().size();
        celldata.reserve(n);
        celldata.resize(0);
        for(int i=0;i<n;++i) {
            double w = file.getWeight().size() > 0 ? file.getWeight()[i] : 1.;
            if (w != 0) 
                celldata.push_back(new CellData<KData,M>(file.getPos()[i],file.getKappa()[i],w));
        }
    }
};

template <int M>
struct BuildCellDataHelper<GData,M>
{
    static void call(const InputFile& file, std::vector<CellData<GData,M>*>& celldata)
    {
        Assert(file.getShear().size() == file.getPos().size());
        Assert(file.getWeight().size() == 0 || 
               file.getWeight().size() == file.getPos().size());
        const int n = file.getPos().size();
        celldata.reserve(n);
        celldata.resize(0);
        for(int i=0;i<n;++i) {
            double w = file.getWeight().size() > 0 ? file.getWeight()[i] : 1.;
            if (w != 0) 
                celldata.push_back(new CellData<GData,M>(file.getPos()[i],file.getShear()[i],w));
        }
    }
};

template <int DC> template <int M>
void Field<DC>::BuildCellData(const InputFile& file, std::vector<CellData<DC,M>*>& celldata)
{ BuildCellDataHelper<DC,M>::call(file,celldata); }

template <int DC>
double DoCalculateVar(const InputFile& file);

template <>
double DoCalculateVar<NData>(const InputFile& file)
{ return 0.; }

template <>
double DoCalculateVar<KData>(const InputFile& file)
{
    Assert(file.getKappa().size() == file.getPos().size());
    Assert(file.getWeight().size() == 0 || 
           file.getWeight().size() == file.getPos().size());
    const int n = file.getPos().size();
    double sumw = 0.;
    double vark = 0.;
    for(int i=0;i<n;++i) {
        double w = file.getWeight().size() > 0 ? file.getWeight()[i] : 1.;
        if (w != 0.) {
            double k = file.getKappa()[i];
            vark += w*w*k*k;
            sumw += w;
        }
    }
    vark /= sumw*sumw/n;
    return vark;
}

template <>
double DoCalculateVar<GData>(const InputFile& file)
{
    Assert(file.getShear().size() == file.getPos().size());
    Assert(file.getWeight().size() == 0 || 
           file.getWeight().size() == file.getPos().size());
    const int n = file.getPos().size();
    double sumw = 0.;
    double varg = 0.;
    for(int i=0;i<n;++i) {
        double w = file.getWeight().size() > 0 ? file.getWeight()[i] : 1.;
        if (w != 0.) {
            varg += w*w*norm(file.getShear()[i]);
            sumw += w;
        }
    }
    varg /= 2.*sumw*sumw/n; // 2 because we want var per component.
    return varg;
}

template <int DC>
double Field<DC>::CalculateVar(const InputFile& file)
{ return DoCalculateVar<DC>(file); }

template <int DC>
Field<DC>::Field(const InputFile& file, const ConfigFile& params,
                 double minsep, double maxsep, double b) :
    _filename(file.getFileName()), _var(0.)
{
    dbg<<"Building Field from file "<<file.getFileName()<<std::endl;

    // Read split_method as a string.
    std::string smstr = params.read<std::string>("split_method","mean");
    SplitMethod sm;
    if (smstr == "mean") sm = MEAN;
    else if (smstr == "median") sm = MEDIAN;
    else if (smstr == "middle") sm = MIDDLE;
    else {
        myerror("Invalid split_method"+smstr);
        sm = MEAN; // To prevent compiler warning
    }
    xdbg<<"Splitting using "<<smstr<<std::endl;

    // We don't bother accumulating the mean information for Cells that would be 
    // too large or too small.

    // The minimum size cell that will be useful is one where two cells that just barely
    // don't split have (d + s1 + s2) = minsep
    // The largest s2 we need to worry about is s2 = 2s1.
    // i.e. d = minsep - 3s1  and s1 = 0.5 * bd
    //      d = minsep - 1.5 bd
    //      d = minsep / (1+1.5 b)
    //      s = 0.5 * b * minsep / (1+1.5 b)
    //        = b * minsep / (2+3b)
    double minsize = minsep * b / (2.+3.*b);
    double minsizesq = minsize * minsize;
    xdbg<<"minsizesq = "<<minsizesq<<std::endl;

    // The maximum size cell that will be useful is one where a cell of size s will
    // be split at the maximum separation even if the other size = 0.
    // i.e. s = b * maxsep
    double maxsize = maxsep * b;
    double maxsizesq = maxsize * maxsize;
    xdbg<<"maxsizesq = "<<maxsizesq<<std::endl;

    // Build the right kind of Cell according to whether the InputFile uses RA,Dec or x,y.
    if (file.useRaDec()) {
        _metric = Sphere;
        std::vector<CellData<DC,Sphere>*> celldata;
        BuildCellData(file,celldata);
        xdbg<<"Built celldata with "<<celldata.size()<<" entries\n";
        buildCells(celldata,_cells_sphere,minsizesq,maxsizesq,sm);
        for (size_t i=0;i<celldata.size();++i) if (celldata[i]) delete celldata[i];
    } else {
        _metric = Flat;
        std::vector<CellData<DC,Flat>*> celldata;
        BuildCellData(file,celldata);
        xdbg<<"Built celldata with "<<celldata.size()<<" entries\n";
        buildCells(celldata,_cells_flat,minsizesq,maxsizesq,sm);
        for (size_t i=0;i<celldata.size();++i) if (celldata[i]) delete celldata[i];
    }
    _var = CalculateVar(file);
    if (DC == KData) {
        Assert(_var >= 0.);
        dbg<<"vark = "<<_var<<": sig_sn = "<<sqrt(_var)<<std::endl;
    }
    if (DC == GData) {
        Assert(_var >= 0.);
        dbg<<"varg = "<<_var<<": sig_sn (per component) = "<<sqrt(_var)<<std::endl;
    }
}

template <int DC> template <int M>
void Field<DC>::buildCells(
    std::vector<CellData<DC,M>*>& vdata,
    std::vector<Cell<DC,M>*>& cells,
    double minsizesq, double maxsizesq, SplitMethod sm)
{
    xdbg<<"Start buildCells\n";
    xdbg<<"minsizesq = "<<minsizesq<<", maxsizesq = "<<maxsizesq<<std::endl;

    // This is done in two parts so that we can do the (time-consuming) second part in 
    // parallel.
    // First we setup what all the top-level cells are going to be.
    // Then we build them and their sub-nodes.
    
    if (maxsizesq == 0.) {
        dbg<<"Doing brute-force calculation (all cells are leaf nodes).\n";
        // If doing a brute-force calculation, the top-level cell data are the same as vdata.
        const int n = vdata.size();
        cells.resize(n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=0;i<n;++i) {
            cells[i] = new Cell<DC,M>(vdata[i]);
            vdata[i] = 0; // Make sure the calling routing doesn't delete this one.
        }
    } else {
        std::vector<CellData<DC,M>*> top_data;
        std::vector<double> top_sizesq;
        std::vector<size_t> top_start;
        std::vector<size_t> top_end;
        setupCells(vdata,minsizesq,maxsizesq,sm,0,vdata.size(),
                   top_data,top_sizesq,top_start,top_end);
        const int n = top_data.size();
        dbg<<"Field has "<<n<<" top-level nodes.  Building lower nodes...\n";
        cells.resize(n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=0;i<n;++i) 
            cells[i] = new Cell<DC,M>(top_data[i],top_sizesq[i],vdata,minsizesq,sm,
                                      top_start[i],top_end[i]);
    }
}

template <int DC> template <int M>
void Field<DC>::setupCells(
    std::vector<CellData<DC,M>*>& vdata,
    double minsizesq, double maxsizesq,
    SplitMethod sm, size_t start, size_t end,
    std::vector<CellData<DC,M>*>& top_data,
    std::vector<double>& top_sizesq,
    std::vector<size_t>& top_start, std::vector<size_t>& top_end)
{
    xdbg<<"Start setupCells: start,end = "<<start<<','<<end<<std::endl;
    xdbg<<"minsizesq = "<<minsizesq<<", maxsizesq = "<<maxsizesq<<std::endl;
    // The structure of this is very similar to the Cell constructor.
    // The difference is that here we only construct a new Cell (and do the corresponding
    // calculation of the averages) if the size is small enough.  At that point, the 
    // rest of the construction is passed onto the Cell class.
    CellData<DC,M>* ave;
    double sizesq;
    if (end-start == 1) {
        xdbg<<"Only 1 CellData entry: size = 0\n";
        ave = vdata[start];
        vdata[start] = 0; // Make sure the calling function doesn't delete this!
        sizesq = 0.;
    } else {
        ave = new CellData<DC,M>(vdata,start,end);
        sizesq = CalculateSizeSq(ave->getPos(),vdata,start,end);
        xdbg<<"size = "<<sqrt(sizesq)<<std::endl;
    }

    if (sizesq <= maxsizesq) {
        xdbg<<"Small enough.  Make a cell.\n";
        if (end-start > 1) ave->finishAverages(vdata,start,end);
        top_data.push_back(ave);
        top_sizesq.push_back(sizesq);
        top_start.push_back(start);
        top_end.push_back(end);
    } else {
        size_t mid = SplitData(vdata,sm,start,end,ave->getPos());
        xdbg<<"Too big.  Recurse with mid = "<<mid<<std::endl;
        setupCells(vdata,minsizesq,maxsizesq,sm,start,mid,top_data,top_sizesq,top_start,top_end);
        setupCells(vdata,minsizesq,maxsizesq,sm,mid,end,top_data,top_sizesq,top_start,top_end);
    }
}

template class Field<NData>;
template class Field<KData>;
template class Field<GData>;
template void Field<NData>::BuildCellData(
    const InputFile& file, std::vector<CellData<NData,Flat>*>& celldata);
template void Field<KData>::BuildCellData(
    const InputFile& file, std::vector<CellData<KData,Flat>*>& celldata);
template void Field<GData>::BuildCellData(
    const InputFile& file, std::vector<CellData<GData,Flat>*>& celldata);
template void Field<NData>::BuildCellData(
    const InputFile& file, std::vector<CellData<NData,Sphere>*>& celldata);
template void Field<KData>::BuildCellData(
    const InputFile& file, std::vector<CellData<KData,Sphere>*>& celldata);
template void Field<GData>::BuildCellData(
    const InputFile& file, std::vector<CellData<GData,Sphere>*>& celldata);
