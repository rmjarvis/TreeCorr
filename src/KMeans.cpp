/* Copyright (c) 2003-2019 by Mike Jarvis
 *
 * TreeCorr is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

//#define DEBUGLOGGING

#include <cstddef>  // for ptrdiff_t
#include <vector>
#include "Field.h"
#include "Cell.h"
#include "dbg.h"

extern "C" {
#include "Field_C.h"
}

// In BinnedCorr2.cpp
extern void SelectRandomFrom(long m, std::vector<long>& selection);

template <int D, int C>
void InitializeCenters(std::vector<Position<C> >& centers, const Cell<D,C>* cell,
                       int first, int ncenters)
{
    dbg<<"Recursive InitializeCenters: "<<first<<"  "<<ncenters<<std::endl;
    xdbg<<"pos = "<<cell->getPos()<<std::endl;
    if (ncenters == 1) {
        xdbg<<"Only 1 center.  Use this cell.\n";
        Assert(first < int(centers.size()));
        dbg<<"initial center["<<first<<"] = "<<cell->getPos()<<std::endl;
        centers[first] = cell->getPos();
    } else if (cell->getLeft()) {
        int m = ncenters / 2;
        xdbg<<"Recurse with m = "<<m<<std::endl;
        InitializeCenters(centers, cell->getLeft(), first, m);
        InitializeCenters(centers, cell->getRight(), first + m, ncenters - m);
    } else {
        dbg<<"BAD!  Shouldn't happen.  Probably shouldn't be running kmeans on this field.\n";
        // Shouldn't ever happen -- this means a leaf is very close to the top.
        // But still, let's do soemthing at least vaguely sensible if it does.
        for (int i=0; i<ncenters; ++i) {
            Assert(first+i < int(centers.size()));
            centers[first+i] = Position<C>(cell->getPos()*(1. + i * 1.e-8));
        }
    }
    xdbg<<"End of Recursive InitializeCenters\n";
}

template <int D, int C>
void InitializeCenters(std::vector<Position<C> >& centers, const std::vector<Cell<D,C>*>& cells)
{
    dbg<<"Initialize centers: "<<centers.size()<<"  "<<cells.size()<<std::endl;
    if (cells.size() > centers.size()) {
        dbg<<"More cells than centers.  Pick from them randomly.\n";
        std::vector<long> selection(centers.size());
        SelectRandomFrom(long(cells.size()), selection);
        for (size_t i=0; i<centers.size(); ++i) {
            Assert(selection[i] < cells.size());
            centers[i] = cells[selection[i]]->getPos();
        }
    } else {
        dbg<<"Fewer cells than centers.  Recurse to get more than one from each.\n";
        int n1 = int(centers.size() / cells.size());
        int k2 = centers.size() % cells.size();
        int n2 = n1 + 1;
        int k1 = cells.size() - k2;
        dbg<<"n1 = "<<n1<<" n2 = "<<n2<<std::endl;
        dbg<<"k1 = "<<k1<<" k2 = "<<k2<<std::endl;
        Assert(n1 >= 1);
        Assert(n1 * k1 + n2 * k2 == centers.size());
        for (int k=0; k<k1; ++k) {
            Assert(k < int(cells.size()));
            InitializeCenters(centers, cells[k], n1*k, n1);
        }
        for (size_t k=k1; k<cells.size(); ++k) {
            Assert(k < int(cells.size()));
            InitializeCenters(centers, cells[k], n1*k1 + n2*(k-k1), n2);
        }
    }
    dbg<<"Done initializing centers\n";
}

template <int D, int C>
struct StoreCells
{
    int npatch;
    std::vector<std::vector<const Cell<D,C>*> > cells_by_patch;

    StoreCells(int _npatch) : npatch(_npatch), cells_by_patch(npatch) {}

    void run(int patch_num, const Cell<D,C>* cell)
    {
        cells_by_patch[patch_num].push_back(cell);
    }

    void combineWith(const StoreCells<D,C>& rhs)
    {
        for (int i=0; i<npatch; ++i)
            cells_by_patch[i].insert(cells_by_patch[i].end(),
                                     rhs.cells_by_patch[i].begin(),
                                     rhs.cells_by_patch[i].end());
    }

    void finalize() {}

    void reset()
    {
        for (int i=0;i<npatch;++i) cells_by_patch[i].clear();
    }
};

template <int D, int C>
struct UpdateCenters
{
    int npatch;
    std::vector<Position<C> > new_centers;
    std::vector<double> w;

    UpdateCenters(int _npatch) : npatch(_npatch), new_centers(npatch), w(npatch) {}

    void run(int patch_num, const Cell<D,C>* cell)
    {
        new_centers[patch_num] += cell->getPos() * cell->getW();
        w[patch_num] += cell->getW();
    }

    void combineWith(const UpdateCenters<D,C>& rhs)
    {
        for (int i=0; i<npatch; ++i) {
            new_centers[i] += rhs.new_centers[i];
            w[i] += rhs.w[i];
        }
    }

    void finalize()
    {
        for (int i=0; i<npatch; ++i) {
            new_centers[i] /= w[i];
            new_centers[i].normalize();
            dbg<<"New center = "<<new_centers[i]<<std::endl;
        }
    }

    void reset()
    {
        for (int i=0;i<npatch;++i) new_centers[i] = Position<C>();
        for (int i=0;i<npatch;++i) w[i] = 0.;
    }
};

template <int D, int C>
struct CalculateInertia
{
    int npatch;
    std::vector<double> inertia;
    double sumw;
    const std::vector<Position<C> >& centers;

    CalculateInertia(int _npatch, const std::vector<Position<C> >& _centers) :
        npatch(_npatch), inertia(npatch,0.), sumw(0.), centers(_centers) {}

    void run(int patch_num, const Cell<D,C>* cell)
    {
        double ssq = cell->getSizeSq();
        double w = cell->getW();
        inertia[patch_num] += (cell->getPos() - centers[patch_num]).normSq() * w;
        // Parallel axis theorem says we should add the inertia of this cell about its own
        // centroid.  We can calculate that with cell->calculateInertia(), but for small cells,
        // it is very close to I = w s^2, and it turns out that this is generally good enough.
        if (ssq > 0.) {
            double I1 = ssq * w;
#ifdef DEBUGLOGGING
            double I2 = cell->calculateInertia();
            dbg<<"ssq, I1, I2 = "<<ssq<<"  "<<I1<<"  "<<I2<<std::endl;
#endif
            inertia[patch_num] += I1;
        }
        sumw += w;
    }

    void combineWith(const CalculateInertia<D,C>& rhs)
    {
        for (int i=0; i<npatch; ++i) {
            inertia[i] += rhs.inertia[i];
        }
        sumw += rhs.sumw;
    }

    void finalize()
    {
#ifdef DEBUGLOGGING
        double mean=0.;
        double rms=0.;
#endif
        double meanw = sumw / npatch;

        for (int i=0; i<npatch; ++i) {
            inertia[i] /= meanw;
#ifdef DEBUGLOGGING
            dbg<<"Inertia["<<i<<"] = "<<inertia[i]<<std::endl;
            mean += inertia[i];
            rms += SQR(inertia[i]);
#endif
        }
#ifdef DEBUGLOGGING
        mean /= npatch;
        rms -= npatch * SQR(mean);
        rms = sqrt(rms / npatch);
        dbg<<"mean inertia = "<<mean<<std::endl;
        dbg<<"rms inertia = "<<rms<<std::endl;
#endif
    }

    void reset()
    {
        for (int i=0;i<npatch;++i) inertia[i] = 0.;
        sumw = 0.;
    }
};

template <int D, int C>
struct AssignPatches
{
    long* patches;
    long n;

    AssignPatches(long* _patches, long _n) : patches(_patches), n(_n) {}

    void run(int patch_num, const Cell<D,C>* cell)
    {
        xdbg<<"Start AssignPatches "<<cell<<" "<<patch_num<<std::endl;
        if (cell->getLeft()) {
            xdbg<<"Not a leaf.  Recurse\n";
            run(patch_num, cell->getLeft());
            run(patch_num, cell->getRight());
        } else if (cell->getN() == 1) {
            long index = cell->getInfo().index;
            xdbg<<"N=1.  index = "<<index<<std::endl;
            Assert(index < n);
            patches[index] = patch_num;
        } else {
            std::vector<long>* indices = cell->getListInfo().indices;
            xdbg<<"Leaf with N>1.  "<<indices->size()<<" indices\n";
            for (size_t j=0; j<indices->size(); ++j) {
                long index = (*indices)[j];
                xdbg<<"    index = "<<index<<std::endl;
                Assert(index < n);
                patches[index] = patch_num;
            }
        }
    }

    void combineWith(const AssignPatches<D,C>& rhs) {}
    void finalize() {}
    void reset() {}
};

template <int D, int C, typename F>
void FindCellsInPatches(const std::vector<Position<C> >& centers,
                        const Cell<D,C>* cell, std::vector<long>& patches, long ncand,
                        std::vector<double>& saved_dsq, F& f,
                        const std::vector<double>* inertia)
{
    //set_verbose(2);
    xdbg<<"Start recursive FindCellsInPatches\n";
    // First find the center that is closest to the current cell's center
    const Position<C> cell_center = cell->getPos();
    double s = cell->getSize();
    xdbg<<"cell = "<<cell_center<<"  "<<s<<"  "<<cell->getN()<<std::endl;

    // Start with a guess that the closest one is in the first position.
    double closest_i = patches[0];
    double min_dsq = (cell_center - centers[closest_i]).normSq();
    saved_dsq[0] = min_dsq;
    if (inertia) {
        xdbg<<"Initial min_dsq = "<<min_dsq<<", I = "<<(*inertia)[closest_i]<<std::endl;
        min_dsq += (*inertia)[closest_i];
        xdbg<<"    min_dsq => "<<min_dsq<<std::endl;
    }
    // Note: saved_dsq is really the d^2 values.
    //       min_dsq is the minimum value of the possibly modified distance:
    //          dsq         for alt = False.
    //          dsq + I_i   for alt = True.

    // Look for closer center
    for (int j=1; j<ncand; ++j) {
        long i=patches[j];
        double dsq = (cell_center - centers[i]).normSq();
        saved_dsq[j] = dsq;
        if (inertia) {
            xdbg<<"dsq = "<<dsq<<", I = "<<(*inertia)[i]<<std::endl;
            dsq += (*inertia)[i];
            xdbg<<"   dsq => "<<dsq<<std::endl;
        }
        xdbg<<"dsq["<<i<<"] = "<<dsq<<", min = "<<min_dsq<<std::endl;
        if (dsq < min_dsq) {
            std::swap(saved_dsq[0], saved_dsq[j]);
            std::swap(patches[0], patches[j]);
            closest_i = i;
            min_dsq = dsq;
        }
    }
    double min_d = sqrt(saved_dsq[0]);
    xdbg<<"closest center = "<<closest_i<<"  "<<centers[closest_i]<<std::endl;
    xdbg<<"min_d = "<<min_d<<"  min_dsq = "<<min_dsq<<std::endl;
    double thresh_dsq;
    if (!inertia) {
        // Can remove any candidate with d - size >  min_d + size
        thresh_dsq = SQR(min_d + 2*s);
    } else {
        // When using inertia, it is slightly more complicated.
        // (d_i-s)^2 + I_i > (min_d+s)^2 + I_0
        // Since this involved the specific I_i in each step, it's simpler to just
        // calculate the left side specifically for each patch.
        thresh_dsq = SQR(min_d + s) + (*inertia)[closest_i];
    }
    xdbg<<"thresh_dsq = "<<thresh_dsq<<std::endl;

    // Update patches to remove any that cannot be the right center from candidate section.
    for (int j=ncand-1; j>0; --j) {
        double dsq = saved_dsq[j];
        if (inertia) {
            double d = sqrt(dsq);
            // If s > d, then the normal calculation doesn't apply.  Use dsq = 0.
            if (s > d)
                dsq = 0.;
            else
                dsq = SQR(d-s) + (*inertia)[patches[j]];
        }
        xdbg<<"Check: "<<dsq<<" >? "<<thresh_dsq<<std::endl;
        if (dsq > thresh_dsq) {
            if (patches[j] < 2) {
                xdbg<<"Remove cell with center "<<centers[patches[j]]<<std::endl;
                xdbg<<"cell = "<<cell_center<<"  "<<s<<"  "<<cell->getN()<<std::endl;
                xdbg<<dsq<<" > "<<thresh_dsq<<std::endl;
                xdbg<<"min_d = "<<min_d<<std::endl;
            }
            --ncand;
            if (j != ncand) {
                std::swap(patches[j], patches[ncand]);
                // Don't bother with saved_dsq, since don't need that anymore.
            }
        }
    }
    xdbg<<"There are "<<ncand<<" patches remaining\n";

    //set_verbose(1);
    if (ncand == 1 || s == 0.) {
        // If we only have one candidate left, we're done.  Use this cell to update this patch.
        f.run(closest_i, cell);
    } else {
        // Otherwise, need to recurse to sub-cells.
        FindCellsInPatches(centers, cell->getLeft(), patches, ncand, saved_dsq, f, inertia);
        // Note: the above call might have swapped some patches around, but only within the
        // first ncand values, so they are still valid for the next call.
        FindCellsInPatches(centers, cell->getRight(), patches, ncand, saved_dsq, f, inertia);
    }
}

// This recurses the tree until if finds a cell that completely belongs in only a single
// patch and then runs f, which can be any of the above function classes.
template <int D, int C, typename F>
void FindCellsInPatches(const std::vector<Position<C> >& centers,
                        const std::vector<Cell<D,C>*>& cells, F& f,
                        const std::vector<double>* inertia=0)
{
#ifdef _OPENMP
#pragma omp parallel
    {
        // Give each thread their own copy of f to use.
        F f2 = f;
#else
        F& f2 = f;
#endif

        // We start with all patches as candidates.
        long npatch = centers.size();
        std::vector<long> patches(npatch);
        for (long i=0; i<npatch; ++i) patches[i] = i;

        // Set up a work space where we save dsq calculations.
        std::vector<double> saved_dsq(npatch);

        // Start a recursion for each top-level cell.
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (size_t k=0; k<cells.size(); ++k) {
            FindCellsInPatches(centers, cells[k], patches, npatch, saved_dsq, f2, inertia);
        }

#ifdef _OPENMP
        // Combine the results
#pragma omp critical
        {
            f.combineWith(f2);
        }
    }
#endif
}



template <int C>
double CalculateShiftSq(const std::vector<Position<C> >& centers,
                        const std::vector<Position<C> >& new_centers)
{
    dbg<<"Start CalculateShiftSq\n";
    double shiftsq=0.;
    for (size_t i=0; i<centers.size(); ++i) {
        double shiftsq_i = (centers[i] - new_centers[i]).normSq();
        dbg<<"Shift for "<<i<<" = "<<centers[i]<<" -> "<<new_centers[i]<<"  d = "<<sqrt(shiftsq_i)<<std::endl;
        shiftsq += shiftsq_i;
    }
    dbg<<"Total shiftsq = "<<shiftsq<<std::endl;
    return shiftsq;
}



// C = Threed or Sphere
template <int C>
void WriteCenters(const std::vector<Position<C> >& centers, double* pycenters, long npatch)
{
    for(int i=0; i<npatch; ++i, pycenters+=3) {
        pycenters[0] = centers[i].getX();
        pycenters[1] = centers[i].getY();
        pycenters[2] = centers[i].getZ();
    }
}
template <int C>
void ReadCenters(std::vector<Position<C> >& centers, const double* pycenters, long npatch)
{
    for(int i=0; i<npatch; ++i, pycenters+=3) {
        centers[i] = Position<C>(pycenters[0], pycenters[1], pycenters[2]);
    }
}

// C = Flat
template <>
void WriteCenters(const std::vector<Position<Flat> >& centers, double* pycenters, long npatch)
{
    for(int i=0; i<npatch; ++i, pycenters+=2) {
        pycenters[0] = centers[i].getX();
        pycenters[1] = centers[i].getY();
    }
}

template <>
void ReadCenters(std::vector<Position<Flat> >& centers, const double* pycenters, long npatch)
{
    for(int i=0; i<npatch; ++i, pycenters+=2) {
        centers[i] = Position<Flat>(pycenters[0], pycenters[1]);
    }
}

template <int D, int C>
void KMeansInit2(Field<D,C>*field, double* pycenters, long npatch)
{
    dbg<<"Start KMeansInit for "<<npatch<<" patches\n";
    const std::vector<Cell<D,C>*> cells = field->getCells();
    std::vector<Position<C> > centers(npatch);
    InitializeCenters(centers, cells);
    WriteCenters(centers, pycenters, npatch);
}


template <int D, int C>
void KMeansRun2(Field<D,C>*field, double* pycenters, long npatch, int max_iter, double tol,
                bool alt)
{
    dbg<<"Start KMeansRun for "<<npatch<<" patches\n";
    const std::vector<Cell<D,C>*> cells = field->getCells();

    // Initialize the centers of the patches smartly according to the tree structure.
    std::vector<Position<C> > centers(npatch);
    ReadCenters(centers, pycenters, npatch);

    // We compare the total shift^2 to this number
    // Want rms shift / field.size < tol
    // So sqrt(total_shift^2 / npatch) / field.size < tol
    // total_shift^2 < tol^2 * size^2 * npatch
    double tolsq = tol*tol * field->getSizeSq() * npatch;
    dbg<<"tolsq = "<<tolsq<<std::endl;

    // The alt version needs to keep track of the inertia of each patch.
    CalculateInertia<D,C> calculate_inertia(alt ? npatch : 0, centers);
    std::vector<double>* pinertia = 0;
    dbg<<"Made calculate_inertia\n";

    // Keep track of which cells belong to which patch.
    UpdateCenters<D,C> update_centers(npatch);
    dbg<<"Made update_centers\n";

    for(int iter=0; iter<max_iter; ++iter) {
        dbg<<"Start iter "<<iter<<std::endl;
        // Update the inertia if we are doing the alternate version
        if (alt) {
            std::vector<double> inertia = calculate_inertia.inertia;
            calculate_inertia.reset();
            FindCellsInPatches(centers, cells, calculate_inertia, &inertia);
            calculate_inertia.finalize();
            pinertia = &calculate_inertia.inertia;
        }

        // Figure out which cells belong to which patch according to the current centers.
        // Note: clear leaves the previous capacity available, so usually won't need much
        // in the way of allocation here.
        update_centers.reset();
        FindCellsInPatches(centers, cells, update_centers, pinertia);
        update_centers.finalize();
        dbg<<"After UpdateCenters\n";

        // Check for convergence
        double shiftsq = CalculateShiftSq(centers, update_centers.new_centers);
        centers = update_centers.new_centers;
        dbg<<"Iter "<<iter<<": shiftsq = "<<shiftsq<<"  tolsq = "<<tolsq<<std::endl;
        // Stop if (rms shift / size) < tol
        if (shiftsq < tolsq) {
            dbg<<"Stopping RunKMeans because rms shift = "<<sqrt(shiftsq/npatch)<<std::endl;
            dbg<<"tol * size = "<<tol * sqrt(field->getSizeSq())<<std::endl;
            break;
        }
        if (iter == max_iter-1) {
            dbg<<"Stopping RunKMeans because hit maximum iterations = "<<max_iter<<std::endl;
        }
    }
    WriteCenters(centers, pycenters, npatch);
}

template <int D, int C>
void KMeansAssign2(Field<D,C>*field, double* pycenters, long npatch, bool alt,
                   long* patches, long n)
{
    dbg<<"Start KMeansAssign for "<<npatch<<" patches\n";
    const std::vector<Cell<D,C>*> cells = field->getCells();

    std::vector<Position<C> > centers(npatch);
    ReadCenters(centers, pycenters, npatch);

    CalculateInertia<D,C> calculate_inertia(alt ? npatch : 0, centers);
    std::vector<double>* pinertia = 0;
    if (alt) {
        // The alt version needs to keep track of the inertia of each patch.
        dbg<<"Calculate inertia.\n";
        const int niter = 1;  // There doesn't seem to be any benefit to more than one pass here.
        for (int i=0;i<niter;++i) {
            std::vector<double> inertia = calculate_inertia.inertia;
            calculate_inertia.reset();
            FindCellsInPatches(centers, cells, calculate_inertia, &inertia);
            calculate_inertia.finalize();
        }
        pinertia = &calculate_inertia.inertia;
    }

    AssignPatches<D,C> assign_patches(patches, n);
    FindCellsInPatches(centers, cells, assign_patches, pinertia);
    dbg<<"After AssignPatches\n";
}

template <int D>
void KMeansInit1(void* field, double* centers, long npatch, int coords)
{
    switch(coords) {
      case Flat:
           KMeansInit2(static_cast<Field<D,Flat>*>(field), centers, npatch);
           break;
      case Sphere:
           KMeansInit2(static_cast<Field<D,Sphere>*>(field), centers, npatch);
           break;
      case ThreeD:
           KMeansInit2(static_cast<Field<D,ThreeD>*>(field), centers, npatch);
           break;
    }
}

void KMeansInit(void* field, double* centers, long npatch, int d, int coords)
{
    switch(d) {
      case NData:
           KMeansInit1<NData>(field, centers, npatch, coords);
           break;
      case KData:
           KMeansInit1<KData>(field, centers, npatch, coords);
           break;
      case GData:
           KMeansInit1<GData>(field, centers, npatch, coords);
           break;
    }
}

template <int D>
void KMeansRun1(void* field, double* centers, long npatch, int max_iter, double tol, bool alt,
                int coords)
{
    switch(coords) {
      case Flat:
           KMeansRun2(static_cast<Field<D,Flat>*>(field), centers, npatch, max_iter, tol, alt);
           break;
      case Sphere:
           KMeansRun2(static_cast<Field<D,Sphere>*>(field), centers, npatch, max_iter, tol, alt);
           break;
      case ThreeD:
           KMeansRun2(static_cast<Field<D,ThreeD>*>(field), centers, npatch, max_iter, tol, alt);
           break;
    }
}

void KMeansRun(void* field, double* centers, long npatch, int max_iter, double tol, int alt,
               int d, int coords)
{
    switch(d) {
      case NData:
           KMeansRun1<NData>(field, centers, npatch, max_iter, tol, bool(alt), coords);
           break;
      case KData:
           KMeansRun1<KData>(field, centers, npatch, max_iter, tol, bool(alt), coords);
           break;
      case GData:
           KMeansRun1<GData>(field, centers, npatch, max_iter, tol, bool(alt), coords);
           break;
    }
}

template <int D>
void KMeansAssign1(void* field, double* centers, long npatch, bool alt, long* patches, long n,
                   int coords)
{
    switch(coords) {
      case Flat:
           KMeansAssign2(static_cast<Field<D,Flat>*>(field), centers, npatch, alt, patches, n);
           break;
      case Sphere:
           KMeansAssign2(static_cast<Field<D,Sphere>*>(field), centers, npatch, alt, patches, n);
           break;
      case ThreeD:
           KMeansAssign2(static_cast<Field<D,ThreeD>*>(field), centers, npatch, alt, patches, n);
           break;
    }
}

void KMeansAssign(void* field, double* centers, long npatch, int alt, long* patches, long n,
                  int d, int coords)
{
    switch(d) {
      case NData:
           KMeansAssign1<NData>(field, centers, npatch, bool(alt), patches, n, coords);
           break;
      case KData:
           KMeansAssign1<KData>(field, centers, npatch, bool(alt), patches, n, coords);
           break;
      case GData:
           KMeansAssign1<GData>(field, centers, npatch, bool(alt), patches, n, coords);
           break;
    }
}


