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
#include <set>
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

#define cell_storage_type std::vector<const Cell<D,C>*>
// Comment: I tried a number of stdlib containers for the cells_by_patch internal container,
// and vector was the fastest.  I kind of expected deque to be faster, and it was close, but
// apparently the amortized reallocations required for push_back were negligible compared to
// the speed advantage from having things contiguous is memory.
// For the record, set and list were both much slower.

template <int D, int C>
void FindCellsInPatches(const std::vector<Position<C> >& centers,
                   const Cell<D,C>* cell, std::vector<long>& patches, long ncand,
                   std::vector<cell_storage_type>& cells_by_patch,
                   std::vector<double>& saved_dsq)
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

    // Look for closer center
    for (int j=1; j<ncand; ++j) {
        long i=patches[j];
        saved_dsq[j] = (cell_center - centers[i]).normSq();
        if (saved_dsq[j] < min_dsq) {
            std::swap(saved_dsq[0], saved_dsq[j]);
            std::swap(patches[0], patches[j]);
            closest_i = i;
            min_dsq = saved_dsq[0];
        }
    }
    double min_d = sqrt(min_dsq);
    // Can remove any candidate with d - size >  min_d + size
    double thresh_dsq = SQR(min_d + 2*s);
    xdbg<<"closest center = "<<closest_i<<"  "<<centers[closest_i]<<" d = "<<min_d<<std::endl;
    xdbg<<"thresh_dsq = "<<thresh_dsq<<std::endl;

    // Update patches to remove any that cannot be the right center from candidate section.
    for (int j=ncand-1; j>0; --j) {
        double dsq = saved_dsq[j];
        xdbg<<"Check: "<<dsq<<" >? "<<thresh_dsq<<std::endl;
        if (dsq > thresh_dsq) {
            xdbg<<"Remove cell with center "<<centers[patches[j]]<<std::endl;
            --ncand;
            if (j != ncand) {
                std::swap(patches[j], patches[ncand]);
                // Don't bother with saved_dsq, since don't need that anymore.
            }
        }
    }
    xdbg<<"There are "<<ncand<<" patches remaining\n";

    //set_verbose(1);
    if (ncand == 1) {
        // If we only have one candidate left, we're done.  Use this cell to update this patch.
        cells_by_patch[closest_i].push_back(cell);
    } else {
        // Otherwise, need to recurse to sub-cells.
        FindCellsInPatches(centers, cell->getLeft(), patches, ncand, cells_by_patch, saved_dsq);
        // Note: the above call might have swapped some patches around, but only within the
        // first ncand values, so they are still valid for the next call.
        FindCellsInPatches(centers, cell->getRight(), patches, ncand, cells_by_patch, saved_dsq);
    }
}

template <int D, int C>
void FindCellsInPatches(const std::vector<Position<C> >& centers,
                   const std::vector<Cell<D,C>*>& cells,
                   std::vector<cell_storage_type>& cells_by_patch)
{
#ifdef _OPENMP
#pragma omp parallel
    {
        // Give each thread their own copy of cells_by_patch to fill in.
        std::vector<cell_storage_type> cbp2 = cells_by_patch;
#else
        std::vector<cell_storage_type>& cbp2 = cells_by_patch;
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
            FindCellsInPatches(centers, cells[k], patches, npatch, cbp2, saved_dsq);
        }

#ifdef _OPENMP
        // Combine the results
#pragma omp critical
        {
            for (int i=0; i<npatch; ++i)
                cells_by_patch[i].insert(cells_by_patch[i].end(), cbp2[i].begin(), cbp2[i].end());
        }
    }
#endif

}

template <int D, int C>
void UpdateCenters(std::vector<Position<C> >& new_centers,
                   std::vector<cell_storage_type>& cells_by_patch)
{
    typedef typename cell_storage_type::iterator iter_type;
    const int npatch = new_centers.size();
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int i=0; i<npatch; ++i) {
        dbg<<"Patch "<<i<<" includes "<<cells_by_patch[i].size()<<" cells\n";
        Position<C> cen;
        double w = 0.;
        for (iter_type it=cells_by_patch[i].begin(); it!=cells_by_patch[i].end(); ++it) {
            cen += (*it)->getPos() * (*it)->getW();
            w += (*it)->getW();
        }
        cen /= w;
        cen.normalize();
        dbg<<"New center = "<<cen<<std::endl;
        new_centers[i] = cen;
    }
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

template <int D, int C>
void FillPatches(const Cell<D,C>* cell, long patch_num, long* patches, long n)
{
    xdbg<<"Start FillPatches "<<cell<<" "<<patch_num<<std::endl;
    if (cell->getLeft()) {
        xdbg<<"Not a leaf.  Recurse\n";
        FillPatches(cell->getLeft(), patch_num, patches, n);
        FillPatches(cell->getRight(), patch_num, patches, n);
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

template <int D, int C>
void RunKMeans2(Field<D,C>*field, int npatch, int max_iter, double tol, long* patches, long n)
{
    dbg<<"Start runKMeans for "<<npatch<<" patches\n";
    const std::vector<Cell<D,C>*> cells = field->getCells();

    // Initialize the centers of the patches smartly according to the tree structure.
    std::vector<Position<C> > centers(npatch);
    InitializeCenters(centers, cells);
    dbg<<"After InitializeCenters\n";

    // We compare the total shift^2 to this number
    // Want rms shift / field.size < tol
    // So (total_shift / npatch) / field.size < tol
    // total_shift^2 < tol^2 * size^2 * npatch^2
    double tolsq = tol*tol * field->getSizeSq() * npatch*npatch;
    dbg<<"tolsq = "<<tolsq<<std::endl;

    // Keep track of which cells belong to which patch.
    std::vector<cell_storage_type> cells_by_patch(npatch);

    for(int iter=0; iter<max_iter; ++iter) {
        // Figure out which cells belong to which patch according to the current centers.
        // Note: clear leaves the previous capacity available, so usually won't need much
        // in the way of allocation here.
        for (int i=0;i<npatch;++i) cells_by_patch[i].clear();
        FindCellsInPatches(centers, cells, cells_by_patch);
        dbg<<"Found cells in patches\n";

        // Calculate the new center positions
        std::vector<Position<C> > new_centers(npatch, Position<C>());
        UpdateCenters(new_centers, cells_by_patch);
        dbg<<"After UpdateCenters\n";

        // Check for convergence
        double shiftsq = CalculateShiftSq(centers, new_centers);
        dbg<<"Iter "<<iter<<": shiftsq = "<<shiftsq<<std::endl;
        // Stop if (rms shift / size) < tol
        if (shiftsq < tol) {
            dbg<<"Stopping RunKMeans because rms shift = "<<sqrt(shiftsq)<<std::endl;
            dbg<<"tol * size = "<<tol * sqrt(field->getSizeSq())<<std::endl;
            break;
        }
        if (iter == max_iter-1) {
            dbg<<"Stopping RunKMeans because hit maximum iterations = "<<max_iter<<std::endl;
        }
        centers = new_centers;
    }

    // Write the patch numbers to the output array.
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
     for (int i=0; i<npatch; ++i) {
        dbg<<"Fill patches for i = "<<i<<std::endl;
        dbg<<"Patch includes "<<cells_by_patch[i].size()<<" cells\n";
        typedef typename cell_storage_type::iterator iter_type;
        for (iter_type it=cells_by_patch[i].begin(); it!=cells_by_patch[i].end(); ++it) {
            FillPatches(*it, i, patches, n);
        }
    }
    dbg<<"After FillPatches\n";
}


template <int D>
void RunKMeans1(void* field, int npatch, int max_iter, double tol,
                int coords, long* patches, long n)
{
    switch(coords) {
      case Flat:
           RunKMeans2(static_cast<Field<D,Flat>*>(field), npatch, max_iter, tol, patches, n);
           break;
      case Sphere:
           RunKMeans2(static_cast<Field<D,Sphere>*>(field), npatch, max_iter, tol, patches, n);
           break;
      case ThreeD:
           RunKMeans2(static_cast<Field<D,ThreeD>*>(field), npatch, max_iter, tol, patches, n);
           break;
    }
}

void RunKMeans(void* field, int npatch, int max_iter, double tol,
               int d, int coords, long* patches, long n)
{
    switch(d) {
      case NData:
           RunKMeans1<NData>(field, npatch, max_iter, tol, coords, patches, n);
           break;
      case KData:
           RunKMeans1<KData>(field, npatch, max_iter, tol, coords, patches, n);
           break;
      case GData:
           RunKMeans1<GData>(field, npatch, max_iter, tol, coords, patches, n);
           break;
    }
}


