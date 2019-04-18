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

template <int D, int C>
void UpdateCenters(const std::vector<Position<C> >& centers,
                   std::vector<Position<C> >& new_centers,
                   std::vector<double>& new_weights,
                   const Cell<D,C>* cell, std::set<long> candidate_patches,
                   std::vector<std::set<const Cell<D,C>*> >& cells_by_patch)
{
    //set_verbose(2);
    xdbg<<"Start recursive UpdateCenters\n";
    // First find the center that is closest to the current cell's center
    const Position<C> cell_center = cell->getPos();
    double s = cell->getSize();
    xdbg<<"cell = "<<cell_center<<"  "<<s<<"  "<<cell->getN()<<std::endl;
    double min_dsq = 1.e300;
    double closest_i = -1;
    std::vector<double> saved_dsq;
    saved_dsq.reserve(candidate_patches.size());
    typedef std::set<long>::iterator set_it;
    for (set_it i=candidate_patches.begin(); i!=candidate_patches.end(); ++i) {
        double dsq = (cell_center - centers[*i]).normSq();
        saved_dsq.push_back(dsq);
        if (closest_i == -1) {
            min_dsq = dsq;
            closest_i = *i;
        } else if (dsq < min_dsq) {
            min_dsq = dsq;
            closest_i = *i;
        }
    }
    xdbg<<"closest center = "<<closest_i<<"  "<<centers[closest_i]<<" d = "<<sqrt(min_dsq)<<std::endl;
    // Can remove any candidate with d - size >  min_d + size
    double thresh_dsq = SQR(sqrt(min_dsq) + 2*s);
    xdbg<<"thresh_dsq = "<<thresh_dsq<<std::endl;

    // Update candidate_patches to remove any that cannot be the right center
    // for anything in the current cell.
    std::vector<double>::const_iterator dsq_it=saved_dsq.begin();
    typedef std::set<long>::iterator set_it;
    for (set_it i=candidate_patches.begin(); i!=candidate_patches.end(); ) {
        double dsq = *dsq_it++;
        xdbg<<"Check: "<<dsq<<" >? "<<thresh_dsq<<std::endl;
        // Note: normally if *i==closest_i, then dsq <= thresh_dsq, but with rounding errors
        // if s==0, sometimes this isn't true, so just make sure we don't remove closest_i.
        if ((dsq > thresh_dsq) && (*i != closest_i)) {
            if (closest_i == 21) {
                dbg<<"distance to 21 center = "<<sqrt(min_dsq)<<std::endl;
                dbg<<"distance to "<<*i<<" center = "<<sqrt(dsq)<<std::endl;
                dbg<<"check "<<sqrt(dsq) - s<<" >? "<<sqrt(min_dsq) + s<<std::endl;
                Assert(std::abs(sqrt(dsq) - (cell_center - centers[*i]).norm()) < 1.e-10);
                Assert(sqrt(dsq) - s >= sqrt(min_dsq) + s + 1.e-10);
            }
            xdbg<<"Remove cell with center "<<centers[*i]<<std::endl;
            candidate_patches.erase(i++);
        }
        else ++i;
    }
    xdbg<<"There are "<<candidate_patches.size()<<" patches remaining\n";

    //set_verbose(1);
    if (candidate_patches.size() == 1) {
        // If we only have one candidate left, we're done.  Use this cell to update this patch.
        cells_by_patch[closest_i].insert(cell);
        dbg<<"cell = "<<cell_center<<"  "<<s<<"  "<<cell->getN()<<"  "<<cell->getW()<<std::endl;
        dbg<<"cell_center * w = "<<cell_center * cell->getW()<<std::endl;
        dbg<<"new_centers["<<closest_i<<"] = "<<new_centers[closest_i]<<std::endl;
        new_centers[closest_i] += cell_center * cell->getW();
        dbg<<"new_centers["<<closest_i<<"] => "<<new_centers[closest_i]<<std::endl;
        new_weights[closest_i] += cell->getW();
        xdbg<<"new center so far = "<<new_centers[closest_i] / new_weights[closest_i]<<std::endl;
        if (closest_i == 21) {
            dbg<<"cell = "<<cell_center<<"  "<<s<<"  "<<cell->getN()<<"  "<<cell->getW()<<std::endl;
            dbg<<"new center so far = "<<new_centers[closest_i]<<" / "<<new_weights[closest_i]<<std::endl;
            dbg<<"                  = "<<new_centers[closest_i] / new_weights[closest_i]<<std::endl;
        }
    } else {
        // Otherwise, need to recurse to sub-cells.
        UpdateCenters(centers, new_centers, new_weights, cell->getLeft(), candidate_patches,
                      cells_by_patch);
        UpdateCenters(centers, new_centers, new_weights, cell->getRight(), candidate_patches,
                      cells_by_patch);
    }
}

template <int D, int C>
void UpdateCenters(const std::vector<Position<C> >& centers,
                   std::vector<Position<C> >& new_centers,
                   const std::vector<Cell<D,C>*>& cells,
                   std::vector<std::set<const Cell<D,C>*> >& cells_by_patch)
{
    // We start with all patches as candidates.
    std::set<long> candidate_patches;
    for (size_t i=0; i<centers.size(); ++i) candidate_patches.insert(i);

    // During the recursion, we just add up Sum (pos * w)
    // At the end, we'll divide by Sum (w)
    std::vector<double> new_weights(new_centers.size(), 0.);

    // Start a recursion for each top-level cell.
    for (size_t k=0; k<cells.size(); ++k) {
        // Note candidate_patches is intentionally passed by value, not reference.
        // It will be modified by the recursive version of this function, so passing by
        // value means each call gets the original version.
        UpdateCenters(centers, new_centers, new_weights, cells[k], candidate_patches,
                      cells_by_patch);
    }

    // Now divide by Sum(w)
    for (size_t i=0; i<new_centers.size(); ++i) {
        dbg<<"Center "<<i<<" = "<<centers[i]<<std::endl;
        dbg<<"New center = "<<new_centers[i]<<" / "<<new_weights[i]<<std::endl;
        new_centers[i] /= new_weights[i];
        new_centers[i].normalize();
        dbg<<"New center = "<<new_centers[i]<<std::endl;
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

    std::vector<Position<C> > centers(npatch);
    InitializeCenters(centers, cells);
    dbg<<"After InitializeCenters\n";

    // We compare the total shift^2 to this number
    // Want rms shift / field.size < tol
    // So (total_shift / npatch) / field.size < tol
    // total_shift^2 < tol^2 * size^2 * npatch^2
    double tolsq = tol*tol * field->getSizeSq() * npatch*npatch;
    dbg<<"tolsq = "<<tolsq<<std::endl;

    std::vector<std::set<const Cell<D,C>*> > cells_by_patch(npatch);

    for(int iter=0; iter<max_iter; ++iter) {
        std::vector<Position<C> > new_centers(npatch, Position<C>());
        for (int i=0;i<npatch;++i) cells_by_patch[i].clear();
        UpdateCenters(centers, new_centers, cells, cells_by_patch);
        dbg<<"After UpdateCenters\n";

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

    typedef typename std::set<const Cell<D,C>*>::iterator set_it;
    for (int i=0; i<npatch; ++i) {
        dbg<<"Fill patches for i = "<<i<<std::endl;
        dbg<<"Patch includes "<<cells_by_patch[i].size()<<" cells\n";
        for (set_it it=cells_by_patch[i].begin(); it!=cells_by_patch[i].end(); ++it) {
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


