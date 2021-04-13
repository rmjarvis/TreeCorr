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

#include <vector>
#include <cmath>
#include "Field.h"
#include "Cell.h"
#include "dbg.h"

extern "C" {
#include "Field_C.h"
}

// In BinnedCorr2.cpp
extern void SelectRandomFrom(long m, std::vector<long>& selection);

template <int D, int C>
void InitializeCentersTree(std::vector<Position<C> >& centers, const Cell<D,C>* cell,
                           long first, int ncenters)
{
    xdbg<<"Recursive InitializeCentersTree: "<<first<<"  "<<ncenters<<std::endl;
    xdbg<<"pos = "<<cell->getPos()<<std::endl;
    if (ncenters == 1) {
        xdbg<<"Only 1 center.  Use this cell.\n";
        Assert(first < long(centers.size()));
        xdbg<<"initial center["<<first<<"] = "<<cell->getPos()<<std::endl;
        centers[first] = cell->getPos();
    } else if (cell->getLeft()) {
        int m1 = ncenters / 2;
        int m2 = ncenters - m1;
        double urd = urand();
        if (urd < 0.5) std::swap(m1,m2);
        // Note: If ncenters is even, this doesn't do anything, but if odd, then 50% chance
        // that left side will get the extra one and 50% chance that right side will get it.
        xdbg<<"Recurse with m1,m2 = "<<m1<<','<<m2<<std::endl;
        InitializeCentersTree(centers, cell->getLeft(), first, m1);
        InitializeCentersTree(centers, cell->getRight(), first + m1, m2);
    } else {
        dbg<<"BAD!  Shouldn't happen.  Probably shouldn't be running kmeans on this field.\n";
        // Shouldn't ever happen -- this means a leaf is very close to the top.
        // But still, let's do soemthing at least vaguely sensible if it does.
        for (int i=0; i<ncenters; ++i) {
            Assert(first+i < long(centers.size()));
            centers[first+i] = Position<C>(cell->getPos()*(1. + urand() * 1.e-8));
        }
    }
    xdbg<<"End of Recursive InitializeCentersTree\n";
}

template <int D, int C>
void InitializeCentersTree(std::vector<Position<C> >& centers, const std::vector<Cell<D,C>*>& cells,
                           long seed)
{
    dbg<<"Initialize centers: "<<centers.size()<<"  "<<cells.size()<<std::endl;
    long ncenters = centers.size();
    long ncells = cells.size();
    urand(seed);  // Make sure rand is seeded properly.
    if (ncells > ncenters) {
        dbg<<"More cells than centers.  Pick from them randomly.\n";
        std::vector<long> selection(ncenters);
        SelectRandomFrom(ncells, selection);
        for (long i=0; i<ncenters; ++i) {
            Assert(selection[i] < long(cells.size()));
            centers[i] = cells[selection[i]]->getPos();
        }
    } else {
        dbg<<"Fewer cells than centers.  Recurse to get more than one from each.\n";
        long n1 = ncenters / ncells;
        long k2 = ncenters % ncells;
        long n2 = n1 + 1;
        long k1 = ncells - k2;
        xdbg<<"n1 = "<<n1<<" n2 = "<<n2<<std::endl;
        xdbg<<"k1 = "<<k1<<" k2 = "<<k2<<std::endl;
        Assert(n1 >= 1);
        Assert(n1 * k1 + n2 * k2 == ncenters);

        // Distribute n1 and n2 in list k1 and k2 times respectively.
        std::vector<long> nvalues(ncells);
        for (long k=0; k<k1; ++k) nvalues[k] = n1;
        for (long k=k1; k<ncells; ++k) nvalues[k] = n2;
        // Shuffle the locations of n1, n2
        // (Fisher-Yates algorithm)
        for (long k=ncells; k>1; --k) {
            long j = int(urand() * k);
            if (j != k-1) std::swap(nvalues[j], nvalues[k-1]);
        }

        // Do the recursive initialization for each value of n1 in the list.
        long first=0;
        for (long k=0; k<ncells; ++k) {
            Assert(first < ncenters);
            InitializeCentersTree(centers, cells[k], first, nvalues[k]);
            first += nvalues[k];
        }
        Assert(first == ncenters);
    }
    xdbg<<"Done initializing centers\n";
}

template <int D, int C>
void InitializeCentersRand(std::vector<Position<C> >& centers, const std::vector<Cell<D,C>*>& cells,
                           long seed)
{
    dbg<<"Initialize centers (random): "<<centers.size()<<"  "<<cells.size()<<std::endl;
    // Pick npatch random numbers from the total number of objecst.
    long ncenters = centers.size();
    long ncells = cells.size();
    long ntot = 0;
    for (long k=0; k<ncells; ++k) ntot += cells[k]->getN();
    xdbg<<"ntot = "<<ntot<<std::endl;
    urand(seed);  // Make sure rand is seeded properly.

    // Find ncenters different indices from 0..ntot-1.
    std::vector<long> index(ncenters);
    SelectRandomFrom(ntot, index);

    // Select the position of the leaf with this index.
    // Note: these aren't the same as the "index" that is stored from the original catalog.
    // These are the index in a kind of arbitrary ordering of the final tree structure.
    for (long i=0; i<ncenters; ++i) {
        dbg<<"Pick random center "<<i<<" index = "<<index[i]<<std::endl;
        long m = index[i];
        for (long k=0; k<ncells; ++k) {
            if (m < cells[k]->getN()) {
                const Cell<D,C>* c = cells[k]->getLeafNumber(m);
                centers[i] = c->getPos();
                xdbg<<"  k = "<<k<<", m = "<<m<<" cen = "<<centers[i]<<std::endl;
                break;
            } else {
                m -= cells[k]->getN();
                xdbg<<"  N["<<k<<"] = "<<cells[k]->getN()<<"  m -> "<<m<<std::endl;
            }
        }

        // It is technically possible that some centers will repeat if a leaf has multiple
        // points in it.  Then two different index values could both point to the same leaf.
        // This should be rare, but guard against it by arbitrarily shifting the second value
        // by a random small value.
        for (long i1=0; i1<i; ++i1) {
            if (centers[i1] == centers[i]) {
                dbg<<"Found duplicate center!\n";
                centers[i] *= (1. + urand() * 1.e-8);
            }
        }
    }
}

template <int D, int C>
Position<C> InitializeCentersKMPP(const Cell<D,C>* cell,
                                  const std::vector<Position<C> >& centers, long ncen)
{
    struct LeafAlreadyUsed {};

    xdbg<<"Start recursive InitializeCentersKMPP\n";

    // If we are in a leaf, then return this position as the next center to use.
    if (cell->getSize() == 0) {
        // Check that we haven't already used this leaf as a center
        for (long j=0; j<ncen; ++j) {
            if (cell->getPos() == centers[j]) {
                dbg<<"Already used this leaf as a center\n";
                throw LeafAlreadyUsed();
            }
        }
        return cell->getPos();
    }

    // If not, then we choose which subcell to recurse to based on the relative dsq values
    // to each daughter's closest center.

    const Cell<D,C>* left = cell->getLeft();
    const Cell<D,C>* right = cell->getRight();
    double dsq1 = (left->getPos() - centers[0]).normSq();
    double dsq2 = (right->getPos() - centers[0]).normSq();

    for (long j=1; j<ncen; ++j) {
        double dsq1j = (left->getPos() - centers[j]).normSq();
        double dsq2j = (right->getPos() - centers[j]).normSq();
        if (dsq1j < dsq1) dsq1 = dsq1j;
        if (dsq2j < dsq2) dsq2 = dsq2j;
    }
    xdbg<<"dsq1 = "<<dsq1<<", dsq2 = "<<dsq2<<std::endl;
    if (dsq1 + dsq2 == 0.) {
        dbg<<"All used.  Throw exception and try other branch.\n";
        xdbg<<"N1 = "<<left->getN()<<"  N2 = "<<right->getN()<<std::endl;
        throw LeafAlreadyUsed();
    }

    double u = urand() * (dsq1 + dsq2);
    if (u < dsq1) {
        xdbg<<"u = "<<u<<", recurse left\n";
        try {
            return InitializeCentersKMPP(left, centers, ncen);
        } catch (LeafAlreadyUsed) {
            dbg<<"Caught LeafAlreadyUsed.\n";
            return InitializeCentersKMPP(right, centers, ncen);
        }
    } else {
        xdbg<<"u = "<<u<<", recurse right\n";
        try {
            return InitializeCentersKMPP(right, centers, ncen);
        } catch (LeafAlreadyUsed) {
            dbg<<"Caught LeafAlreadyUsed.\n";
            return InitializeCentersKMPP(left, centers, ncen);
        }
    }
}

template <int D, int C>
void InitializeCentersKMPP(std::vector<Position<C> >& centers, const std::vector<Cell<D,C>*>& cells,
                           long seed)
{
    // cf. https://en.wikipedia.org/wiki/K-means%2B%2B
    // The basic KMeans++ algorithm is as follows:
    // 1. Pick the first center at random.
    // 2. For subsequent centers, calculate all d_i = the distance to the nearest center
    //    picked so far.
    // 3. Select the next center randomly with probability proportional to d_i^2.
    // 4. Repeat 2,3 until all centers are selected.
    //
    // We modify this algorithm slightly to make it reasonably efficient.  Rather than actually
    // calculating all the distances each time, we traverse the tree and pick the direction to
    // go according to the distances to the centers of the cell at each point.  It's not quite the
    // same thing, but it's pretty close.

    dbg<<"Initialize centers (kmeans++): "<<centers.size()<<"  "<<cells.size()<<std::endl;
    long ncenters = centers.size();
    long ncells = cells.size();
    urand(seed);  // Make sure rand is seeded properly.

    long ntot = 0;
    for (long k=0; k<ncells; ++k) ntot += cells[k]->getN();
    xdbg<<"ntot = "<<ntot<<std::endl;

    // Keep track of how many centers are assigned from each cell.
    std::vector<long> centers_per_cell(ncells, 0);

    // Pick the first point
    long m = long(urand() * ntot);
    xdbg<<"m = "<<m<<std::endl;
    for (long k=0; k<ncells; ++k) {
        if (m < cells[k]->getN()) {
            dbg<<"Pick first center from cell "<<k<<std::endl;
            centers[0] = cells[k]->getLeafNumber(m)->getPos();
            xdbg<<"center[0] = "<<centers[0]<<std::endl;
            centers_per_cell[k] += 1;
            break;
        } else {
            m -= cells[k]->getN();
        }
    }

    // Pick the rest of the points
    for (long i=1; i<ncenters; ++i) {
        xdbg<<"Start work on center "<<i<<std::endl;
        // Calculate the dsq for each top level cell, to calculate the probability of choosing it.
        std::vector<double> p(ncells);
        double sump=0.;  // to normalize probabilities
        for (long k=0; k<ncells; ++k) {
            double dsq1 = (centers[0] - cells[k]->getPos()).normSq();
            xdbg<<"   dsq[0] = "<<dsq1<<std::endl;
            for (long i1=1; i1<i; ++i1) {
                double dsq2 = (centers[i1] - cells[k]->getPos()).normSq();
                xdbg<<"   dsq["<<i1<<"] = "<<dsq2<<std::endl;
                if (dsq2 < dsq1) dsq1 = dsq2;
            }
            xdbg<<"Cell "<<k<<" has dsq = "<<dsq1<<std::endl;
            // The probability of picking each point is proportional to its dsq.
            // Approximate that all points in a cell are closest to the same center, and that
            // the points are uniformly distibuted in either a circle or a sphere (according to C)
            // Then the sum of all the dsq of a cell is
            // dsq = N (d^2 + 1/2 s^2)   for a circle
            //     = N (d^2 + 3/5 s^2)   for a sphere
            // If N is smallish, then we want to make sure we subtract off the number of centers
            // that are already in the cell.
            // TODO: This could be improved for the case where there is more than 1 center already
            //       in the cell.  The current formula is a significant over-estimate in that case.
            //       Not sure how important it is though, since I think the Tree initialization is
            //       normally better, so this is not a very important use case.
            const double coef = C == ThreeD ? 0.6 : 0.5;
            p[k] = (cells[k]->getN() - centers_per_cell[k]) * (dsq1 + coef * cells[k]->getSizeSq());
            sump += p[k];
        }
        double u = urand();
        xdbg<<"u = "<<u<<std::endl;
        for (long k=0; k<ncells; ++k) {
            p[k] /= sump;
            xdbg<<"Prob("<<k<<") = "<<p[k]<<std::endl;
            if (u < p[k]) {
                dbg<<"Choose next center from cell "<<k<<std::endl;
                xdbg<<"N = "<<cells[k]->getN()<<" ncen so far = "<<centers_per_cell[k]<<std::endl;
                centers[i] = InitializeCentersKMPP(cells[k], centers, ncenters);
                xdbg<<"center["<<i<<"] = "<<centers[i]<<std::endl;
                centers_per_cell[k] += 1;
                break;
            } else {
                u -= p[k];
                xdbg<<"u -> "<<u<<std::endl;
                Assert(k != ncells-1);
            }
        }
    }
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
            if (w[i] > 0.) {
                new_centers[i] /= w[i];
                new_centers[i].normalize();
            }
            xdbg<<"New center = "<<new_centers[i]<<std::endl;
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
        // it is close to I = w s^2, and for large cells (assuming uniform density of points)
        // it is close to I = 0.5 w s^2.  Splitting the difference works fairly well and is much
        // faster than the full calculateInertia() method.
        if (ssq > 0.) {
            double I1 = 0.75 * ssq * w;
#ifdef DEBUGLOGGING
            double I2 = cell->calculateInertia();
            xdbg<<"ssq, I1, I2 = "<<ssq<<"  "<<I1<<"  "<<I2<<std::endl;
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
        // The 3 here is empirical, but fairly good results were obtained with this factor
        // being anywhere from 2 to 4.  4 was actually slgihtly better, but above 4, the
        // iterations could turn unstable and neighboring patches could alternate grabbing
        // more and more area back and forth until one patch had no points.
        // So to avoid getting near this instability, we use 3 rather than 4, which we believe
        // is usually safe.
        double meanw = sumw / npatch / 3.;

        for (int i=0; i<npatch; ++i) {
#ifdef DEBUGLOGGING
            xdbg<<"Inertia["<<i<<"] = "<<inertia[i]<<std::endl;
            mean += inertia[i];
            rms += SQR(inertia[i]);
#endif
            inertia[i] /= meanw;
        }
#ifdef DEBUGLOGGING
        mean /= npatch;
        rms -= npatch * SQR(mean);
        rms = sqrt(rms / npatch);
        xdbg<<"mean inertia = "<<mean<<std::endl;
        xdbg<<"rms inertia = "<<rms<<std::endl;
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
#ifdef DEBUGLOGGING
    const std::vector<Position<C> >& centers;
    std::vector<double> inertia;
#endif

    AssignPatches(long* _patches, long _n, const std::vector<Position<C> >& _c) :
        patches(_patches), n(_n)
#ifdef DEBUGLOGGING
        , centers(_c), inertia(_c.size())
#endif
    {}

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
#ifdef DEBUGLOGGING
            inertia[patch_num] += (cell->getPos() - centers[patch_num]).normSq() * cell->getW();
#endif
        } else {
            std::vector<long>* indices = cell->getListInfo().indices;
            xdbg<<"Leaf with N>1.  "<<indices->size()<<" indices\n";
            for (size_t j=0; j<indices->size(); ++j) {
                long index = (*indices)[j];
                xdbg<<"    index = "<<index<<std::endl;
                Assert(index < n);
                patches[index] = patch_num;
            }
#ifdef DEBUGLOGGING
            inertia[patch_num] += (cell->getPos() - centers[patch_num]).normSq() * cell->getW();
#endif
        }
    }

    void combineWith(const AssignPatches<D,C>& rhs)
    {
#ifdef DEBUGLOGGING
        int npatch = centers.size();
        for (int i=0; i<npatch; ++i) {
            inertia[i] += rhs.inertia[i];
        }
#endif
    }
    void finalize()
    {
#ifdef DEBUGLOGGING
        double mean=0.;
        double rms=0.;
        int npatch = centers.size();
        dbg<<"Finalize AssignPatches\n";
        for (int i=0; i<npatch; ++i) {
            xdbg<<"Inertia["<<i<<"] = "<<inertia[i]<<std::endl;
            mean += inertia[i];
            rms += SQR(inertia[i]);
        }
        mean /= npatch;
        rms -= npatch * SQR(mean);
        rms = sqrt(rms / npatch);
        dbg<<"mean inertia = "<<mean<<std::endl;
        dbg<<"rms inertia = "<<rms<<std::endl;
#endif
    }
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
    for (long j=1; j<ncand; ++j) {
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
    for (long j=ncand-1; j>0; --j) {
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
        int npatch = centers.size();
        std::vector<long> patches(npatch);
        for (int i=0; i<npatch; ++i) patches[i] = i;

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
    xdbg<<"Start CalculateShiftSq\n";
    double shiftsq=0.;
    for (size_t i=0; i<centers.size(); ++i) {
        double shiftsq_i = (centers[i] - new_centers[i]).normSq();
        xdbg<<"Shift for "<<i<<" = "<<centers[i]<<" -> "<<new_centers[i]<<"  d = "<<sqrt(shiftsq_i)<<std::endl;
        shiftsq += shiftsq_i;
    }
    xdbg<<"Total shiftsq = "<<shiftsq<<std::endl;
    return shiftsq;
}



// C = Threed or Sphere
template <int C>
void WriteCenters(const std::vector<Position<C> >& centers, double* pycenters, int npatch)
{
    for(int i=0; i<npatch; ++i, pycenters+=3) {
        pycenters[0] = centers[i].getX();
        pycenters[1] = centers[i].getY();
        pycenters[2] = centers[i].getZ();
    }
}
template <int C>
void ReadCenters(std::vector<Position<C> >& centers, const double* pycenters, int npatch)
{
    for(int i=0; i<npatch; ++i, pycenters+=3) {
        centers[i] = Position<C>(pycenters[0], pycenters[1], pycenters[2]);
    }
}

// C = Flat
template <>
void WriteCenters(const std::vector<Position<Flat> >& centers, double* pycenters, int npatch)
{
    for(int i=0; i<npatch; ++i, pycenters+=2) {
        pycenters[0] = centers[i].getX();
        pycenters[1] = centers[i].getY();
    }
}

template <>
void ReadCenters(std::vector<Position<Flat> >& centers, const double* pycenters, int npatch)
{
    for(int i=0; i<npatch; ++i, pycenters+=2) {
        centers[i] = Position<Flat>(pycenters[0], pycenters[1]);
    }
}

template <int D, int C>
void KMeansInitTree2(Field<D,C>*field, double* pycenters, int npatch, long seed)
{
    dbg<<"Start KMeansInitTree for "<<npatch<<" patches\n";
    const std::vector<Cell<D,C>*> cells = field->getCells();
    std::vector<Position<C> > centers(npatch);
    InitializeCentersTree(centers, cells, seed);
    WriteCenters(centers, pycenters, npatch);
}

template <int D, int C>
void KMeansInitRand2(Field<D,C>*field, double* pycenters, int npatch, long seed)
{
    dbg<<"Start KMeansInitRand for "<<npatch<<" patches\n";
    const std::vector<Cell<D,C>*> cells = field->getCells();
    std::vector<Position<C> > centers(npatch);
    InitializeCentersRand(centers, cells, seed);
    WriteCenters(centers, pycenters, npatch);
}

template <int D, int C>
void KMeansInitKMPP2(Field<D,C>*field, double* pycenters, int npatch, long seed)
{
    dbg<<"Start KMeansInitKMPP for "<<npatch<<" patches\n";
    const std::vector<Cell<D,C>*> cells = field->getCells();
    std::vector<Position<C> > centers(npatch);
    InitializeCentersKMPP(centers, cells, seed);
    WriteCenters(centers, pycenters, npatch);
}

template <int D, int C>
void KMeansRun2(Field<D,C>*field, double* pycenters, int npatch, int max_iter, double tol,
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
    xdbg<<"tolsq = "<<tolsq<<std::endl;

    // The alt version needs to keep track of the inertia of each patch.
    CalculateInertia<D,C> calculate_inertia(alt ? npatch : 0, centers);
    std::vector<double>* pinertia = 0;
    xdbg<<"Made calculate_inertia\n";

    // Keep track of which cells belong to which patch.
    UpdateCenters<D,C> update_centers(npatch);
    xdbg<<"Made update_centers\n";

    for(int iter=0; iter<max_iter; ++iter) {
        xdbg<<"Start iter "<<iter<<std::endl;
        // Update the inertia if we are doing the alternate version
        if (alt) {
            calculate_inertia.reset();
            FindCellsInPatches(centers, cells, calculate_inertia);
            calculate_inertia.finalize();
            pinertia = &calculate_inertia.inertia;
        }

        // Figure out which cells belong to which patch according to the current centers.
        // Note: clear leaves the previous capacity available, so usually won't need much
        // in the way of allocation here.
        update_centers.reset();
        FindCellsInPatches(centers, cells, update_centers, pinertia);
        update_centers.finalize();
        xdbg<<"After UpdateCenters\n";

        // Check for convergence
        double shiftsq = CalculateShiftSq(centers, update_centers.new_centers);
        centers = update_centers.new_centers;
        xdbg<<"Iter "<<iter<<": shiftsq = "<<shiftsq<<"  tolsq = "<<tolsq<<std::endl;
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
void KMeansAssign2(Field<D,C>*field, double* pycenters, int npatch, long* patches, long n)
{
    dbg<<"Start KMeansAssign for "<<npatch<<" patches\n";
    const std::vector<Cell<D,C>*> cells = field->getCells();

    std::vector<Position<C> > centers(npatch);
    ReadCenters(centers, pycenters, npatch);

    xdbg<<"Start AssignPatches\n";
    AssignPatches<D,C> assign_patches(patches, n, centers);
    FindCellsInPatches(centers, cells, assign_patches);
    assign_patches.finalize();
    xdbg<<"After AssignPatches\n";
}

template <int D>
void KMeansInitTree1(void* field, double* centers, int npatch, int coords, long seed)
{
    switch(coords) {
      case Flat:
           KMeansInitTree2(static_cast<Field<D,Flat>*>(field), centers, npatch, seed);
           break;
      case Sphere:
           KMeansInitTree2(static_cast<Field<D,Sphere>*>(field), centers, npatch, seed);
           break;
      case ThreeD:
           KMeansInitTree2(static_cast<Field<D,ThreeD>*>(field), centers, npatch, seed);
           break;
    }
}

void KMeansInitTree(void* field, double* centers, int npatch, int d, int coords, long seed)
{
    switch(d) {
      case NData:
           KMeansInitTree1<NData>(field, centers, npatch, coords, seed);
           break;
      case KData:
           KMeansInitTree1<KData>(field, centers, npatch, coords, seed);
           break;
      case GData:
           KMeansInitTree1<GData>(field, centers, npatch, coords, seed);
           break;
    }
}

template <int D>
void KMeansInitRand1(void* field, double* centers, int npatch, int coords, long seed)
{
    switch(coords) {
      case Flat:
           KMeansInitRand2(static_cast<Field<D,Flat>*>(field), centers, npatch, seed);
           break;
      case Sphere:
           KMeansInitRand2(static_cast<Field<D,Sphere>*>(field), centers, npatch, seed);
           break;
      case ThreeD:
           KMeansInitRand2(static_cast<Field<D,ThreeD>*>(field), centers, npatch, seed);
           break;
    }
}

void KMeansInitRand(void* field, double* centers, int npatch, int d, int coords, long seed)
{
    switch(d) {
      case NData:
           KMeansInitRand1<NData>(field, centers, npatch, coords, seed);
           break;
      case KData:
           KMeansInitRand1<KData>(field, centers, npatch, coords, seed);
           break;
      case GData:
           KMeansInitRand1<GData>(field, centers, npatch, coords, seed);
           break;
    }
}

template <int D>
void KMeansInitKMPP1(void* field, double* centers, int npatch, int coords, long seed)
{
    switch(coords) {
      case Flat:
           KMeansInitKMPP2(static_cast<Field<D,Flat>*>(field), centers, npatch, seed);
           break;
      case Sphere:
           KMeansInitKMPP2(static_cast<Field<D,Sphere>*>(field), centers, npatch, seed);
           break;
      case ThreeD:
           KMeansInitKMPP2(static_cast<Field<D,ThreeD>*>(field), centers, npatch, seed);
           break;
    }
}

void KMeansInitKMPP(void* field, double* centers, int npatch, int d, int coords, long seed)
{
    switch(d) {
      case NData:
           KMeansInitKMPP1<NData>(field, centers, npatch, coords, seed);
           break;
      case KData:
           KMeansInitKMPP1<KData>(field, centers, npatch, coords, seed);
           break;
      case GData:
           KMeansInitKMPP1<GData>(field, centers, npatch, coords, seed);
           break;
    }
}

template <int D>
void KMeansRun1(void* field, double* centers, int npatch, int max_iter, double tol, bool alt,
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

void KMeansRun(void* field, double* centers, int npatch, int max_iter, double tol, int alt,
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
void KMeansAssign1(void* field, double* centers, int npatch, long* patches, long n, int coords)
{
    switch(coords) {
      case Flat:
           KMeansAssign2(static_cast<Field<D,Flat>*>(field), centers, npatch, patches, n);
           break;
      case Sphere:
           KMeansAssign2(static_cast<Field<D,Sphere>*>(field), centers, npatch, patches, n);
           break;
      case ThreeD:
           KMeansAssign2(static_cast<Field<D,ThreeD>*>(field), centers, npatch, patches, n);
           break;
    }
}

void KMeansAssign(void* field, double* centers, int npatch, long* patches, long n,
                  int d, int coords)
{
    switch(d) {
      case NData:
           KMeansAssign1<NData>(field, centers, npatch, patches, n, coords);
           break;
      case KData:
           KMeansAssign1<KData>(field, centers, npatch, patches, n, coords);
           break;
      case GData:
           KMeansAssign1<GData>(field, centers, npatch, patches, n, coords);
           break;
    }
}

void QuickAssign(double* centers, int npatch,
                 double* x, double* y, double* z, long* patches, long n)
{
    if (z) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i=0; i<n; ++i) {
            int kmin = 0;
            double min_rsq = SQR(x[i]-centers[0]) + SQR(y[i]-centers[1]) + SQR(z[i]-centers[2]);
            for (int k=1; k<npatch; ++k) {
                double rsq = SQR(x[i]-centers[3*k]) + SQR(y[i]-centers[3*k+1]) + SQR(z[i]-centers[3*k+2]);
                if (rsq < min_rsq) {
                    kmin = k;
                    min_rsq = rsq;
                }
            }
            patches[i] = kmin;
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i=0; i<n; ++i) {
            int kmin = 0;
            double min_rsq = SQR(x[i]-centers[0]) + SQR(y[i]-centers[1]);
            for (int k=1; k<npatch; ++k) {
                double rsq = SQR(x[i]-centers[2*k]) + SQR(y[i]-centers[2*k+1]);
                if (rsq < min_rsq) {
                    kmin = k;
                    min_rsq = rsq;
                }
            }
            patches[i] = kmin;
        }
    }
}


void SelectPatch(int patch, double* centers, int npatch, double* x, double* y, double* z,
                 long* use, long n)
{
    // Notation: p = the good patch we are looking for
    //           q = other patches
    //           if p is the closest, then use = 1, else use = 0.
    if (z) {
        // 3d version
        double px = centers[3*patch];
        double py = centers[3*patch+1];
        double pz = centers[3*patch+2];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i=0; i<n; ++i) {
            double p_dsq = SQR(x[i]-px) + SQR(y[i]-py) + SQR(z[i]-pz);
            use[i] = 1;
            for (int q=0; q<npatch; ++q) {
                if (q == patch) continue;
                double qx = centers[3*q];
                double qy = centers[3*q+1];
                double qz = centers[3*q+2];
                double q_dsq = SQR(x[i]-qx) + SQR(y[i]-qy) + SQR(z[i]-qz);
                if (q_dsq < p_dsq) {
                    use[i] = 0;
                    break;
                }
            }
        }
    } else {
        // 2d version
        double px = centers[2*patch];
        double py = centers[2*patch+1];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i=0; i<n; ++i) {
            double p_dsq = SQR(x[i]-px) + SQR(y[i]-py);
            use[i] = 1;
            for (int q=0; q<npatch; ++q) {
                if (q == patch) continue;
                double qx = centers[2*q];
                double qy = centers[2*q+1];
                double q_dsq = SQR(x[i]-qx) + SQR(y[i]-qy);
                if (q_dsq < p_dsq) {
                    use[i] = 0;
                    break;
                }
            }
        }
    }
}

void GenerateXYZ(double* x, double* y, double* z, double* ra, double* dec, double* r, long n)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i=0; i<n; ++i) {
        double sr, cr, sd, cd;
#ifdef _GLIBCXX_HAVE_SINCOS
        ::sincos(ra[i],&sr,&cr);
        ::sincos(dec[i],&sd,&cd);
#else
        // If the native sincos function isn't available, then most compilers will still optimize
        // these into a single trig call for each.
        sr = std::sin(ra[i]);
        cr = std::cos(ra[i]);
        sd = std::sin(dec[i]);
        cd = std::cos(dec[i]);
#endif
        x[i] = cd * cr;
        y[i] = cd * sr;
        z[i] = sd;
        if (r) {
            x[i] *= r[i];
            y[i] *= r[i];
            z[i] *= r[i];
        }
    }
}
