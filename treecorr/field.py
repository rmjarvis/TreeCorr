# Copyright (c) 2003-2019 by Mike Jarvis
#
# TreeCorr is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

"""
.. module:: field
"""

import numpy as np
import weakref

from . import _lib, _ffi
from .util import get_omp_threads, parse_xyzsep, coord_enum
from .util import long_ptr as lp
from .util import double_ptr as dp
from .util import depr_pos_kwargs

def _parse_split_method(split_method):
    if split_method == 'middle': return 0
    elif split_method == 'median': return 1
    elif split_method == 'mean': return 2
    else: return 3  # random


class Field(object):
    r"""A Field in TreeCorr is the object that stores the tree structure we use for efficient
    calculation of the correlation functions.

    The root "cell" in the tree has information about the whole field, including the total
    number of points, the total weight, the mean position, the size (by which we mean the
    maximum distance of any point from the mean position), and possibly more information depending
    on which kind of field we have.

    It also points to two sub-cells which each describe about half the points.  These are commonly
    referred to as "daughter cells".  They in turn point to two more cells each, and so on until
    we get to cells that are considered "small enough" according to the ``min_size`` parameter given
    in the constructor.  These lowest level cells are referred to as "leaves".

    Technically, a Field doesn't have just one of these trees.  To make parallel computation
    more efficient, we actually skip the first few layers of the tree as described above and
    store a list of root cells.  The three parameters that determine how many of these there
    will be are ``max_size``, ``min_top``, and ``max_top``:

        - ``max_size`` sets the maximum size cell that we want to make sure we have in the trees,
          so the root cells will be at least this large.  The default is None, which means
          we care about all sizes, so there may be only one root cell (but typically more
          because of ``min_top``).
        - ``min_top`` sets the minimum number of initial levels to skip.  The default is either 3
          or :math:`\log_2(N_{cpu})`, whichever is larger.  This means there will be at least 8
          (or :math:`N_{cpu}`) root cells (assuming ntot is at least this large of course).
        - ``max_top`` sets the maximum number of initial levels to skip.  The default is 10,
          which means there could be up to 1024 root cells.

    Finally, the ``split_method`` parameter sets how the points in a cell should be divided
    when forming the two daughter cells.  The split is always done according to whichever
    dimension has the largest extent.  E.g. if max(\|x - meanx\|) is larger than max(\|y - meany\|)
    and (for 3d) max(\|z - meanz\|), then it will split according to the x values.  But then
    it may split in different ways according to ``split_method``.  The allowed values are:

        - 'mean' means to divide the points at the average (mean) value of x, y or z.
        - 'median' means to divide the points at the median value of x, y, or z.
        - 'middle' means to divide the points at midpoint between the minimum and maximum values.
        - 'random' means to divide the points randomly somewhere between the 40th and 60th
          percentile locations in the sorted list.

    Field itself is an abstract base class for the specific types of field classes.
    As such, it cannot be constructed directly.  You should make one of the concrete subclasses:

        - `NField` describes a field of objects to be counted only.
        - `KField` describes a field of points sampling a scalar field (e.g. kappa in the
          weak lensing context).  In addition to the above values, cells keep track of
          the mean kappa value in the given region.
        - `GField` describes a field of points sampling a spinor field (e.g. gamma in the
          weak lensing context).  In addition to the above values, cells keep track of
          the mean (complex) gamma value in the given region.
    """
    def __init__(self):
        raise NotImplementedError("Field is an abstract base class.  It cannot be instantiated.")

    def _determine_top(self, min_top, max_top):
        if min_top is None:
            n_cpu = get_omp_threads()
            # int.bit_length is a trick to avoid going through float.
            # bit_length(n-1) == ceil(log2(n)), which is what we want.
            min_top = max(3, int.bit_length(n_cpu-1))
        else:
            min_top = int(min_top)
        max_top = int(max_top)
        min_top = min(min_top, max_top)  # If min_top > max_top favor max_top.
        return min_top, max_top

    @property
    def nTopLevelNodes(self):
        """The number of top-level nodes.
        """
        return _lib.FieldGetNTopLevel(self.data, self._d, self._coords)

    @property
    def cat(self):
        """The catalog from which this field was constructed.

        It is stored as a weakref, so if the Catalog has already been garbage collected, this
        might be None.
        """
        # _cat is a weakref.  This gets back to a Catalog object.
        return self._cat()

    def count_near(self, *args, **kwargs):
        """Count how many points are near a given coordinate.

        Use the existing tree structure to count how many points are within some given separation
        of a target coordinate.

        There are several options for how to specify the reference coordinate, which depends
        on the type of coordinate system this field implements.

        1. For flat 2-dimensional coordinates:

        Parameters:
            x (float):       The x coordinate of the target location
            y (float):       The y coordinate of the target location
            sep (float):     The separation distance

        2. For 3-dimensional Cartesian coordinates:

        Parameters:
            x (float):       The x coordinate of the target location
            y (float):       The y coordinate of the target location
            z (float):       The z coordinate of the target location
            sep (float):     The separation distance

        3. For spherical coordinates:

        Parameters:
            ra (float or Angle):    The right ascension of the target location
            dec (float or Angle):   The declination of the target location
            c (CelestialCorod):     A ``coord.CelestialCoord`` object in lieu of (ra, dec)
            sep (float or Angle):   The separation distance
            ra_units (str):         The units of ra if given as a float
            dec_units (str):        The units of dec if given as a float
            sep_units (str):        The units of sep if given as a float

        4. For spherical coordinates with distances:

        Parameters:
            ra (float or Angle):    The right ascension of the target location
            dec (float or Angle):   The declination of the target location
            c (CelestialCorod):     A ``coord.CelestialCoord`` object in lieu of (ra, dec)
            r (float):              The distance to the target location
            sep (float):            The separation distance
            ra_units (str):         The units of ra if given as a float
            dec_units (str):        The units of dec if given as a float

        In all cases, for parameters that are angles (ra, dec, sep for 'spherical'), you may either
        provide this quantity as a ``coord.Angle`` instance, or you may provide ra_units, dec_units
        or sep_units respectively to specify which angular units are providing.

        Finally, in cases where ra, dec are allowed, you may instead provide a
        ``coord.CelestialCoord`` instance as the first argument to specify both RA and Dec.
        """
        if self.min_size == 0:
            # If min_size = 0, then regular method is already exact.
            x,y,z,sep = parse_xyzsep(args, kwargs, self._coords)
            return self._count_near(x, y, z, sep)
        else:
            # Otherwise, we need to expand the radius a bit and then check the actual radii
            # using the catalog values.  This is already done in get_near, so just do that
            # and take the len of the result.
            return len(self.get_near(*args, **kwargs))

    def _count_near(self, x, y, z, sep):
        # If self.min_size > 0, these results may be approximate, since the tree will have
        # grouped points within this separation together.
        return _lib.FieldCountNear(self.data, x, y, z, sep, self._d, self._coords)

    def get_near(self, *args, **kwargs):
        """Get the indices of points near a given coordinate.

        Use the existing tree structure to find the points that are within some given separation
        of a target coordinate.

        There are several options for how to specify the reference coordinate, which depends
        on the type of coordinate system this field implements.

        1. For flat 2-dimensional coordinates:

        Parameters:
            x (float):       The x coordinate of the target location
            y (float):       The y coordinate of the target location
            sep (float):     The separation distance

        2. For 3-dimensional Cartesian coordinates:

        Parameters:
            x (float):       The x coordinate of the target location
            y (float):       The y coordinate of the target location
            z (float):       The z coordinate of the target location
            sep (float):     The separation distance

        3. For spherical coordinates:

        Parameters:
            ra (float or Angle):    The right ascension of the target location
            dec (float or Angle):   The declination of the target location
            c (CelestialCorod):     A ``coord.CelestialCoord`` object in lieu of (ra, dec)
            sep (float or Angle):   The separation distance
            ra_units (str):         The units of ra if given as a float
            dec_units (str):        The units of dec if given as a float
            sep_units (str):        The units of sep if given as a float

        4. For spherical coordinates with distances:

        Parameters:
            ra (float or Angle):    The right ascension of the target location
            dec (float or Angle):   The declination of the target location
            c (CelestialCorod):     A ``coord.CelestialCoord`` object in lieu of (ra, dec)
            r (float):              The distance to the target location
            sep (float):            The separation distance
            ra_units (str):         The units of ra if given as a float
            dec_units (str):        The units of dec if given as a float

        In all cases, for parameters that are angles (ra, dec, sep for 'spherical'), you may either
        provide this quantity as a ``coord.Angle`` instance, or you may provide ra_units, dec_units
        or sep_units respectively to specify which angular units are providing.

        Finally, in cases where ra, dec are allowed, you may instead provide a
        ``coord.CelestialCoord`` instance as the first argument to specify both RA and Dec.
        """
        x,y,z,sep = parse_xyzsep(args, kwargs, self._coords)
        if self.min_size == 0:
            # If min_size == 0, then regular method is already exact.
            ind = self._get_near(x, y, z, sep)
        else:
            # Expand the radius by the minimum size of the cells.
            sep1 = sep + self.min_size
            # Get those indices
            ind = self._get_near(x, y, z, sep1)
            # Now check the actual radii of these points using the catalog x,y,z values.
            rsq = (self.cat.x[ind]-x)**2 + (self.cat.y[ind]-y)**2
            if self._coords != _lib.Flat:
                rsq += (self.cat.z[ind]-z)**2
            # Select the ones with r < sep
            near = rsq < sep**2
            ind = ind[near]
        # It comes back unsorted, so sort it.  (Not really required, but nicer output.)
        return np.sort(ind)

    def _get_near(self, x, y, z, sep):
        # If self.min_size > 0, these results may be approximate, since the tree will have
        # grouped points within this separation together.
        # First count how many there are, so we can allocate the array for the indices.
        n = self._count_near(x, y, z, sep)
        ind = np.empty(n, dtype=int)
        # Now fill the array with the indices of the nearby points.
        _lib.FieldGetNear(self.data, x, y, z, sep, self._d, self._coords, lp(ind), n)
        return ind

    @depr_pos_kwargs
    def run_kmeans(self, npatch, *, max_iter=200, tol=1.e-5, init='tree', alt=False, rng=None):
        r"""Use k-means algorithm to set patch labels for a field.

        The k-means algorithm (cf. https://en.wikipedia.org/wiki/K-means_clustering) identifies
        a center position for each patch.  Each point is then assigned to the patch whose center
        is closest.  The centers are then updated to be the mean position of all the points
        assigned to the patch.  This process is repeated until the center locations have converged.

        The process tends to converge relatively quickly.  The convergence criterion we use
        is a tolerance on the rms shift in the centroid positions as a fraction of the overall
        size of the whole field.  This is settable as ``tol`` (default 1.e-5).  You can also
        set the maximum number of iterations to allow as ``max_iter`` (default 200).

        The upshot of the k-means process is to minimize the total within-cluster sum of squares
        (WCSS), also known as the "inertia" of each patch.  This tends to produce patches with
        more or less similar inertia, which make them useful for jackknife or other sampling
        estimates of the errors in the correlation functions.

        More specifically, if the points :math:`j` have vector positions :math:`\vec x_j`,
        and we define patches :math:`S_i` to comprise disjoint subsets of the :math:`j`
        values, then the inertia :math:`I_i` of each patch is defined as:

        .. math::

            I_i = \sum_{j \in S_i} \left| \vec x_j - \vec \mu_i \right|^2,

        where :math:`\vec \mu_i` is the center of each patch:

        .. math::

            \vec \mu_i \equiv \frac{\sum_{j \in S_i} \vec x_j}{N_i},

        and :math:`N_i` is the number of points assigned to patch :math:`S_i`.
        The k-means algorithm finds a solution that is a local minimum in the total inertia,
        :math:`\sum_i I_i`.

        In addition to the normal k-means algorithm, we also offer an alternate algorithm, which
        can produce slightly better patches for the purpose of patch-based covariance estimation.
        The ideal patch definition for such use would be to minimize the standard deviation (std)
        of the inertia of each patch, not the total (or mean) inertia.  It turns out that it is
        difficult to devise an algorithm that literally does this, since it has a tendancy to
        become unstable and not converge.

        However, adding a penalty term to the patch assignment step of the normal k-means
        algorithm turns out to work reasonably well.  The penalty term we use is :math:`f I_i`,
        where :math:`f` is a scaling constant (see below).  When doing the assignment step we assign
        each point :math:`j` to the patch :math:`i` that gives the minimum penalized distance

        .. math::

            d_{ij}^{\prime\;\! 2} = \left| \vec x_j - \mu_i \right|^2 + f I_i.

        The penalty term means that patches with less inertia get more points on the next
        iteration, and vice versa, which tends to equalize the inertia values somewhat.
        The resulting patches have significantly lower std inertia, but typically only slightly
        higher total inertia.

        For the scaling constant, :math:`f`, we chose

        .. math::

            f = \frac{3}{\langle N_i\rangle},

        three times the inverse of the mean number of points in each patch.

        The :math:`1/\langle N_i\rangle` factor makes the two terms of comparable magnitude
        near the edges of the patches, so patches still get most of the points near their previous
        centers, even if they already have larger than average inertia, but some of the points in
        the outskirts of the patch might switch to a nearby patch with smaller inertia.  The
        factor of 3 is purely empirical, and was found to give good results in terms of std
        inertia on some test data (the DES SV field).

        The alternate algorithm is available by specifying ``alt=True``.  Despite it typically
        giving better patch centers than the standard algorithm, we don't make it the default,
        because it may be possible for the iteration to become unstable, leading to some patches
        with no points in them. (This happened in our tests when the arbitrary factor in the
        scaling constant was 5 instead of 3, but I could not prove that 3 would always avoid this
        failure mode.) If this happens for you, your best bet is probably to switch to the
        standard algorithm, which can never suffer from this problem.

        Parameters:
            npatch (int):       How many patches to generate
            max_iter (int):     How many iterations at most to run. (default: 200)
            tol (float):        Tolerance in the rms centroid shift to consider as converged
                                as a fraction of the total field size. (default: 1.e-5)
            init (str):         Initialization method. Options are:

                                    - 'tree' (default) =  Use the normal tree structure of the
                                      field, traversing down to a level where there are npatch
                                      cells, and use the centroids of these cells as the initial
                                      centers.  This is almost always the best choice.
                                    - 'random' =  Use npatch random points as the intial centers.
                                    - 'kmeans++' =  Use the k-means++ algorithm.
                                      cf. https://en.wikipedia.org/wiki/K-means%2B%2B

            alt (bool):         Use the alternate assignment algorithm to minimize the standard
                                deviation of the inertia rather than the total inertia (aka WCSS).
                                (default: False)
            rng (RandomState):  If desired, a numpy.random.RandomState instance to use for random
                                number generation. (default: None)

        Returns:
            Tuple containing

                - patches (array): An array of patch labels, all integers from 0..npatch-1.
                  Size is self.ntot.
                - centers (array): An array of center coordinates used to make the patches.
                  Shape is (npatch, 2) for flat geometries or (npatch, 3) for 3d or
                  spherical geometries.  In the latter case, the centers represent
                  (x,y,z) coordinates on the unit sphere.
        """
        centers = self.kmeans_initialize_centers(npatch, init=init, rng=rng)
        self.kmeans_refine_centers(centers, max_iter=max_iter, tol=tol, alt=alt)
        patches = self.kmeans_assign_patches(centers)
        return patches, centers

    @depr_pos_kwargs
    def kmeans_initialize_centers(self, npatch, init='tree', *, rng=None):
        """Use the field's tree structure to assign good initial centers for a K-Means run.

        The classic K-Means algorithm involves starting with random points as the initial
        centers of the patches.  This has a tendency to result in rather poor results in
        terms of having similar sized patches at the end.  Specifically, the standard deviation
        of the inertia at the local minimum that the K-Means algorithm settles into tends to be
        fairly high for typical geometries.

        A better approach is to use the existing tree structure to start out with centers that
        are fairly evenly spread out through the field.  This algorithm traverses the tree
        until we get to a level that has enough cells for the requested number of patches.
        Then it uses the centroids of these cells as the initial patch centers.

        Parameters:
            npatch (int):       How many patches to generate initial centers for
            init (str):         Initialization method. Options are:

                                    - 'tree' (default) =  Use the normal tree structure of the
                                      field, traversing down to a level where there are npatch
                                      cells, and use the centroids of these cells as the initial
                                      centers.  This is almost always the best choice.
                                    - 'random' =  Use npatch random points as the intial centers.
                                    - 'kmeans++' =  Use the k-means++ algorithm.
                                      cf. https://en.wikipedia.org/wiki/K-means%2B%2B

            rng (RandomState):  If desired, a numpy.random.RandomState instance to use for random
                                number generation. (default: None)

        Returns:
            An array of center coordinates.
            Shape is (npatch, 2) for flat geometries or (npatch, 3) for 3d or
            spherical geometries.  In the latter case, the centers represent
            (x,y,z) coordinates on the unit sphere.
        """
        if npatch > self.ntot:
            raise ValueError("Invalid npatch.  Cannot be greater than self.ntot.")
        if npatch < 1:
            raise ValueError("Invalid npatch.  Cannot be less than 1.")
        if self._coords == _lib.Flat:
            centers = np.empty((npatch, 2))
        else:
            centers = np.empty((npatch, 3))
        seed = 0 if rng is None else int(rng.random_sample() * 2**63)
        if init == 'tree':
            _lib.KMeansInitTree(self.data, dp(centers), int(npatch), self._d, self._coords, seed)
        elif init == 'random':
            _lib.KMeansInitRand(self.data, dp(centers), int(npatch), self._d, self._coords, seed)
        elif init == 'kmeans++':
            _lib.KMeansInitKMPP(self.data, dp(centers), int(npatch), self._d, self._coords, seed)
        else:
            raise ValueError("Invalid init: %s. "%init +
                             "Must be one of 'tree', 'random', or 'kmeans++.'")

        return centers

    @depr_pos_kwargs
    def kmeans_refine_centers(self, centers, *, max_iter=200, tol=1.e-5, alt=False):
        """Fast implementation of the K-Means algorithm

        The standard K-Means algorithm is as follows
        (cf. https://en.wikipedia.org/wiki/K-means_clustering):

        1. Choose centers somehow.  Traditionally, this is done by just selecting npatch random
           points from the full set, but we do this more smartly in `kmeans_initialize_centers`.
        2. For each point, measure the distance to each current patch center, and assign it to the
           patch that has the closest center.
        3. Update all the centers to be the centroid of the points assigned to each patch.
        4. Repeat 2, 3 until the rms shift in the centers is less than some tolerance or the
           maximum number of iterations is reached.
        5. Assign the corresponding patch label to each point (`kmeans_assign_patches`).

        In TreeCorr, we use the tree structure to massively increase the speed of steps 2 and 3.
        For a given cell, we know both its center and its size, so we can quickly check whether
        all the points in the cell are closer to one center than another.  This lets us quickly
        cull centers from consideration as we traverse the tree.  Once we get to a cell where only
        one center can be closest for any of the points in it, we stop traversing and assign the
        whole cell to that patch.

        Further, it is also fast to update the new centroid, since the sum of all the positions
        for a cell is just N times the cell's centroid.

        As a result, this algorithm typically takes a fraction of a second for ~a million points.
        Indeed most of the time spent in the full kmeans calculation is in building the tree
        in the first place, rather than actually running the kmeans code.  With the alternate
        algorithm (``alt=True``), the time is only slightly slower from having to calculate
        the sizes at each step.

        Parameters:
            centers (array):    An array of center coordinates. (modified by this function)
                                Shape is (npatch, 2) for flat geometries or (npatch, 3) for 3d or
                                spherical geometries.  In the latter case, the centers represent
                                (x,y,z) coordinates on the unit sphere.
            max_iter (int):     How many iterations at most to run. (default: 200)
            tol (float):        Tolerance in the rms centroid shift to consider as converged
                                as a fraction of the total field size. (default: 1.e-5)
            alt (bool):         Use the alternate assignment algorithm to minimize the standard
                                deviation of the inertia rather than the total inertia (aka WCSS).
                                (default: False)
        """
        npatch = centers.shape[0]
        _lib.KMeansRun(self.data, dp(centers), npatch, int(max_iter), float(tol),
                       bool(alt), self._d, self._coords)

    def kmeans_assign_patches(self, centers):
        """Assign patch numbers to each point according to the given centers.

        This is final step in the full K-Means algorithm.  It assignes patch numbers to each
        point in the field according to which center is closest.

        Parameters:
            centers (array):    An array of center coordinates.
                                Shape is (npatch, 2) for flat geometries or (npatch, 3) for 3d or
                                spherical geometries.  In the latter case, the centers represent
                                (x,y,z) coordinates on the unit sphere.

        Returns:
            An array of patch labels, all integers from 0..npatch-1.  Size is self.ntot.
        """
        patches = np.empty(self.ntot, dtype=int)
        npatch = centers.shape[0]
        centers = np.ascontiguousarray(centers)
        _lib.KMeansAssign(self.data, dp(centers), npatch,
                          lp(patches), self.ntot, self._d, self._coords)
        return patches


class NField(Field):
    r"""This class stores the positions and number of objects in a tree structure from which it is
    efficient to compute correlation functions.

    An NField is typically created from a Catalog object using

        >>> nfield = cat.getNField(min_size=min_size, max_size=max_size)

    Parameters:
        cat (Catalog):      The catalog from which to make the field.
        min_size (float):   The minimum radius cell required (usually min_sep). (default: 0)
        max_size (float):   The maximum radius cell required (usually max_sep). (default: None)
        split_method (str): Which split method to use ('mean', 'median', 'middle', or 'random').
                            (default: 'mean')
        brute (bool):       Whether to force traversal to the leaves for this field.
                            (default: False)
        min_top (int):      The minimum number of top layers to use when setting up the field.
                            (default: :math:`\max(3, \log_2(N_{\rm cpu}))`)
        max_top (int):      The maximum number of top layers to use when setting up the field.
                            (default: 10)
        coords (str):       The kind of coordinate system to use. (default: cat.coords)
        rng (RandomState):  If desired, a numpy.random.RandomState instance to use for random
                            number generation. (default: None)
        logger (Logger):    A logger file if desired. (default: None)
    """
    @depr_pos_kwargs
    def __init__(self, cat, *, min_size=0, max_size=None, split_method='mean', brute=False,
                 min_top=None, max_top=10, coords=None, rng=None, logger=None):
        if logger:
            if cat.name != '':
                logger.info('Building NField from cat %s',cat.name)
            else:
                logger.info('Building NField')

        self._cat = weakref.ref(cat)
        self.ntot = cat.ntot
        self.min_size = float(min_size) if not brute else 0.
        self.max_size = float(max_size) if max_size is not None else np.inf
        self.split_method = split_method
        self._sm = _parse_split_method(split_method)
        self._d = 1  # NData
        self.brute = bool(brute)
        self.min_top, self.max_top = self._determine_top(min_top, max_top)
        self.coords = coords if coords is not None else cat.coords
        self._coords = coord_enum(self.coords)  # These are the C++-layer enums
        seed = 0 if rng is None else int(rng.random_sample() * 2**63)

        self.data = _lib.BuildNField(dp(cat.x), dp(cat.y), dp(cat.z),
                                     dp(cat.w), dp(cat.wpos), cat.ntot,
                                     self.min_size, self.max_size, self._sm, seed,
                                     self.brute, self.min_top, self.max_top, self._coords)
        if logger:
            logger.debug('Finished building NField (%s)',self.coords)

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        # In case __init__ failed to get that far
        if hasattr(self,'data'):  # pragma: no branch
            # I don't get this, but sometimes it gets here when the ffi.lock is already locked.
            # When that happens, this will freeze in a `with ffi._lock` line in the ffi api.py.
            # So, don't do that, and just accept the memory leak instead.
            if not _ffi._lock.locked(): # pragma: no branch
                _lib.DestroyNField(self.data, self._coords)


class KField(Field):
    r"""This class stores the values of a scalar field (kappa in the weak lensing context) in a
    tree structure from which it is efficient to compute correlation functions.

    A KField is typically created from a Catalog object using

        >>> kfield = cat.getKField(min_size, max_size, b)

    Parameters:
        cat (Catalog):      The catalog from which to make the field.
        min_size (float):   The minimum radius cell required (usually min_sep). (default: 0)
        max_size (float):   The maximum radius cell required (usually max_sep). (default: None)
        split_method (str): Which split method to use ('mean', 'median', 'middle', or 'random').
                            (default: 'mean')
        brute (bool):       Whether to force traversal to the leaves for this field.
                            (default: False)
        min_top (int):      The minimum number of top layers to use when setting up the field.
                            (default: :math:`\max(3, \log_2(N_{\rm cpu}))`)
        max_top (int):      The maximum number of top layers to use when setting up the field.
                            (default: 10)
        coords (str):       The kind of coordinate system to use. (default: cat.coords)
        rng (RandomState):  If desired, a numpy.random.RandomState instance to use for random
                            number generation. (default: None)
        logger (Logger):    A logger file if desired. (default: None)
    """
    @depr_pos_kwargs
    def __init__(self, cat, *, min_size=0, max_size=None, split_method='mean', brute=False,
                 min_top=None, max_top=10, coords=None, rng=None, logger=None):
        if logger:
            if cat.name != '':
                logger.info('Building KField from cat %s',cat.name)
            else:
                logger.info('Building KField')

        self._cat = weakref.ref(cat)
        self.ntot = cat.ntot
        self.min_size = float(min_size) if not brute else 0.
        self.max_size = float(max_size) if max_size is not None else np.inf
        self.split_method = split_method
        self._sm = _parse_split_method(split_method)
        self._d = 2  # KData
        self.brute = bool(brute)
        self.min_top, self.max_top = self._determine_top(min_top, max_top)
        self.coords = coords if coords is not None else cat.coords
        self._coords = coord_enum(self.coords)  # These are the C++-layer enums
        seed = 0 if rng is None else int(rng.random_sample() * 2**63)

        self.data = _lib.BuildKField(dp(cat.x), dp(cat.y), dp(cat.z),
                                     dp(cat.k),
                                     dp(cat.w), dp(cat.wpos), cat.ntot,
                                     self.min_size, self.max_size, self._sm, seed,
                                     self.brute, self.min_top, self.max_top, self._coords)
        if logger:
            logger.debug('Finished building KField (%s)',self.coords)

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        # In case __init__ failed to get that far
        if hasattr(self,'data'):  # pragma: no branch
            if not _ffi._lock.locked(): # pragma: no branch
                _lib.DestroyKField(self.data, self._coords)


class GField(Field):
    r"""This class stores the values of a spinor field (gamma in the weak lensing context) in a
    tree structure from which it is efficient to compute correlation functions.

    A GField is typically created from a Catalog object using

        >>> gfield = cat.getGField(min_size, max_size, b)

    Parameters:
        cat (Catalog):      The catalog from which to make the field.
        min_size (float):   The minimum radius cell required (usually min_sep). (default: 0)
        max_size (float):   The maximum radius cell required (usually max_sep). (default: None)
        split_method (str): Which split method to use ('mean', 'median', 'middle', or 'random').
                            (default: 'mean')
        brute (bool):       Whether to force traversal to the leaves for this field.
                            (default: False)
        min_top (int):      The minimum number of top layers to use when setting up the field.
                            (default: :math:`\max(3, \log_2(N_{\rm cpu}))`)
        max_top (int):      The maximum number of top layers to use when setting up the field.
                            (default: 10)
        coords (str):       The kind of coordinate system to use. (default: cat.coords)
        rng (RandomState):  If desired, a numpy.random.RandomState instance to use for random
                            number generation. (default: None)
        logger (Logger):    A logger file if desired. (default: None)
    """
    @depr_pos_kwargs
    def __init__(self, cat, *, min_size=0, max_size=None, split_method='mean', brute=False,
                 min_top=None, max_top=10, coords=None, rng=None, logger=None):
        if logger:
            if cat.name != '':
                logger.info('Building GField from cat %s',cat.name)
            else:
                logger.info('Building GField')

        self._cat = weakref.ref(cat)
        self.ntot = cat.ntot
        self.min_size = float(min_size) if not brute else 0.
        self.max_size = float(max_size) if max_size is not None else np.inf
        self.split_method = split_method
        self._sm = _parse_split_method(split_method)
        self._d = 3  # GData
        self.brute = bool(brute)
        self.min_top, self.max_top = self._determine_top(min_top, max_top)
        self.coords = coords if coords is not None else cat.coords
        self._coords = coord_enum(self.coords)  # These are the C++-layer enums
        seed = 0 if rng is None else int(rng.random_sample() * 2**63)

        self.data = _lib.BuildGField(dp(cat.x), dp(cat.y), dp(cat.z),
                                     dp(cat.g1), dp(cat.g2),
                                     dp(cat.w), dp(cat.wpos), cat.ntot,
                                     self.min_size, self.max_size, self._sm, seed,
                                     self.brute, self.min_top, self.max_top, self._coords)
        if logger:
            logger.debug('Finished building GField (%s)',self.coords)

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        # In case __init__ failed to get that far
        if hasattr(self,'data'):  # pragma: no branch
            if not _ffi._lock.locked(): # pragma: no branch
                _lib.DestroyGField(self.data, self._coords)


class SimpleField(object):
    """A SimpleField is like a Field, but only stores the leaves as a list, skipping all the
    tree stuff.

    Again, this is an abstract base class, which cannot be instantiated.  You should
    make one of the concrete subclasses:

        - NSimpleField describes a field of objects to be counted only.
        - KSimpleField describes a field of points sampling a scalar field.
        - GSimpleField describes a field of points sampling a spinor field.

    .. warning::

        .. deprecated:: 4.1

            This function is deprecated and slated to be removed.
            If you have a need for it, please open an issue to describe your use case.
    """
    def __init__(self):
        raise NotImplementedError(
            "SimpleField is an abstract base class.  It cannot be instantiated.")


class NSimpleField(SimpleField):
    """This class stores the positions as a list, skipping all the tree stuff.

    An NSimpleField is typically created from a Catalog object using

        >>> nfield = cat.getNSimpleField()

    .. warning::

        .. deprecated:: 4.1

            This function is deprecated and slated to be removed.
            If you have a need for it, please open an issue to describe your use case.

    Parameters:
        cat (Catalog):      The catalog from which to make the field.
        logger (Logger):    A logger file if desired. (default: None)
    """
    @depr_pos_kwargs
    def __init__(self, cat, *, logger=None):
        if logger:
            if cat.name != '':
                logger.info('Building NSimpleField from cat %s',cat.name)
            else:
                logger.info('Building NSimpleField')
        self._d = 1  # NData
        self.coords = cat.coords
        self._coords = coord_enum(self.coords)  # These are the C++-layer enums

        self.data = _lib.BuildNSimpleField(dp(cat.x), dp(cat.y), dp(cat.z),
                                           dp(cat.w), dp(cat.wpos), cat.ntot,
                                           self._coords)
        if logger:
            logger.debug('Finished building NSimpleField (%s)',self.coords)

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        # In case __init__ failed to get that far
        if hasattr(self,'data'):  # pragma: no branch
            if not _ffi._lock.locked(): # pragma: no branch
                _lib.DestroyNSimpleField(self.data, self._coords)


class KSimpleField(SimpleField):
    """This class stores the kappa field as a list, skipping all the tree stuff.

    A KSimpleField is typically created from a Catalog object using

        >>> kfield = cat.getKSimpleField()

    .. warning::

        .. deprecated:: 4.1

            This function is deprecated and slated to be removed.
            If you have a need for it, please open an issue to describe your use case.

    Parameters:
        cat (Catalog):      The catalog from which to make the field.
        logger (Logger):    A logger file if desired. (default: None)
    """
    @depr_pos_kwargs
    def __init__(self, cat, *, logger=None):
        if logger:
            if cat.name != '':
                logger.info('Building KSimpleField from cat %s',cat.name)
            else:
                logger.info('Building KSimpleField')
        self._d = 2  # KData
        self.coords = cat.coords
        self._coords = coord_enum(self.coords)  # These are the C++-layer enums

        self.data = _lib.BuildKSimpleField(dp(cat.x), dp(cat.y), dp(cat.z),
                                           dp(cat.k),
                                           dp(cat.w), dp(cat.wpos), cat.ntot,
                                           self._coords)
        if logger:
            logger.debug('Finished building KSimpleField (%s)',self.coords)

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        # In case __init__ failed to get that far
        if hasattr(self,'data'):  # pragma: no branch
            if not _ffi._lock.locked(): # pragma: no branch
                _lib.DestroyKSimpleField(self.data, self._coords)


class GSimpleField(SimpleField):
    """This class stores the shear field as a list, skipping all the tree stuff.

    A GSimpleField is typically created from a Catalog object using

        >>> gfield = cat.getGSimpleField()

    .. warning::

        .. deprecated:: 4.1

            This function is deprecated and slated to be removed.
            If you have a need for it, please open an issue to describe your use case.

    Parameters:
        cat (Catalog):      The catalog from which to make the field.
        logger (Logger):    A logger file if desired. (default: None)
    """
    @depr_pos_kwargs
    def __init__(self, cat, *, logger=None):
        if logger:
            if cat.name != '':
                logger.info('Building GSimpleField from cat %s',cat.name)
            else:
                logger.info('Building GSimpleField')
        self._d = 3  # GData
        self.coords = cat.coords
        self._coords = coord_enum(self.coords)  # These are the C++-layer enums

        self.data = _lib.BuildGSimpleField(dp(cat.x), dp(cat.y), dp(cat.z),
                                           dp(cat.g1), dp(cat.g2),
                                           dp(cat.w), dp(cat.wpos), cat.ntot,
                                           self._coords)
        if logger:
            logger.debug('Finished building KSimpleField (%s)',self.coords)

    def __del__(self):
        # Using memory allocated from the C layer means we have to explicitly deallocate it
        # rather than being able to rely on the Python memory manager.

        # In case __init__ failed to get that far
        if hasattr(self,'data'):  # pragma: no branch
            if not _ffi._lock.locked(): # pragma: no branch
                _lib.DestroyGSimpleField(self.data, self._coords)
