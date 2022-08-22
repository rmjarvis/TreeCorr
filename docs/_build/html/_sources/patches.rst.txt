Patches
=======

Normally, TreeCorr is used to compute the auto-correlation function
of data in a single input `Catalog` or the cross-correlation of data
from two `Catalogs <Catalog>`.
However, there are a number of reasons that it might make sense to
divide up a region into several smaller patches for computing the
correlation function:

1. To compute a more accurate covariance matrix.
   There are a number of ways to compute more accurate covariance estimates
   from the data than the default method.  All of them require dividing
   up the data into patches and doing different things with the
   patch-to-patch correlations.  See `Covariance Estimates` for details.
2. To save memory.
   The entire data set might be too large to fit in memory, so you might
   want to divide it up so less data is required to be in memory at a time.
   See `Reducing Memory Use` below.
3. To split the job among multiple machines.
   TreeCorr does a good job of utilizing many cores on a single machine
   using OpenMP.  However, for very large jobs, you may want to also
   split the work among more than one node on a cluster.  The most
   effective way to do this is to split the data into patches.
   See `Using MPI` below.
4. To run k-means on some data set for non-correlation reasons.
   TreeCorr happens to have an extremely efficient implementation of the
   k-means algorithm.  So if you want to perform k-means clustering on
   some data that can be represnted in a TreeCorr `Catalog` (i.e.
   only 2 or 3 spatial dimensions), then using TreeCorr may be a
   particularly efficient way to do the clustering.
   See `Running K-Means` below.

Below we describe how to split up an input `Catalog` into patches and
a few things you can do with it once you have done so.

Defining Patches on Input
-------------------------

The most straightforward way to define which object goes in which patch
is to just tell TreeCorr the patch number for each object explicitly.

If passing in numpy arrays for everything, then just pass in a ``patch``
parameter with integer values indicating the patch number.

If reading in data from a file, then set a ``patch_col`` to use which
should have these values.

The next simplest way to define the patches is to tell TreeCorr how many
patches you want using ``npatch``.
TreeCorr will then run the K-Means algorithm to split up the full area
into this many patches.
See `Running K-Means` below for more details.

Finally, to make sure multiple catalogs are using the same definition for
where patches are on the sky, you would probably want to have a single
set of patch centers and have all of your catalogs use that via
the ``patch_centers`` option.  See `Using Patch Centers` below for details.


Running K-Means
---------------

One standard way to split up a set of objects into roughly equal area
patches is an algorithm called
`k-means clustering <https://en.wikipedia.org/wiki/K-means_clustering>`_.

The basic idea of the algorithm is to divide the points :math:`\vec x_j` into
:math:`k` patches, :math:`S_i`, such that the total "inertia" is minimized.
Inertia :math:`I_i` of each patch is defined as follows:

.. math::

    I_i = \sum_{j \in S_i} \left| \vec x_j - \vec \mu_i \right|^2,

where :math:`\vec \mu_i` is the center of each patch:

.. math::

    \vec \mu_i \equiv \frac{\sum_{j \in S_i} \vec x_j}{N_i},

and :math:`N_i` is the number of points assigned to patch :math:`S_i`.
The k-means algorithm finds a solution that is a local minimum in the total inertia,
:math:`\sum_i I_i`, or equivalently the mean inertia :math:`\langle I_i \rangle`
of all the patches.

This definition of inertia is a relatively good proxy for area on the
sky that has objects, so this algorithm is a good choice for dividing up a
catalog of astronomical objects into fairly uniform patches.

To use the TreeCorr implementation of k-means, simply
set the ``npatch`` parameter in the `Catalog` constructor to specifiy
how many patches you want TreeCorr to split the data into.

.. note::

    If the input catalog has (ra, dec, r) positions, then the patches will
    be made using only the ra,dec location on the sky, not the full 3-D
    positions.  This is usually what you want for making patches over an
    astronomical survey area.  If you really want to make patches according
    to 3-D clustering of points, then you should input x,y,z values instead.

There are also two additional options which can affect how the k-means
algorithm runs:

* ``kmeans_init`` specifies what procedure to use for the initialization
  of the patches.  Options are:

   * 'random' = Choose initial centers randomly from among the input points.
     This is the traditional k-means initialization algorithm.
   * 'kmeans++' = Use `k-means++ <https://en.wikipedia.org/wiki/K-means%2B%2B>`_,
     an improved algorithm by Arthur and Vassilvitskii
     with a provable upper bound for how close the final result will
     be to the global minimum possible total inertia.
   * 'tree' = Use the upper layers of the TreeCorr ball tree to define
     the initial centers.  This is the default, and in practice,
     it will almost always yield the best final patches.
     (See :ref:`Comparison with other implementations <Comparison>` below.)

* ``kmeans_alt`` specifies whether to use an alternate iteration algorithm
  similar to k-means, which often produces somewhat more uniform patches.

  This alternate algorithm specifically targets minimizing the standard deviation
  of the inertia rather than the mean inertia, so it tends to lead to patches that
  have a smaller final size variation than the regular k-means algorithm.

  This is not the default algorithm because it is not provably (at least by
  me) stable.  It is possible that the iteration can get into a failure mode
  where one patch will end up with zero objects.  The regular k-means
  provably cannot fail in this way.

  So if you care especially about having very uniform patch sizes, you might
  want to try this option, but be careful about inspecting the results that
  they don't look crazy.

See also `Field.run_kmeans`, which has more information about these options,
where these parameters are called simply ``init`` and ``alt`` respectively.

.. _Comparison:
.. admonition:: Comparison with other implementations

    Before implementing k-means in TreeCorr, I investigated what other options
    there were in the Python landscape.  I found the following implementations:

    * `scipy.cluster.vq.kmeans <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html>`_
    * `scipy.cluster.vq.kmeans2 <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html#scipy.cluster.vq.kmeans2>`_
    * `kmeans_radec <https://github.com/esheldon/kmeans_radec>`_
    * `pyclustering.cluster.kmeans <https://pyclustering.github.io/docs/0.8.2/html/da/d97/namespacepyclustering_1_1cluster_1_1kmeans.html>`_
    * `sklearn.cluster.KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans>`_
    * `sklearn.cluster.MiniBatchKMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans>`_

    I made a `notebook <https://github.com/rmjarvis/TreeCorr/blob/main/devel/kmeans.ipynb>`_
    comparing the different algorithms using a random million galaxies from the DES SV
    (Dark Energy Survey, Science Verification) footprint, chosen because it is a
    real-life use case that has some ratty edges to deal with, so it seemed like it would
    provide a reasonable challenge without being crazy.

    The ideal patches would be essentially uniform in size according to some measure of the
    effective area of the patch. To make things simple, I just used the inertia as my
    proxy for area, since that's the thing that k-means algorithms natively work with.

    However, we don't really care about the total inertia being minimized.  For most purposes
    here, we really want the patches to be all close to the *same* size.  So rather than
    the total inertia, my metric for quality was the rms variation of the intertia
    (aka the standard deviation).

    Fortunately, the process of minimizing the total inertia does tend to select patches with
    small rms variation as well, but it is worth noting that this is not directly targeted by the
    normal k-means algorithm. And furthermore, the k-means algorithm almost never finds the true
    global minimum inertia. The quality of the local minimum depends pretty strongly on the
    choice of initial centers to seed the iterative part of the algorithm.

    Comparing the results of the various k-means implementations, I found that they all tend
    to be either fairly slow, taking a minute or more for just 1 million objects, or they have
    very high rms variation in the inertia.
    I reran each code multiple times using a different random million objects selected from the original
    catalog (of around 16 million objects). Here is a scatter plot of the time vs rms variation
    in the inertia for the various codes.

    .. image:: https://user-images.githubusercontent.com/623887/57647337-ac6bd800-7590-11e9-80bc-900bda3bf66b.png

    Since there was no existing implementation I was particularly happy with,
    I implemented it myself in TreeCorr. It turns out (not surprisingly) that the ball tree
    data structure that TreeCorr uses for efficient calculation of correlation functions
    also enables a very efficient implementation of the k-means iteration step.
    Furthermore, the quality of the k-means result is pretty dependent
    on the choice of the initial centers, and using the ball tree for the initialization turns
    out to produce reliably better results than the initialization methods used by other packages.

    The big red dots in the lower left corner are the TreeCorr implementation of the standard
    k-means clustering algorithm. It typically takes about 1 or 2 seconds to classify these
    1 million points into 40 patches, and the rms variation is usually less than any other
    implementation.

    The `notebook <https://github.com/rmjarvis/TreeCorr/blob/main/devel/kmeans.ipynb>`_ also
    includes plots of total inertia, variation in size according to the mean d^2 rather than
    sum, and variation in the counts. The TreeCorr algorithm tends to be the best k-means
    implementation according to any of these metrics.

    In addition, you can see some slightly smaller orange dots, which have even lower rms
    variation but take very slightly longer to run. These are the alternate algorithm I mentioned
    above.  This alternate algorithm is similar to k-means, but it penalizes patches with a
    larger-than-average inertia, so they give up some of their outer points to patches with
    smaller inertia. In other words, it explicitly targets making the rms variation as small as
    possible.  But in practice, it is not much worse in terms of total inertia either.

    The alternate algorithm is available using ``alt=True`` in `Field.run_kmeans`.
    I left this as a non-default option for two reasons. First, it's not actually the real
    k-means, so I didn't want to confuse people who just want to use this for regular k-means
    clustering. But second, I'm not completely sure that it is always stable. There is a free
    parameter in the penalty function I chose, which I set to 3. Setting it to 4 gave even better
    results (slightly), but at 5 the algorithm broke down with neighboring patches trading
    escalating numbers of points between each other until one of them had no points left.

    I couldn't convince myself that 4 was actually a magic number and not just the particular
    value for this data set. So 3 might be safe, or there might be data sets where that also
    leads to this runaway trading failure mode. I know the regular k-means algorithm can't get
    into this mode, so it's always safe. Therefore, I think it's better to force the user to
    intentionally select the alternate algorithm if they really care about having a low rms
    size variation, with the normal algorithm being the backup if the alternate one fails for them.


Using Patch Centers
-------------------

If you are doing a cross correlation, and you want to use patches for computing
a jackknife covariance for instance, you cannot
just set ``npatch`` in both and expect it to work properly.  The two catalogs
would end up with patches arranged very differently on the sky.  Patch 2
for one catalog would not be in the same place as patch 2 in the other one.
Thus, the jackknife calculation would be messed up.

Instead, you should define the patches using one of the two (or more)
catalogs you want to work with,
and then use its ``patch centers`` attribute as the ``patch_centers``
parameter when building the other catalog(s)::

    >>> cat1 = treecorr.Catalog(cat_file1, config1, npatch=N)
    >>> cat2 = treecorr.Catalog(cat_file2, config2, patch_centers=cat1.patch_centers)

You can also save the patches to a file using `Catalog.write_patch_centers`
and use that file name as the ``patch_centers`` parameter::

    >>> cat1 = treecorr.Catalog(cat_file1, config1, npatch=N)
    >>> cat1.write_patch_centers(cen_file)
    >>> cat2 = treecorr.Catalog(cat_file2, config2, patch_centers=cen_file)

With either method, cat2 will have patches assigned according to which patch
center each object is closest to.


Reducing Memory Use
-------------------

One reason you might want to use patches is if the full `Catalog` doesn't fit
in memory.  (Or possibly by itself it fits, but when performing the correlation function,
the additional memory from building the tree overflows the memory.)
Then you can potentially perform the calculation over patches
with less data loaded into memory at any given time.
The overall procedure for doing this is as follows:

1. First define your patch centers using some smaller `Catalog`, which
   does fit in memory.  This could be a catalog over the same survey
   geometry, which is intrinsically sparser (say a catalog of red sequence
   galaxies or clusters or even stars).  Or it could be the large catalog
   you want to use, but sampled using the ``every_nth`` option to read
   in only a fraction of the rows.  Run k-means on the smaller catalog
   and write the patch_centers to a file, as describe `above <Using Patch Centers>`.
2. Set up a directory somewhere that TreeCorr can use as temporary
   space for writing the individual patch files.
3. Define the full `Catalog`, specifying to use the above centers file for the
   ``patch_centers`` and the temp directory as ``save_patch_dir``.
4. Make sure not to do anything that requires the catalog be loaded from disk.
   TreeCorr will delay doing the actual load until it needs to do so.
   Here, we want to make sure it never loads the full data.
5. Run the `process <NNCorrelation.process>` function (for whichever correlation
   type you need) using the ``low_mem=True`` option.

Here are some worked examples.  First, an auto-correlation of a
single large shear catalog::

    >>> small_cat = treecorr.Catalog(cat_file, config, every_nth=100, npatch=N)
    >>> small_cat.write_patch_centers(cen_file)
    >>> del small_cat
    >>> full_cat = treecorr.Catalog(cat_file, config, patch_centers=cen_file,
    ...                             save_patch_dir=tmp_dir)
    >>> gg = treecorr.GGCorrelation(ggconfig)
    >>> gg.process(full_cat, low_mem=True)

Second, a cross-correlation, where the lens catalog is small enough not to
be a problem, but the source catalog is too large to hold in memory::

    >>> len_cat = treecorr.Catalog(lens_file, lens_config, npatch=N)
    >>> source_cat = treecorr.Catalog(source_file, source_config,
    ...                               patch_centers=lens_cat.patch_centers,
    ...                               save_patch_dir=tmp_dir)
    >>> ng = treecorr.NGCorrelation(ngconfig)
    >>> ng.process(lens_cat, source_cat, low_mem=True)

In both cases, the result should be equivalent to what you would get if you could
hold the catalogs fully in memory, but the peak memory will be much lower.
The downside is that this usage will generally take somewhat longer --
probably something like a factor of 2 for typical scenarios, but this of course
depends heavily on the nature of your calculation, how fast your disk I/O is
compared to your CPUs, and how many cores you are using.

.. note::

    Technically, the ``save_patch_dir`` parameter is not required, but it is
    recommended.  The first time a given patch is loaded, it will find the right
    rows in the full catalog and load the ones you need.  If you give it a
    directory, then it will write these data to disk, which will make subsequent
    reads of that patch much faster.

.. warning::

    One caveat with respect to the ``save_patch_dir`` parameter is that if there
    are already files present in the directory with the right names, then it
    will go ahead and use them, rather than make new patch files.  This is usually
    an efficiency gain, since repeated runs with the same data will already have
    the right patch files present.  However, if you use the same file name and
    save directory for a different data set, or if you make new patches for the
    same input file, then TreeCorr won't notice.

    To get TreeCorr to make new patch files, you can either manually delete
    everything in the save directory before starting, or (easier) call::

        >>> cat.write_patch_files()

    which will overwrite any existing files that may be there with the same names.

Using MPI
---------

Another use case that is enabled by using patches is
to divide up the work of calculating a correlation function
over multiple machines with MPI using `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_.

For this usage, the `process <NNCorrelation.process>` functions take an optional ``comm``
parameter.  When running in an MPI job, you can pass in ``comm=MPI.COMM_WORLD``,
and TreeCorr will divide up the work among however many nodes you are using.
The results will be sent back the the rank 0 node and combined to produce the
complete answer:

.. code-block:: python
    :linenos:

    # File name: run_with_mpi.py
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Define stuff
    fname = ...
    centers_file = ...
    config = ...
    ggconfig = ...

    # All machines read the catalog
    cat = treecorr.Catalog(fname, config, patch_centers=centers_file)

    # All machines define the same correlation object
    gg = treecorr.GGCorrelation(ggconfig)

    # Pass the comm object to the process function
    gg.process(cat, comm=comm)

    # rank 0 has the completed result.
    if rank == 0:
        # Probably do something more interesting with this now...
        print('xip = ',gg.xip)

You would then run this script using (e.g. with 4 processes)::

    $ mpiexec -n 4 python run_with_mpi.py

The file defining the patch centers should already be written to make sure
that each machine is using the same patch definitions.  There is some level of
randomness in the k-means calculation, so if you use ``npatch=N``, then each
machine may end up with different patch definitions, which would definitely
mess things up.

If you wanted to have it all run in a single script, you should have only
the rank 0 process define the patches.  Then send ``cat.patch_centers`` to the
other ranks, who can build their catalogs using this.
But it's probably easier to just precompute the centers and save them to a file
before starting the MPI run.

A more complete worked example is
`available <https://github.com/rmjarvis/TreeCorr/blob/main/devel/mpi_example.py>`_
in the TreeCorr devel directory.
