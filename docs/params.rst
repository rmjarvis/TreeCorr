
Configuration Parameters
========================

This section describes the various configuration parameters for controlling
what the `corr2` and `corr3` scripts (or functions) do:

Parameters about the input file(s)
----------------------------------

:file_name: (str or list)
    The file(s) with the data to be correlated.

    For an auto-correlation, like cosmic shear, this will be the only file
    name you need to specify.  This parameter is always required, and
    depending on what kind of correlation you are doing, you may need to
    specify others below.

    Normally, there would only be a single file name here, but sometimes
    the galaxy data comes in multiple files.  To treat them all as though
    they were a single large catalog, you may specify a list of file names
    here::

        file_name : [ file1.dat, file2.dat, file3.dat ]

    If you are specifying this on the command line, you'll need to put
    quotes around the names, or it won't be parsed correctly::

        file_name="[file1.dat,file2.dat,file3.dat]"

:file_name2: (str or list)
    The file(s) to use for the second field for a cross-correlation.

    If you want to cross-correlate one file (or set of files) with another, then
    ``file_name2`` is used to specify the second thing being correlated.  e.g.
    for galaxy-galaxy lensing, ``file_name`` should be the catalog of lenses, and
    ``file_name2`` should be the catalog of source shear values.

:file_name3: (str or list)
    The file(s) to use for the third field for a three-point cross-correlation.

:rand_file_name: (str or list)
    If necessary, a list of random files with the same masking as the ``file_name`` catalog.
:rand_file_name2: (str or list)
    If necessary, a list of random files with the same masking as the ``file_name2`` catalog.
:rand_file_name3: (str or list)
    If necessary, a list of random files with the same masking as the ``file_name3`` catalog.

    When doing NN and NNN correlations, you need to account for masks and variable
    depth by providing a file or list of files that correspond to a uniform-
    density field as observed with the same masking and other observational
    details.  For cross-correlations, you need to provide both of the above
    values to separately calibrate the first and second fields.

    ``rand_file_name`` may also be used for NG and NK correlations, but it is not
    required in those cases.

:file_list: (str) A text file with file names in lieu of ``file_name``.
:file_list2: (str) A text file with file names in lieu of ``file_name2``.
:file_list3: (str) A text file with file names in lieu of ``file_name3``.
:rand_file_list: (str) A text file with file names in lieu of ``rand_file_name``.
:rand_file_list2: (str) A text file with file names in lieu of ``rand_file_name2``.
:rand_file_list3: (str) A text file with file names in lieu of ``rand_file_name3``.

    If you have a list of file names, it may be cumbersome to list them all
    in the ``file_name`` (etc) parameter.  It may be easier to do something like
    ``ls *.cat > catlist`` and then use ``file_list=catlist`` as the list of
    file names to use.  Of course, it is an error to specify both ``file_list``
    and ``file_name`` (or any of the other corresponding pairs).

:file_type: (ASCII, FITS, HDF5, or Parquet) The file type of the input files.
:delimiter: (str, default = '\0') The delimeter between input values in an ASCII catalog.
:comment_marker: (str, default = '#') The first (non-whitespace) character of comment lines in an input ASCII catalog.

    The default file type is normally ASCII.  However, if the file name
    includes ".fit" in it, then a fits binary table is assumed.
    You can override this behavior using ``file_type``.

    Furthermore, you may specify a delimiter for ASCII catalogs if desired.
    e.g. delimiter=',' for a comma-separated value file.  Similarly,
    comment lines usually begin with '#', but you may specify something
    different if necessary.

:ext: (int/str, default=1 for FITS or root for HDF5) The extension (fits) or group (hdf) to read from

    Normally if you are using a fits file, the binary fits table is
    taken from the first extension, HDU 1.  If you want to read from a
    different HDU, you can specify which one to use here. For HDF files,
    the default is to read from the root of the file, but you can also
    specify group names like "/data/cat1"

:first_row: (int, default=1)
:last_row: (int, default=-1)
:every_nth: (int, default=1)

    You can optionally not use all the rows in the input file.
    You may specify ``first_row``, ``last_row``, or both to limit the rows being used.
    The rows are numbered starting with 1.  If ``last_row`` is not positive, it
    means to use to the end of the file.  If ``every_nth`` is set, it will skip
    rows, selecting only 1 out of every n rows.

:npatch: (int, default=1)

    How many patches to split the catalog into (using kmeans if no other
    patch information is provided) for the purpose of jackknife variance
    or other options that involve running via patches. (default: 1)

    .. note::

        If the catalog has ra,dec,r positions, the patches will
        be made using just ra,dec.

:kmeans_init: (str, default='tree')
:kmeans_alt: (bool, default=False)

    If using kmeans to make patches, these two parameters specify which init method
    to use and whether to use the alternate kmeans algorithm.
    cf. `Field.run_kmeans`

:patch_centers: (str)

    Alternative to setting patch by hand or using kmeans, you
    may instead give patch_centers either as a file name or an array
    from which the patches will be determined.

:x_col: (int/str) Which column to use for x.
:y_col: (int/str) Which column to use for y.
:ra_col: (int/str) Which column to use for ra.
:dec_col: (int/str) Which column to use for dec.

    For the positions of the objects, you can specify either x,y values, which
    imply a flat-sky approximation has already been performed (or ignored),
    or ra,dec values, which are of course positions on the curved sky.

    For ASCII files, the columns are specified by number, starting with 1 being
    the first column (not 0!).
    For FITS files, the columns are specified by name, not number.

:x_units: (str, default=None) The units of x values.
:y_units: (str, default=None) The units of y values.
:ra_units: (str) The units of ra values.
:dec_units: (str) The units of dec values.

    All distances on the sky include a "units" parameter to specify what in
    units the values are specified.  Options for units are radians, hours,
    degrees, arcmin, arcsec.  For ra, dec the units field is required.
    But for x,y, you can ignore all the unit issues, in which case the
    output distances will be in the same units as the input positions.

:r_col: (int/str) Which column to use for r.

    When using spherical coordinates, ra,dec, you can optionally provide a
    distance to the object.  In this case, the calculation will be done in
    three dimensional distances rather than angular distances.  The distances
    between objects will be the 3-D Euclidean distance, so you should define
    your r values appropriately, given whatever cosmology you are assuming.

    ``r_col`` is invalid in conjunction with ``x_col``, ``y_col``.

:z_col: (int/str) Which column to use for z.

    Rather than specifying 3-D coordinates as (ra, dec, r), you may instead
    specify them as (x, y, z).

    ``z_col`` is invalid in conjunction with ``ra_col``, ``dec_col``.

:g1_col: (int/str) Which column to use for g1.
:g2_col: (int/str) Which column to use for g2.

    If you are doing one of the shear correlation functions (i.e. NG, KG, GG),
    then you need to specify the shear estimates of the corresponding galaxies.
    The g1,g2 values are taken to be reduced shear values.  They should be
    unbiases estimators of g1,g2, so they are allowed to exceed :math:`|g| = 1`.
    (This is required for some methods to produce unbiased estimates.

:k_col: (int/str) Which column to use for kappa.

    If you are doing one of the kappa correlation functions (i.e. NK, KG, KK),
    then you need to specify the column to use for kappa.  While kappa is
    nominally the lensing convergence, it could really be any scalar quantity,
    like temperature, size, etc.

:patch_col: (int/str) Which column to use for patch.

    Use precalculated patch numbers to split the catalog into patches.

:w_col: (int/str) Which column to use for the weight (if any).
:wpos_col: (int/str) Which column to use for the position weight (if any).

    The weight column is optional. If omitted, all weights are taken to be 1.

:flag_col: (int/str) Which column to use for the weight (if any).
:ignore_flag: (int) What flag(s) should be ignored.
:ok_flag: (int) What flag(s) are ok to use.

    The code can be set to ignore objects with a particular flag value if desired.
    Some codes output a flag along with the shear value.  Typically any flag != 0
    should be ignored, but you can optionally have the code ignore only particular
    flags, treating the flag value as a bit mask.  If ``ignore_flag`` is set to
    something, then objects with ``(flag & ignore_flag != 0)`` will be ignored.
    If ``ok_flag`` is set, then objects with ``(flag & ~ok_flag != 0)`` will be ignored.
    The default is equivalent to ``ok_flag = 0``, which ignores any flag != 0.

:x_ext: (int/str) Which HDU (fits) or group (HDF) to use for the ``x_col``.
:y_ext: (int/str) Which HDU (fits) or group (HDF) to use for the ``y_col``.
:z_ext: (int/str) Which HDU (fits) or group (HDF) to use for the ``z_col``.
:ra_ext: (int/str) Which HDU (fits) or group (HDF) to use for the ``ra_col``.
:dec_ext: (int/str) Which HDU (fits) or group (HDF) to use for the ``dec_col``.
:r_ext: (int/str) Which HDU (fits) or group (HDF) to use for the ``r_col``.
:g1_ext: (int/str) Which HDU (fits) or group (HDF) to use for the ``g1_col``.
:g2_ext: (int/str) Which HDU (fits) or group (HDF) to use for the ``g2_col``.
:k_ext: (int/str) Which HDU (fits) or group (HDF) to use for the ``k_col``.
:patch_ext: (int/str) Which HDU (fits) or group (HDF) to use for the ``patch_col``.
:w_ext: (int/str) Which HDU (fits) or group (HDF) to use for the ``w_col``.
:wpos_ext: (int/str) Which HDU (fits) or group (HDF) to use for the ``wpos_col``.
:flag_ext: (int/str) Which HDU (fits) or group (HDF) to use for the ``flag_col``.

    If you want to use an extension other than the first one, normally you would
    specify which fits extension or HDF5 group to use with the ``ext`` parameter.
    However, if different columns need to come from different HDUs, then you can
    override the default (given by ``ext``, or '1' (fits), or '/' (HDF) if there
    is no ``ext`` parameter) for each column separately.

:allow_xyz: (bool, default=False)

    Whether to allow x,y,z columns in conjunction with ra, dec.

:flip_g1: (bool, default=False) Whether to flip the sign of g1.
:flip_g2: (bool, default=False) Whether to flip the sign of g2.

    Sometimes there are issues with the sign conventions of gamma.  If you
    need to flip the sign of g1 or g2, you may do that with ``flip_g1`` or ``flip_g2``
    (or both).

:keep_zero_weight: (bool, default=False)

    Whether to keep objects with wpos=0 in the catalog (including
    any objects that indirectly get wpos=0 due to NaN or flags), so they
    would be included in ntot and also in npairs calculations that use
    this Catalog, although of course not contribute to the accumulated
    weight of pairs.

.. note::

    - If you are cross-correlating two files with different formats, you may
      set any of the above items from ``file_type`` to ``flip_g2`` as a two element
      list (i.e. two values separated by a space).  In this case, the first
      item refers to the file(s) in ``file_name``, and the second item refers
      to the file(s) in files_name2.

    - You may not mix (x,y) columns with (ra,dec) columns, since its meaning
      would be ambiguous.

    - If you don't need a particular column for one of the files, you may
      use 0 to indicate not to read that column.  This is true for
      any format of input catalog.

    - Also, if the given column only applies to one of the two input files
      (e.g. k_col for an n-kappa cross-correlation) then you may specify just
      the column name or number for the file to which it does apply.


Parameters about the binned correlation function to be calculated
-----------------------------------------------------------------


:bin_type: (str, default='Log') Which type of binning should be used.

    See `Metrics` for details.

:min_sep: (float) The minimum separation to include in the output.
:max_sep: (float) The maximum separation to include in the output.
:nbins: (int) The number of output bins to use.
:bin_size: (float) The size of the output bins in log(sep).

    The bins for the histogram may be defined by setting any 3 of the above 4
    parameters.  The fourth one is automatically calculated from the values
    of the other three.

    See `Binning` for details about how these parameters are used for the
    different choice of ``bin_type``.

:sep_units: (str, default=None) The units to use for ``min_sep`` and ``max_sep``.

    ``sep_units`` is also the units of R in the output file.  For ra, dec values,
    you should always specify ``sep_units`` explicitly to indicate what angular
    units you want to use for the separations.  But if your catalogs use x,y,
    or if you specify 3-d correlations with r, then the output separations are
    in the same units as the input positions.

    See `sep_units` for more discussion about this parameter.

:bin_slop: (float, default=1) The fraction of a bin width by which it is ok to let the pairs miss the correct bin.

    The code normally determines when to stop traversing the tree when all of the
    distance pairs for the two nodes have a spread in distance that is less than the
    bin size.  i.e. the error in the tree traversal is less than the uncertainty
    induced by just binning the results into a histogram.  This factor can be changed
    by the parameter ``bin_slop``.  It is probably best to keep it at 1, but if you want to
    make the code more conservative, you can decrease it, in which case the error
    from using the tree nodes will be less than the error in the histogram binning.
    (In practice, if you are going to do this, you are probably better off just
    decreasing the ``bin_size`` instead and leaving ``bin_slop=1``.)

    See `bin_slop` for more discussion about this parameter.

:brute: (bool/int, default=False) Whether to do the "brute force" algorithm, where the
    tree traversal always goes to the leaf cells.

    In addition to True or False, whose meanings are obvious, you may also set
    ``brute`` to 1 or 2, which means to go to the leaves for cat1 or cat2, respectively,
    but stop traversing the other catalog according to the normal ``bin_slop`` criterion.

    See `brute` for more discussion about this parameter.

:min_u: (float) The minimum u=d3/d2 to include for three-point functions.
:max_u: (float) The maximum u=d3/d2 to include for three-point functions.
:nubins: (int) The number of output bins to use for u.
:ubin_size: (float) The size of the output bins for u.

:min_v: (float) The minimum positive v=(d1-d2)/d3 to include for three-point functions.
:max_v: (float) The maximum positive v=(d1-d2)/d3 to include for three-point functions.
:nvbins: (int) The number of output bins to use for positive v.
    The total number of bins in the v direction will be twice this number.
:vbin_size: (float) The size of the output bins for v.

:metric: (str, default='Euclidean') Which metric to use for distance measurements.

    See `Metrics` for details.

:min_rpar: (float) If the metric supports it, the minimum Rparallel to allow for pairs
    to be included in the correlation function.
:max_rpar: (float) If the metric supports it, the maximum Rparallel to allow for pairs
    to be included in the correlation function.

:period: (float) For the 'Periodic' metric, the period to use in all directions.
:xperiod: (float) For the 'Periodic' metric, the period to use in the x directions.
:yperiod: (float) For the 'Periodic' metric, the period to use in the y directions.
:zperiod: (float) For the 'Periodic' metric, the period to use in the z directions.


Parameters about the output file(s)
-----------------------------------

The kind of correlation function that the code will calculate is based on
which output file(s) you specify.  It will do the calculation(s) relevant for
each output file you set.  For each output file, the first line of the output
says what the columns are.  See the descriptions below for more information
about the output columns.

:nn_file_name: (str) The output filename for count-count correlation function.

    This is the normal density two-point correlation function.

    The output columns are:

    - ``R_nom`` = The center of the bin
    - ``meanR`` = The mean separation of the points that went into the bin.
    - ``meanlogR`` = The mean log(R) of the points that went into the bin.
    - ``xi`` = The correlation function.
    - ``sigma_xi`` = The 1-sigma error bar for xi.
    - ``DD``, ``RR`` = The raw numbers of pairs for the data and randoms
    - ``DR`` (if ``nn_statistic=compensated``) = The cross terms between data and random.
    - ``RD`` (if ``nn_statistic=compensated`` cross-correlation) = The cross term between random and data, which for a cross-correlation is not equivalent to ``DR``.

:nn_statistic: (str, default='compensated') Which statistic to use for xi as the estimator of the NN correlation function.

    Options are (D = data catalog, R = random catalog)

    - 'compensated' is the now-normal Landy-Szalay statistic:  xi = (DD-2DR+RR)/RR, or for cross-correlations, xi = (DD-DR-RD+RR)/RR
    - 'simple' is the older version: xi = (DD/RR - 1)

:ng_file_name: (str) The output filename for count-shear correlation function.

    This is the count-shear correlation function, often called galaxy-galaxy
    lensing.

    The output columns are:

    - ``R_nom`` = The center of the bin
    - ``meanR`` = The mean separation of the points that went into the bin.
    - ``meanlogR`` = The mean log(R) of the points that went into the bin.
    - ``gamT`` = The mean tangential shear with respect to the point in question.
    - ``gamX`` = The shear component 45 degrees from the tangential direction.
    - ``sigma`` = The 1-sigma error bar for ``gamT`` and ``gamX``.
    - ``weight`` = The total weight of the pairs in each bin.
    - ``npairs`` = The total number of pairs in each bin.

:ng_statistic: (str, default='compensated' if ``rand_files`` is given, otherwise 'simple') Which statistic to use for the mean shear as the estimator of the NG correlation function.

    Options are:

    - 'compensated' is simiar to the Landy-Szalay statistic:
      Define:

      - NG = Sum(gamma around data points)
      - RG = Sum(gamma around random points), scaled to be equivalent in effective number as the number of pairs in NG.
      - npairs = number of pairs in NG.

      Then this statistic is gamT = (NG-RG)/npairs
    - 'simple' is the normal version: gamT = NG/npairs

:gg_file_name: (str) The output filename for shear-shear correlation function.

    This is the shear-shear correlation function, used for cosmic shear.

    The output columns are:

    - ``R_nom`` = The center of the bin
    - ``meanR`` = The mean separation of the points that went into the bin.
    - ``meanlogR`` = The mean log(R) of the points that went into the bin.
    - ``xip`` = <g1 g1 + g2 g2> where g1 and g2 are measured with respect to the line joining the two galaxies.
    - ``xim`` = <g1 g1 - g2 g2> where g1 and g2 are measured with respect to the line joining the two galaxies.
    - ``xip_im`` = <g2 g1 - g1 g2>.

        In the formulation of xi+ using complex numbers, this is the imaginary component.
        It should normally be consistent with zero, especially for an
        auto-correlation, because if every pair were counted twice to
        get each galaxy in both positions, then this would come out
        exactly zero.

    - ``xim_im`` = <g2 g1 + g1 g2>.

        In the formulation of xi- using complex
        numbers, this is the imaginary component.
        It should be consistent with zero for parity invariant shear
        fields.

    - ``sigma_xi`` = The 1-sigma error bar for xi+ and xi-.
    - ``weight`` = The total weight of the pairs in each bin.
    - ``npairs`` = The total number of pairs in each bin.

:nk_file_name: (str) The output filename for count-kappa correlation function.

    This is nominally the kappa version of the ne calculation.  However, k is
    really any scalar quantity, so it can be used for temperature, size, etc.

    The output columns are:

    - ``R_nom`` = The center of the bin
    - ``meanR`` = The mean separation of the points that went into the bin.
    - ``meanlogR`` = The mean log(R) of the points that went into the bin.
    - ``kappa`` = The mean kappa this distance from the foreground points.
    - ``sigma`` = The 1-sigma error bar for <kappa>.
    - ``weight`` = The total weight of the pairs in each bin.
    - ``npairs`` = The total number of pairs in each bin.

:nk_statistic: (str, default='compensated' if ``rand_files`` is given, otherwise 'simple') Which statistic to use for the mean shear as the estimator of the NK correlation function.

    Options are:

    - 'compensated' is simiar to the Landy-Szalay statistic:
      Define:

      - NK = Sum(kappa around data points)
      - RK = Sum(kappa around random points), scaled to be equivalent in effective number as the number of pairs in NK.
      - npairs = number of pairs in NK.

      Then this statistic is ``<kappa>`` = (NK-RK)/npairs
    - 'simple' is the normal version: ``<kappa>`` = NK/npairs

:kk_file_name: (str) The output filename for kappa-kappa correlation function.

    This is the kappa-kappa correlation function.  However, k is really any
    scalar quantity, so it can be used for temperature, size, etc.

    The output columns are:

    - ``R_nom`` = The center of the bin
    - ``meanR`` = The mean separation of the points that went into the bin.
    - ``meanlogR`` = The mean log(R) of the points that went into the bin.
    - ``xi`` = The correlation function <k k>
    - ``sigma_xi`` = The 1-sigma error bar for xi.
    - ``weight`` = The total weight of the pairs in each bin.
    - ``npairs`` = The total number of pairs in each bin.

:kg_file_name: (str) The output filename for kappa-shear correlation function.

    This is the kappa-shear correlation function.  Essentially, this is just
    galaxy-galaxy lensing, weighting the tangential shears by the foreground
    kappa values.

    The output columns are:

    - ``R_nom`` = The center of the bin
    - ``meanR`` = The mean separation of the points that went into the bin.
    - ``meanlogR`` = The mean log(R) of the points that went into the bin.
    - ``kgamT`` = The kappa-weighted mean tangential shear.
    - ``kgamX`` = The kappa-weighted shear component 45 degrees from the tangential direction.
    - ``sigma`` = The 1-sigma error bar for ``kgamT`` and ``kgamX``.
    - ``weight`` = The total weight of the pairs in each bin.
    - ``npairs`` = The total number of pairs in each bin.

:nnn_file_name: (str) The output filename for count-count-count correlation function.

    This is three-point correlation function of number counts.

    The output columns are:

    - ``R_nom`` = The center of the bin in R = d2 where d1 > d2 > d3
    - ``u_nom`` = The center of the bin in u = d3/d2
    - ``v_nom`` = The center of the bin in v = +-(d1-d2)/d3
    - ``meand1`` = The mean value of d1 for the triangles in each bin
    - ``meanlogd1`` = The mean value of log(d1) for the triangles in each bin
    - ``meand2`` = The mean value of d2 for the triangles in each bin
    - ``meanlogd2`` = The mean value of log(d2) for the triangles in each bin
    - ``meand3`` = The mean value of d3 for the triangles in each bin
    - ``meanlogd3`` = The mean value of log(d3) for the triangles in each bin
    - ``zeta`` = The correlation function.
    - ``sigma_zeta`` = The 1-sigma error bar for zeta.
    - ``DDD``, ``RRR`` = The raw numbers of triangles for the data and randoms
    - ``DDR``, ``DRD``, ``RDD``, ``DRR``, ``RDR``, ``RRD`` (if ``nn_statistic=compensated``) = The cross terms between data and random.

:nnn_statistic: (str, default='compensated') Which statistic to use for xi as the estimator of the NNN correlation function.

    Options are:

    - 'compensated' is the Szapudi & Szalay (1998) estimator:
      zeta = (DDD-DDR-DRD-RDD+DRR+RDR+RRD-RRR)/RRR
    - 'simple' is the older version: zeta = (DDD/RRR - 1), although this is not actually
      an estimator of zeta.  Rather, it estimates zeta(d1,d2,d3) + xi(d1) + xi(d2) + xi(d3).

:ggg_file_name: (str) The output filename for shear-shear-shear correlation function.

    This is the shear three-point correlation function.  We use the "natural components"
    as suggested by Schenider & Lombardi (2003): Gamma_0, Gamma_1, Gamma_2, Gamma_3.
    All are complex-valued functions of (d1,d2,d3).  The offer several options for the projection
    direction.  We choose to use the triangle centroid as the reference point.

    The output columns are:

    - ``R_nom`` = The center of the bin in R = d2 where d1 > d2 > d3
    - ``u_nom`` = The center of the bin in u = d3/d2
    - ``v_nom`` = The center of the bin in v = +-(d1-d2)/d3
    - ``meand1`` = The mean value of d1 for the triangles in each bin
    - ``meanlogd1`` = The mean value of log(d1) for the triangles in each bin
    - ``meand2`` = The mean value of d2 for the triangles in each bin
    - ``meanlogd2`` = The mean value of log(d2) for the triangles in each bin
    - ``meand3`` = The mean value of d3 for the triangles in each bin
    - ``meanlogd3`` = The mean value of log(d3) for the triangles in each bin
    - ``gam0r`` = The real part of Gamma_0.
    - ``gam0i`` = The imag part of Gamma_0.
    - ``gam1r`` = The real part of Gamma_1.
    - ``gam1i`` = The imag part of Gamma_1.
    - ``gam2r`` = The real part of Gamma_2.
    - ``gam2i`` = The imag part of Gamma_2.
    - ``gam3r`` = The real part of Gamma_3.
    - ``gam3i`` = The imag part of Gamma_3.
    - ``sigma_gam`` = The 1-sigma error bar for the Gamma values.
    - ``weight`` = The total weight of the triangles in each bin.
    - ``ntri`` = The total number of triangles in each bin.

:kkk_file_name: (str) The output filename for kappa-kappa-kappa correlation function.

    This is the three-point correlation function of a scalar field.

    The output columns are:

    - ``R_nom`` = The center of the bin in R = d2 where d1 > d2 > d3
    - ``u_nom`` = The center of the bin in u = d3/d2
    - ``v_nom`` = The center of the bin in v = +-(d1-d2)/d3
    - ``meand1`` = The mean value of d1 for the triangles in each bin
    - ``meanlogd1`` = The mean value of log(d1) for the triangles in each bin
    - ``meand2`` = The mean value of d2 for the triangles in each bin
    - ``meanlogd2`` = The mean value of log(d2) for the triangles in each bin
    - ``meand3`` = The mean value of d3 for the triangles in each bin
    - ``meanlogd3`` = The mean value of log(d3) for the triangles in each bin
    - ``zeta`` = The correlation function.
    - ``sigma_zeta`` = The 1-sigma error bar for zeta.
    - ``weight`` = The total weight of the triangles in each bin.
    - ``ntri`` = The total number of triangles in each bin.

:precision: (int) The number of digits after the decimal in the output.

    All output quantities are printed using scientific notation, so this sets
    the number of digits output for all values.  The default precision is 4.
    So if you want more (or less) precise values, you can set this to something
    else.


Derived output quantities
-------------------------

The rest of these output files are calculated based on one or more correlation
functions.

:m2_file_name: (str) The output filename for the aperture mass statistics.

    This file outputs the aperture mass variance and related quantities,
    derived from the shear-shear correlation function.

    The output columns are:

    - ``R`` = The radius of the aperture.  (Spaced the same way as  ``R_nom`` is in the correlation function output files.
    - ``Mapsq`` = The E-mode aperture mass variance for each radius R.
    - ``Mxsq`` = The B-mode aperture mass variance.
    - ``MMxa``, ``MMxb`` = Two semi-independent estimate for the E-B cross term.  (Both should be consistent with zero for parity invariance shear fields.)
    - ``sig_map`` = The 1-sigma error bar for these values.
    - ``Gamsq`` = The variance of the top-hat weighted mean shear in apertures of the given radius R.
    - ``sig_gam`` = The 1-sigma error bar for ``Gamsq``.

:m2_uform: (str, default='Crittenden') The function form of the aperture

    The form of the aperture mass statistic popularized by Schneider is

        U = 9/Pi (1-r^2) (1/3-r^2)
        Q = 6/Pi r^2 (1-r^2)

    However, in many ways the form used by Crittenden:

        U = 1/2Pi (1-r^2) exp(-r^2/2)
        Q = 1/4Pi r^2 exp(-r^2/2)

    is easier to use.  For example, the skewness of the aperture mass
    has a closed form solution in terms of the 3-point function for the
    Crittenden form, but no such formula is known for the Schneider form.

    The ``m2_uform`` parameter allows you to switch between the two forms,
    at least for 2-point applications.  (You will get an error if you
    try to use 'Schneider' with the m3 output.)

:nm_file_name: (str) The output filename for <N Map> and related values.

    This file outputs the correlation of the aperture mass with the
    aperture-smoothed density field, derived from the count-shear correlation
    function.

    The output columns are:

    - ``R`` = The radius of the aperture.  (Spaced the same way as  ``R_nom`` is in the correlation function output files.
    - ``NMap`` = The E-mode aperture mass correlated with the density smoothed with the same aperture profile as the aperture mass statistic uses.
    - ``NMx`` = The corresponding B-mode statistic.
    - ``sig_nmap`` = The 1-sigma error bar for these values.

:norm_file_name: (str) The output filename for <Nap Map>^2/<Nap^2><Map^2> and related values.

    This file outputs the <Nap Map> values normalized by <Nap^2><Map^2>.  This
    provides an estimate of the correlation coefficient, r.

    The output columns are:

    - ``R`` = The radius of the aperture.  (Spaced the same way as  ``R_nom`` is in the correlation function output files.
    - ``NMap`` = The E-mode aperture mass correlated with the density smoothed with the same aperture profile as the aperture mass statistic uses.
    - ``NMx`` = The corresponding B-mode statistic.
    - ``sig_nmap`` = The 1-sigma error bar for these values.
    - ``Napsq`` = The variance of the aperture-weighted galaxy density.
    - ``sig_napsq`` = The 1-sigma error bar for <Nap^2>.
    - ``Mapsq`` = The aperture mass variance.
    - ``sig_mapsq`` = The 1-sigma error bar for <Map^2>.
    - ``NMap_norm`` = <Nap Map>^2 / (<Nap^2> <Map^2>)
    - ``sig_norm`` = The 1-sigma error bar for this value.
    - ``Nsq_Mapsq`` = <Nap^2> / <Map^2>
    - ``sig_nn_mm`` = The 1-sigma error bar for this value.


Miscellaneous parameters
------------------------

:verbose: (int, default=1) How verbose the code should be during processing.

    - 0 = no output unless there is an error
    - 1 = output warnings
    - 2 = output progress information
    - 3 = output extra debugging lines

    This is overridden by the ``-v`` command line argument for the `corr2` executable.

:log_file: (str, default=None) Where to write the logging information.

    The default is to write lines to the screen, but this option allows you to
    write them to a file instead.  With the `corr2` executable, this can also be
    specified with the ``-l`` command line argument.

:output_dots: (bool, default=(``verbose``>=2)) Whether to output progress dots during the
    calculation of the correlation function.

:split_method: (str, default='mean') Which method to use for splitting cells.

    When building the tree, there are three obvious choices for how to split a set
    of points into two chld cells.  The direction is always taken to be the
    coordinate direction with the largest extent.  Then, in that direction,
    you can split at the mean value, the median value, or the "middle" =
    (xmin+xmax)/2.  To select among these, ``split_method`` may be given as
    "mean", "median", or "middle" respectively.

    In addition, sometimes it may be useful to inject some randomness into the
    tree construction to study how much the results depend on the specific splitting
    used.  For that purpose, there is also the option to set ``split_method`` = 'random',
    which will choose a random point in the middle two quartiles of the range.

:min_top: (int, default=3) The minimum number of top layers to use when setting up the field.

    The OpenMP parallelization happens over the top level cells, so setting this > 0
    ensures that there will be multiple jobs to be run in parallel.  For systems with
    very many cores, it may be helpful to set this larger than the default value of 3.

:max_top: (int, default=10) The maximum number of top layers to use when setting up the field.

    The top-level cells are the cells where each calculation job starts.  There will
    typically be of order 2^max_top top-level cells.

:num_threads: (int, default=0) How many (OpenMP) threads should be used.

    The default is to try to determine the number of cpu cores your system has
    and use that many threads.

