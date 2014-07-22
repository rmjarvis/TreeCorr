TreeCorr
========

The repository will become live with the release of version 3.0, which is still in 
development.  Until then, please use the latet 2.x release from 

https://code.google.com/p/mjarvis/

The code is licensed under a FreeBSD license.  Essentially, you can use the 
code in any way you want, but if you distribute it, you need to include the 
file `TreeCorr_LICENSE` with the distribution.  See that file for details.

Overview
========

Code for efficiently computing 2-point correlation functions on the celestial sphere.

This software package uses ball trees (similar to kd trees) to efficiently
compute 2 and 3-point correlation functions.

- Current version is 2.6, which is hosted at: 
    https://code.google.com/p/mjarvis/
- 2-point correlations may be auto-correlations or cross-correlations.
- Includes shear-shear, count-shear, count-count, kappa-kappa, etc.
  (Any combination of shear, kappa, and counts.)
- 2-point functions can be done with correct curved-sky calculation using
  RA, Dec coordinates or on a Euclidean tangent plane.
- Can use OpenMP to run in parallel on multi-core machines.
- Approximate running time for 2-point shear-shear is ~30 sec * (N/10^6) / core.
- 3-point functions have not yet been migrated to the new API, but they should be
  available soon.
- Reference: Jarvis, Bernstein, & Jain, 2004, MNRAS, 352, 338


Two-point Correlations
======================

This software is able to compute several varieties of two-point correlations:

NN = the normal two point correlation function of things like 2dF that
     correlate the galaxy counts at each position.
NG = correlation of counts with shear.  This is what is often called
     galaxy-galaxy lensing.
GG = two-point shear correlation function.
NK = correlation of counts with kappa.  While kappa is nominally the lensing
     convergence, it could really be any scalar quantity, like temperature,
     size, etc.
KG = correlation of convergence with shear.  Like the NG calculation, but 
     weighting the pairs by the convergence values the foreground points.
KK = two-point kappa correlation function.


Running corr2
=============

The executable corr2 takes one required command-line argument, which is the 
name of a configuration file:

    corr2 config_file

A sample configuration file is provided, called default.params, and see below 
for a fairly complete list of 
includes much of this 

You can also specify parameters on the command line after the name of 
the configuration file. e.g.:

    corr2 config_file file_name=file1.dat g2_file_name=file1.out
    corr2 config_file file_name=file2.dat g2_file_name=file2.out
    ...

This can be useful when running the program from a script for lots of input 
files.


# Parameters about the input file

    file_name = (str or list) The file(s) with the galaxy data.

        You an also specify two files here, in which case the program calculates a 
        cross-correlation between the two sets of values.  e.g.
        file_name = file1.dat file2.dat

        If you are specifying this on the command line, you'll need to put 
        quotes around the names, or it won't be parsed correctly:
        filename="file1.dat file2.dat"


    do_auto_corr = (bool, default=false) Whether to do auto-correlations within
        a list of files.
    do_cross_corr = (bool, default=true)) Whether to do cross-correlations within 
        a list of files.

        If there are more than two names in the file_name paramter, then the code 
        will normally calculate all the pair-wise cross-correlations of each pair 
        i,j in the list.  The default is to only do cross-correlations, but not 
        auto-correlations.  You may change that behavior by changing do_auto_corr 
        or do_cross_corr.  


    file_name2 = (str or list) The file(s) to use for the second field for a 
        cross-correlation.

        If you want to cross-correlate one set of files with another, then the 
        above file_name parameter isn't sufficient.  Instead, you would list 
        the first set of files in file_name, and the second set in file_name2.
        Of course, each list may only contain one file each, so this is another
        way to specify two files to be cross-correlated.


    rand_file_name = (str or list) For NN correlations, a list of random files.
    rand_file_name2 = (str or list) The randoms for the second field for a 
        cross correlation.

        When doing NN correlations, you need to account for masks and variable
        depth by providing a file or list of files that correspond to a uniform-
        density field as observed with the same masking and other observational
        details.  For cross-correlations, you need to provide both of the above
        values to separately calibrate the first and second fields.
    
    file_list = (str) A text file with file names in lieu of file_name.
    file_list2 = (str) A text file with file names in lieu of file_name2.
    rand_file_list = (str) A text file with file names in lieu of rand_file_name.
    rand_file_list2 = (str) A text file with file names in lieu of rand_file_name2.

        If you have a list of file names, it may be cumbersome to list them all
        in the file_name (etc) parameter.  It may be easier to do something like
        `ls *.cat > catlist` and then use `file_list=catlist` as the list of 
        file names to use.  Of course, it is an error to specify both file_list
        and file_name (or any of the other corresponding pairs).

    file_type = (ASCII or FITS) The file type of the input files.
    delimiter = (str, default = '\0') The delimeter between input values in an 
        ASCII catalog.
    comment_marker = (str, default = '#') The first (non-whitespace) character 
        of comment lines in an input ASCII catalog.

        The default file type is normally ASCII.  However, if the file name 
        includes ".fit" in it, then a fits binary table is assumed.
        You can override this behavior using file_type.

        Furthermore, you may specify a delimiter for ASCII catalogs if desired.
        e.g. delimiter=',' for a comma-separated value file.  Similarly, 
        comment lines usually begin with '#', but you may specify something 
        different if necessary.

    hdu = (int) Normally if you are using a fits file, the binary fits table is
        taken from the first extension, HDU 1.  If you want to read from a
        different HDU, you can specify which one to use with the hdu parameter.

    first_row = (int, default=1)
    last_row = (int, default=-1)

        You can optionally not use all the rows in the input file.
        You may specify first_row, last_row, or both to limit the rows being used.
        The rows are numbered starting with 1.  If last_row is not positive, it 
        means to use all the rows (starting with first_row).

    x_col = (int/str) Which column to use for x
    y_col = (int/str) Which column to use for y
    ra_col = (int/str) Which column to use for ra
    dec_col = (int/str) Which column to use for dec

        For the positions of the objects, you can specify either x,y values, which 
        imply a flat-sky approximation has already been performed (or ignored),
        or ra,dec values, which are of course positions on the curved sky.

        For ASCII files, the columns are specified by number, starting with 1 being
        the first column (not 0!).  
        For FITS files, the columns are specified by name, not number.

    x_units = (str, default=arcsec) The units of x values.
    y_units = (str, default=arcsec) The units of y values.
    ra_units = (str) The units of ra values.
    dec_units = (str) The units of dec values.

        All distances on the sky include a "units" parameter to specify what in 
        units the values are specified.  Options for units are radians, hours, 
        degrees, arcmin, arcsec.  If omitted, arcsec is assumed if you are using 
        x,y.  But for ra, dec the units field is required.


    g1_col = (int/str) Which column to use for g1
    g2_col = (int/str) Which column to use for g2

        If you are doing one of the shear correlation functions (i.e. NG, KG, GG),
        then you need to specify the shear estimates of the corresponding galaxies.
        The g1,g2 values are taken to be reduced shear values.  They should be
        unbiases estimators of g1,g2, so they are allowed to exceed |g| = 1.
        (This is required for some methods to produce unbiased estimates.

    k_col = (int/str) Which column to use for kappa 

        If you are doing one of the kappa correlation functions (i.e. NK, KG, KK),
        then you need to specify the column to use for kappa.  While kappa is 
        nominally the lensing convergence, it could really be any scalar quantity,
        like temperature, size, etc.
    
    w_col = (int/str) Which column to use for the weight (if any)

        The weight column is optional. If omitted, all weights are taken to be 1.

    flag_col = (int/str) Which column to use for the weight (if any)
    ignore_flag = (int) What flag(s) should be ignored.
    ok_flag = (int) What flag(s) are ok to use.

        The code can be set to ignore objects with a particular flag value if desired.
        Some codes output a flag along with the shear value.  Typically any flag != 0
        should be ignored, but you can optionally have the code ignore only particular 
        flags, treating the flag value as a bit mask.  If ignore_flag is set to 
        something, then objects with (flag & ignore_flag != 0) will be ignored.
        If ok_flag is set, then objects with (flag & ~ok_flag != 0) will be ignored.
        The default is equivalent to ok_flag = 0, which ignores any flag != 0.

    flip_g1 = (bool, default=false) Whether to flip the sign of g1
    flip_g2 = (bool, default=false) Whether to flip the sign of g2

        Sometimes there are issues with the sign conventions of gamma.  If you 
        need to flip the sign of g1 or g2, you may do that with flip_g1 or flip_g2 
        (or both).

    x_hdu = (int) Which HDU to use for the x_col
    y_hdu = (int) Which HDU to use for the y_col
    ra_hdu = (int) Which HDU to use for the ra_col
    dec_hdu = (int) Which HDU to use for the dec_col
    g1_hdu = (int) Which HDU to use for the g1_col
    g2_hdu = (int) Which HDU to use for the g2_col
    k_hdu = (int) Which HDU to use for the k_col
    w_hdu = (int) Which HDU to use for the w_col
    flag_hdu = (int) Which HDU to use for the flag_col

        If you want to use an HDU other than the first one, normally you would 
        specify which fits extension to use with the hdu parameter.  However, if 
        different columns need to come from different HDUs, then you can override
        the default (given by hdu or 1 if there is no hdu parameter) for each
        column separately.  

    Notes about the above parameters: 
        - If you are cross-correlating two files with different formats, you may 
        set any of the above items from file_type to flip_g2 as a two element 
        list (i.e. two values separated by a space).  In this case, the first 
        item refers to the file(s) in file_name, and the second item refers 
        to the file(s) in files_name2.
      
        - You may not mix (x,y) columns with (ra,dec) columns , since its meaning 
        would be ambiguous.

        - If you don't need a particular column for one of the files, you may 
        use 0 to indicate not to read that column.  This is true both for 
        ASCII and FITS input catalogs.

        - Also, if the given column only applies to one of the two input files
        (e.g. k_col for an n-kappa cross-correlation) then you may specify just
        the column name or number for the file to which it does apply.

    pairwise = (bool, default=false) Whether to do a pair-wise cross-correlation 
        of the corresponding lines in the two files, rather than correlating
        all lines in one with all the lines in the other.
           
        That is, the data in each (non-comment) line of the first file is 
        correlated with the same corresponding line in the second file.  This only 
        applies to cross-correlations.  

        An example of why this might be useful is to measure the tangent shear
        of galaxies around the field center of the exposures in which they were
        observed.  You can build a catalog of the field centers corresponding to
        each galaxy in the catalog, then a pairwise correlation option will only 
        use the corresponding centers where the galaxy was observed, rather than
        using all the field centers in the whole survey.

    project = (bool, default=false) Whether to do a tangent plane projection for
        handling the curved sky values.

        The native RA, Dec code is almost as fast as the flat-sky code, so it is
        generally preferable to let the code handle RA and Dec using the correct
        spherical geometry formulae.  But if you want, you can instead project the 
        RA, Dec values onto a tangent plane.  This is probable most useful for 
        investigating bugs in the curved sky code, so I suspect most users will
        not want to use this feature.

    project_ra = (float) The ra of the tangent point for projection.
    project_dec = (float) The dec of the tangent point for projection.

        The default tangent point for the projection is the average position.
        However, this may be inappropriate for some reason, so you may specify the
        projection point with project_ra, project_dec.
        (These use the same units specified for ra_units, dec_units.)


# Parameters about the binned correlation function to be calculated

    min_sep = (float) The minimum separation to include in the output.
    max_sep = (float) The maximum separation to include in the output.
    nbins = (int) The number of output bins to use.
    bin_size = (float) The size of the output bins in log(sep).

        The bins for the histogram may be defined by setting any 3 of the above 4 
        parameters.  The fourth one is automatically calculated from the values
        of the other three.

        There is one exception.  If you set min_sep, max_sep, and bin_size, 
        then it won't generally be the case that the corresponding number of 
        bins is an integer.  So the code will increase max_sep slightly to make
        sure the total range is an integer number of bins.

    sep_units = (str, default=arcsec) The units to use for min_sep and max_sep.

        sep_units is also the units of R in the output file.  The default only
        makes sense if using x,y.  For Ra, Dec values, you need to specify
        sep_units explicitly.

    bin_slop = (float, default=1) The fraction of a bin width by which it is 
        ok to let the pairs miss the correct bin.

        The code normally determines when to stop traversing the tree when all of the 
        distance pairs for the two nodes have a spread in distance that is less than the 
        bin size.  i.e. the error in the tree traversal is less than the uncertainty 
        induced by just binning the results into a histogram.  This factor can be changed
        by the parameter bin_slop.  It is probably best to keep it at 1, but if you want to
        make the code more conservative, you can decrease it, in which case the error 
        from using the tree nodes will be less than the error in the histogram binning.
        (In practice, if you are going to do this, you are probably better off just 
        decreasing the bin_size instead and leaving bin_slop=1.)

        Note, if you set bin_slop=0, then the code will effectively do a brute-force
        calculation, since it will branch all the way to each leaf of the tree.

    smooth_scale = (float) An optional smoothing scale to smooth the output values.

        In addition to the raw output, the code will also optionally output a smoothed 
        version of the correlation functions, which is better for plotting.
        The smoothing scale is specified as smooth_scale.
        If omitted or smooth_scale = 0, then no smoothing will be done.


# Parameters about the output file(s)

    The kind of correlation function that the code will calculate is based on 
    which output file(s) you specify.  It will do the calculation(s) relevant for 
    each output file you set.  For each output file, the first line of the output 
    says what the columns are.  See the descriptions below for more information
    about the output columns.

    CAVEAT: The error estimates for all quantities only include the propagation 
            of the shot noise and shape noise through the calculation.  It 
            does not include sample variance, which is almost always important.
            So the error values should always be treated as an underestimate
            of the true error bars.

    n2_file_name = (str) The output filename for point-point correlation function.

        This is the normal density two-point correlation function.
        
        The output columns are:
        - R_nominal = The center of the bin
        - <R> = The mean separation of the points that went into the bin.  
            Technically, since we bin in log(R), this is really exp( <log(R)> ).
        - omega = The NN correlation function.
        - sig_omega = The 1-sigma error bar for omega.
        - DD, RR = The raw numbers of pairs for the data and randoms
        - DR, RD (if n2_statistic=compensated) = The cross terms between data and 
            random.  Note: For an auto-correlation, DR and RD are identical, but 
            for cross-correlations, they may be different.


    n2_statistic = (str, default=compensated) Which statistic to use for omega as
        the estimator of the NN correlation function.  

        Options are:
        - "compensated" is the now-normal Landy-Szalay statistic:
            omega = (DD-2DR+RR)/RR, or for cross-correlations, (DD-DR-RD+RR)/RR
        - "simple" is the older version: omega = (DD/RR - 1)


    ng_file_name = (str) The output filename for point-shear correlation function.

        This is the point-shear correlation function, often called galaxy-galaxy
        lensing.
        
        The output columns are:
        - R_nominal = The center of the bin
        - <R> = The mean separation of the points that went into the bin.  
          Technically, since we bin in log(R), this is really exp( <log(R)> ).
        - <gamT> = The mean tangential shear with respect to the point in question.
        - <gamX> = The shear component 45 degrees from the tangential direction.
        - sig = The 1-sigma error bar for <gamT> and <gamX>.
        - weight (if ng_statistic = simple) = The total weight of the pairs in each bin.
        - npairs (if ng_statistic = simple) = The total number of pairs in each bin.
        - gamT_d (if ng_statistic = compensated) = The raw <gamT> from just the data.
        - gamX_d (if ng_statistic = compensated) = The raw <gamX> from just the data.
        - weight_d (if ng_statistic = compensated) = The raw weight from just the data.
        - npairs_d (if ng_statistic = compensated) = The raw npairs from just the data.
        - gamT_r (if ng_statistic = compensated) = The <gamT> from the randoms.
        - gamX_r (if ng_statistic = compensated) = The <gamX> from the randoms.
        - weight_r (if ng_statistic = compensated) = The weight from the randoms.
        - npairs_r (if ng_statistic = compensated) = The npairs from the randoms.
        - R_sm (if smooth_scale is set) = <R> for the smoothed values
        - gamT_sm (if smooth_scale is set) = <gamT> smoothed over the appropriate scale.
        - sig_sm (if smooth_scale is set) = The 1-sigma error bar for gamT_sm.


    ng_statistic = (str, default=compensated if rand_files is given, otherwise 
        simple) Which statistic to use for the mean shear as the 
        estimator of the NG correlation function. 

        Options are:
        - "compensated" is simiar to the Landy-Szalay statistic:
            Let DG = Sum(gamma around data points)
                RG = Sum(gamma around random points), scaled to be equivalent in
                     effective number as the number of pairs in DG.
                npairs = number of pairs in DG.
            Then this statistic is <gamma> = (DG-RG)/npairs
        - "simple" is the normal version: <gamma> = DG/npairs


    g2_file_name = (str) The output filename for shear-shear correlation function.

        This is the shear-shear correlation function, used for cosmic shear.
        
        The output columns are:
        - R_nominal = The center of the bin
        - <R> = The mean separation of the points that went into the bin.  
          Technically, since we bin in log(R), this is really exp( <log(R)> ).
        - xi+ = <g1 g1 + g2 g2> where g1 and g2 are measured with respect to the
          line joining the two galaxies.
        - xi- = <g1 g1 - g2 g2> where g1 and g2 are measured with respect to the
          line joining the two galaxies.
        - xi+_im = <g2 g1 - g1 g2>.  In the formulation of xi+ using complex 
            numbers, this is the imaginary component. 
            It should normally be consistent with zero, especially for an
            auto-correlation, because if every pair were counted twice to 
            get each galaxy in both positions, then this would come out 
            exactly zero.
        - xi-_im = <g2 g1 + g1 g2>.  In the formulation of xi- using complex 
            numbers, this is the imaginary component.
            It should be consistent with zero for parity invariant shear 
            fields.
        - sig_xi = The 1-sigma error bar for xi+ and xi-.
        - weight = The total weight of the pairs in each bin.
        - npairs = The total number of pairs in each bin.
        - R_sm (if smooth_scale is set) = <R> for the smoothed values
        - xi+_sm, xi-_sm (if smooth_scale is set) = xi+, xi- smoothed over the 
            appropriate scale.
        - sig_sm (if smooth_scale is set) = The 1-sigma error bar for xi+_sm, xi-_sm.

    nk_file_name = (str) The output filename for point-kappa correlation function.

        This is nominally the kappa version of the ne calculation.  However, k is
        really any scalar quantity, so it can be used for temperature, size, etc.

        The output columns are:
        - R_nominal = The center of the bin
        - <R> = The mean separation of the points that went into the bin.  
          Technically, since we bin in log(R), this is really exp( <log(R)> ).
        - <kappa> = The mean kappa this distance from the foreground points.
        - sig = The 1-sigma error bar for <kappa>.
        - weight (if nk_statistic = simple) = The total weight of the pairs in each bin.
        - npairs (if nk_statistic = simple) = The total number of pairs in each bin.
        - kappa_d (if nk_statistic = compensated) = The raw <kappa> from just the data.
        - weight_d (if nk_statistic = compensated) = The raw weight from just the data.
        - npairs_d (if nk_statistic = compensated) = The raw npairs from just the data.
        - kappa_r (if nk_statistic = compensated) = The <kappa> from the randoms.
        - weight_r (if nk_statistic = compensated) = The weight from the randoms.
        - npairs_r (if nk_statistic = compensated) = The npairs from the randoms.
        - R_sm (if smooth_scale is set) = <R> for the smoothed values
        - kappa_sm (if smooth_scale is set) = <kappa> smoothed over the appropriate scale.
        - sig_sm (if smooth_scale is set) = The 1-sigma error bar for kappasm.


    nk_statistic = (str, default=compensated if rand_files is given, otherwise 
        simple) Which statistic to use for the mean shear as the 
        estimator of the NK correlation function. 

        Options are:
        - "compensated" is simiar to the Landy-Szalay statistic:
            Let DK = Sum(kappa around data points)
                RK = Sum(kappa around random points), scaled to be equivalent in
                    effective number as the number of pairs in DK.
                npairs = number of pairs in DK.
            Then this statistic is <kappa> = (DK-RK)/npairs
        - "simple" is the normal version: <kappa> = DK/npairs


    k2_file_name = (str) The output filename for kappa-kappa correlation function.

        This is the kappa-kappa correlation function.  However, k is really any 
        scalar quantity, so it can be used for temperature, size, etc.
        
        The output columns are:
        - R_nominal = The center of the bin
        - <R> = The mean separation of the points that went into the bin.  
          Technically, since we bin in log(R), this is really exp( <log(R)> ).
        - xi = The correlation function <k k> 
        - sig_xi = The 1-sigma error bar for xi.
        - weight = The total weight of the pairs in each bin.
        - npairs = The total number of pairs in each bin.
        - R_sm (if smooth_scale is set) = <R> for the smoothed values
        - xi_sm (if smooth_scale is set) = xi smoothed over the appropriate scale.
        - sig_sm (if smooth_scale is set) = The 1-sigma error bar for xi.


    kg_file_name = (str) The output filename for kappa-shear correlation function.

        This is the kappa-shear correlation function.  Essentially, this is just
        galaxy-galaxy lensing, weighting the tangential shears by the foreground
        kappa values.
        
        The output columns are:
        - R_nominal = The center of the bin

        - <R> = The mean separation of the points that went into the bin.  
            Technically, since we bin in log(R), this is really exp( <log(R)> ).
        - <kgamT> = The kappa-weighted mean tangential shear.
        - <kgamX> = The kappa-weighted shear component 45 degrees from the 
            tangential direction.
        - sig = The 1-sigma error bar for <kgamT> and <kgamX>.
        - weight (if ng_statistic = simple) = The total weight of the pairs in each bin.
        - npairs (if ng_statistic = simple) = The total number of pairs in each bin.
        - kgamT_d (if ng_statistic = compensated) = The raw <kgamT> from just the data.
        - kgamX_d (if ng_statistic = compensated) = The raw <kgamX> from just the data.
        - weight_d (if ng_statistic = compensated) = The raw weight from just the data.
        - npairs_d (if ng_statistic = compensated) = The raw npairs from just the data.
        - kgamT_r (if ng_statistic = compensated) = The <kgamT> from the randoms.
        - kgamX_r (if ng_statistic = compensated) = The <kgamX> from the randoms.
        - weight_r (if ng_statistic = compensated) = The weight from the randoms.
        - npairs_r (if ng_statistic = compensated) = The npairs from the randoms.
        - R_sm (if smooth_scale is set) = <R> for the smoothed values
        - kgamT_sm (if smooth_scale is set) = <kgamT> smoothed over the appropriate 
            scale.
        - sig_sm (if smooth_scale is set) = The 1-sigma error bar for kgamT_sm.


    precision = (int) The number of digits after the decimal in the output.

        All output quantities are printed using scientific notation, so this sets 
        the number of digits output for all values.  The default precision is 3. 
        So if you want more (or less) precise values, you can set this to something 
        else. 


# Derived output quantities

The rest of these output files are calculated based on one or more correlation 
functions.


    m2_file_name = (str) The output filename for the aperture mass statistics.

        This file outputs the aperture mass variance and related quantities, 
        derived from the shear-shear correlation function.
        
        The output columns are:

        - R = The radius of the aperture.  (Spaced the same way as  R_nominal is 
            in the correlation function output files.
        - <Map^2> = The E-mode aperture mass variance for each radius R.
        - <Mx^2> = The B-mode aperture mass variance.
        - <MMx>(a), <MMx>(b) = Two semi-independent estimate for the E-B cross term.
            (Both should be consistent with zero for parity invariance shear fields.)
        - sig_map = The 1-sigma error bar for these values.
        - <Gam^2> = The variance of the top-hat weighted mean shear in apertures of 
            the given radius R.
        - sig_gam = The 1-sigma error bar for <Gam^2>.

    m2_uform = (str, default=Crittenden) The function form of the aperture

        The form of the aperture mass statistic popularized by Schneider is
            U = 9/Pi (1-r^2) (1/3-r^2)
            Q = 6/Pi r^2 (1-r^2)
        However, in many ways the form used by Crittenden:
            U = 1/2Pi (1-r^2) exp(-r^2/2)
            Q = 1/4Pi r^2 exp(-r^2/2)
        is easier to use.  For example, the skewness of the aperture mass
        has a closed form solution in terms of the 3-point function for the 
        Crittenden form, but no such formula is known for the Schneider form.
    
        The m2_uform parameter allows you to switch between the two forms,
        at least for 2-point applications.  (You will get an error if you
        try to use Schneider with the m3 output.)


    nm_file_name = (str) The output filename for <N Map> and related values.

        This file outputs the correlation of the aperture mass with the 
        aperture-smoothed density field, derived from the point-shear correlation 
        function.
        
        The output columns are:
        - R = The radius of the aperture.  (Spaced the same way as  R_nominal is 
            in the correlation function output files.
        - <NMap> = The E-mode aperture mass correlated with the density smoothed
            with the same aperture profile as the aperture mass statistic uses.
        - <NMx> = The corresponding B-mode statistic.
        - sig_nmap = The 1-sigma error bar for these values.


    norm_file_name = (str) The output filename for <N Map>^2/<N^2><Map^2>
        and related values.

        This file outputs the <N Map> values normalized by <N^2><Map^2>.  This 
        provides an estimate of the correlation coefficient, r.
        
        The output columns are:
        - R = The radius of the aperture.  (Spaced the same way as  R_nominal is 
            in the correlation function output files.
        - <NMap> = The E-mode aperture mass correlated with the density smoothed
            with the same aperture profile as the aperture mass statistic uses.
        - <NMx> = The corresponding B-mode statistic.
        - sig_nm = The 1-sigma error bar for these values.
        - <N^2> = The variance of the aperture-weighted galaxy density.
        - sig_nm = The 1-sigma error bar for <N^2>.
        - <Map^2> = The aperture mass variance.
        - sig_mm = The 1-sigma error bar for <Map^2>.
        - nmnorm = <NMap>^2 / (<N^2> <Map^2> )
        - sig_nmnorm = The 1-sigma error bar for this value.
        - nnnorm = <NN> / <Map^2> 
        - sig_nnnorm = The 1-sigma error bar for this value.


# Miscellaneous parameters

    verbose = (int, default=0) How verbose the code should be during processing.

        0 = no output
        1 = normal output
        2 = extra output


    num_threads = (int, default=auto) How many (OpenMP) threads should be used.

        The default is to let OpenMP determine an appropriate number of threads 
        automatically.  Usually this matches the number of cores your system has.


    split_method = (str, default=mean) Which method to use for splitting cells.

        When building the tree, there are three choices for how to split a set
        of points into two chld cells.  The direction is always taken to be the 
        coordinate direction with the largest extent.  Then, in that direction,
        you can split at the mean value, the median value, or the "middle" =
        (xmin+xmax)/2.  To select among these, split_method may be given as
        "mean", "median", or "middle" respectively.


# Reporting bugs

If you find a bug running the code, please report it at:

    http://code.google.com/p/mjarvis/issues/list

Click "New Issue", which will open up a form for you to fill in with the
details of the problem you are having.

If you would like to request a new feature, then after clicking "New Issue",
you can change the Template to "Feature Request", which will ask more
appropriate questions about the feature you would like me to add.

