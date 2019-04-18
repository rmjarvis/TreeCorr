// A list of all the valid key words that can appear in the config file.
// This follows the order that the parameters appear in the Read.me file.
// Ideally, I would create this file automatically by parsing the Read.me
// file and extracting the valid key words to make sure the documentation
// keeps up to date with the code.  But I don't do that yet.
// (Even better would be to automatically scan to code for usages of config 
// and make sure the two are in sync, but that's a lot harder to do.)

const int n_valid_keys = 76;
std::string ar_valid_keys[n_valid_keys] = {

// Parameters about the input file
"file_name" , // (str or list) The file(s) with the galaxy data.
"do_auto_corr" , // (bool, default=false) Whether to do auto-correlations within
                 // a list of files.
"do_cross_corr" , // (bool, default=true)) Whether to do cross-correlations within 
                  // a list of files.
"file_name2" , // (str or list) The file(s) to use for the second field for a 
               // cross-correlation.
"rand_file_name" , // (str or list) For NN correlations, a list of random files.
"rand_file_name2" , // (str or list) The randoms for the second field for a 
                    // cross-correlation.
"file_list" , // (str) A text file with file names in lieu of file_name.
"file_list2" , // (str) A text file with file names in lieu of file_name2.
"rand_file_list" , // (str) A text file with file names in lieu of rand_file_name.
"rand_file_list2" , // (str) A text file with file names in lieu of rand_file_name2.
"file_type" , // (ASCII or FITS) The file type of the input files.
"delimiter" , // (str, default = '\0') The delimeter between input valus in an 
              // ASCII catalog.
"comment_marker" , // (str, default = '#') The first (non-whitespace) character 
                   // of comment lines in an input ASCII catalog.
"first_row" , // (int, default=1)
"last_row" , // (int, default=-1)
"x_col" , // (int/str) Which column to use for x
"y_col" , // (int/str) Which column to use for y
"ra_col" , // (int/str) Which column to use for ra
"dec_col" , // (int/str) Which column to use for dec
"x_units" , // (str, default=arcsec) The units of x values.
"y_units" , // (str, default=arcsec) The units of y values.
"ra_units" , // (str) The units of ra values.
"dec_units" , // (str) The units of dec values.
"g1_col" , // (int/str) Which column to use for g1
"g2_col" , // (int/str) Which column to use for g2
"k_col" , // (int/str) Which column to use for kappa 
"w_col" , // (int/str) Which column to use for the weight (if any)
"flag_col" , // (int/str) Which column to use for a flag (if any)
"ignore_flag", // (int) Ignore objects with flag & ignore_flag != 0 (bitwise &)
"ok_flag", // (int) Ignore objects with flag & ~ok_flag != 0 (bitwise &, ~)
"hdu", // (int) Which HDU in a fits file to use rather than hdu=1
"x_hdu", // (int) Which HDU to use for the x_col
"y_hdu", // (int) Which HDU to use for the y_col
"ra_hdu", // (int) Which HDU to use for the ra_col
"dec_hdu", // (int) Which HDU to use for the dec_col
"g1_hdu", // (int) Which HDU to use for the g1_col
"g2_hdu", // (int) Which HDU to use for the g2_col
"k_hdu", // (int) Which HDU to use for the k_col
"w_hdu", // (int) Which HDU to use for the w_col
"flag_hdu", // (int) Which HDU to use for the flag_col
"flip_g1" , // (bool, default=false) Whether to flip the sign of g1
"flip_g2" , // (bool, default=false) Whether to flip the sign of g2
"pairwise" , // (bool, default=false) Whether to do a pair-wise cross-correlation 
"project" , // (bool, default=false) Whether to do a tangent plane projection
"project_ra" , // (float) The ra of the tangent point for projection.
"project_dec" , // (float) The dec of the tangent point for projection.


//Parameters about the binned correlation function to be calculated

"min_sep" , // (float) The minimum separation to include in the output.
"max_sep" , // (float) The maximum separation to include in the output.
"nbins" , // (int) The number of output bins to use.
"bin_size" , // (float) The size of the output bins in log(sep).
"sep_units" , // (str, default=arcsec) The units to use for min_sep and max_sep.
"bin_slop" , // (float, default=1) The fraction of a bin width by which it is 
             // ok to let the pairs miss the correct bin.
"smooth_scale" , // (float) An optional smoothing scale to smooth the output values.

//Parameters about the output file(s)

"n2_file_name" , // (str) The output filename for point-point correlation function.
"n2_statistic" , // (str, default=compensated) Which statistic to use for omega as
                 // the estimator fo the NN correlation function.
"ng_file_name" , // (str) The output filename for point-shear correlation function.
"ng_statistic" , // (str, default=compensated if rand_files is given, otherwise 
                 // simple) Which statistic to use for the mean shear as the
                 // estimator of the NE correlation function.
"g2_file_name" , // (str) The output filename for shear-shear correlation function.
"nk_file_name" , // (str) The output filename for point-kappa correlation function.
"nk_statistic" , // (str, default=compensated if rand_files is given, otherwise 
                 // simple) Which statistic to use for the mean shear as the
                 // estimator of the NK correlation function.
"k2_file_name" , // (str) The output filename for kappa-kappa correlation function.
"kg_file_name" , // (str) The output filename for kappa-shear correlation function.
"precision"    , // (int) The number of digits after the decimal in the output.

//Derived output quantities

"m2_file_name" , // (str) The output filename for the aperture mass statistics.
"m2_uform" , // (str, default=Crittenden) The function form of the aperture
"nm_file_name" , // (str) The output filename for <N Map> and related values.
"norm_file_name" , // (str) The output filename for <N Map>^2/<N^2><Map^2>
                   // and related values.

//Miscellaneous parameters

"verbose" , // (int, default=0) How verbose the code should be during processing.
"num_threads" , // (int, default=auto) How many (OpenMP) threads should be used.
"split_method" , // (str, default=mean) Which method to use for splitting cells.

// Deprecated, but still allowed:
"e1_col" ,
"e2_col" ,
"e2_file_name" ,
"ne_file_name" ,
"ne_statistic" ,
"ke_file_name" ,

};
