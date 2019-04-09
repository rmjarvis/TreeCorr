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

extern void* BuildCorr2(int d1, int d2, int bin_type,
                        double minsep, double maxsep, int nbins, double binsize, double b,
                        double minrpar, double maxrpar, double xp, double yp, double zp,
                        double* xip, double* xip_im, double* xim, double* xim_im,
                        double* meanr, double* meanlogr, double* weight, double* npairs);

extern void DestroyCorr2(void* corr, int d1, int d2, int bin_type);

extern void ProcessAuto2(void* corr, void* field, int dots,
                         int d, int coord, int bin_type, int metric);

extern void ProcessCross2(void* corr, void* field1, void* field2, int dots,
                          int d1, int d2, int coord, int bin_type, int metric);

extern void ProcessPair(void* corr, void* field1, void* field2, int dots,
                        int d1, int d2, int coord, int bin_type, int metric);

extern int SetOMPThreads(int num_threads);

extern long SamplePairs(void* corr, void* field1, void* field2, double min_sep, double max_sep,
                        int d1, int d2, int coords, int bin_type, int metric,
                        long* i1, long* i2, double* sep, int n);
