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

extern void* BuildCorr3(int d1, int d2, int d3, int bin_type,
                        double minsep, double maxsep, int nbins, double binsize, double b,
                        double minu, double maxu, int nubins, double ubinsize, double bu,
                        double minv, double maxv, int nvbins, double vbinsize, double bv,
                        double minrpar, double maxrpar, double xp, double yp, double zp,
                        double* gam0, double* gam0_im, double* gam1, double* gam1_im,
                        double* gam2, double* gam2_im, double* gam3, double* gam3_im,
                        double* meand1, double* meanlogd1, double* meand2, double* meanlogd2,
                        double* meand3, double* meanlogd3, double* meanu, double* meanv,
                        double* weight, double* ntri);

extern void DestroyCorr3(void* corr, int d1, int d2, int d3, int bin_type);

extern void ProcessAuto3(void* corr, void* field, int dots,
                         int d, int coord, int bin_type, int metric);

extern void ProcessCross3(void* corr, void* field1, void* field2, void* field3, int dots,
                          int d1, int d2, int d3, int coord, int bin_type, int metric);
