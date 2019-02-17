/* Copyright (c) 2003-2015 by Mike Jarvis
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

extern void* BuildNNNCorr(int bin_type,
                          double minsep, double maxsep, int nbins, double binsize, double b,
                          double minu, double maxu, int nubins, double ubinsize, double bu,
                          double minv, double maxv, int nvbins, double vbinsize, double bv,
                          double minrpar, double maxrpar,
                          double* meand1, double* meanlogd1, double* meand2, double* meanlogd2,
                          double* meand3, double* meanlogd3, double* meanu, double* meanv,
                          double* weight, double* ntri);
extern void* BuildKKKCorr(int bin_type,
                          double minsep, double maxsep, int nbins, double binsize, double b,
                          double minu, double maxu, int nubins, double ubinsize, double bu,
                          double minv, double maxv, int nvbins, double vbinsize, double bv,
                          double minrpar, double maxrpar,
                          double* zeta,
                          double* meand1, double* meanlogd1, double* meand2, double* meanlogd2,
                          double* meand3, double* meanlogd3, double* meanu, double* meanv,
                          double* weight, double* ntri);
extern void* BuildGGGCorr(int bin_type,
                          double minsep, double maxsep, int nbins, double binsize, double b,
                          double minu, double maxu, int nubins, double ubinsize, double bu,
                          double minv, double maxv, int nvbins, double vbinsize, double bv,
                          double minrpar, double maxrpar,
                          double* gam0, double* gam0_im, double* gam1, double* gam1_im,
                          double* gam2, double* gam2_im, double* gam3, double* gam3_im,
                          double* meand1, double* meanlogd1, double* meand2, double* meanlogd2,
                          double* meand3, double* meanlogd3, double* meanu, double* meanv,
                          double* weight, double* ntri);

extern void DestroyNNNCorr(void* corr, int bin_type);
extern void DestroyKKKCorr(void* corr, int bin_type);
extern void DestroyGGGCorr(void* corr, int bin_type);

extern void ProcessAutoNNN(void* corr, void* field, int dots, int coord, int bin_type, int metric);
extern void ProcessAutoKKK(void* corr, void* field, int dots, int coord, int bin_type, int metric);
extern void ProcessAutoGGG(void* corr, void* field, int dots, int coord, int bin_type, int metric);

extern void ProcessCrossNNN(void* corr, void* field1, void* field2, void* field3, int dots,
                            int coord, int bin_type, int metric);
extern void ProcessCrossKKK(void* corr, void* field1, void* field2, void* field3, int dots,
                            int coord, int bin_type, int metric);
extern void ProcessCrossGGG(void* corr, void* field1, void* field2, void* field3, int dots,
                            int coord, int bin_type, int metric);
