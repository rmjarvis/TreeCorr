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

extern void* BuildNNCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                         double minrpar, double maxrpar,
                         double* meanr, double* meanlogr, double* weight, double* npairs);
extern void* BuildNKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                         double minrpar, double maxrpar,
                         double* xi,
                         double* meanr, double* meanlogr, double* weight, double* npairs);
extern void* BuildNGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                         double minrpar, double maxrpar,
                         double* xi, double* xi_im,
                         double* meanr, double* meanlogr, double* weight, double* npairs);
extern void* BuildKKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                         double minrpar, double maxrpar,
                         double* xi,
                         double* meanr, double* meanlogr, double* weight, double* npairs);
extern void* BuildKGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                         double minrpar, double maxrpar,
                         double* xi, double* xi_im,
                         double* meanr, double* meanlogr, double* weight, double* npairs);
extern void* BuildGGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                         double minrpar, double maxrpar,
                         double* xip, double* xip_im, double* xim, double* xim_im,
                         double* meanr, double* meanlogr, double* weight, double* npairs);

extern void DestroyNNCorr(void* corr);
extern void DestroyNKCorr(void* corr);
extern void DestroyNGCorr(void* corr);
extern void DestroyKKCorr(void* corr);
extern void DestroyKGCorr(void* corr);
extern void DestroyGGCorr(void* corr);

extern void ProcessAutoNN(void* corr, void* field, int dots, int coord, int metric);
extern void ProcessAutoKK(void* corr, void* field, int dots, int coord, int metric);
extern void ProcessAutoGG(void* corr, void* field, int dots, int coord, int metric);

extern void ProcessCrossNN(void* corr, void* field1, void* field2, int dots, int coord, int metric);
extern void ProcessCrossNK(void* corr, void* field1, void* field2, int dots, int coord, int metric);
extern void ProcessCrossNG(void* corr, void* field1, void* field2, int dots, int coord, int metric);
extern void ProcessCrossKK(void* corr, void* field1, void* field2, int dots, int coord, int metric);
extern void ProcessCrossKG(void* corr, void* field1, void* field2, int dots, int coord, int metric);
extern void ProcessCrossGG(void* corr, void* field1, void* field2, int dots, int coord, int metric);

extern void ProcessPairNN(void* corr, void* field1, void* field2, int dots, int coord, int metric);
extern void ProcessPairNK(void* corr, void* field1, void* field2, int dots, int coord, int metric);
extern void ProcessPairNG(void* corr, void* field1, void* field2, int dots, int coord, int metric);
extern void ProcessPairKK(void* corr, void* field1, void* field2, int dots, int coord, int metric);
extern void ProcessPairKG(void* corr, void* field1, void* field2, int dots, int coord, int metric);
extern void ProcessPairGG(void* corr, void* field1, void* field2, int dots, int coord, int metric);

extern int SetOMPThreads(int num_threads);

