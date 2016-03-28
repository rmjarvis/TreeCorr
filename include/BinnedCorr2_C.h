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
                         double* meanr, double* meanlogr, double* weight, double* npairs);
extern void* BuildNKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                         double* xi,
                         double* meanr, double* meanlogr, double* weight, double* npairs);
extern void* BuildNGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                         double* xi, double* xi_im,
                         double* meanr, double* meanlogr, double* weight, double* npairs);
extern void* BuildKKCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                         double* xi,
                         double* meanr, double* meanlogr, double* weight, double* npairs);
extern void* BuildKGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                         double* xi, double* xi_im,
                         double* meanr, double* meanlogr, double* weight, double* npairs);
extern void* BuildGGCorr(double minsep, double maxsep, int nbins, double binsize, double b,
                         double* xip, double* xip_im, double* xim, double* xim_im,
                         double* meanr, double* meanlogr, double* weight, double* npairs);

extern void DestroyNNCorr(void* corr);
extern void DestroyNKCorr(void* corr);
extern void DestroyNGCorr(void* corr);
extern void DestroyKKCorr(void* corr);
extern void DestroyKGCorr(void* corr);
extern void DestroyGGCorr(void* corr);

extern void ProcessAutoNNFlat(void* corr, void* field, int dots, int metric);
extern void ProcessAutoNN3D(void* corr, void* field, int dots, int metric);
extern void ProcessAutoKKFlat(void* corr, void* field, int dots, int metric);
extern void ProcessAutoKK3D(void* corr, void* field, int dots, int metric);
extern void ProcessAutoGGFlat(void* corr, void* field, int dots, int metric);
extern void ProcessAutoGG3D(void* corr, void* field, int dots, int metric);

extern void ProcessCrossNNFlat(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessCrossNN3D(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessCrossNKFlat(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessCrossNK3D(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessCrossNGFlat(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessCrossNG3D(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessCrossKKFlat(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessCrossKK3D(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessCrossKGFlat(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessCrossKG3D(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessCrossGGFlat(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessCrossGG3D(void* corr, void* field1, void* field2, int dots, int metric);

extern void ProcessPairwiseNNFlat(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessPairwiseNN3D(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessPairwiseNKFlat(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessPairwiseNK3D(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessPairwiseNGFlat(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessPairwiseNG3D(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessPairwiseKKFlat(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessPairwiseKK3D(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessPairwiseKGFlat(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessPairwiseKG3D(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessPairwiseGGFlat(void* corr, void* field1, void* field2, int dots, int metric);
extern void ProcessPairwiseGG3D(void* corr, void* field1, void* field2, int dots, int metric);

extern int SetOMPThreads(int num_threads);

