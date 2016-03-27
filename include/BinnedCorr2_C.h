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

extern void ProcessAutoNNFlat(void* corr, void* field, int dots);
extern void ProcessAutoNN3D(void* corr, void* field, int dots);
extern void ProcessAutoNNPerp(void* corr, void* field, int dots);
extern void ProcessAutoNNLens(void* corr, void* field, int dots);
extern void ProcessAutoKKFlat(void* corr, void* field, int dots);
extern void ProcessAutoKK3D(void* corr, void* field, int dots);
extern void ProcessAutoKKPerp(void* corr, void* field, int dots);
extern void ProcessAutoKKLens(void* corr, void* field, int dots);
extern void ProcessAutoGGFlat(void* corr, void* field, int dots);
extern void ProcessAutoGG3D(void* corr, void* field, int dots);
extern void ProcessAutoGGPerp(void* corr, void* field, int dots);
extern void ProcessAutoGGLens(void* corr, void* field, int dots);

extern void ProcessCrossNNFlat(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossNN3D(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossNNPerp(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossNNLens(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossNKFlat(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossNK3D(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossNKPerp(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossNKLens(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossNGFlat(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossNG3D(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossNGPerp(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossNGLens(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossKKFlat(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossKK3D(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossKKPerp(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossKKLens(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossKGFlat(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossKG3D(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossKGPerp(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossKGLens(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossGGFlat(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossGG3D(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossGGPerp(void* corr, void* field1, void* field2, int dots);
extern void ProcessCrossGGLens(void* corr, void* field1, void* field2, int dots);

extern void ProcessPairwiseNNFlat(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseNN3D(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseNNPerp(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseNNLens(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseNKFlat(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseNK3D(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseNKPerp(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseNKLens(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseNGFlat(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseNG3D(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseNGPerp(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseNGLens(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseKKFlat(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseKK3D(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseKKPerp(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseKKLens(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseKGFlat(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseKG3D(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseKGPerp(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseKGLens(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseGGFlat(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseGG3D(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseGGPerp(void* corr, void* field1, void* field2, int dots);
extern void ProcessPairwiseGGLens(void* corr, void* field1, void* field2, int dots);

extern int SetOMPThreads(int num_threads);

