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

extern void* BuildGField(double* x, double* y, double* z, double* g1, double* g2,
                         double* w, double* wpos, long nobj,
                         double minsize, double maxsize,
                         int sm_int, long seed, int brute, int mintop, int maxtop, int coords);

extern void* BuildKField(double* x, double* y, double* z, double* k,
                         double* w, double* wpos, long nobj,
                         double minsize, double maxsize,
                         int sm_int, long seed, int brute, int mintop, int maxtop, int coords);

extern void* BuildNField(double* x, double* y, double* z,
                         double* w, double* wpos, long nobj,
                         double minsize, double maxsize,
                         int sm_int, long seed, int brute, int mintop, int maxtop, int coords);

extern void DestroyGField(void* field, int coords);
extern void DestroyKField(void* field, int coords);
extern void DestroyNField(void* field, int coords);

extern long FieldGetNTopLevel(void* field, int d, int coords);
extern long FieldCountNear(void* field, double x, double y, double z, double sep,
                           int d, int coords);
extern void FieldGetNear(void* field, double x, double y, double z, double sep,
                         int d, int coords, long* indices, long n);

extern void* BuildGSimpleField(double* x, double* y, double* z, double* g1, double* g2,
                               double* w, double* wpos, long nobj, int coords);

extern void* BuildKSimpleField(double* x, double* y, double* z, double* k,
                               double* w, double* wpos, long nobj, int coords);

extern void* BuildNSimpleField(double* x, double* y, double* z,
                               double* w, double* wpos, long nobj, int coords);

extern void DestroyGSimpleField(void* field, int coords);
extern void DestroyKSimpleField(void* field, int coords);
extern void DestroyNSimpleField(void* field, int coords);

extern void KMeansInitTree(void* field, double* centers, int npatch, int d, int coords, long seed);
extern void KMeansInitRand(void* field, double* centers, int npatch, int d, int coords, long seed);
extern void KMeansInitKMPP(void* field, double* centers, int npatch, int d, int coords, long seed);
extern void KMeansRun(void* field, double* centers, int npatch, int max_iter, double tol,
                      int alt, int d, int coords);
extern void KMeansAssign(void* field, double* centers, int npatch,
                         long* patches, long n, int d, int coords);

// These aren't field functions, but I'm putting them here anyway, since they're related to patches.
extern void QuickAssign(double* centers, int npatch,
                        double* x, double* y, double* z, long* patches, long n);
extern void SelectPatch(int patch, double* centers, int npatch, double* x, double* y, double* z,
                        long* use, long n);
extern void GenerateXYZ(double* x, double* y, double* z, double* ra, double* dec, double* r,
                        long n);
