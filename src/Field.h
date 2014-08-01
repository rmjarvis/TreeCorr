/* Copyright (c) 2003-2014 by Mike Jarvis
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
 * 3. Neither the name of the {organization} nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 */

extern "C" {

    extern void* BuildGFieldFlat(double* x, double* y, double* g1, double* g2, double* w,
                                 int nobj, double minsep, double maxsep, double b, int sm);

    extern void* BuildGFieldSphere(double* ra, double* dec, double* g1, double* g2, double* w,
                                   int nobj, double minsep, double maxsep, double b, int sm);


    extern void* BuildGFieldFlat(double* x, double* y, double* g1, double* g2, double* w,
                                 int nobj, double minsep, double maxsep, double b, int sm_int);

    extern void* BuildGFieldSphere(double* ra, double* dec, double* g1, double* g2, double* w,
                                   int nobj, double minsep, double maxsep, double b, int sm_int);

    extern void* BuildKFieldFlat(double* x, double* y, double* k, double* w,
                                 int nobj, double minsep, double maxsep, double b, int sm_int);

    extern void* BuildKFieldSphere(double* ra, double* dec, double* k, double* w,
                                   int nobj, double minsep, double maxsep, double b, int sm_int);

    extern void* BuildNFieldFlat(double* x, double* y, double* w,
                                 int nobj, double minsep, double maxsep, double b, int sm_int);

    extern void* BuildNFieldSphere(double* ra, double* dec, double* w,
                                   int nobj, double minsep, double maxsep, double b, int sm_int);

    extern void DestroyGFieldFlat(void* cells);

    extern void DestroyGFieldSphere(void* cells);

    extern void DestroyKFieldFlat(void* cells);

    extern void DestroyKFieldSphere(void* cells);

    extern void DestroyNFieldFlat(void* cells);

    extern void DestroyNFieldSphere(void* cells);


}

