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

extern void* BuildGFieldFlat(double* x, double* y, double* g1, double* g2,
                             double* w, double* wpos, long nobj,
                             double minsize, double maxsize,
                             int sm_int, int maxtop);
extern void* BuildGField3D(double* x, double* y, double* z, double* g1, double* g2,
                           double* w, double* wpos, long nobj,
                           double minsize, double maxsize,
                           int sm_int, int maxtop);
extern void* BuildGFieldSphere(double* x, double* y, double* z, double* g1, double* g2,
                               double* w, double* wpos, long nobj,
                               double minsize, double maxsize,
                               int sm_int, int maxtop);

extern void* BuildKFieldFlat(double* x, double* y, double* k,
                             double* w, double* wpos, long nobj,
                             double minsize, double maxsize,
                             int sm_int, int maxtop);
extern void* BuildKField3D(double* x, double* y, double* z, double* k,
                           double* w, double* wpos, long nobj,
                           double minsize, double maxsize,
                           int sm_int, int maxtop);
extern void* BuildKFieldSphere(double* x, double* y, double* z, double* k,
                               double* w, double* wpos, long nobj,
                               double minsize, double maxsize,
                               int sm_int, int maxtop);

extern void* BuildNFieldFlat(double* x, double* y,
                             double* w, double* wpos, long nobj,
                             double minsize, double maxsize,
                             int sm_int, int maxtop);
extern void* BuildNField3D(double* x, double* y, double* z,
                           double* w, double* wpos, long nobj,
                           double minsize, double maxsize,
                           int sm_int, int maxtop);
extern void* BuildNFieldSphere(double* x, double* y, double* z,
                               double* w, double* wpos, long nobj,
                               double minsize, double maxsize,
                               int sm_int, int maxtop);

extern void DestroyGFieldFlat(void* field);
extern void DestroyGField3D(void* field);
extern void DestroyGFieldSphere(void* field);

extern void DestroyKFieldFlat(void* field);
extern void DestroyKField3D(void* field);
extern void DestroyKFieldSphere(void* field);

extern void DestroyNFieldFlat(void* field);
extern void DestroyNField3D(void* field);
extern void DestroyNFieldSphere(void* field);


extern void* BuildGSimpleFieldFlat(double* x, double* y, double* g1, double* g2,
                                   double* w, double* wpos, long nobj);
extern void* BuildGSimpleField3D(double* x, double* y, double* z, double* g1, double* g2,
                                 double* w, double* wpos, long nobj);
extern void* BuildGSimpleFieldSphere(double* x, double* y, double* z, double* g1, double* g2,
                                     double* w, double* wpos, long nobj);

extern void* BuildKSimpleFieldFlat(double* x, double* y, double* k,
                                   double* w, double* wpos, long nobj);
extern void* BuildKSimpleField3D(double* x, double* y, double* z, double* k,
                                 double* w, double* wpos, long nobj);
extern void* BuildKSimpleFieldSphere(double* x, double* y, double* z, double* k,
                                     double* w, double* wpos, long nobj);

extern void* BuildNSimpleFieldFlat(double* x, double* y,
                                   double* w, double* wpos, long nobj);
extern void* BuildNSimpleField3D(double* x, double* y, double* z,
                                 double* w, double* wpos, long nobj);
extern void* BuildNSimpleFieldSphere(double* x, double* y, double* z,
                                     double* w, double* wpos, long nobj);

extern void DestroyGSimpleFieldFlat(void* field);
extern void DestroyGSimpleField3D(void* field);
extern void DestroyGSimpleFieldSphere(void* field);

extern void DestroyKSimpleFieldFlat(void* field);
extern void DestroyKSimpleField3D(void* field);
extern void DestroyKSimpleFieldSphere(void* field);

extern void DestroyNSimpleFieldFlat(void* field);
extern void DestroyNSimpleField3D(void* field);
extern void DestroyNSimpleFieldSphere(void* field);

extern long NFieldFlatGetNTopLevel(void* field);
extern long NField3DGetNTopLevel(void* field);
extern long NFieldSphereGetNTopLevel(void* field);
extern long KFieldFlatGetNTopLevel(void* field);
extern long KField3DGetNTopLevel(void* field);
extern long KFieldSphereGetNTopLevel(void* field);
extern long GFieldFlatGetNTopLevel(void* field);
extern long GField3DGetNTopLevel(void* field);
extern long GFieldSphereGetNTopLevel(void* field);
