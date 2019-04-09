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

// This is the only thing about Metric that we need in python.

// We use a code for the metric to use:
// Euclidean is Euclidean in (x,y) or (x,y,z)
// Rperp uses the perpendicular component of the separation as the distance
// Rlens uses the perpendicular component at the lens (the first catalog) distance
// Arc uses the great circle distance between two points on the sphere

// This is the C++ way to do it.
//enum Metric { Euclidean=1, Rperp=2, Rlens=3, Arc=4 };

// But this is how we need to do it in C.
typedef enum { Euclidean=1, Rperp=2, Rlens=3, Arc=4, OldRperp=5, Periodic=6 } Metric;
