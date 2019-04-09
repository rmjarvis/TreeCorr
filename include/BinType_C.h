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

// This is the only thing about BinType that we need in python.

// We use a code for the type of binning to use:
// Log is logarithmic spacing in r
// Linear is linear spacing in r
// TwoD is linear spacing in x,y

// This is the C++ way to do it.
//enum BinType { Log=1, Linear=2, TwoD=3 };

// But this is how we need to do it in C.
typedef enum { Log=1, Linear=2, TwoD=3 } BinType;
