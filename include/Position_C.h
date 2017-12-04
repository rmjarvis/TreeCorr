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


// There are three kinds of coordinate systems we can use:
// 1 = Flat = (x,y) coordinates
// 2 = ThreeD (called 3d in python) = (x,y,z) coordinates
// 3 = Sphere = (ra,dec)  These are stored as (x,y,z), but normalized to have |r| = 1.
enum Coord { Flat=1, ThreeD=2, Sphere=3 };

