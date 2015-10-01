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

#ifndef TreeCorr_dbg_H
#define TreeCorr_dbg_H

/* Put the following in the main program file:

   std::ostream* dbgout=0;
   bool XDEBUG=false;

*/

//
// Set up debugging stuff
//

#include <cstdlib>
#include <iostream>

extern std::ostream* dbgout;
extern bool XDEBUG;

#ifdef NDEBUG
#define dbg if (false) (*dbgout)
#define xdbg if (false) (*dbgout)
#define xxdbg if (false) (*dbgout)
#define Assert(x)
#else
#define dbg if (dbgout) (*dbgout)
#define xdbg if (dbgout && XDEBUG) (*dbgout)
#define xxdbg if (false) (*dbgout)
#define Assert(x) \
    do { \
        if(!(x)) { \
            dbg << "Error - Assert " #x " failed"<<std::endl; \
            dbg << "on line "<<__LINE__<<" in file "<<__FILE__<<std::endl; \
            std::cerr << "Error - Assert " #x " failed"<<std::endl; \
            std::cerr << "on line "<<__LINE__<<" in file "<<__FILE__<<std::endl; \
            exit(1); \
        }  \
    } while (false)
#endif

inline void myerror(const std::string& s)
{
    dbg << "Error: " << s << std::endl;
    std::cerr << "Error: " << s << std::endl;
    exit(1);
}

#endif
