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

#ifndef TreeCorr_dbg_H
#define TreeCorr_dbg_H

//#define DEBUGLOGGING

#ifdef DEBUGLOGGING
// Python usually turns this on automatically.  If we are debugging, turn it off so assert works.
#ifdef NDEBUG
#undef NDEBUG
#endif
#endif

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <cstddef>
#include <cassert>


// Convenient debugging.
// Use as a normal C++ stream:
// dbg << "Here x = "<<x<<" and y = "<<y<<std::endl;
// If DEBUGLOGGING is not enabled, the compiler optimizes it away, so it
// doesn't take any CPU cycles.
//

#ifdef DEBUGLOGGING
class Debugger // Use a Singleton model so it can be included multiple times.
{
public:
    std::ostream& get_dbgout() { return *dbgout; }
    void set_dbgout(std::ostream* new_dbgout) { dbgout = new_dbgout; }
    void set_verbose(int level) { verbose_level = level; }
    bool do_level(int level) { return verbose_level >= level; }
    int get_level() { return verbose_level; }

    static Debugger& instance()
    {
        static Debugger _instance;
        return _instance;
    }

private:
    std::ostream* dbgout;
    int verbose_level;

    Debugger() : dbgout(&std::cout), verbose_level(1) {}
    Debugger(const Debugger&);
    void operator=(const Debugger&);
};

#define dbg if(Debugger::instance().do_level(1)) Debugger::instance().get_dbgout()
#define xdbg if(Debugger::instance().do_level(2)) Debugger::instance().get_dbgout()
#define xxdbg if(Debugger::instance().do_level(3)) Debugger::instance().get_dbgout()
#define set_dbgout(dbgout) Debugger::instance().set_dbgout(dbgout)
#define get_dbgout() Debugger::instance().get_dbgout()
#define set_verbose(level) Debugger::instance().set_verbose(level)
#define verbose_level Debugger::instance().get_level()
#define Assert(x) do { if (!(x)) { dbg<<"Failed Assert: "<<#x<<std::endl; throw std::runtime_error("Failed Assert: " #x); } } while (false)
#define XAssert(x) do { if (verbose_level >= 3) Assert(x); } while (false)
#else
#define dbg if(false) (std::cerr)
#define xdbg if(false) (std::cerr)
#define xxdbg if(false) (std::cerr)
#define set_dbgout(dbgout)
#define get_dbgout()
#define set_verbose(level)
// When not in debug mode, don't bomb out for assert failures.  Just print to screen.
#define Assert(x) do { if (!(x)) std::cerr<<"Failed Assert: "<<(#x); } while (false)
#define XAssert(x)
#endif

#endif
