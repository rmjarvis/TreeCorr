//---------------------------------------------------------------------------
#ifndef MemDebugH
#define MemDebugH
//---------------------------------------------------------------------------


/* Put the following in the main program file:

#ifdef MEMDEBUG
AllocList* allocList;
#endif

and somewhere in main:

atexit(&DumpUnfreed);

*/

#ifdef MEMDEBUG

#include <list>
#include <string>
#include <iostream>

// Here is some code for tracking memory leaks I got from
// http://www.flipcode.com/tutorials/tut_memleak.shtml
// Code is originally (and still mostly) by Dion Picco (c) 23 May 2000

// Remember to use atexit(&DumpUnfreed) to print results

struct AllocInfo 
{
    AllocInfo(long long a, long s, const char* f, int l) :
        address(a), size(s), file(f), line(l) {}

    long long address;
    long size;
    std::string file;
    int line;
};

typedef std::list<AllocInfo*> AllocList;

extern AllocList *allocList;

inline void AddTrack(long long addr, long asize, const char* fname, int lnum)
{
    AllocInfo *info;

    if(!allocList) {
        allocList = new AllocList;
    }

    info = new AllocInfo(addr,asize,fname,lnum);
    allocList->insert(allocList->begin(), info);
};

inline void RemoveTrack(long long addr)
{
    if(!allocList) return;
    for(AllocList::iterator i=allocList->begin();i!=allocList->end();i++) {
        if((*i)->address == addr) {
            allocList->remove((*i));
            break;
        }
    }
};

inline void DumpUnfreed()
{
    long totalSize = 0;
    if(!allocList) return;
    for(AllocList::iterator i=allocList->begin();i!=allocList->end();i++) {
        std::cerr << (*i)->file <<"  LINE "<<(*i)->line<<",\t\tADDRESS ";
        std::cerr << (*i)->address <<"  "<<(*i)->size<<" bytes unfreed\n";
#ifdef dbgH
        dbg << (*i)->file <<"  LINE "<<(*i)->line<<",\t\tADDRESS ";
        dbg << (*i)->address <<"  "<<(*i)->size<<" bytes unfreed\n";
#endif
        totalSize += (*i)->size;
    }
    std::cerr << "-----------------------------------------------------------\n";
    std::cerr << "Total Unfreed: "<<totalSize<<" bytes\n";
#ifdef dbgH
    dbg << "-----------------------------------------------------------\n";
    dbg << "Total Unfreed: "<<totalSize<<" bytes\n";
#endif
};

inline void* operator new(unsigned int size,const char* file,int line)
{ 
    void* ptr = (void*)malloc(size);
    AddTrack((long long)ptr, size, file, line);
    return(ptr);
};

inline void operator delete(void* p)
{
    RemoveTrack((long long)p);
    free(p);
};

inline void* operator new[](unsigned int size,const char* file,int line)
{ 
    void* ptr = (void*)malloc(size);
    AddTrack((long long)ptr, size, file, line);
    return(ptr);
};

inline void operator delete[](void* p)
{
    RemoveTrack((long long)p);
    free(p);
};


#define MEMDEBUG_NEW new(__FILE__, __LINE__)
#define new MEMDEBUG_NEW

#endif


#endif
