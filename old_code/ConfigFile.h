// ConfigFile.h
//
// Based on a class, originally written by Richard J. Wagner.
// Modified substantially by Mike Jarvis.
//
// Copyright notice in original version:
//
// Copyright (c) 2004 Richard J. Wagner
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef ConfigFile_H
#define ConfigFile_H

#include <string>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <typeinfo>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <limits>

#include "dbg.h"

template <class T>
struct ConvertibleStringTraits
{
    enum { is_bool = false };
    enum { is_vector = false };
    enum { is_string = false };
    enum { is_cstring = false };
};

template <>
struct ConvertibleStringTraits<bool>
{
    enum { is_bool = true };
    enum { is_vector = false };
    enum { is_string = false };
    enum { is_cstring = false };
};

template <class T>
struct ConvertibleStringTraits<std::vector<T> >
{
    enum { is_bool = false };
    enum { is_vector = true };
    enum { is_string = false };
    enum { is_cstring = false };
};

template <class T>
struct ConvertibleStringTraits<std::basic_string<T> >
{
    enum { is_bool = false };
    enum { is_vector = false };
    enum { is_string = true };
    enum { is_cstring = false };
};

template <>
struct ConvertibleStringTraits<const char*>
{
    enum { is_bool = false };
    enum { is_vector = false };
    enum { is_string = false };
    enum { is_cstring = true };
};

// Get rid of const or ref:
template <class T>
struct ConvertibleStringTraits<T&> : public ConvertibleStringTraits<T> {};
template <class T>
struct ConvertibleStringTraits<const T> : public ConvertibleStringTraits<T> {};
template <class T>
struct ConvertibleStringTraits<const T&> : public ConvertibleStringTraits<T> {};


template <int algo, class T>
struct ConvertibleStringOpT;

template <class T>
struct ConvertibleStringOpT<1,T> // Normal case
{
    static T call(const std::string& s) 
    {
#ifdef Use_Zero_Default
        if (s == "") return T();
#endif

        T ret;
        std::stringstream ss(s);

        // This bit is needed to get the oct and hex to work:
        // I haven't incorporated it into the below vector conversion,
        // so those always need to be decimal.
        if (std::numeric_limits<T>::is_integer) {
            if (s[0] == '0') {
                if ((s)[1] == 'x' || s[1] == 'X') {
                    ss >> std::hex;
                } else {
                    ss >> std::oct;
                }
            }
        }
        ss >> ret;
        if (!ss) {
            std::string err="Could not convert ConvertibleString to input type ";
            err += typeid(T).name();
            err += std::string(": s = ") + s;
            throw std::runtime_error(err);
        }
        return ret;
    }
};

template <class T>
struct ConvertibleStringOpT<2,std::vector<T> > // T is a vector
{
    static std::vector<T> call(const std::string& s) 
    {
#ifdef Use_Zero_Default
        if (s == "") return std::vector<T>();
#endif

        // Two syntaxes: "{1.,2.,3.}" or "1. 2. 3."
        if (s[0] == '{') {
            // Using "{1.,2.,3.}" syntax

            int i1 = s.find_first_not_of(" \t\n\r\f",1);
            if (i1 == int(std::string::npos)) {
                std::string err="Could not convert ConvertibleString to input type ";
                err += std::string("std::vector<")+typeid(T).name()+">";
                err += std::string(": s = ") + s;
                throw std::runtime_error(err);
            }
            if (s[i1] == '}') {
                // string is empty: "{ }"
                return std::vector<T>();
            }

            int nComma = std::count(s.begin(),s.end(),',');
            std::vector<T> ret(nComma+1);

            int i2 = s.find_first_of("},",i1);
            int j = 0;

            while (s[i2] != '}') {
                std::string s2 = s.substr(i1,i2-i1);
                std::stringstream ss(s2);
                ss >> ret[j++];
                i1 = i2+1;
                i2 = s.find_first_of("},",i1);
            }
            {
                // Do last element
                std::string s2 = s.substr(i1,i2-i1);
                std::stringstream ss(s2);
                ss >> ret[j];
            }
            if (j != nComma) {
                std::string err="Could not convert ConvertibleString to input type ";
                err += std::string("std::vector<")+typeid(T).name()+">";
                err += std::string(": s = ") + s;
                throw std::runtime_error(err);
            }
            return ret;
        } else {
            // Using "1. 2. 3." syntax
            std::stringstream ss(s);
            std::vector<T> ret;
            T x;
            while (ss >> x) ret.push_back(x);
            return ret;
        }
    }
};

template <class T>
struct ConvertibleStringOpT<3,T> // T is bool
{
    static T call(const std::string& s) 
    {
#ifdef Use_Zero_Default
        if (*this == "") return false;
#endif

        // make string all caps
        std::string sup = s;
        for ( std::string::iterator p = sup.begin(); p != sup.end(); ++p )
            *p = toupper(*p); 

        if ( sup=="FALSE" || sup=="F" || sup=="NO" || sup=="N" ||
             sup=="0" || sup=="NONE" ) {
            return false;
        } else if ( sup=="TRUE" || sup=="T" || sup=="YES" || sup=="Y" ||
                    sup=="1" ) {
            return true;
        } else {
            std::string err=
                "Could not convert ConvertibleString to input type bool"
                ": s = " + s;
            throw std::runtime_error(err);
        }
    }
};

template <class T>
struct ConvertibleStringOpT<4,T> // T is std::string
{ static T call(const std::string& s) { return s; } };

template <class T>
struct ConvertibleStringOpT<5,T> // T is C-style string
{ static T call(const std::string& s) { return s.c_str(); } };

#ifdef __INTEL_COMPILER
#pragma warning (disable : 444)
// Disable "destructor for base class ... is not virtual"
// Technically, it is bad form to inherit from a class that doesn't have
// a virtual destructor.  The reason is that an object that is only
// known as a reference or pointer to the base class won't call the 
// derived class's destructor when it is deleted.
// A) This isn't a problem here, since ConvertibleString has no
//    data elements that need to be cleaned up by a destructor.
// B) I can't find a way to avoid inheriting from std::string.  When I
//    instead keep a string variable, then I can't find a way to 
//    make assignment from ConvertibleString to string work correctly.
//    The operator T() technique that we use for all other types
//    fails, since the compiler can't disambiguate which string
//    assignment operator to use this with.
//    Specializing an operator string() case doesn't help.
// So the easiest solution is to leave this as is and just disable the warning.
#endif
class ConvertibleString : public std::string
{

public:
    ConvertibleString() : std::string("") {}
    ConvertibleString(const std::string& s) : std::string(s) {}

    template <typename T> 
    explicit ConvertibleString(const T& x)
    { *this = x; }

    ~ConvertibleString() {}

    ConvertibleString& operator=(const std::string& rhs)
    {
        std::string::operator=(rhs);
        return *this;
    }

    template <typename T> 
    ConvertibleString& operator=(const T& x)
    {
        std::stringstream oss;
        oss << x;
        *this = oss.str();
        return *this;
    }

    template <typename T> 
    ConvertibleString& operator=(const std::vector<T>& x)
    {
        std::stringstream oss;
        const int n = x.size();
        if (n > 0) oss << x[0];
        for(int i=1;i<n;++i) oss << ' ' << x[i];
        *this = oss.str();
        return *this;
    }

    template <typename T> 
    operator T() const 
    {
        const int algo =
            ConvertibleStringTraits<T>::is_vector ? 2 :
            ConvertibleStringTraits<T>::is_bool ? 3 :
            ConvertibleStringTraits<T>::is_string ? 4 :
            ConvertibleStringTraits<T>::is_cstring ? 5 :
            1;
        return ConvertibleStringOpT<algo,T>::call(*this);
    }
};
#ifdef __INTEL_COMPILER
#pragma warning (default : 444)
#endif

class ConfigFile 
{

public:
    // Create a blank config file with default values of delimter, etc.
    ConfigFile();

    // Create and read from the specified file
    ConfigFile( const std::string file_name,
                const std::string delimiter = "=",
                const std::string comment = "#",
                const std::string include = "+",
                const std::string sentry = "EndConfigFile",
                const std::string cont = "&");

    // Load more value from a file.
    void load( const std::string file_name );
    //{ std::ifstream fs(file_name.c_str()); read(fs); }

    // Load a file that uses different delimiter or comment or ...
    // Note: these delimiter, comment, etc. are temporary for this load only.
    // "" means use existing values
    void load( const std::string file_name,
               const std::string delimiter,
               const std::string comment = "",
               const std::string include = "",
               const std::string sentry = "",
               const std::string cont = "");

    // Read more values from stream or a string:
    void read(std::istream& is);
    void append(const std::string& s)
    { std::istringstream ss(s); read(ss); }

    // Write configuration
    void write(std::ostream& os) const;
    void writeAsComment(std::ostream& os) const;

    // Search for key and read value or optional default value
    ConvertibleString& getNoCheck( const std::string& key );
    ConvertibleString get( const std::string& key ) const;

    inline ConvertibleString& operator[]( const std::string& key )
    { return getNoCheck(key); }
    inline ConvertibleString operator[]( const std::string& key ) const
    { return get(key); }

    template <typename T> inline T read( const std::string& key ) const;
    template <typename T> inline T read(
        const std::string& key, const T& value ) const;
    inline std::string read(const std::string& key, const char* value ) const
    {
        std::string ret = read(key,std::string(value)); 
        return ret;
    }

    template <typename T> inline bool readInto(T& var, const std::string& key ) const;
    template <typename T> inline bool readInto(
        T& var, const std::string& key, const T& value ) const;

    // If the ConfigFile is not const, store the provided default.
    template <typename T> inline T read(const std::string& key, const T& value );

    // special string getter.  This is really for the python
    // bindings for just viewing quickly the contents.  Hence
    // also throwing const char* for now, which swig can easily
    // deal with
    std::string getstr(const std::string key) const throw (const char*);
    // with default value
    std::string getstr(const std::string key, const std::string defval);

    // Modify keys and values
    template <typename T> inline void add( std::string key, const T& value );
    void remove( const std::string& key );

    // Check whether key exists in configuration
    bool keyExists( const std::string& key ) const;

    // Check or change configuration syntax
    std::string getDelimiter() const { return _delimiter; }
    std::string getComment() const { return _comment; }
    std::string getInclude() const { return _include; }
    std::string getSentry() const { return _sentry; }
    std::string getContinue() const { return _continue; }

    std::string setDelimiter( const std::string& s )
    { std::string old = _delimiter;  _delimiter = s;  return old; }  
    std::string setComment( const std::string& s )
    { std::string old = _comment;  _comment = s;  return old; }
    std::string setInclude( const std::string& s )
    { std::string old = _include;  _include = s;  return old; }  
    std::string setSentry( const std::string& s )
    { std::string old = _sentry;  _sentry = s;  return old; }  
    std::string setContinue( const std::string& s )
    { std::string old = _continue;  _continue = s;  return old; }  

    int size() { return _contents.size(); }

    // Check that the ConfigFile only includes keys in the list of valid_keys.
    // The returned value is a list of any invalid keys found.
    std::set<std::string> checkValid(const std::set<std::string>& valid_keys) const;

protected:

    static void trim( std::string& s );

    std::string _delimiter;  // separator between key and value
    std::string _comment;    // separator between value and comments
    std::string _include;    // directive for including another file
    std::string _sentry;     // optional string to signal end of file
    std::string _continue;   // marker of a continuation line
    std::map<std::string,ConvertibleString> _contents;   
    // extracted keys and values

    typedef std::map<std::string,ConvertibleString>::iterator MapIt;
    typedef std::map<std::string,ConvertibleString>::const_iterator MapCIt;
};

inline std::ostream& operator<<( std::ostream& os, const ConfigFile& cf )
{ cf.write(os); return os; }
inline std::istream& operator>>( std::istream& is, ConfigFile& cf )
{ cf.read(is); return is; }

template <typename T>
T ConfigFile::read(const std::string& key) const
{
    // Read the value corresponding to key
    std::string key2 = key;
    trim(key2);
    MapCIt p = _contents.find(key2);
    if(p == _contents.end()) {
        throw std::runtime_error("ConfigFile error: key "+key2+" not found");
    }
    T ret;
    try {
        ret = p->second;
    } catch (std::runtime_error& e) {
        xdbg<<"Caught exception: \n"<<e.what()<<std::endl;
        throw std::runtime_error(
            "ConfigFile error: Could not convert entry for key " +
            key2 +
            " to given type.\nCaught error from ConvertibleString: \n" +
            e.what());
    }
    return ret;
}


template <typename T>
T ConfigFile::read( const std::string& key, const T& value ) const
{
    // Return the value corresponding to key or given default value
    // if key is not found
    std::string key2 = key;
    trim(key2);
    MapCIt p = _contents.find(key2);
    if(p == _contents.end()) {
        return value;
    } else {
        T ret;
        try {
            ret = p->second;
        } catch (std::runtime_error& e) {
            xdbg<<"Caught exception: \n"<<e.what()<<std::endl;
            throw std::runtime_error(
                "ConfigFile error: Could not convert entry for key " +
                key2 +
                " to given type.\nCaught error from ConvertibleString: \n" +
                e.what());
        }
        return ret;
    }
}

template <typename T>
T ConfigFile::read( const std::string& key, const T& value )
{
    // Return the value corresponding to key or given default value
    // if key is not found
    std::string key2 = key;
    trim(key2);
    MapCIt p = _contents.find(key2);
    if(p == _contents.end()) {
        _contents[key] = value;
        return value;
    } else {
        T ret;
        try {
            ret = p->second;
        } catch (std::runtime_error& e) {
            xdbg<<"Caught exception: \n"<<e.what()<<std::endl;
            throw std::runtime_error(
                "ConfigFile error: Could not convert entry for key " +
                key2 +
                " to given type.\nCaught error from ConvertibleString: \n" +
                e.what());
        }
        return ret;
    }
}

template <typename T>
bool ConfigFile::readInto( T& var, const std::string& key ) const
{
    // Get the value corresponding to key and store in var
    // Return true if key is found
    // Otherwise leave var untouched
    std::string key2 = key;
    trim(key2);
    MapCIt p = _contents.find(key2);
    bool is_found = ( p != _contents.end() );
    if(is_found) {
        try {
            var = p->second;
        } catch (std::runtime_error& e) {
            xdbg<<"Caught exception: \n"<<e.what()<<std::endl;
            throw std::runtime_error(
                "ConfigFile error: Could not convert entry for key " +
                key2 +
                " to given type.\nCaught error from ConvertibleString: \n" +
                e.what());
        }
    }
    return is_found;
}


template <typename T>
bool ConfigFile::readInto( 
    T& var, const std::string& key, const T& value ) const
{
    // Get the value corresponding to key and store in var
    // Return true if key is found
    // Otherwise set var to given default
    std::string key2 = key;
    trim(key2);
    MapCIt p = _contents.find(key2);
    bool is_found = ( p != _contents.end() );
    if(is_found) {
        try {
            var = p->second;
        } catch (std::runtime_error& e) {
            xdbg<<"Caught exception: \n"<<e.what()<<std::endl;
            throw std::runtime_error(
                "ConfigFile error: Could not convert entry for key " +
                key2 +
                " to given type.\nCaught error from ConvertibleString: \n" +
                e.what());
        }
    } else {
        var = value;
    }
    return is_found;
}

template <typename T>
void ConfigFile::add( std::string key, const T& value )
{
    // Add a key with given value
    trim(key);
    _contents[key] = value;
}

#endif  // CONFIGFILE_H
