
#include "dbg.h"
#include "ConfigFile.h"

ConfigFile::ConfigFile() :
    _delimiter("="), _comment("#"), _include("+"), _sentry("EndConfigFile"), _continue("&")
{
    // Construct an empty ConfigFile
}

ConfigFile::ConfigFile( 
    const std::string file_name, const std::string delimiter,
    const std::string comment, const std::string inc,
    const std::string sentry, const std::string cont ) : 
    _delimiter(delimiter), _comment(comment), _include(inc), _sentry(sentry), _continue(cont)
{ load(file_name); }


void ConfigFile::load( const std::string file_name )
{ 
    std::ifstream fin(file_name.c_str()); 
    if(!fin) throw std::runtime_error("Unable to open config file "+file_name);
    read(fin); 
}

void ConfigFile::load( 
    const std::string file_name, const std::string delimiter,
    const std::string comment, const std::string inc,
    const std::string sentry, const std::string cont )
{
    // Construct a ConfigFile, getting keys and values from given file

    // the old values
    std::string delimiter1 = _delimiter;
    std::string comment1 = _comment;
    std::string inc1 = _include;
    std::string sentry1 = _sentry;
    std::string cont1 = _continue;

    if (delimiter != "") _delimiter = delimiter;
    if (comment != "") _comment = comment;
    if (inc != "") _include = inc;
    if (sentry != "") _sentry = sentry;
    if (cont != "") _continue = cont;

    load(file_name);

    _delimiter = delimiter1;
    _comment = comment1;
    _include = inc1;
    _sentry = sentry1;
    _continue = cont1;
}

ConvertibleString& ConfigFile::getNoCheck( const std::string& key )
{
    std::string key2 = key;
    trim(key2);
    return _contents[key2]; 
}

ConvertibleString ConfigFile::get( const std::string& key ) const
{
    std::string key2 = key;
    trim(key2);
    MapCIt p = _contents.find(key2);
    if (p == _contents.end()) {
        throw std::runtime_error("ConfigFile error: key "+key2+" not found");
    } else {
        return p->second;
    }
}

// special string getter.  This is really for the python
// bindings for just viewing quickly the contents.  Hence
// also throwing const char* for now, which swig can easily
// deal with
std::string ConfigFile::getstr(const std::string key) const throw (const char*) 
{
    MapCIt p = _contents.find(key);

    if (p == _contents.end()) {
        std::stringstream err;
        err<<"ConfigFile error: key '"<<key<<"' not found";
        throw err.str().c_str();
    }

    std::string val = get(key);
    return val;
}

// with default value
std::string ConfigFile::getstr(const std::string key, 
                               const std::string defval) 
{
    MapCIt p = _contents.find(key);

    std::string val;
    if (p == _contents.end()) {
        val = defval;
    } else {
        val = get(key);
    }
    return val;
}


void ConfigFile::remove( const std::string& key )
{
    // Remove key and its value
    _contents.erase( _contents.find( key ) );
    return;
}

bool ConfigFile::keyExists( const std::string& key ) const
{
    // Indicate whether key is found
    MapCIt p = _contents.find( key );
    return ( p != _contents.end() );
}

void ConfigFile::trim( std::string& s )
{
    // Remove leading and trailing whitespace
    std::string whiteSpace = " \n\t\v\r\f";
    s.erase( 0, s.find_first_not_of(whiteSpace) );
    s.erase( s.find_last_not_of(whiteSpace) + 1);
}

void ConfigFile::write(std::ostream& os) const
{
    // Save a ConfigFile to os
    for(MapCIt p = _contents.begin(); p != _contents.end(); ++p ) {
        os << p->first << " " << _delimiter << " " << p->second << std::endl;
    }
}

void ConfigFile::writeAsComment(std::ostream& os) const
{
    // Save a ConfigFile to os
    for(MapCIt p = _contents.begin(); p != _contents.end(); ++p ) {
        std::string f = p->first;
        std::string s = p->second;
        std::replace(f.begin(),f.end(),'\n',' ');
        std::replace(s.begin(),s.end(),'\n',' ');
        os << _comment << " " << f << " " << _delimiter << " " << s << std::endl;
    }
}

void ConfigFile::read(std::istream& is)
{
    // Load a ConfigFile from is
    // Read in keys and values, keeping internal whitespace

    std::string nextLine = "";  
    // might need to read ahead to see where value ends

    while( is || nextLine.size() > 0 ) {
        // Read an entire line at a time
        std::string line;
        if( nextLine.size() > 0 ) {
            line = nextLine;  // we read ahead; use it now
            nextLine = "";
        } else {
            std::getline( is, line );
        }
        std::string orig_line = line;

        // Ignore comments
        //line = line.substr( 0, line.find(_comment) );
        std::string::size_type commPos = line.find( _comment );
        if (commPos != std::string::npos)
            line.replace(commPos,std::string::npos,"");

        // Remove leading and trailing whitespace
        trim(line);

        // If line is blank, go on to next line.
        if (line.size() == 0) continue;

        // Check for include directive (only at start of line)
        if (line.find(_include) == 0) { 
            line.erase(0,_include.size());
            std::stringstream ss(line);
            std::string file_name;
            ss >> file_name;
            load(file_name);
            // implcitly skip the rest of the line.
            continue;
        }

        // Check for end of file sentry
        if( _sentry != "" && line.find(_sentry) != std::string::npos ) return;

        // At this point, a line without a delimiter is an error.
        std::string::size_type delimPos = line.find( _delimiter );
        if( delimPos == std::string::npos ) {
            throw std::runtime_error("Invalid config line: " + orig_line);
        }

        // Extract the key
        std::string key = line.substr( 0, delimPos );

        // Extract the value
        line.replace( 0, delimPos+_delimiter.size(), "" );

        // See if value continues on the next line(s)
        while (is) {
            std::getline( is, nextLine );
            ConfigFile::trim(nextLine);
            std::string::size_type continPos = nextLine.find(_continue);
            if (continPos == 0) { // Only counts if its the first thing.
                nextLine.replace(line.find(_comment),std::string::npos,"");
                ConfigFile::trim(nextLine);
                line += "\n";
                line += nextLine;
            } else{
                break;
            }
        }

        // Store key and value
        ConfigFile::trim(key);
        ConfigFile::trim(line);
        _contents[key] = line;  // overwrites if key is repeated
    }
}

std::set<std::string> ConfigFile::checkValid(const std::set<std::string>& valid_keys) const
{
    std::set<std::string> invalid_keys;
    for (MapCIt p = _contents.begin(); p != _contents.end(); ++p ) {
        std::string key = p->first;
        if (valid_keys.find(key) == valid_keys.end()) invalid_keys.insert(key);
    }
    return invalid_keys;
}
