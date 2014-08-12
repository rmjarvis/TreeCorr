# Copyright (c) 2003-2014 by Mike Jarvis
#
# TreeCorr is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.


def get_aardvark():
    """We host the Aardvark.fit file separately so it people don't need to download it with
    the code when checking out the repo.  Most people don't run the tests after all.
    """
    import os
    file_name = os.path.join('data','Aardvark.fit')
    url = 'https://github.com/rmjarvis/TreeCorr/wiki/Aardvark.fit'
    if not os.path.isfile(file_name):
        import urllib
        print 'downloading %s from %s...'%(file_name,url)
        urllib.urlretrieve(url,file_name)
        print 'done.'
    
