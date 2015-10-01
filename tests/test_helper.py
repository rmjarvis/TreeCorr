# Copyright (c) 2003-2015 by Mike Jarvis
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

from __future__ import print_function

def get_from_wiki(file_name):
    """We host some larger files used for the test suite separately on the TreeCorr wiki repo
    so people don't need to download them with the code when checking out the repo.
    Most people don't run the tests after all.
    """
    import os
    local_file_name = os.path.join('data',file_name)
    url = 'https://github.com/rmjarvis/TreeCorr/wiki/' + file_name
    if not os.path.isfile(local_file_name):
        try:
            from urllib.request import urlopen
        except ImportError:
            from urllib import urlopen
        import shutil

        print('downloading %s from %s...'%(local_file_name,url))
        # urllib.request.urlretrieve(url,local_file_name)
        # The above line doesn't work very well with the SSL certificate that github puts on it.
        # It works fine in a web browser, but on my laptop I get:
        # urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:600)>
        # The solution is to open a context that doesn't do ssl verification.
        # But that can only be done with urlopen, not urlretrieve.  So, here is the solution.
        # cf. http://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
        #     http://stackoverflow.com/questions/27835619/ssl-certificate-verify-failed-error
        try:
            import ssl
            context = ssl._create_unverified_context()
            u = urlopen(url, context=context)
        except (AttributeError, TypeError):
            # Note: prior to 2.7.9, there is no such function or even the context keyword.
            u = urlopen(url)
        with open(local_file_name, 'wb') as out:
            shutil.copyfileobj(u, out)
        u.close()
        print('done.')
