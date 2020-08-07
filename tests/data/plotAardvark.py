#!/sw64/bin/python2.7
# vim: set filetype=python :

import matplotlib.pyplot as plt
import numpy

# Read truth file
if True:
    truth = numpy.loadtxt('Aardvark.bs0.out')
    truth_r = truth[:,1]
    truth_xip = truth[:,2]
    truth_xim = truth[:,3]
else:
    truth = numpy.loadtxt('Aardvark.direct.dat')
    truth_r = truth[:,2]
    truth_xip = truth[:,3]
    truth_xim = truth[:,4]

# Read my output file
mine = numpy.loadtxt('Aardvark.out')
my_r = mine[:,1]
my_xip = mine[:,2]
my_xim = mine[:,3]

#
# Make the plot
#

plt.clf()
plt.loglog(my_r,my_xip,"bo")
plt.loglog(my_r,my_xim,"ro")
plt.loglog(truth_r,truth_xip,"k-")
plt.loglog(truth_r,truth_xim,"k-")
plt.loglog(my_r,[-x for x in my_xip],"b+")
plt.loglog(my_r,[-x for x in my_xim],"r+")
plt.loglog(truth_r,[-x for x in truth_xip],"k:")
plt.loglog(truth_r,[-x for x in truth_xim],"k:")
plt.xlabel('R (arcmin)')
plt.ylabel('xi+(blue), xi-(red)')
plt.title('Aardvark: Brute-force (line) vs. Tree code (circles)')
plt.savefig('Aardvark.png')
