#!/sw64/bin/python2.7
# vim: set filetype=python :

import math
import matplotlib.pyplot as plt

# Read truth file
#truth = open('Aardvark.direct.dat')
truth = open('Aardvark.bs0.out')
truth_r = []
truth_xip = []
truth_xim = []

for line in truth:
    if line[0] == '#':
        continue
    cols = line.strip().split()
    truth_r.append(float(cols[1]))
    truth_xip.append(float(cols[2]))
    truth_xim.append(float(cols[3]))

# Read my output file
mine = open('Aardvark.out')
my_r = []
my_xip = []
my_xim = []

for line in mine:
    if line[0] == '#':
        continue
    cols = line.strip().split()
    my_r.append(float(cols[1]))
    my_xip.append(float(cols[2]))
    my_xim.append(float(cols[3]))

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
