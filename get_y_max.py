import numpy as np
import sys

from load_setup import read_setup
from read_sim_output import read_plotinfo, populate_current

## # load the main experiment setup data
exp_b = read_setup('data/hdf5/all.h5', print_output=False)

loc = 'output/hdf5/'
plti = read_plotinfo(loc+'plotinfo.h5', read_dim=True, quiet=False)
# load the last timestep data
tc_ind = ('tc_%05d' % plti.lc)
h5_filename = loc+tc_ind+".h5";
t_exp_b = populate_current(h5_filename, exp_b, read_CurrPos=True)

PArr = t_exp_b.PArr

M = []
for i in range(len(PArr)):
    P = PArr[i]
    for j in range(len(P.CurrPos)):
        M.append(np.max(P.CurrPos[:,1]))
y_max = np.max(M)



# add a height to the y_max for output
t_i = len(sys.argv)
if t_i > 1:
    extra = float(sys.argv[1])
else:
    extra = 0

# output
print(y_max + extra)

