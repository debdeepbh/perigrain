import sys, os
sys.path.append(os.getcwd())

import numpy as np
from multiprocessing import Pool
import time

from sys import argv

import load_setup
from read_sim_output import read_plotinfo, read_force_val

#######################################################################

if len(argv) == 4:
    loc = str(argv[3])

    # saving to file
    npy_file = loc+'V.npy'
    plti = read_plotinfo(loc+'plotinfo.h5')
    dim = plti.dim

    plti.fc = int(argv[1])
    plti.lc = int(argv[2])
else:
    print('Incorrect number of arguments')



## # load the main experiment setup data
exp_b = load_setup.read_setup(loc+'all.h5')
vol = exp_b.total_volume()
mass = exp_b.total_mass()
print('Total volume:', vol)
print('Total mass:', mass)
######################################################################

def compute_all(t):
    return read_force_val(t, loc, dim, get_f_sum=False, input_exp_b=exp_b)

start = time.time()
# parallel formulation
a_pool = Pool()
V = a_pool.map(compute_all, range(plti.fc, plti.lc+1))
a_pool.close()
print('')
print('Time taken: ', time.time() - start)
print('Saving to disk: ', npy_file)
np.save(npy_file, V)