import sys, os
sys.path.append(os.getcwd())

import numpy as np
from multiprocessing import Pool
import time

from sys import argv

import load_setup
from read_sim_output import read_plotinfo
from exp_dict import Wall, Wall3d
from read_sim_output import update_wallinfo, populate_current

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')


if len(argv) == 4:
    loc = str(argv[3])
    # saving to file
    npy_file = loc+'damage.npy'
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
    print(t, end = ' ', flush=True)

    ## update wall data
    # if (dim ==2):
        # wall = Wall()
    # else:
        # wall = Wall3d()
    # print(t, end = ' ', flush=True)
    # wall_ind = ('wall_%05d' % t)
    # wall_filename = loc+wall_ind+".h5";
    # update_wallinfo(wall_filename, wall)
    

    # update PArr
    tc_ind = ('tc_%05d' % t)
    h5_filename = loc+tc_ind+".h5";

    t_exp_b = populate_current(h5_filename, exp_b, q='damage', read_CurrPos=False, read_vel=False, read_acc=False, read_force=False, read_connectivity=True)

    PArr = t_exp_b.PArr
    ## total damage of the bulk of particles 
    # d_sum = np.zeros((1, dim))

    # d_sum = 0
    d_sum = np.zeros(len(PArr))
    for i in range(len(PArr)):
        # d_sum += np.sum(PArr[i].q, axis =0)

        ## total damage of the particle
        # d_sum[i] = np.sum(PArr[i].q, axis =0)

        ## mean damage of the particle
        d_sum[i] = np.mean(PArr[i].q, axis =0)

    # print(d_sum)
    return d_sum

start = time.time()
# parallel formulation
a_pool = Pool()
V = a_pool.map(compute_all, range(plti.fc, plti.lc+1))
a_pool.close()
print('')
print('Time taken: ', time.time() - start)

print('Saving to disk: ', npy_file)
np.save(npy_file, V)

