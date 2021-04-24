import sys, os
sys.path.append(os.getcwd())

import numpy as np
from multiprocessing import Pool
import time

from sys import argv

import load_setup
from read_sim_output import read_plotinfo, update_wallinfo, populate_current
from exp_dict import Wall, Wall3d

# saving to file
npy_file = 'output/V.npy'

#######################################################################
loc = 'output/hdf5/'
plti = read_plotinfo(loc+'plotinfo.h5')
dim = plti.dim

# override here
if len(argv) == 2:
    plti.fc = int(argv[1])
if len(argv) == 3:
    plti.lc = int(argv[2])

## # load the main experiment setup data
exp_b = load_setup.read_setup('data/hdf5/all.h5')
vol = exp_b.total_volume()
mass = exp_b.total_mass()
print('Total volume:', vol)
print('Total mass:', mass)

######################################################################

class Val(object):
    """docstring for ClassName"""
    def __init__(self, f_sum, wall_v, wall_h, wall_reaction, bar_l):
        self.f_sum = f_sum
        self.wall_reaction = wall_reaction
        self.wall_v = wall_v
        self.wall_h = wall_h

        self.bar_l = bar_l
        
## smoothing
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def read_force_val(t, loc, dim, get_f_sum=False, input_exp_b=None, get_bar_l=True):
    """ Reads the wall reaction force.
    : get_f_sum: if true, reads the total force of all particles, provided input_exp_b is given
    : returns: class Val
    """
    if (dim ==2):
        wall = Wall()
    else:
        wall = Wall3d()
    print(t, end = ' ', flush=True)
    wall_ind = ('wall_%05d' % t)
    wall_filename = loc+wall_ind+".h5";
    update_wallinfo(wall_filename, wall)


    if get_f_sum:
        tc_ind = ('tc_%05d' % t)
        h5_filename = loc+tc_ind+".h5";
        # t_exp_b = populate_current(h5_filename, input_exp_b, q = None, read_CurrPos=False, read_vel=False, read_acc=False, read_force=True)
        t_exp_b = populate_current(h5_filename, input_exp_b, q = None, read_CurrPos=True, read_vel=False, read_acc=True, read_force=False)
        PArr = t_exp_b.PArr
        f_sum = np.zeros((1, dim))
        for i in range(len(PArr)):
            # f_sum += np.sum(PArr[i].vol * PArr[i].force, axis =0)
            f_sum += np.sum(PArr[i].vol * PArr[i].rho * PArr[i].acc, axis =0)
        f_sum = f_sum[0]
        bar_l = PArr[i].CurrPos[1][0] - PArr[i].CurrPos[0][0]
    else:
        f_sum = None



    return Val(f_sum=f_sum, wall_v=wall.get_v(), wall_h=wall.get_h(), wall_reaction=wall.reaction, bar_l=bar_l)


#######################################################################


def compute_all(t):
    return read_force_val(t, loc, dim, get_f_sum=True, input_exp_b=exp_b, get_bar_l=True)

start = time.time()
# parallel formulation
a_pool = Pool()
V = a_pool.map(compute_all, range(plti.fc, plti.lc+1))
a_pool.close()
print('')
print('Time taken: ', time.time() - start)
print('Saving to disk: ', npy_file)
np.save(npy_file, V)
