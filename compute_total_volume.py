import numpy as np
from multiprocessing import Pool
import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import load_setup
from read_sim_output import read_plotinfo, read_run_time, populate_current

## # load the main experiment setup data
exp_b = load_setup.read_setup('data/hdf5/all.h5')

PArr = exp_b.PArr

vol = 0
mass = 0
for i in range(len(PArr)):
    this_vol =  np.sum(PArr[i].vol)
    vol += this_vol
    mass += PArr[i].rho * this_vol 

print('Total volume, mass: ', vol, mass)
