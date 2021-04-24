import sys, os
sys.path.append(os.getcwd())

from sys import argv

from load_setup import read_setup
from read_sim_output import read_plotinfo, populate_current

import numpy as np

#######################################################################
exp_b = read_setup('data/hdf5/all.h5')

loc = 'output/hdf5/'
plti = read_plotinfo(loc+'plotinfo.h5')

ua = []
# ub = []

va = []
# vb = []

aa = []
# ab = []

for t in range(plti.fc, plti.lc+1):

    tc_ind = ('tc_%05d' % t)
    h5_filename = loc+tc_ind+".h5";

    t_exp_b = populate_current(h5_filename, exp_b, q = None, read_CurrPos=True, read_vel=True, read_acc=True)

    # t_exp_b = read_current(i)
    A = t_exp_b.PArr[0]
    # B = t_exp_b.PArr[1]

    ua.append(np.mean(A.CurrPos[:,1]))
    # ub.append(np.mean(B.CurrPos[:,1]))

    va.append(np.mean(A.vel[:,1]))
    # vb.append(np.mean(B.vel[:,1]))

    aa.append(np.mean(A.acc[:,1]))

# in micro sec
tt = np.array(range(plti.fc, plti.lc+1)) * plti.dt * plti.modulo * 1e6

np.savetxt(argv[1], np.c_[tt, ua, va, aa], delimiter=',')

