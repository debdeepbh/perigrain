import sys, os
sys.path.append(os.getcwd())

from load_setup import read_setup
from read_sim_output import read_plotinfo, populate_current

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

import time
from multiprocessing import Pool

#######################################################################
exp_b = read_setup('data/hdf5/all.h5')

loc = 'output/hdf5/'
plti = read_plotinfo(loc+'plotinfo.h5')


ua = []
ub = []

va = []
vb = []

aa = []
ab = []

for t in range(plti.fc, plti.lc+1):

    tc_ind = ('tc_%05d' % t)
    h5_filename = loc+tc_ind+".h5";

    t_exp_b = populate_current(h5_filename, exp_b, q = None, read_CurrPos=True, read_vel=True, read_acc=True)

    # t_exp_b = read_current(i)
    A = t_exp_b.PArr[0]
    B = t_exp_b.PArr[1]

    ua.append(np.mean(A.CurrPos[:,1]))
    ub.append(np.mean(B.CurrPos[:,1]))

    va.append(np.mean(A.vel[:,1]))
    vb.append(np.mean(B.vel[:,1]))

# in micro sec
tt = np.array(range(plti.fc, plti.lc+1)) * plti.dt * plti.modulo * 1e6


#######################################################################
# position

save_filename = '/home/debdeep/gdrive/work/peridynamics/granular/data/2d-collision-elastic-CurrPos.png'
plt.plot(tt, ua, label = 'A')
plt.plot(tt, ub, label = 'B')
plt.xlabel(r'Time ($\mu$s)')
plt.ylabel(r'$y$-position of center (m)')

plt.xlim(min(tt), max(tt))
plt.gca().legend()
plt.gca().grid(True)
plt.savefig(save_filename, dpi=300, bbox_inches='tight')
# plt.show()
plt.close()

#######################################################################
# vel

save_filename = '/home/debdeep/gdrive/work/peridynamics/granular/data/2d-collision-elastic-vel.png'
plt.plot(tt, va, label = 'A')
plt.plot(tt, vb, label = 'B')
plt.xlabel(r'Time ($\mu$s)')
plt.ylabel(r'$y$-velocity of center (m)')

plt.xlim(min(tt), max(tt))
plt.gca().legend()
plt.gca().grid(True)
plt.savefig(save_filename, dpi=300, bbox_inches='tight')
# plt.show()
plt.close()
