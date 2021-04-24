import sys, os
sys.path.append(os.getcwd())

import numpy as np
from multiprocessing import Pool
import time

import matplotlib.pyplot as plt

from sys import argv

import load_setup
from read_sim_output import read_plotinfo, read_bar_val

# saving to file
npy_file = 'output/V.npy'

#######################################################################
loc = 'output/hdf5/'
plti = read_plotinfo(loc+'plotinfo.h5')
dim = plti.dim

## # load the main experiment setup data
exp_b = load_setup.read_setup('data/hdf5/all.h5')
vol = exp_b.total_volume()
mass = exp_b.total_mass()
print('Total volume:', vol)
print('Total mass:', mass)
######################################################################

nodes_right_edge = np.load('output/2d_bar_r_edge.npy')
E = np.load('output/youngs.npy')

def compute_all(t):
    # return read_bar_val(t, loc, dim, get_f_sum=True, input_exp_b=exp_b)
    return read_bar_val(t, loc, dim, input_exp_b=exp_b, nodes_right_edge=nodes_right_edge)

start = time.time()
# parallel formulation
a_pool = Pool()
V = a_pool.map(compute_all, range(plti.fc, plti.lc+1))
a_pool.close()

# cut off half
# V = V[50:]

# force = plti.fc * 

# print('Saving to disk: ', npy_file)
# np.save(npy_file, V)


l = float(argv[1])
s = float(argv[2])
Force = float(argv[3])

modulo = 200
extforce_maxstep = 10000


gradients = np.linspace(1, 100, num=100)*modulo/extforce_maxstep
gradients[gradients > 1] = 1
# plt.plot(gradients)

# strain = np.array([((v.x_max - (-10e-3)) - 20e-3)/20e-3 for v in V])
strain = np.array([((v.r_avg - (-l)) - 2*l)/(2*l) for v in V])
# strain = np.array([((v.r_min - (-l)) - 2*l)/(2*l) for v in V])

# multiplying by the number of nodes on which the force is applied
# total force on the body
extforce = gradients * Force * len(nodes_right_edge)

stress = np.array([f/(2*s) for f in extforce])

# print(stress)

Es = [f/(2*s)/eps for (f, eps) in zip(extforce, strain)]

# plt.plot( strain, stress)
# plt.plot([v.x_max/1e-3 for v in V])
# plt.plot(strain)
# plt.plot(stress)

# plt.plot(E*strain)
# plt.plot(stress)
# plt.plot(Es)
# plt.plot(extforce)
plt.plot(E*strain/stress)

fixed = 50
# plt.plot( strain, stress[fixed] + (strain - strain[fixed])*E)


plt.grid()
plt.gca().legend()
filename = 'output/img/bar_pull.png'
print('Saving to :', filename)
plt.savefig(filename, dpi=300, bbox_inches='tight')

plt.show()
plt.close()
