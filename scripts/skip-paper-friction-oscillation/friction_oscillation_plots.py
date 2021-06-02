import sys, os
sys.path.append(os.getcwd())

import h5py
import numpy as np
## regex matching
import re

from multiprocessing import Pool
import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from read_sim_output import read_plotinfo

dim = 2
# gravity
# g_val = -10
g_val = -5e-4

linewidth = 1

#######################################################################
loc = 'output/hdf5/'
plti = read_plotinfo(loc+'plotinfo.h5')

# override
plti.fc = 200
# plti.lc = 45

## # load the main experiment setup data
import load_setup
exp_b = load_setup.read_setup('data/hdf5/all.h5')
mPArr = exp_b.PArr
vol = 0
mass = 0
for i in range(len(mPArr)):
    this_vol =  np.sum(mPArr[i].vol)
    vol += this_vol
    mass += mPArr[i].rho * this_vol 

weight = mass * g_val

print('Total volume:', vol)
print('Total mass:', mass)
print('Total weight:', weight)
#######################################################################

## smoothing
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def force_compute(t):
    print(t, end = ' ', flush=True)
    tc_ind = ('tc_%05d' % t)
    h5_filename = loc+tc_ind+".h5";
    f = h5py.File(h5_filename, "r")

    f_sum = np.zeros((1,dim))
    for name in f:
        # if re.match(r'P_[0-9]+', name):
        if (name == 'P_00000'):
            pid = int(name[2:])
            vol = mPArr[pid].vol
            rho = mPArr[pid].rho
            # this is actually the force density, Not the Force
            # multiply by volume to get the force
            # f_sum += np.sum(vol * np.array(f[name+'/force']), axis = 0)
            f_sum += np.sum(vol * rho * np.array(f[name+'/acc']), axis = 0)
    return f_sum[0]

def vel_compute(t):
    print(t, end = ' ', flush=True)
    tc_ind = ('tc_%05d' % t)
    h5_filename = loc+tc_ind+".h5";
    f = h5py.File(h5_filename, "r")

    v_mean = np.zeros((1,dim))
    for name in f:
        # if re.match(r'P_[0-9]+', name):
        if (name == 'P_00000'):
            pid = int(name[2:])
            v_mean += np.mean(np.array(f[name+'/vel']), axis = 0)
    return v_mean[0]

def u_compute(t):
    print(t, end = ' ', flush=True)
    tc_ind = ('tc_%05d' % t)
    h5_filename = loc+tc_ind+".h5";
    f = h5py.File(h5_filename, "r")

    v_mean = np.zeros((1,dim))
    for name in f:
        # if re.match(r'P_[0-9]+', name):
        if (name == 'P_00000'):
            pid = int(name[2:])
            pid = 0
            v_mean += np.mean(np.array(f[name+'/CurrPos']), axis = 0)
    return v_mean[0]

start = time.time()

# parallel formulation
a_pool = Pool()
f_tot = a_pool.map(force_compute, range(plti.fc, plti.lc+1))
v_mean = a_pool.map(vel_compute, range(plti.fc, plti.lc+1))
u_mean = a_pool.map(u_compute, range(plti.fc, plti.lc+1))
print('')

f_tot = np.array(f_tot)
v_mean = np.array(v_mean)
u_mean = np.array(u_mean)

f_norm = np.sqrt(np.sum(f_tot**2, axis = 1))
np.savetxt("output/csv/force_total.csv", f_tot, delimiter=",")

# tt = np.array(range(plti.fc, plti.lc+1)) * plti.dt * plti.modulo * 1e6
tt = np.array(range(plti.fc, plti.lc+1))

plt.plot(tt, f_tot[:,0], label = r'$F_x$', linewidth=linewidth)
plt.plot(tt, f_tot[:,1], label = r'$F_y$', linewidth=linewidth)
if (dim == 3):
    plt.plot(tt, f_tot[:,2], label = r'$F_z$', linewidth=linewidth)


f_norm_avg = movingaverage(f_norm, 10)

# plt.plot(tt, f_norm, label = 'norm', linewidth=linewidth)
# plt.plot(tt, f_norm_avg, label = 'norm smooth',linewidth=linewidth)

plt.gca().legend()
plt.xlim(min(tt), max(tt))
# plt.show()
save_filename = '/home/debdeep/gdrive/work/peridynamics/granular/data/friction_oscillation_force.png'
print('Saving file:', save_filename)
plt.savefig(save_filename, dpi=300, bbox_inches='tight')
plt.close()

plt.plot(tt, v_mean[:,0], label = r'$v_x$', linewidth=linewidth)
# plt.plot(tt, np.abs(v_mean[:,0]), label = r'$|v_x|$', linewidth=linewidth)
plt.plot(tt, v_mean[:,1], label = r'$v_y$', linewidth=linewidth)
if (dim == 3):
    plt.plot(tt, v_mean[:,2], label = r'$v_z$', linewidth=linewidth)

plt.gca().legend()
plt.xlim(min(tt), max(tt))
# plt.show()
save_filename = '/home/debdeep/gdrive/work/peridynamics/granular/data/friction_oscillation_vel.png'
print('Saving file:', save_filename)
plt.savefig(save_filename, dpi=300, bbox_inches='tight')
plt.close()

plt.plot(tt, u_mean[:,0], label = r'$u_x$', linewidth=linewidth)
# plt.plot(tt, np.abs(v_mean[:,0]), label = r'$|v_x|$', linewidth=linewidth)
plt.plot(tt, u_mean[:,1], label = r'$u_y$', linewidth=linewidth)
if (dim == 3):
    plt.plot(tt, u_mean[:,2], label = r'$u_z$', linewidth=linewidth)

plt.gca().legend()
plt.xlim(min(tt), max(tt))
# plt.show()
save_filename = '/home/debdeep/gdrive/work/peridynamics/granular/data/friction_oscillation_pos.png'
print('Saving file:', save_filename)
plt.savefig(save_filename, dpi=300, bbox_inches='tight')
plt.close()

print('time taken ', time.time() - start)
#######################################################################

