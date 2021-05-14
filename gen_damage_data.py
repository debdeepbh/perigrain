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
    plti.fc = int(argv[1])
    plti.lc = int(argv[2])

# load the main experiment setup data
exp_b = load_setup.read_setup('data/hdf5/all.h5')
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

# print('Saving to disk: ', npy_file)
# np.save(npy_file, V)

# damage of individual particles in time
# for i in range(len(V[0])):
    # vv = [v[i] for v in V]
    # plt.stem(vv)
    # plt.savefig('stem'+str(i)+'.png')
    # plt.close()

#######################################################################
# damage of all particles

plt.matshow(V)
plt.xlabel('particle')
plt.ylabel('timestep')
plt.colorbar()
plt.clim(0, 1)
# plt.show()
plt.savefig('output/img/damage_mat.png')
plt.close()

#######################################################################
# mean damage of the bulk
bulk_d = [np.mean(d) for d in V]
plt.plot(bulk_d)
plt.savefig('output/img/bulk_damage.png')
plt.close()

#######################################################################
# damage distribution at the end
bins = 10
plt.hist(V[-1], bins=bins)
plt.savefig('output/img/end_damage_hist.png')
plt.close()

#######################################################################
# damage distribution at different time points
for i in range(len(V)):
    plt.hist(V[i], bins=bins)
    plt.xlim([0,1])
    plt.ylim([0,len(V[0])])
    plt.savefig('output/img/damage_hist_'+('%05d' % i)+'.png')
    plt.close()

#######################################################################
# initial particle size and damage
