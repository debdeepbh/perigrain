import copy
import h5py
import numpy as np
from multiprocessing import Pool
## regex matching
import re
import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import load_setup

from load_setup import Particle_brief, Experiment_brief


# run_parallel = 0
run_parallel = 1

## # load the setup data
exp_b = load_setup.read_setup('data/hdf5/all.h5')

# wall_color = 'red'
wall_color = ['cyan', 'red', 'yellow', 'blue', 'green', 'black']
# wall_color = ['none', 'none', 'none', 'none', 'none', 'none']
wall_alpha = 0.1

camera_angle = [0, 0]
# camera_angle = [0, 90]

#######################################################################

loc = 'output/hdf5/'
print('dir read for data: ', loc)

p = h5py.File(loc+'plotinfo.h5', "r")
# convert to int, since it is read as double, I think (strange, since hdf5 specifies type)
first_counter = int(p['f_l_counter'][0])
last_counter = int(p['f_l_counter'][1])

# override
# first_counter = 1
# last_counter = 100

print('first counter: ', first_counter)
print('last counter: ', last_counter)


def read_current(t):
    """ Copies the experiment setup class to a new Experiment_brief and outputs it
    :returns: Experiment_brief, with copied setup data and updated motion info
    """
    print(t, end = ' ', flush=True)
    tc_ind = ('tc_%05d' % t)
    filename = loc+tc_ind+".h5";
    f = h5py.File(filename, "r")

    # copy the setup experiment
    # I suspect this will take a while
    t_exp_b = copy.deepcopy(exp_b)
    # t_exp_b = exp_b

    for name in f:
        if re.match(r'P_[0-9]+', name):
            # get particle id (int, starts from 0)
            pid = int(name[2:])
            P = t_exp_b.PArr[pid]

            P.CurrPos = np.array(f[name+'/CurrPos'])
            P.vel     = np.array(f[name+'/vel'])
            # t_exp_b.PArr[pid].acc    = np.array(f[name+'/acc'])
            P.force   = np.array(f[name+'/force'])

            ## Quantity to plot
            # print(P.force)
            P.q = np.sqrt(np.sum(np.square(P.force), axis=1)) #norm
            # print(P.q)
            # q = np.sqrt(np.sum(np.square(P.vel), axis=1)) #norm
            # q = np.abs(P.vel[:,0]) #norm

            # print('mean =', np.mean(P.CurrPos, axis = 0)/1e-3)
            # print('mean =', np.mean(P.CurrPos, axis = 0)/1e-3)
            # rr = print('range of CurrPos = ', (np.max(P.CurrPos, axis = 0) - np.min(P.CurrPos, axis=0))/1e-3)
            # rr = print('range of force = ', (np.max(P.force, axis = 0) - np.min(P.force, axis=0))/1e-3)

    return t_exp_b

def read_current_new(t):
    """ Creates a new Experiment_brief instance
    :returns: Experiment_brief, with copied setup data and updated motion info
    """
    print(t, end = ' ', flush=True)
    tc_ind = ('tc_%05d' % t)
    filename = loc+tc_ind+".h5";
    f = h5py.File(filename, "r")

    PArr= []
    for name in f:
        if re.match(r'P_[0-9]+', name):
            P = Particle_brief( 
                    CurrPos = np.array(f[name+'/CurrPos']),
                    vel     = np.array(f[name+'/vel']),
                    acc     = np.array(f[name+'/acc']),
                    force   = np.array(f[name+'/force'])
                    )
            ## for extracting dimension, in case P.Pos is not loaded (via load_setup.read_setup())
            ## There should be a better way
            # P.pos = np.zeros((len(P.CurrPos), 2))

            ## Quantity to plot
            P.q = np.sqrt(np.sum(np.square(P.force), axis=1)) #norm
            # q = np.sqrt(np.sum(np.square(P.vel), axis=1)) #norm
            # q = np.abs(P.vel[:,0]) #norm

            PArr.append(P)

    return Experiment_brief(PArr = PArr, contact = exp_b.contact, wall = exp_b.wall)

def write_img(t):
    # print(t, end = ' ', flush=True)

    tc_ind = ('tc_%05d' % t)
    out_png = 'output/img/img_'+tc_ind+'.png'

    ## copy: deep or shallow, specify within the function
    t_exp_b = read_current(t)

    ## New instance: slowest
    # t_exp_b = read_current_new(t)

    # limits = np.array([ [-6e-3, 6e-3], [-6e-3, 6e-3] ])
    # limits = np.array([ [-11e-3, 11e-3], [-1e-3, 11e-3] ])
    limits = None
    # camera_angle = None
    camera_angle = [0, t/(last_counter - first_counter)*30]
    t_exp_b.plot(by_CurrPos=True, plot_scatter = True, plot_delta = 0, plot_contact_rad = 0, plot_bonds = 0, plot_bdry_nodes = 0, plot_bdry_edges= 1, edge_alpha = 0.2, plot_wall = 1, plot_wall_faces = False, wall_color=wall_color, wall_alpha=wall_alpha, camera_angle = camera_angle, do_plot = False, do_save = 1, save_filename = out_png, dotsize = 10, plot_vol = False, linewidth = 0.3, limits = limits)

    # dotsize = 10
    # # for P in enumerate(t_exp_b.PArr):
    # for i in range(len(t_exp_b.PArr)):
        # P = t_exp_b.PArr[i]
        # q = np.sqrt(np.sum(np.square(P.force), axis=1)) #norm
        # # q = np.sqrt(np.sum(np.square(vel), axis=1)) #norm
        # # q = np.abs(vel[:,0]) #norm

        # plt.scatter(P.CurrPos[:,0], P.CurrPos[:,1], c = q, s = dotsize, marker = '.', linewidth = 0, cmap='viridis')

    # # saving plot
    # matplotlib.pyplot.savefig(out_png, dpi=200, bbox_inches='tight')
    # plt.close()


## Test
# write_img(1)
# t_exp_b = read_current(1)

start = time.time()

# Serial formulation
# for i in range(first_counter, last_counter+1):
    # write_img(i)


if run_parallel:
    print('Running in parallel.')
    # parallel formulation
    a_pool = Pool()
    a_pool.map(write_img, range(first_counter, last_counter+1))
    print('')
else:
    print('Running in serial.')
    # Serial formulation
    for i in range(first_counter, last_counter+1):
        write_img(i)

# write_img(14)

print('time taken ', time.time() - start)



