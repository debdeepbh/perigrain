
import sys, os
sys.path.append(os.getcwd())

import numpy as np
from multiprocessing import Pool
import time
import os.path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import load_setup
from read_sim_output import read_plotinfo, read_run_time, populate_current, read_wallinfo

# run_parallel = 0
run_parallel = 1

## # load the main experiment setup data
exp_b = load_setup.read_setup('data/hdf5/all.h5')

# wall_color = 'red'
wall_color = ['cyan', 'red', 'yellow', 'blue', 'green', 'black']
wall_alpha = 0.5

camera_angle = [0, 0]
# camera_angle = [0, 90]

plot_run_time = 0
#######################################################################

loc = 'output/hdf5/'
plti = read_plotinfo(loc+'plotinfo.h5', read_dim=True)

# plot runtime
if plot_run_time:
    rt = read_run_time(loc+'run_time.h5')
    plt.plot(rt[:,0], rt[:,1])
    plt.xlim(min(rt[:,0]), max(rt[:,0]))
    plt.savefig('output/img/run_time_'+str(rt[0,0])+'.png', dpi=300, bbox_inches='tight')


# plt.show()
# override
# plti.fc = 30
# plti.lc = 60

# plti.print()

def write_img(t):
    print(t, end = ' ', flush=True)

    tc_ind = ('tc_%05d' % t)
    out_png = 'output/img/img_'+tc_ind+'.png'

    h5_filename = loc+tc_ind+".h5";
    t_exp_b = populate_current(h5_filename, exp_b, q = 'force_norm')

    
    # load wall info
    wall_ind = ('wall_%05d' % t)
    wall_filename = loc+wall_ind+".h5"
    if os.path.isfile(wall_filename):
        wi = read_wallinfo(wall_filename)[:,0]
        if (plti.dim ==2):

            # print(wi)
            t_exp_b.wall.left        = wi[0]
            t_exp_b.wall.right       = wi[1]
            t_exp_b.wall.top         = wi[2] 
            t_exp_b.wall.bottom      = wi[3]
        else:
            print('3D moving wall plotting is not implemented.')
        



    # limits = np.array([ [-6e-3, 6e-3], [-6e-3, 6e-3] ])

    # wall_left   = -25e-3
    # wall_right  = 25e-3
    # wall_top    = 50e-3
    # wall_bottom = -50e-3
    # limits = None
    limits = np.array([ [-25e-3, 25e-3], [-50e-3, 0e-3] ])

    # camera_angle = None
    camera_angle = [0, t/(plti.lc - plti.fc)*30]

    # colorlim = [0, 1e9]
    colorlim =None

    t_exp_b.plot(by_CurrPos=True, plot_scatter = True, plot_delta = 0, plot_contact_rad = 0, plot_bonds = 0, plot_bdry_nodes = 0, plot_bdry_edges= 0, edge_alpha = 0.2, plot_wall = 1, plot_wall_faces = False, wall_color=wall_color, wall_linewidth=1, wall_alpha=wall_alpha, camera_angle = camera_angle, do_plot = False, do_save = 1, save_filename = out_png, dotsize = 1, plot_vol = False, linewidth = 0.5, limits = limits, remove_axes=False, grid =True, colorbar=False, colorlim=colorlim)

    # t_exp_b.plot(by_CurrPos=True, plot_scatter = True, plot_delta = 0, plot_contact_rad = 0, plot_bonds = 0, plot_bdry_nodes = 0, plot_bdry_edges= 0, edge_alpha=0.2, plot_wall = 1, plot_wall_faces = False, wall_color=wall_color, wall_alpha=wall_alpha, do_plot = False, do_save = 1, save_filename = out_png, dotsize = 10, linewidth = 1, remove_axes = True, grid = False)

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

start = time.time()

if run_parallel:
    print('Running in parallel.')
    # parallel formulation
    a_pool = Pool()
    a_pool.map(write_img, range(plti.fc, plti.lc+1))
    print('')
else:
    print('Running in serial.')
    # Serial formulation
    for i in range(plti.fc, plti.lc+1):
        write_img(i)

# write_img(14)

print('time taken ', time.time() - start)



