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

## # load the main experiment setup data
exp_b = load_setup.read_setup('data/hdf5/all.h5')

# wall_color = 'red'
wall_color = ['cyan', 'red', 'yellow', 'blue', 'green', 'black']
wall_alpha = 0.5

limits = np.array([ [-25e-3, 25e-3], [-50e-3, 50e-3] ])
# limits = None
camera_angle = None
# colorlim = [0, 1e9]
colorlim =None

plot_run_time = 1
#######################################################################

loc = 'output/hdf5/'
plti = read_plotinfo(loc+'plotinfo.h5', read_dim=True)

# plot runtime
if plot_run_time:
    rt = read_run_time(loc+'run_time.h5')
    plt.plot(rt[:,0], rt[:,1])
    plt.xlim(min(rt[:,0]), max(rt[:,0]))
    plt.savefig('output/img/run_time_'+str(rt[0,0])+'.png', dpi=300, bbox_inches='tight')

# override
plti.fc = 11
# plti.lc = 450

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
            t_exp_b.wall.left        = wi[0]
            t_exp_b.wall.right       = wi[1]
            t_exp_b.wall.top         = wi[2] 
            t_exp_b.wall.bottom      = wi[3]
        else:
            print('3D moving wall plotting is not implemented.')

    t_exp_b.plot(by_CurrPos=True, plot_scatter = True, plot_delta = 0, plot_contact_rad = 1, plot_bonds = 0, plot_bdry_nodes = 0, plot_bdry_edges= 1, edge_alpha = 0.2, plot_wall = 1, plot_wall_faces = False, wall_color=wall_color, wall_linewidth=1, wall_alpha=wall_alpha, camera_angle = camera_angle, do_plot = False, do_save = 1, save_filename = out_png, dotsize = 1, plot_vol = False, linewidth = 0.5, limits = limits, remove_axes=True, grid =True, colorbar=False, colorlim=colorlim)


start = time.time()
# parallel formulation
a_pool = Pool()
a_pool.map(write_img, range(plti.fc, plti.lc+1))
print('')
print('time taken ', time.time() - start)



