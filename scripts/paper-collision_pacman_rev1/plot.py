import sys, os
sys.path.append(os.getcwd())

import load_setup
from read_sim_output import read_plotinfo, read_run_time, populate_current, read_wallinfo

# import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

## # load the main experiment setup data
exp_b = load_setup.read_setup('data/hdf5/all.h5')

# limits = [ [-2e-3,3e-3], [ -2e-3,6e-3]]
limits = [ [-2e-3,3e-3], [ -2e-3, 3e-3]]

# t = 1

def plot(t):
    print(t, end = ' ', flush=True)

    tc_ind = ('%05d' % t)
    h5_filename = 'output/hdf5/tc_'+tc_ind+'.h5'
    t_exp_b = populate_current(h5_filename, exp_b, q = 'damage', read_CurrPos=True, read_vel=False, read_acc=False, read_force=True, read_connectivity=True)

    wall_filename = 'output/hdf5/wall_'+tc_ind+'.h5'
    wi = read_wallinfo(wall_filename)[:,0]
    t_exp_b.wall.left        = wi[0]
    t_exp_b.wall.right       = wi[1]
    t_exp_b.wall.top         = wi[2] 
    t_exp_b.wall.bottom      = wi[3]

    out_png = 'output/img/img_tc_'+tc_ind+'.png'

    t_exp_b.plot(by_CurrPos=True, plot_scatter = True, plot_delta = 0, plot_contact_rad = 0, plot_bonds = 0, plot_bdry_nodes = 0, plot_bdry_edges= 0, edge_alpha = 0.2, plot_wall = 1, plot_wall_faces = False, wall_color=None, wall_linewidth=1, wall_alpha=1, camera_angle = None, do_plot = False, do_save = 1, save_filename = out_png, dotsize = 5, plot_vol = False, linewidth = 0.5, limits=limits, remove_axes=False, grid =True, colorbar=True, colorlim=[0,1], seaborn=True)


a_pool = Pool()
# a_pool.map(plot, range(1, 21))
a_pool.map(plot, range(15, 26))
a_pool.close()
