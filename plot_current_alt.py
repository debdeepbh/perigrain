## Full implementation of plotting 
import numpy as np
from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool
import time
import os.path

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')

from sys import argv

import load_setup
from read_sim_output import read_plotinfo, read_run_time, populate_current, read_wallinfo

run_parallel = 0
# run_parallel = 1

## quantity to plot in color
# q_col = 'force_norm'
q_col = 'damage'

# cmap_name = 'viridis'
# cmap_name = 'Greys'
cmap_name = 'cividis'

## # load the main experiment setup data
exp_b = load_setup.read_setup('data/hdf5/all.h5')

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



# override here
if len(argv) == 2:
    plti.fc = int(argv[1])
if len(argv) == 3:
    plti.fc = int(argv[1])
    plti.lc = int(argv[2])
    print('argv1,2=', argv[1], argv[2])
    print('setting plti', plti.fc, 'to', plti.lc)

def write_img(t):
    print(t, end = ' ', flush=True)

    tc_ind = ('tc_%05d' % t)
    out_png = 'output/img/img_'+tc_ind+'.png'

    h5_filename = loc+tc_ind+".h5";
    # t_exp_b = populate_current(h5_filename, exp_b, q = 'force_norm')
    t_exp_b = populate_current(h5_filename, exp_b, q = q_col, read_CurrPos=True, read_vel=False, read_acc=False, read_force=False, read_connectivity=True)

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
        
    PArr = t_exp_b.PArr

    for i in range(len(PArr)):
        P = PArr[i]
            plt.scatter(P.CurrPos[:,0], P.CurrPos[:,1], s = 5, c = P.q, linewidth=0)
            plt.clim(0, 1)


    plt.colorbar()
    plt.clim(0, 1)
    plt.axis('scaled')
    # plt.savefig('testimg_'+str(t)+'.png', bbox_inches='tight')
    plt.show()
    plt.close()

start = time.time()

if run_parallel:
    print('Running in parallel.', plti.fc, 'to', plti.lc)
    # parallel formulation
    a_pool = Pool()
    a_pool.map(write_img, range(plti.fc, plti.lc+1))
    a_pool.close()
    print('')
else:
    print('Running in serial.')
    # Serial formulation
    for i in range(plti.fc, plti.lc+1):
        write_img(i)

print('time taken ', time.time() - start)

