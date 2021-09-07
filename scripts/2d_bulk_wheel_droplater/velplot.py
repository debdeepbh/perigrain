import numpy as np
import h5py

import re
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.collections import LineCollection
from sys import argv

# last arg will be the data directory (with path)
datadir = argv[-1]
print('datadir', datadir)

shapes = []
for i in range(len(argv)-2):
    shapes.append(argv[i+1])
print('shapes', shapes)

def func(t, dirname, modulo, dt):
    # print(t)
    tc_ind = ('%05d' % t)
    filename = dirname+'/h5/tc_'+tc_ind+'.h5'
    f = h5py.File(filename, "r")

    mean_v = []
    for name in f:
        if re.match(r'P_[0-9]+', name):
            vel = np.array(f['P_00000'+'/vel'])
            mean_v = np.mean(vel, axis = 0)
    # time
    tt = t * modulo * dt/1e-6
    return np.array([tt, mean_v[0]]) 



a_pool = Pool()
for i, shape in enumerate(shapes):
    dirname = datadir+'/'+shape
    print('dirname', dirname)

    p = h5py.File(dirname+'/h5/plotinfo.h5', "r")
    fc = int(p['f_l_counter'][0])
    lc = int(p['f_l_counter'][1])
    dt = float(p['dt'][0])
    modulo = int(p['modulo'][0])

    ## parallel
    t_range = range(fc, lc+1)
    n = len(t_range)
    out = np.array(a_pool.map(func, t_range, [dirname]*n, [modulo]*n, [dt]*n))
    # print(out)
    plt.plot(out[:,0], out[:,1], label=shape)

plt.xlabel('time')
plt.ylabel(r'$v_x$')
plt.gca().legend()
png_filename = datadir+'/velplot.png'
print('Saving plot to', png_filename)
plt.savefig(png_filename, bbox_inches='tight')
plt.close()
