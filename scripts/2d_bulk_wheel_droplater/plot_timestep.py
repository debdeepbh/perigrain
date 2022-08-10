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

import argparse
# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')


# Optional argument
parser.add_argument('--data_dir', type=str, help='input data directory', default='output/hdf5/')
parser.add_argument('--img_dir', type=str, help='output image directory', default='output/img/')
parser.add_argument('--setup_file', type=str, help='output image directory', default='data/hdf5/all.h5')
parser.add_argument('--fc', type=int, help='first counter', default=1)
parser.add_argument('--lc', type=int, help='last counter')
# finish parsing
args = parser.parse_args()

data_dir = args.data_dir
img_dir = args.img_dir
setup_filename = args.setup_file


plot_damage = 1

# plot info
plotinfo_file = data_dir+'/plotinfo.h5'
p = h5py.File(plotinfo_file, "r")
fc = int(p['f_l_counter'][0])
lc = int(p['f_l_counter'][1])
dt = float(p['dt'][0])
modulo = int(p['modulo'][0])

if args.fc:
    fc = args.fc
if args.lc:
    lc = args.lc

def total_neighbors(conn, N):
    """Compute the total number of neighbors from connectivity data
    :conn: connectivity matrix, mx2, m=total intact bonds
    :N: total number of nodes
    :returns: TODO
    """
    deg = np.zeros(N)
    for i in range(len(conn)):
        deg[conn[i][0]] += 1
        deg[conn[i][1]] += 1
    return deg



# ref_connectivity 
f = h5py.File(setup_filename, "r")
ref_n = {}
for name in f:
    if re.match(r'P_[0-9]+', name):
        # pid = int(name[2:])
        ref_connectivity = np.array(f[name+'/Connectivity']).astype(int)
        N = len(f[name+'/Pos'])
        orig_tot_nbrs = total_neighbors(ref_connectivity, N)
        ref_n[name] = orig_tot_nbrs


def genplot(t):
    print(t)
    tc_ind = ('%05d' % t)
    filename = data_dir+'/tc_'+tc_ind+'.h5'
    wall_filename = data_dir+'/wall_'+tc_ind+'.h5'
    out_png = img_dir+'/img_'+tc_ind+'.png'
    f = h5py.File(filename, "r")
    for name in f:
        if re.match(r'P_[0-9]+', name):
            pid = int(name[2:])
            Pos = np.array(f[name+'/CurrPos'])
            c = 'blue'
            if plot_damage:
                # damage
                P_conn = np.array(f[name+'/Connectivity']).astype(int)
                N = len(Pos)
                now_nbrs = total_neighbors(P_conn, N)
                orig_nbrs = np.array(ref_n[name])
                damage = (orig_nbrs - now_nbrs)/ orig_nbrs
                c = damage
                vmin=0
                vmax=1

            plt.scatter(Pos[:,0], Pos[:,1], c=c, s=2, marker='.', linewidth=0, cmap='viridis', vmin=0, vmax=1)


    # wall
    w = h5py.File(wall_filename, "r")
    wi = np.array(w['wall_info'])[:,0]
    a = [wi[0],  wi[3]]
    b = [wi[1], wi[3]]
    c = [wi[1], wi[2]]
    d = [wi[0],  wi[2]]
    ls = [ [a, b], [b,c], [c,d], [d,a] ]
    lc = LineCollection(ls, linewidths=1, colors='b')
    plt.gca().add_collection(lc)

    # colorbar
    plt.colorbar()

    plt.axis('scaled')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()



## serial j
# for t in range(fc, lc+1):
    # genplot(t)

## parallel
a_pool = Pool()
a_pool.map(genplot, range(fc, lc+1))
a_pool.close()
