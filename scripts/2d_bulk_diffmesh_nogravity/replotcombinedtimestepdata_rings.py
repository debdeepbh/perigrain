import sys, os
sys.path.append(os.getcwd())

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from figsize import set_size
from matplotlib.pyplot import figure
fig_size = set_size('amsart')

from sys import argv

avgwindow = 2

thr = 0.99

tag = 'new_rings'

# datadir = '/home/debdeep/from_lockett145/perigrain-data'
# datadir = '/home/debdeep/from_lockett145'

datadir = '/home/debdeep/Downloads/data_granular_rev2_bak/img_paper_rev2'

f = []
#dirs = [  'ring0.3frac', 'ring0.4frac', 'ring0.5frac', 'ring0.6frac', 'ring0.7frac' ]
#dirs = [  'ring0.3frac_oldercommit', 'ring0.4frac_oldercommit', 'ring0.5frac_oldercommit', 'ring0.6frac_oldercommit', 'ring0.7frac' ]
dirs = [  'ring0.3frac', 'ring0.4frac', 'ring0.5frac', 'ringfrac', 'ring0.7frac' ]
#shape = [  '0.3', '0.4', '0.5', '0.6', '0.7' ]
shape = [  r'$\gamma$=0.3', r'$\gamma$=0.4', r'$\gamma$=0.5', r'$\gamma$=0.6', r'$\gamma$=0.7' ]

def movingaverage(array, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(array, window, 'same')

## load all the files
for i,thisdir in enumerate(dirs):
    # input_file = datadir+'/timestep_'+thisdir+'.h5'
    input_file = datadir+'/'+thisdir+'/h5/timestep_data.h5'
    print('input file:', input_file)
    f.append(h5py.File(input_file, 'r'))

#######################################################################
lpair = [r'Volume fraction ($\phi$)', 'Bulk damage']
png_file = 'volfrac_v_damage.png'

figure(figsize=fig_size)
for i in range(len(f)):
    ff = f[i]
    vf = np.array(ff['volume_fraction'])
    bd = np.array(ff['bulk_damage'])
    # threshold for bulk damage to cut off plots
    plt.plot(vf[bd < thr], bd[bd < thr], label=shape[i])

plt.xlim(0.3,0.85)
plt.xlabel(lpair[0])
plt.ylabel(lpair[1])
plt.gca().legend()
full_filename = datadir+'/'+tag+'_'+png_file
print('Saving plot to', full_filename)
plt.savefig(full_filename, bbox_inches='tight')
plt.close()

#######################################################################
lpair = [r'Volume fraction ($\phi$)', 'Top wall force (N)']
png_file = 'volfrac_v_wallf.png'

figure(figsize=fig_size)
for i in range(len(f)):
    ff = f[i]
    
    vf = np.array(ff['volume_fraction'])
    bd = np.array(ff['bulk_damage'])
    wall_reaction = np.array(ff['wall_force'])
    wf = wall_reaction[:,2,1]

    wf = movingaverage(wf, avgwindow)
    # threshold for bulk damage to cut off plots
    # thr = 1.99

    # plt.plot( vf[vf < thr], wf[vf < thr], label=shape[i])
    plt.plot( vf[bd < thr], wf[bd < thr], label=shape[i])

plt.xlim(0.3,0.85)
plt.ylim(-0.02e6,0.25e6)
plt.xlabel(lpair[0])
plt.ylabel(lpair[1])
plt.gca().legend()
full_filename = datadir+'/'+tag+'_'+png_file
print('Saving plot to', full_filename)
plt.savefig(full_filename, bbox_inches='tight')
plt.close()

#######################################################################
lpair = [r'Time (ns)', 'Top wall force (N)']
png_file = 'time_v_wallf.png'

figure(figsize=fig_size)
for i in range(len(f)):
    ff = f[i]
    
    vf = np.array(ff['time'])
    bd = np.array(ff['bulk_damage'])
    wall_reaction = np.array(ff['wall_force'])
    wf = wall_reaction[:,2,1]
    wf = movingaverage(wf, avgwindow)
    # threshold for bulk damage to cut off plots
    # thr = 1.99

    # plt.plot( vf[vf < thr], wf[vf < thr], label=shape[i])
    plt.plot( vf[bd < thr]*1e3, wf[bd < thr], label=shape[i])

plt.xlim(0,0.018*1e3)
plt.ylim(-0.02e6,0.25e6)
plt.xlabel(lpair[0])
plt.ylabel(lpair[1])
plt.gca().legend()
full_filename = datadir+'/'+tag+'_'+png_file
print('Saving plot to', full_filename)
plt.savefig(full_filename, bbox_inches='tight')
plt.close()


#######################################################################
lpair = [r'Time (ns)', 'Bulk damage']
png_file = 'time_v_damage.png'

figure(figsize=fig_size)
for i in range(len(f)):
    ff = f[i]
    vf = np.array(ff['time'])
    bd = np.array(ff['bulk_damage'])
    # threshold for bulk damage to cut off plots
    plt.plot(vf[bd < thr]*1e3, bd[bd < thr], label=shape[i])

plt.xlim(0,0.018*1e3)
plt.xlabel(lpair[0])
plt.ylabel(lpair[1])
plt.gca().legend()
full_filename = datadir+'/'+tag+'_'+png_file
print('Saving plot to', full_filename)
plt.savefig(full_filename, bbox_inches='tight')
plt.close()
#######################################################################
# close all the files
figure(figsize=fig_size)
for i in range(len(f)):
    f[i].close()
