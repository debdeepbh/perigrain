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

tag = 'new_rings'

datadir = '/home/debdeep/from_lockett145/perigrain-data'

f = []
dirs = [  'ring0.3frac', 'ring0.4frac', 'ring0.5frac', 'ring0.6frac', 'ring0.7frac' ]
shape = [  '0.3', '0.4', '0.5', '0.6', '0.7' ]


## load all the files
for i,thisdir in enumerate(dirs):
    input_file = datadir+'/timestep_'+thisdir+'.h5'
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
    thr = 0.99
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
    # threshold for bulk damage to cut off plots
    thr = 0.99
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
lpair = [r'Time ($\mu s$)', 'Top wall force (N)']
png_file = 'time_v_wallf.png'

figure(figsize=fig_size)
for i in range(len(f)):
    ff = f[i]
    
    vf = np.array(ff['time'])
    bd = np.array(ff['bulk_damage'])
    wall_reaction = np.array(ff['wall_force'])
    wf = wall_reaction[:,2,1]
    # threshold for bulk damage to cut off plots
    thr = 0.99
    # thr = 1.99

    # plt.plot( vf[vf < thr], wf[vf < thr], label=shape[i])
    plt.plot( vf[bd < thr], wf[bd < thr], label=shape[i])

plt.xlim(0,0.018)
plt.ylim(-0.02e6,0.25e6)
plt.xlabel(lpair[0])
plt.ylabel(lpair[1])
plt.gca().legend()
full_filename = datadir+'/'+tag+'_'+png_file
print('Saving plot to', full_filename)
plt.savefig(full_filename, bbox_inches='tight')
plt.close()


#######################################################################
lpair = [r'Time ($\mu$s)', 'Bulk damage']
png_file = 'time_v_damage.png'

figure(figsize=fig_size)
for i in range(len(f)):
    ff = f[i]
    vf = np.array(ff['time'])
    bd = np.array(ff['bulk_damage'])
    # threshold for bulk damage to cut off plots
    thr = 0.99
    plt.plot(vf[bd < thr], bd[bd < thr], label=shape[i])

plt.xlim(0,0.018)
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
