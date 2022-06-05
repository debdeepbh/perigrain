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

tag = 'new_shapes'

datadir = '/home/debdeep/perigrain/scripts/2d_bulk_diffmesh_nogravity/data'

f = []
dirs = [  'plusfrac', '0.4frac',        'n4frac', 'ringfrac' ]
shape = ['Plus',     'Perturbed disk', 'Square', 'Ring']

## load all the files
for i,thisdir in enumerate(dirs):
    input_file = datadir+'/timestep_'+thisdir+'.h5'
    print('input file:', input_file)
    f.append(h5py.File(input_file, 'r'))

#######################################################################
pair = ['volume_fraction', 'bulk_damage']
lpair = [r'Volume fraction ($\phi$)', 'Bulk damage']
png_file = 'volfrac_v_damage.png'

figure(figsize=fig_size)
for i in range(len(f)):
    ff = f[i]
    plt.plot(
            np.array(ff[pair[0]]),
            np.array(ff[pair[1]]),
            label=shape[i],
            )

plt.xlim(0,1)
plt.xlabel(lpair[0])
plt.ylabel(lpair[1])
plt.gca().legend()
plt.show()
full_filename = datadir+'/'+tag+'_'+png_file
print('Saving plot to', full_filename)
plt.savefig(full_filename, bbox_inches='tight')
plt.close()

#######################################################################
# close all the files
figure(figsize=fig_size)
for i in range(len(f)):
    f[i].close()
