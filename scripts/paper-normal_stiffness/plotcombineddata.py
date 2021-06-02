import sys, os
sys.path.append(os.getcwd())

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sys import argv

from figsize import set_size
from matplotlib.pyplot import figure
fig_size = set_size('amsart')

# input filenames are all except last two args
t_i = len(argv)-3

output_plot_dir = argv[-2]
print('output plot dir ', output_plot_dir)
tag = argv[-1]
print('filename tag', tag)

f = []
shape = []
## load all the files
for i in range(t_i):
    shape.append(argv[i+1])
    input_file = output_plot_dir+argv[i+1]
    print('input file:', input_file)
    f.append(h5py.File(input_file, 'r'))

#######################################################################
pair = ['time', 'particle_mean_CurrPos']
png_file = 'pos_'+tag+'.png'

print(tag)

if tag=='nodamp':
    ll = [
            r'$K_0$',
            r'$K_1$',
            r'$K_2$',
            ]
elif tag == 'damp':
    ll = [
            r'$d_0$',
            r'$d_1$',
            r'$d_2$',
            ]

figure(figsize=fig_size)
for i in range(len(f)):
    ff = f[i]
    wall_reaction = np.array(ff[pair[1]])
    plt.plot(
            1e6 * np.array(ff[pair[0]]),
            # y-value of top wall force
            1e3 * np.array(wall_reaction[:,0,1]),
            label=ll[i],
            )

    tt = 1e6 * np.array(ff[pair[0]])

plt.xlabel(r'Time ($\mu$s)')
plt.ylabel('y-position of centroid (mm/s)')
plt.xlim(min(tt), max(tt))
plt.gca().legend()
# plt.show()
full_filename = output_plot_dir+png_file
print('Saving plot to', full_filename)
plt.savefig(full_filename, bbox_inches='tight')
plt.close()

#######################################################################
pair = ['time', 'particle_mean_vel']
png_file = 'vel_'+tag+'.png'

figure(figsize=fig_size)
for i in range(len(f)):
    ff = f[i]
    wall_reaction = np.array(ff[pair[1]])
    plt.plot(
            1e6 * np.array(ff[pair[0]]),
            # y-value of top wall force
            np.array(wall_reaction[:,0,1]),
            label=ll[i],
            )

    tt = 1e6 * np.array(ff[pair[0]])

plt.xlabel(r'Time ($\mu$s)')
plt.ylabel('y-velocity of centroid (m/s)')
plt.xlim(min(tt), max(tt))
plt.gca().legend()
# plt.show()
full_filename = output_plot_dir+png_file
print('Saving plot to', full_filename)
plt.savefig(full_filename, bbox_inches='tight')
plt.close()

#######################################################################
pair = ['time', 'particle_mean_acc']
png_file = 'acc_'+tag+'.png'

figure(figsize=fig_size)
for i in range(len(f)):
    ff = f[i]
    wall_reaction = np.array(ff[pair[1]])
    plt.plot(
            1e6 * np.array(ff[pair[0]]),
            # y-value of top wall force
            np.array(wall_reaction[:,0,1]),
            label=ll[i],
            )

    tt = 1e6 * np.array(ff[pair[0]])

plt.xlabel(r'Time ($\mu$s)')
plt.ylabel('y-accocity of centroid (m/s)')
plt.xlim(min(tt), max(tt))
plt.gca().legend()
# plt.show()
full_filename = output_plot_dir+png_file
print('Saving plot to', full_filename)
plt.savefig(full_filename, bbox_inches='tight')
plt.close()

#######################################################################
# pair = ['bulk_damage', 'wall_force']
# png_file = 'damage_v_wallf.png'

# for i in range(len(f)):
    # ff = f[i]
    # wall_reaction = np.array(ff[pair[1]])
    # plt.plot(
            # np.array(ff[pair[0]]),
            # # y-value of top wall force
            # np.array(wall_reaction[:,2,1]),
            # label=shape[i],
            # )

# plt.xlabel(pair[0])
# plt.ylabel(pair[1])
# plt.gca().legend()
# plt.show()
# full_filename = output_plot_dir+tag+'_'+png_file
# print('Saving plot to', full_filename)
# plt.savefig(full_filename, bbox_inches='tight')
# plt.close()

# #######################################################################
# pair = ['volume_fraction', 'bulk_damage']
# png_file = 'volfrac_v_damage.png'

# for i in range(len(f)):
    # ff = f[i]
    # plt.plot(
            # np.array(ff[pair[0]]),
            # np.array(ff[pair[1]]),
            # label=shape[i],
            # )

# plt.xlabel(pair[0])
# plt.ylabel(pair[1])
# plt.gca().legend()
# plt.show()
# full_filename = output_plot_dir+tag+'_'+png_file
# print('Saving plot to', full_filename)
# plt.savefig(full_filename, bbox_inches='tight')
# plt.close()

# #######################################################################
# pair = ['volume_fraction', 'wall_force']
# png_file = 'volfrac_v_wallf.png'

# for i in range(len(f)):
    # ff = f[i]
    # wall_reaction = np.array(ff[pair[1]])
    # plt.plot(
            # np.array(ff[pair[0]]),
            # # y-value of top wall force
            # np.array(wall_reaction[:,2,1]),
            # label=shape[i],
            # )

# plt.xlabel(pair[0])
# plt.ylabel(pair[1])
# plt.gca().legend()
# plt.show()
# full_filename = output_plot_dir+tag+'_'+png_file
# print('Saving plot to', full_filename)
# plt.savefig(full_filename, bbox_inches='tight')
# plt.close()

#######################################################################
# close all the files
for i in range(len(f)):
    f[i].close()
