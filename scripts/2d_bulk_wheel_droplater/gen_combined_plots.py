import numpy as np
# import pandas as pd
import h5py

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.append(os.getcwd())

from sys import argv

from load_setup import read_setup

# last two arg will be the output filename
t_i = len(argv)-3
str_pref = argv[-2]
print('str_pref is = ', str_pref)

# print('range: ', range(t_i))

d = []
damage = []
labels = []
vols = []
for i in range(t_i):
    shape = argv[i+1]
    print('shape', shape)
    # dir=${str_pref}$std$1
    d.append(np.load(str_pref+shape+'/h5/V.npy', allow_pickle=True))
    damage.append(np.load(str_pref+shape+'/h5/damage.npy', allow_pickle=True))
    # print('loading'+str_pref+shape+'.npy')
    labels.append(shape)


    ## # load the main experiment setup data
    exp_b = read_setup(str_pref+shape+'/h5/all.h5')
    vols.append(exp_b.total_volume())
    # mass = exp_b.total_mass()
    # print('Total volume:', vol)
    # print('Total mass:', mass)

# random identifying name to attach to filename in the beginning
t_name = argv[-1]
    
filename = str_pref+t_name+'_df_forces.h5'
print('saving to', filename)
df_forces = h5py.File(filename, 'w')

## avg pressure vs volume fraction
for i in range(t_i):
    # lower bound
    lb = 1
    ub = len(d[i])

    # if labels[i]=='disk':
        # ub = 40
    # if labels[i]=='n8':
        # lb = 1
    # if labels[i]=='n4':
        # lb = 100
    # if labels[i]=='plus0.8':
        # # ub = 100
        # pass

    d[i] = d[i][lb:ub]

    phi = [vols[i]/(v.wall_v * v.wall_h) for v in d[i]]
    # print('phi = ', phi)

    # top wall force 
    # plt.plot([ np.abs(v.wall_reaction[2,1]) for v in d[i]],  label = r'$\sigma=$ '+labels[i])
    
    # volume fraction vs top wall force
    data = [ np.abs(v.wall_reaction[2,1]) for v in d[i]]
    plt.plot(phi, data,  label = r'$\sigma=$ '+labels[i])

    # df_forces.create_dataset(labels[i], data=np.array([phi, data]))
    df_forces.create_dataset(labels[i]+'/volfrac', data=phi)
    df_forces.create_dataset(labels[i]+'/topforce', data=data)

    # total force in time
    # plt.plot([ (np.abs(v.wall_reaction[3,1]) +
    # np.abs(v.wall_reaction[2,1]) +
    # np.abs(v.wall_reaction[0,0]) +
    # np.abs(v.wall_reaction[1,0]))/4
        # for v in d[i]],  label = r'$\sum |F|$ '+labels[i])

    # avg pressure vs volume fraction
    # plt.plot([ (np.abs(v.wall_reaction[3,1]/v.wall_h) +
    # np.abs(v.wall_reaction[2,1]/v.wall_h) +
    # np.abs(v.wall_reaction[0,0]/v.wall_v) +
    # np.abs(v.wall_reaction[1,0]/v.wall_v))/4
        # for v in d[i]], phi,  label = r'$|p|$ avg '+labels[i])

    # total force vs vol frac
    # plt.plot([ (np.abs(v.wall_reaction[3,1]) +
    # np.abs(v.wall_reaction[2,1]) +
    # np.abs(v.wall_reaction[0,0]) +
    # np.abs(v.wall_reaction[1,0]))/4
        # for v in d[i]], phi,  label = r'$\sum |F|$ '+labels[i])

    # Deviatoric
    # plt.plot(
            # [
                # (np.abs(v.wall_reaction[2,1])/v.wall_h) /
                # (np.abs(v.wall_reaction[1,0])/v.wall_v)
                # for v in d[i]],
            # label = r'$\sum |F|$ '+labels[i]
            # )

df_forces.close()
# force limits x axis
# plt.xlim(20, 40)

plt.ylabel(r'$|F_y^{top}|$ (N)')
# plt.ylabel('Volume fraction')


plt.grid()
# plt.autoscale(tight=True)
plt.gca().legend()
# plt.show()
# plt.savefig(argv[-1], dpi=300, bbox_inches='tight')
filename = str_pref+t_name+'_force_plot.png'
print('saving to', filename)
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

#######################################################################
# df_damage = pd.DataFrame()

filename = str_pref+t_name+'_df_damage.h5'
print('saving to', filename)
df_damage = h5py.File(filename, 'w')

for i in range(t_i):
    phi = [vols[i]/(v.wall_v * v.wall_h) for v in d[i]]
    data = [ np.mean(dam) for dam in damage[i]]
    plt.plot(data,  label = r'$\sigma=$ '+labels[i])

    # df_damage[labels[i]] =  data
    df_damage.create_dataset(labels[i], data=data)
df_damage.close()

plt.ylabel('damage')

plt.grid()
plt.gca().legend()
filename = str_pref+t_name+'_damage_plot.png'
print('saving to', filename)
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

