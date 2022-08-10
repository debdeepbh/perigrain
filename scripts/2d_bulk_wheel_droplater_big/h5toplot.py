# Run is from it path

import h5py
import seaborn as sns
import numpy as np
sns.set()
import matplotlib.pyplot as plt

print('Run it from its path, i.e. cd to scripts/2d_bulk_diffmesh_nogravity/, and then run python3 h5toplot.py')

# filename prefix 
# t_name = 'shapes'
# legends = ['Perturbed disk', 'Square', 'Plus']

t_name = 'roundness'
legends = [
        r'$\sigma=$'+str(0.1),
        r'$\sigma=$'+str(0.2),
        r'$\sigma=$'+str(0.3),
        r'$\sigma=$'+str(0.4),
        ]
#######################################################################

filename = t_name+'_df_forces.h5'

with h5py.File(filename, 'r') as f:
    print(f.keys())
    png_file = t_name+'_pl_forces_v_damage.png'

    # for name in f:
    for name, legend in zip(f, legends):
        # print(f[name])
        # plt.plot(np.array(f[name+'/volfrac']), np.array(f[name+'/topforce']))
        plt.plot(
                np.array(f[name+'/volfrac']),
                np.array(f[name+'/topforce']),
                label=legend,
                )


    plt.xlim(0, 1)

    plt.xlabel('Volume fraction')
    plt.ylabel('Force (N)')

    plt.gca().legend()
    # plt.show()
    plt.savefig(png_file, bbox_inches='tight')
    plt.close()
    
    ############################
    png_file = t_name+'_pl_forces.png'
    # for name in f:
    for name, legend in zip(f, legends):
        # print(f[name])
        # plt.plot(np.array(f[name+'/volfrac']), np.array(f[name+'/topforce']))
        plt.plot(
                np.array(f[name+'/topforce']),
                label=legend,
                )


    plt.xlabel('Timesteps')
    plt.ylabel('Force (N)')

    plt.gca().legend()
    # plt.show()
    plt.savefig(png_file, bbox_inches='tight')
    plt.close()

#######################################################################

filename = t_name+'_df_damage.h5'
png_file = t_name+'_pl_damage.png'

with h5py.File(filename, 'r') as f:
    print(f.keys())


    # for name in f:
    for name, legend in zip(f, legends):
        plt.plot(
                np.array(f[name]),
                label=legend,
                )


    plt.xlabel('Timestep')
    plt.ylabel('Damage')

    plt.gca().legend()
    # plt.show()
    plt.savefig(png_file, bbox_inches='tight')
    plt.close()

