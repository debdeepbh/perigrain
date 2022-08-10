## Add the project root as a path
import sys, os
sys.path.append(os.getcwd())

import exp_dict

import numpy as np
import matplotlib.pyplot as plt

import shape_dict

# wall info
wall_left   = -10e-3
wall_right  = 10e-3
wall_top    = 10e-3
wall_bottom = 0e-3

# particle generation boundary
# P = np.array([
    # [wall_left, wall_bottom],
    # [wall_right, wall_bottom],
    # [wall_right, wall_top],
    # [wall_left, wall_top]
    # ])

# shape=shape_dict.small_disk(steps=30)
shape=shape_dict.small_disk_fromfile()
# shape=shape_dict.test()
# shape=shape_dict.vase(l=10e-3, s=5e-3, steps=20, n_pi=3, phase_1=0, phase_2=np.pi/2)

# print(shape)
# P = shape.P
# print(P)
# plt.plot(P[:,0], P[:,1])
# plt.show()
        

# particle generation spacing
# P_meshsize = 3.5e-3

# P_meshsize = 3.5e-3
# P_meshsize = 0.8e-3
# P_meshsize = 0.2e-3

# msh = exp_dict.get_incenter_mesh_loc(P, P_meshsize, msh_file=None, modify_nodal_circles= False, gen_bdry = False )

shape = shape_dict.test()
msh_file = shape.msh_file
print(msh_file)
msh = exp_dict.get_incenter_mesh_loc(P=[], meshsize=[], msh_file=msh_file, dimension=2, modify_nodal_circles= False, gen_bdry = False )

savefile='scripts/arr_no_modify.png'
# def plot(self, do_plot = True, plot_edge = True, plot_mesh = True, remove_axes = True, save_file = None):
msh.plot(save_file = savefile, plot_mesh=True, remove_axes=True, plot_circles=False)
msh.info()
# plt.show()
plt.close()


# msh = exp_dict.get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= True, gen_bdry = False )
# msh.plot(save_file = '/home/debdeep/gdrive/work/peridynamics/granular/data/incircles_nodal.png' )
# plt.close()

msh = exp_dict.get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= True, gen_bdry = True )
savefile='scripts/arr_modified.png'
msh.plot(save_file = savefile, plot_mesh=False, remove_axes=True, plot_circles=True)
msh.info()
# plt.show()
plt.close()
