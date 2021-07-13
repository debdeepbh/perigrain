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
P = np.array([
    [wall_left, wall_bottom],
    [wall_right, wall_bottom],
    [wall_right, wall_top],
    [wall_left, wall_top]
    ])
# particle generation spacing
P_meshsize = 3.5e-3/2

 
shape =  shape_dict.unif_rect(x_min=-1, y_min=-1, length_x=2, length_y=2, meshsize=2/8, ny=8, filename_suffix='00')
msh = exp_dict.get_incenter_mesh_loc(P, P_meshsize, msh_file=shape.msh_file, modify_nodal_circles= False, gen_bdry = False )


# msh = exp_dict.get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= False, gen_bdry = False )
# msh.plot(save_file = '/home/debdeep/gdrive/work/peridynamics/granular/data/incircles_orig.png' )
msh.plot()
msh.info()
plt.close()


# msh = exp_dict.get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= True, gen_bdry = False )
# msh.plot(save_file = '/home/debdeep/gdrive/work/peridynamics/granular/data/incircles_nodal.png' )
# plt.close()

msh = exp_dict.get_incenter_mesh_loc(P, P_meshsize, msh_file=shape.msh_file, modify_nodal_circles= True, gen_bdry = False )
# msh = exp_dict.get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= True, gen_bdry = True )
# msh.plot(save_file = '/home/debdeep/gdrive/work/peridynamics/granular/data/incircles_bdry.png' )
msh.plot()
msh.info()
plt.close()
