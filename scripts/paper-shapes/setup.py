import numpy as np
# import time
from random import seed
from random import random

import sys, os
sys.path.append(os.getcwd())

import shape_dict, material_dict
from exp_dict import ShapeList
# from genmesh import genmesh
# from exp_dict import ShapeList, Wall, Contact, Experiment, get_incenter_mesh_loc
from genmesh import genmesh

meshsize = 1e-3/15
delta = 1e-3


pacman_angle = np.pi/2

# SL = ShapeList()
# SL.append(shape=shape_dict.wheel_annulus(scaling=1e-3, inner_circle_ratio=0.5,  meshsize=meshsize/2, nci_steps=20), count=2, meshsize=meshsize, material=material_dict.peridem_softer_1(delta))
# SL.append(shape=shape_dict.wheel_annulus(scaling=1e-3, inner_circle_ratio=0.5,  meshsize=meshsize/2, nci_steps=20), count=2, meshsize=meshsize, material=material_dict.peridem_softer_1(delta))
# SL.append(shape=shape_dict.pacman(angle = pacman_angle), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
# particles = SL.generate_mesh(dimension = 2, contact_radius = None, plot_mesh=False, plot_shape=True)

# shape=shape_dict.pacman(angle = pacman_angle) 
# shape.plot(bdry_arrow=False, extended_bdry=False, angle_bisector=False)

# filename_suffix='00'
# shape=shape_dict.wheel_annulus(scaling=1e-3, inner_circle_ratio=0.5,  meshsize=meshsize, nci_steps=20, filename_suffix=filename_suffix)
# shape=shape_dict.wheel_annulus(scaling=1e-3, inner_circle_ratio=0.5,  meshsize=meshsize, nci_steps=20, filename_suffix=filename_suffix)
shape=shape_dict.perturbed_disk(steps=22, seed=1, scaling=1e-3, std=0.4, angle_drift_amp=1e-3, angle_drift_std_ratio=0.25)


# msh_file = 'meshdata/ring_'+str(filename_suffix)+'.msh'
# mesh = genmesh(P_bdry=None, meshsize=meshsize, pygmsh_geom=None, msh_file = msh_file, dimension = 2, mesh_optimize=True)

# shape.plot()
shape.plot(bdry_arrow=False, extended_bdry=False, angle_bisector=False, plot_bounding_ball=True, bounding_ball_steps=50)
# mesh.plot(dotsize=10, plot_node_text=False, highlight_bdry_nodes=False)
