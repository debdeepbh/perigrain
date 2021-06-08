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

meshsize = 1e3
delta = 1e3

SL = ShapeList()

# SL.append(shape=shape_dict.wheel_annulus(scaling=1e-3, inner_circle_ratio=0.5,  meshsize=meshsize/2, nci_steps=20), count=2, meshsize=meshsize, material=material_dict.peridem_softer_1(delta))
# SL.append(shape=shape_dict.wheel_annulus(scaling=1e-3, inner_circle_ratio=0.5,  meshsize=meshsize/2, nci_steps=20), count=2, meshsize=meshsize, material=material_dict.peridem_softer_1(delta))
pacman_angle = np.pi/2
# SL.append(shape=shape_dict.pacman(angle = pacman_angle), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
# particles = SL.generate_mesh(dimension = 2, contact_radius = None, plot_mesh=False, plot_shape=True)

shape=shape_dict.pacman(angle = pacman_angle) 
# shape.plot(bdry_arrow=False, extended_bdry=False, angle_bisector=False)
