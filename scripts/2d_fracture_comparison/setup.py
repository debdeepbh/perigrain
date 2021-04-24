import sys, os
sys.path.append(os.getcwd())

import material_dict
import shape_dict
from exp_dict import ShapeList, Wall, Contact, Experiment, get_incenter_mesh_loc
from genmesh import genmesh
import numpy as np
import time
from random import seed
from random import random

""" Two particles colliding 
"""

delta = 1e-3
meshsize = 1e-3/16
contact_radius = delta/3

SL = ShapeList()

SL.append(shape=shape_dict.box_prenotch(), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, rho_scale=1, Gnot_scale=2))

SL.append(shape=shape_dict.plank(l=1e-3/2, s=0.5e-3/2), count=2, meshsize=meshsize, material=material_dict.peridem_deformable(delta, rho_scale=1.2))

particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)

# apply transformation
particles[0][0].shift([-1e-3/2, -1e-3/2])

particles[1][0].shift([0, 1.5e-3])
particles[1][1].shift([0, -1.5e-3])

# Initial data
# particles[1][0].vel += [0, -20]
# particles[1][1].vel += [0, 20]

av = 10e5
particles[1][0].acc += [0, -av]
particles[1][0].extforce += [0, -av * particles[1][0].material.rho]
particles[1][1].acc += [0, av]
particles[1][1].extforce += [0, av * particles[1][1].material.rho]


# wall info
wall_left   = -3e-3
wall_right  = 3e-3
wall_top    = 3e-3
wall_bottom = -3e-3
wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

# contact properties
normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
damping_ratio = 0.9
friction_coefficient = 0.8


contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)


exp= Experiment(particles, wall, contact)

# save the data
exp.save('meshdata/all.h5')

# import load_setup
# # ## # load the setup data
# exp_b = load_setup.read_setup('meshdata/all.h5')

# wall_color = ['cyan', 'red', 'yellow', 'blue', 'green', 'black']
# # wall_color = ['none', 'none', 'none', 'none', 'none', 'none']
# wall_alpha = 0.1

# exp_b.plot(by_CurrPos=False, plot_scatter = True, plot_delta = 1, plot_contact_rad = 1, plot_bonds = 1, plot_bdry_nodes = 0, plot_bdry_edges= 0, plot_wall = 1, plot_wall_faces = False, wall_color=wall_color, wall_linewidth = 1, wall_alpha=wall_alpha, do_plot = True, do_save = 0, save_filename = 'setup.png', dotsize = 10, linewidth = 0.3, remove_axes = True, grid = False)
