import numpy as np
from multiprocessing import Pool
import time
from random import seed
from random import random

import sys, os
sys.path.append(os.getcwd())

import shape_dict, material_dict
from exp_dict import ShapeList, Wall, Contact, Experiment


""" Two particles colliding in 2D
"""

delta = 1e-3
meshsize = delta/20
contact_radius = 1e-3/3;

## shape: 0
# shapes.append(shape_dict.small_disk())
# shapes.append(shape_dict.plus())

SL = ShapeList()

# SL.append(shape=shape_dict.small_disk(), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
SL.append(shape=shape_dict.pacman(), count=2, meshsize=meshsize, material=material_dict.peridem_deformable(delta, Gnot_scale=0.1))

particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)

# apply transformation
particles[0][0].rotate(-np.pi/4)
particles[0][1].rotate(3*np.pi/4)
particles[0][1].shift([1e-3+contact_radius, 1e-3])

# Initial data
particles[0][1].vel += [0, -30]
# particles[0][1].acc += [0, -5e4]
# particles[0][1].extforce += [0, -5e4 * particles[0][1].material.rho]

# wall info
wall_left   = -6e-3
wall_right  = 6e-3
wall_top    = 6e-3
wall_bottom = -6e-3
wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)


# contact properties
normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
damping_ratio = 0.9
friction_coefficient = 0.8


contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)


exp = Experiment(particles, wall, contact)

#######################################################################

# save the data
exp.save('meshdata/all.h5')

## # plot the setup data
# import load_setup
# exp_b = load_setup.read_setup('meshdata/all.h5')
# exp_b.plot(by_CurrPos=False, plot_scatter = True, plot_delta = 1, plot_contact_rad = 1, plot_bonds = 1, plot_bdry_nodes = 0, plot_bdry_edges= 0, plot_wall = 1, plot_wall_faces = False, wall_color='blue', wall_alpha=0.1, do_plot = True, do_save = 1, save_filename = 'setup.png', dotsize = 10, linewidth = 0.3, remove_axes = False, grid = False, show_particle_index=True)
