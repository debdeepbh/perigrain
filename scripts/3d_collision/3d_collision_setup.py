import numpy as np
from multiprocessing import Pool
import time
from random import seed
from random import random

import sys, os
sys.path.append(os.getcwd())

import shape_dict, material_dict
from genmesh import genmesh
from exp_dict import ShapeList, Wall3d, Contact, Experiment


""" Two particles colliding in 3D
"""

delta = 1e-3
meshsize = None  # set in .geo and generated .msh
# contact_radius = 1e-3/3;
contact_radius = 1e-3/(2.5)    # conserves momentum better (than delta/3)

SL = ShapeList()

## shape: 0

SL.append(shape=shape_dict.sphere_small_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))
# SL.append(shape=shape_dict.disk_w_hole_3d(), count=2,
          # meshsize=meshsize, material=material_dict.peridem_3d(delta))
# SL.append(shape=shape_dict.plus_small_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))

mat = material_dict.peridem_3d(delta)
mat.print()

particles = SL.generate_mesh(dimension=3, contact_radius=contact_radius, plot_mesh = False)

# apply transformation
particles[0][0].rotate3d('z', np.pi/2)
particles[0][1].rotate3d('z', -np.pi/2)
particles[0][1].shift([0, 0, 4e-3])
# particles[0][1].shift([0, 0, 2.6e-3])

# Initial data
particles[0][1].vel += [0, 0, -20]
# particles[0][1].acc += [0, 0, -16e4]
# particles[0][1].extforce += [0, 0, -16e4 * particles[0][1].material.rho]

# wall info

x_min = -6e-3
y_min = -6e-3
z_min = -6e-3
x_max = 6e-3
y_max = 6e-3
z_max = 6e-3
wall = Wall3d(1, x_min, y_min, z_min, x_max, y_max, z_max)
# wall = Wall3d(0)

# contact properties
normal_stiffness = 18 * \
    material_dict.peridem(delta).bulk_modulus / \
    (np.pi * np.power(delta, 5))
# normal_stiffness = 15 * mat.E /( np.pi * np.power(delta,5) * (1 - 2*mat.nu));

damping_ratio = 0.8
friction_coefficient = 0.8

contact = Contact(contact_radius, normal_stiffness,
                  damping_ratio, friction_coefficient)

exp = Experiment(particles, wall, contact)

#######################################################################

# save the data
exp.save('meshdata/all.h5')

## # plot the setup data
# exp_b.plot(by_CurrPos=False, plot_scatter = True, plot_delta = 1, plot_contact_rad = 1, plot_bonds = 0, plot_bdry_nodes = 0, plot_bdry_edges= 0, plot_wall = 1, plot_wall_faces = False, wall_color=wall_color, wall_alpha=wall_alpha, do_plot = True, do_save = 0, save_filename = 'setup.png', dotsize = 10, linewidth = 0.3)
