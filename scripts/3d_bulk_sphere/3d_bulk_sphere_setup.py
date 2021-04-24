import numpy as np
import time
from random import seed
from random import random

import sys, os
sys.path.append(os.getcwd())

import shape_dict, material_dict
from genmesh import genmesh
from exp_dict import ShapeList, Wall3d, Contact, Experiment, get_incenter_mesh_loc

""" bulk particle generation for 3D region from tetrahedrons
"""

delta = 1e-3
meshsize = None # set in .geo and generated .msh
# contact_radius = 1e-3/3;
contact_radius = 1e-3/(2.5)    # conserves momentum better (than delta/3)

# at least this much space to leave between particles. Will get halved while selecting max radius
min_particle_spacing = contact_radius*1.001

# wall info
x_min = -20e-3
y_min = -20e-3
z_min = -20e-3
x_max = 20e-3
y_max = 20e-3
z_max = 20e-3
wall = Wall3d(1, x_min, y_min, z_min, x_max, y_max, z_max)

####
# Bulk generation

# particle generation spacing
msh = get_incenter_mesh_loc(P = None, meshsize=None, dimension = 3, msh_file='meshdata/3d/closepacked_cube.msh', modify_nodal_circles= False, gen_bdry = False )

msh.info()

# Adding min_particle_spacing/2 here ensures that later, after reducing their radii by min_particle_spacing/2 produces the same range (and max, min, mean) of particle radii as set here
# msh.trim(min_rad = 0.8*1e-3 + min_particle_spacing/2, max_rad = 1.2*1e-3 + min_particle_spacing/2)
msh.trim(min_rad = 0.95*1e-3 + min_particle_spacing/2, max_rad = 1.05*1e-3 + min_particle_spacing/2)

msh.info()
msh.plot(plot_edge = True)

# reduce radius to avoid contact
msh.incircle_rad -= min_particle_spacing/2
msh.info()
msh.plot(plot_edge = True)
# radius of base object is 1e-3
scaling_list = msh.incircle_rad/1e-3

## Create a list of shapes
SL = ShapeList()

## shape: 0
SL.append(shape=shape_dict.sphere_small_3d(), count=msh.count(), meshsize=meshsize, material=material_dict.peridem_3d(delta))

mat = material_dict.peridem_3d(delta)
mat.print()

particles = SL.generate_mesh(dimension = 3, contact_radius=contact_radius)

## apply transformation
#particles[0][0].rotate3d('z', np.pi/2)
#particles[0][1].rotate3d('z', -np.pi/2)
#particles[0][1].shift([0, 0, 3e-3])
## particles[0][1].shift([0, 0, 2.6e-3])

## Apply transformation
seed(1)
for i in range(msh.count()):
    particles[0][i].scale(scaling_list[i])

    # particles[0][i].trasnform_bymat(np.array([ [0.8, 0], [0, 1] ]))
    # particles[0][i].trasnform_bymat(np.array([ [0.5 + (random() * (1 - 0.5)), 0], [0, 1] ]))

    ## generate random rotation
    # particles[0][i].rotate(0 + (random() * (2*np.pi - 0)) )
    particles[0][i].shift(msh.incenter[i])

## Initial data
for i in range(msh.count()):
    particles[0][i].acc += [0, 0, -5e3]
    particles[0][i].extforce += [0, 0, -5e3 * particles[0][i].material.rho]

## wall = Wall3d(0)

# contact properties
normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,5));
# normal_stiffness = 15 * mat.E /( np.pi * np.power(delta,5) * (1 - 2*mat.nu));

damping_ratio = 0.8
friction_coefficient = 0.8

contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

exp =  Experiment(particles, wall, contact)
#######################################################################

# save the data
exp.save('meshdata/all.h5')

## # plot the setup data
import load_setup
## # load the setup data
exp_b = load_setup.read_setup('meshdata/all.h5')

wall_color = ['cyan', 'red', 'yellow', 'blue', 'green', 'black']
# wall_color = ['none', 'none', 'none', 'none', 'none', 'none']
wall_alpha = 0.1

exp_b.plot(by_CurrPos=False, plot_scatter = True, plot_delta = 1, plot_contact_rad = 1, plot_bonds = 0, plot_bdry_nodes = 0, plot_bdry_edges= 1, plot_wall = 1, plot_wall_faces = False, wall_color=wall_color, wall_linewidth = 1, wall_alpha=wall_alpha, do_plot = True, do_save = 0, save_filename = 'setup.png', dotsize = 10, linewidth = 0.3, remove_axes = True, grid = False)
