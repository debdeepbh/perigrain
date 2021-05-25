import numpy as np
# import time
from random import seed
from random import random

import sys, os
sys.path.append(os.getcwd())

import shape_dict, material_dict
# from genmesh import genmesh
from exp_dict import ShapeList, Wall, Contact, Experiment, get_incenter_mesh_loc


""" A bulk of particles, locations are generated from a mesh
Each particle mesh is generated using scaling factor, hence the neighborhood and boundary nodes are computed separately
"""

delta = 1e-3
meshsize = 1e-3/5
contact_radius = delta/4;

# at least this much space to leave between particles. Will get halved while selecting max radius
min_particle_spacing = contact_radius*1.001

# wall info
# wall_left   = -10e-3
# wall_right  = 10e-3
# wall_top    = 10e-3
# wall_bottom = 0e-3

# wall_left   = -25e-3
# wall_right  = 25e-3
# wall_top    = 50e-3
# wall_bottom = -50e-3

# wall_left   = -10e-3
# wall_right  = 10e-3
# wall_top    = 50e-3
# wall_bottom = -50e-3

## smallest
# wall_left   = -10e-3
# wall_right  = 10e-3
# wall_top    = 10e-3
# wall_bottom = -10e-3

## slightly bigger, needs twice the time 
wall_left   = -20e-3
wall_right  = 20e-3
wall_top    = 20e-3
wall_bottom = -20e-3

# wall_bottom = -5e-3
wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

# particle generation boundary
c = min_particle_spacing/2
P = np.array([
    [wall_left + c, wall_bottom + c ],
    [wall_right - c, wall_bottom + c],
    [wall_right - c, wall_top - c],
    [wall_left + c, wall_top - c]
    ])

# particle generation spacing
P_meshsize = 3.5e-3
msh = get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= True, gen_bdry = False )

# trim minimum radius
#msh.trim(min_rad = 0.7e-3 + min_particle_spacing/2)

# reduce radius to avoid contact
msh.incircle_rad -= min_particle_spacing/2
msh.info()
# msh.plot(plot_edge = True)

# Create a list of shapes
SL = ShapeList()


# append each shape with own scaling
for i in range(msh.count()):
    if sys.argv[1] == 'pertdisk':
        pertdisk_std = 0.3
        SL.append(shape=shape_dict.perturbed_disk(seed=i, steps=16, scaling=msh.incircle_rad[i], std=pertdisk_std ), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, Gnot_scale=0.5, rho_scale=0.8))
    elif sys.argv[1] == 'plus':
        plus_ratio = 0.2
        SL.append(shape=shape_dict.plus_inscribed(notch_dist=plus_ratio), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, Gnot_scale=0.5, rho_scale=0.8))
# plus_inscribed(notch_dist=0.2)
    elif sys.argv[1] == 'n4':
        SL.append(shape=shape_dict.small_disk(steps=4), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, Gnot_scale=0.5, rho_scale=0.8))

# generate the mesh for each shape
particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False, shapes_in_parallel=False)

## Apply transformation
seed(1)
g_val = -5e4
for i in range(msh.count()):
    ## scaling is done while generating the mesh
    ## generate random rotation
    particles[i][0].rotate(0 + (random() * (2*np.pi - 0)) )
    particles[i][0].shift(msh.incenter[i])

    # Initial data
    # particles[0][1].vel += [0, -20]
    # particles[i][0].acc += [0, g_val]
    # particles[i][0].extforce += [0, g_val * particles[i][0].material.rho]

# contact properties
normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
damping_ratio = 0.8
friction_coefficient = 0.8
contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

# return Experiment(particles, wall, contact)
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

exp_b.plot(by_CurrPos=False, plot_scatter = True, plot_delta = 1, plot_contact_rad = 1, plot_bonds = 0, plot_bdry_nodes = 0, plot_bdry_edges= 1, plot_wall = 1, plot_wall_faces = False, wall_color=wall_color, wall_linewidth = 1, wall_alpha=wall_alpha, do_plot = True, do_save = 1, save_filename = 'setup.png', dotsize = 10, linewidth = 0.3, remove_axes = True, grid = False)
