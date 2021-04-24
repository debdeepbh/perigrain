import numpy as np
import time
from random import seed
from random import random

import sys, os
sys.path.append(os.getcwd())

import shape_dict, material_dict
from exp_dict import ShapeList, Wall, Contact, Experiment, get_incenter_mesh_loc


""" A bulk of particles, locations are generated from a mesh
"""

delta = 1e-3
meshsize = 1e-3/3
contact_radius = delta/5;

# at least this much space to leave between particles. Will get halved while selecting max radius
min_particle_spacing = contact_radius*1.001

# wall info
wall_left   = -10e-3
wall_right  = 10e-3
wall_top    = 5e-3
wall_bottom = -5e-3
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
msh.info()
# msh.plot(plot_edge = True)

# reduce radius to avoid contact
msh.incircle_rad -= min_particle_spacing/2
msh.info()
# msh.plot(plot_edge = True)
# radius of base object is 1e-3
scaling_list = msh.incircle_rad/1e-3

# Create a list of shapes
SL = ShapeList()

if sys.argv[1]=='disk':
    SL.append(shape=shape_dict.small_disk(), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))
elif sys.argv[1]=='pacman':
    SL.append(shape=shape_dict.pacman(), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))
elif sys.argv[1]=='box':
    SL.append(shape=shape_dict.plank(l=1e-3/np.sqrt(2), s=1e-3/np.sqrt(2)), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))
elif sys.argv[1]=='plus0.8':
    SL.append(shape=shape_dict.plus_inscribed(notch_dist=0.8), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))
elif sys.argv[1]=='plus0.2':
    SL.append(shape=shape_dict.plus_inscribed(notch_dist=0.2), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))

elif sys.argv[1]=='n3':
    SL.append(shape=shape_dict.polygon_inscribed(sides=3), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))
elif sys.argv[1]=='n4':
    SL.append(shape=shape_dict.polygon_inscribed(sides=4), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))
elif sys.argv[1]=='n5':
    SL.append(shape=shape_dict.polygon_inscribed(sides=5), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))
elif sys.argv[1]=='n6':
    SL.append(shape=shape_dict.polygon_inscribed(sides=6), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))
elif sys.argv[1]=='n8':
    SL.append(shape=shape_dict.polygon_inscribed(sides=8), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))

# generate the mesh for each shape
particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)

## Apply transformation
seed(1)


for i in range(msh.count()):
    particles[0][i].scale(scaling_list[i])

    # particles[0][i].trasnform_bymat(np.array([ [0.8, 0], [0, 1] ]))
    # particles[0][i].trasnform_bymat(np.array([ [0.5 + (random() * (1 - 0.5)), 0], [0, 1] ]))

    ## generate random rotation
    particles[0][i].rotate(0 + (random() * (2*np.pi - 0)) )
    particles[0][i].shift(msh.incenter[i])


# Initial data
# particles[0][1].vel += [0, -20]
# particles[0][1].acc += [0, -5e4]
# particles[0][1].extforce += [0, -5e4 * particles[0][1].material.rho]
for i in range(msh.count()):
    # particles[0][i].acc += [0, -5e3]
    # particles[0][i].extforce += [0, -5e3 * particles[0][i].material.rho]
    pass

# contact properties
normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
damping_ratio = 0.8
friction_coefficient = 0.8
contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

# contact.print()
# material_dict.peridem(delta).print()

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

# exp_b.plot(by_CurrPos=False, plot_scatter = True, plot_delta = 1, plot_contact_rad = 1, plot_bonds = 0, plot_bdry_nodes = 0, plot_bdry_edges= 1, plot_wall = 1, plot_wall_faces = False, wall_color=wall_color, wall_linewidth = 1, wall_alpha=wall_alpha, do_plot = True, do_save = 0, save_filename = 'setup.png', dotsize = 10, linewidth = 0.3, remove_axes = True, grid = False)
