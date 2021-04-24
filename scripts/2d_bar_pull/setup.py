import numpy as np
import time
from random import seed
from random import random
from sys import argv

import sys, os
sys.path.append(os.getcwd())

import shape_dict, material_dict
from genmesh import genmesh
from exp_dict import ShapeList, Wall, Contact, Experiment, get_incenter_mesh_loc

plot_setup = False

""" elastic bar for stress strain curve
"""

delta = 1e-3
meshsize = 1e-3/2
contact_radius = 1e-3/3;

SL = ShapeList()

# l= 20e-3
# l= 10e-3
# s = 2e-3

l = float(argv[1])
s = float(argv[2])
Force = float(argv[3])

# SL.append(shape=shape_dict.line_1d(), count=1, meshsize=meshsize, material=material_dict.peridem_1d_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
# SL.append(shape=shape_dict.tie(l=10e-3, s=0.25e-3), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
# SL.append(shape=shape_dict.plank(l=l, s=s), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, G_scale=1, K_scale=0.5, Gnot_scale=0.1, rho_scale=50))
SL.append(shape=shape_dict.plank(l=l, s=s), count=1, meshsize=meshsize, material=material_dict.sodalime_similar_to_peridem(delta, rho_scale=1))


SL.material_list[0].print()

particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)

# apply transformation
# particles[0][0].rotate(-np.pi/2)
# particles[1][0].shift([0, -3e-3])

# particles[0][1].shift([0, -1e-3])

# Initial data
# particles[0][0].vel += [0, -20]
# particles[0][0].vel += [0, -2]

# particles[0][0].acc += [0, -5e4]
# particles[0][0].extforce += [0, -5e4 * particles[0][0].material.rho]

# clamped nodes
part = particles[0][0]

# part.clamped_nodes.append(0)

# F_acc = 5e6

## To observe the stress-strain curve, we specify Force and compute the force density and initial acceleration from it (as opposed to specifying acceleration)
## That way, for any rho, the elongation of the rod will be the same

nodes_right_edge = []
# clamp the left and stretch the right
for i in range(len(part.pos)):
    # clamp the left
    if np.abs(part.pos[i][0] - (-l))< meshsize/2:
        part.clamped_nodes.append(i)

    # apply force on the right
    # Use with (in config)
    ## gradient_extforce = 1
    ## extforce_maxstep = 20000

    if np.abs(part.pos[i][0] - (l))< meshsize/2:
        # part.acc[i] += [F_acc, 0]
        part.acc[i] += [Force/(part.vol[i][0] * part.material.rho), 0]
        # force density
        # part.extforce[i] += [F_acc * part.material.rho, 0]
        part.extforce[i] += [Force/part.vol[i][0], 0]

        nodes_right_edge.append(i)

np.save('output/2d_bar_r_edge.npy', nodes_right_edge)
np.save('output/youngs.npy', part.material.E)

# print('node w force', node_w_force)

# particles[0][0].vol[0] *= 2
# particles[0][0].clamped_nodes.append(3)
# particles[0][0].clamped_nodes.append(81)


# wall info
# wall_left   = -l - contact_radius*1.5
# wall_right  = l + contact_radius*1.5
wall_left   = -2*l
wall_right  = 2*l
wall_top    = 2*l
wall_bottom = -2*l
wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

# contact properties
normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
damping_ratio = 0.9
friction_coefficient = 0.9

contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

exp =  Experiment(particles, wall, contact)

#######################################################################

# save the data
exp.save('meshdata/all.h5')

if plot_setup:
    ## # plot the setup data
    import load_setup
    ## # load the setup data
    exp_b = load_setup.read_setup('meshdata/all.h5')

    wall_color = ['cyan', 'red', 'yellow', 'blue', 'green', 'black']
    # wall_color = ['none', 'none', 'none', 'none', 'none', 'none']
    wall_alpha = 0.1

    exp_b.plot(by_CurrPos=False, plot_scatter = True, plot_delta = 1, plot_contact_rad = 1, plot_bonds = 1, plot_bdry_nodes = 0, plot_bdry_edges= 1, plot_wall = 1, plot_wall_faces = False, wall_color=wall_color, wall_linewidth = 1, wall_alpha=wall_alpha, do_plot = True, do_save = 1, save_filename = 'setup.png', dotsize = 10, linewidth = 0.3, remove_axes = False, grid = True)
