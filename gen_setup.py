import exp_dict

import load_setup

# get the experiment
# exp = exp_dict.hopper()
# exp = exp_dict.hopper_2()
# exp = exp_dict.bar()
# exp = exp_dict.flow()
# exp = exp_dict.worm()
# exp = exp_dict.pendulum()
# exp = exp_dict.own_weight_float_force()
# exp = exp_dict.own_weight()
# exp = exp_dict.particle_confined()
# exp = exp_dict.wall_particle()
# exp = exp_dict.particle_plank()
# exp = exp_dict.collision()
exp = exp_dict.collision_pacman()
# exp = exp_dict.crack_prenotch()
# exp = exp_dict.fixed_prenotch()
# exp = exp_dict.bulk_generated()
# exp = exp_dict.bulk_generated_mixed()
# exp = exp_dict.bulk_generated_diffmesh()
# exp = exp_dict.coffee_diffmesh()
# exp = exp_dict.collision_3d()
# exp = exp_dict.collision_3d_bad()
# exp = exp_dict.bulk_generated_3d()

exp.save('meshdata/all.h5')


## # load the setup data
exp_b = load_setup.read_setup('meshdata/all.h5')

# print(exp_b.PArr[0].vol)
# print(exp_b.PArr[1].vol)

# print(exp_b.PArr[0].bdry_edges)
# print(exp_b.PArr[1].bdry_edges)


# wall_color = 'red'
wall_color = ['cyan', 'red', 'yellow', 'blue', 'green', 'black']
# wall_color = ['none', 'none', 'none', 'none', 'none', 'none']
wall_alpha = 0.1

## # plot the setup data
exp_b.plot(by_CurrPos=False, plot_scatter = True, plot_delta = 1, plot_contact_rad = 1, plot_bonds = 0, plot_bdry_nodes = 0, plot_bdry_edges= 0, plot_wall = 1, plot_wall_faces = False, wall_color=wall_color, wall_alpha=wall_alpha, do_plot = True, do_save = 0, save_filename = 'setup.png', dotsize = 10, linewidth = 0.3, remove_axes = False, grid = False)


