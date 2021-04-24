import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

import h5py
import numpy as np
import time

from multiprocessing import Pool

loc = 'output/hdf5/'

#######################################################################


print(loc)
p = h5py.File(loc+'plotinfo.h5', "r")
# convert to int, since it is read as double, I think (strange, since hdf5 specifies type)
first_counter = int(p['f_l_counter'][0])
last_counter = int(p['f_l_counter'][1])

print(first_counter)
print(last_counter)

wall_specified = p['wall/allow_wall'][0]

if wall_specified:
    wall_lrtb = p['wall/geom_wall_info'];
    wall_left = wall_lrtb[0]
    wall_right = wall_lrtb[1]
    wall_top = wall_lrtb[2]
    wall_bottom = wall_lrtb[3]

print("wall dim: ", wall_left, wall_right, wall_top, wall_bottom)

dotsize = 10
# dotsize = 0.1

# override
# first_counter = 1
# last_counter = 100

print('fc: ', first_counter)
print('lc: ', last_counter)

def write_img(t):
    tc_ind = ('tc_%05d' % t)
    print(t, end = ' ', flush=True)

    filename = loc+tc_ind+".h5";
    f = h5py.File(filename, "r")

    # for i in range(0, total_particles_univ):
        # part_ind = ('_particle_%05d' % i)
        # # print ('i = ', i)
        # filename = loc+tc_ind+part_ind+'.csv'
        # df  = pd.read_csv(filename, names=cols, header=None, delimiter=' ')
        # df_holder = pd.concat([df_holder, df])

    for name in f:
        CurrPos = f[name+"/CurrPos"]
        force = f[name+"/force"]
        vel = f[name+"/vel"]
        # print(vel[:])
        q = np.sqrt(np.sum(np.square(force), axis=1)) #norm
        # q = np.sqrt(np.sum(np.square(vel), axis=1)) #norm
        # q = np.abs(vel[:,0]) #norm

        plt.scatter(CurrPos[:,0], CurrPos[:,1], c = q, s = dotsize, marker = '.', linewidth = 0, cmap='viridis')
        # plt.scatter(CurrPos[:,0], CurrPos[:,1], c = q, s = dotsize, marker = '.', linewidth = 0, cmap='viridis', alpha = 0.3)

    # plot properties
    ## Note: plt.axis('scaled') goes _before_ setting plt.xlim() and plt.ylim()
    plt.axis('scaled')

    # limit of the axes
    if wall_specified:
        plt.xlim(wall_left, wall_right)
        plt.ylim(wall_bottom, wall_top)

    else:
        ## generate_particles_from_mesh_geom_wall
        # plt.xlim(-0.01, 0.01)
        # plt.ylim(0, 0.01)

        ## for sodalime_prenotch
        # plt.xlim(0.1, 0.2)
        # plt.ylim(0, 0.04)

        ## Collision test
        plt.xlim(-5e-3, 5e-3)
        plt.ylim(-5e-3, 5e-3)

    ## remove axes
    # plt.tick_params(
        # axis='x',          # changes apply to the x-axis
        # which='both',      # both major and minor ticks are affected
        # bottom=False,      # ticks along the bottom edge are off
        # top=False,         # ticks along the top edge are off
        # labelbottom=False)
    # plt.tick_params(
        # axis='y',          # changes apply to the x-axis
        # which='both',      # both major and minor ticks are affected
        # left=False,      # ticks along the bottom edge are off
        # top=False,         # ticks along the top edge are off
        # labelleft=False)

    ## remove white space around the plot
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # ax.set_xticks(numpy.arange(0, 1, 0.1))
    # ax.set_yticks(numpy.arange(0, 1., 0.1))
    plt.grid()

    # plt.colorbar()

    # remove colorbar
    # cb=plt.colorbar()
    # cb.remove()

    out_png = 'output/img/img_'+tc_ind+'.png'
    # plt.tight_layout()
    # plt.gca().set_axis_off()


    # saving plot
    matplotlib.pyplot.savefig(out_png, dpi=200, bbox_inches='tight')

    # close the plot
    plt.close()


start = time.time()

# for t in range(first_counter, last_counter):
    # write_img(t)

# parallel formulation
a_pool = Pool()
a_pool.map(write_img, range(first_counter, last_counter+1))
print('\n')
print('time taken ', time.time() - start)


