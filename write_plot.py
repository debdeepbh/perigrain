import matplotlib
matplotlib.use('Agg')

from multiprocessing import Pool

import matplotlib.pyplot as plt

import pandas as pd
# import matplotlib.pyplot as plt
# cols=['x', 'y'] 
cols=['x', 'y', 'q'] 


loc = 'output/csv/'

# load plotinfo, a column
plti  = pd.read_csv(loc+'plotinfo.csv', names=['n'], header=None, delimiter=' ')
first_counter = int(plti['n'][0])
last_counter = int(plti['n'][1])
total_particles_univ = int(plti['n'][2])
# total_particles_univ = 53
wall_specified = bool(plti['n'][3])

if wall_specified:
    wall_left = plti['n'][3]
    wall_right = plti['n'][4]
    wall_bottom = plti['n'][5]
    wall_top = plti['n'][6]

dotsize = 5
# dotsize = 0.1

first_counter = 1
last_counter = 50


def write_img(t):
    tc_ind = ('tcounter_%05d' % t)
    print(t, end = ' ', flush=True)
    df_holder = pd.DataFrame({'x' : [], 'y' : [], 'z' : []})

    for i in range(0, total_particles_univ):
        part_ind = ('_particle_%05d' % i)
        # print ('i = ', i)

        filename = loc+tc_ind+part_ind+'.csv'
        df  = pd.read_csv(filename, names=cols, header=None, delimiter=' ')

        df_holder = pd.concat([df_holder, df])

        # df.plot()  # plots all columns against index
        # df.plot(kind='scatter',x='x',y='y', s=dotsize) # scatter plot
        # df.plot(kind='scatter',x='x',y='y', s=dotsize, c='q')
    # df_holder.plot(kind='scatter',x='x',y='y', s=dotsize, c='q', colormap='viridis') # scatter plot
    # df_holder.plot(kind='scatter',x='x',y='y', c='q', s = dotsize, marker = '.', linewidth = 0, colormap='viridis') # scatter plot
    # df_holder.plot.scatter(x='x', y='y', c='q', s = dotsize, marker = '.', linewidth = 0, colormap='viridis') # scatter plot
    # plt.scatter(df_holder.x, df_holder.y, c = df_holder.q, s = dotsize, marker = '.', linewidth = 0)
    plt.scatter(df_holder.x, df_holder.y, c = df_holder.q, s = dotsize, marker = '.', linewidth = 0, cmap='viridis')
    # df.plot(kind='scatter')
    # df.plot(kind='density')  # estimate density function
    # df.plot(kind='hist')  # histogram
    plt.axis('scaled')

    # plt.xlim(-5e-3, 5e-3)
    # plt.ylim(-5e-3, 5e-3)

    ## for sodalime_prenotch
    # plt.xlim(0.1, 0.2)
    # plt.ylim(0, 0.04)

    if wall_specified:
        plt.xlim(wall_left, wall_right)
        plt.ylim(wall_bottom, wall_top)

    else:
        ## generate_particles_from_mesh_geom_wall
        plt.xlim(-0.01, 0.01)
        plt.ylim(0, 0.01)

        ## Collision test
        # plt.xlim(-5e-3, 5e-3)
        # plt.ylim(-5e-3, 5e-3)

    # remove axis 
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=False)

    # remove colorbar
    # cb=plt.colorbar()
    # cb.remove()

    out_png = 'output/img/pic_'+tc_ind+'.png'
    # plt.tight_layout()
    # plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    matplotlib.pyplot.savefig(out_png, dpi=200, bbox_inches='tight')
    # matplotlib.pyplot.clf()
    matplotlib.pyplot.close()

# for t in range(first_counter, last_counter):
    # write_img(t)
# parallel formulation
a_pool = Pool()
a_pool.map(write_img, range(first_counter, last_counter+1))


