import numpy as np
from multiprocessing import Pool
import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from load_setup import read_setup
from read_sim_output import read_plotinfo, wall_reaction_val

#######################################################################
loc = 'output/hdf5/'
plti = read_plotinfo(loc+'plotinfo.h5')

# override here
# plti.fc = 1
# plti.fc = 200
# plti.fc = 380 
# plti.fc = 400
# plti.lc = 400 # settles until
# plti.lc = 450 # pressure stops
# plti.lc = 650 # end
# plti.lc = 50

## # load the main experiment setup data
exp_b = read_setup('data/hdf5/all.h5')
vol = exp_b.total_volume()
mass = exp_b.total_mass()

print('Total volume:', vol)
print('Total mass:', mass)
######################################################################

def compute_2d(t):
    return wall_reaction_val(t, loc, 2)

# tt = np.array(range(plti.fc, plti.lc+1)) * plti.dt * plti.modulo * 1e6
tt = np.array(range(plti.fc, plti.lc+1))

start = time.time()

# parallel formulation
a_pool = Pool()
V = a_pool.map(compute_2d, range(plti.fc, plti.lc+1))
print('')

# print(V[10].wall_reaction)

# plt.plot(tt, f_tot[:,0], label = r'F_x')
# plt.plot(tt, f_tot[:,1], label = r'F_y top')
# # plt.plot(tt, np.abs(f_tot[:,1]), label = r'F_y')
# # plt.plot(tt, movingaverage(f_tot[:,1],20), label = r'F_y')
# plt.plot(tt, f_right[:,1], label = r'F_y right')
# # plt.plot(tt, vol/cont_vol, label = r'$\phi$')
# if (len(f_tot[0]) == 3):
    # plt.plot(tt, f_tot[:,2], label = r'F_z')

# # plt.plot(f_tot, vol/cont_vol, label = 'frac-pres')

### Total force 
# plt.plot(tt, [v.wall_reaction[2,1] for v in V], label = r'$F_y/V$ top')
# plt.plot(tt, [v.wall_reaction[3,1] for v in V], label = r'$F_y/V$ bottom')
# plt.plot(tt, [v.wall_reaction[0,0] for v in V], label = r'$F_x/V$ left')
# plt.plot(tt, [v.wall_reaction[1,0] for v in V], label = r'$F_x/V$ right')

### Pressure, average force
# plt.plot(tt, [v.wall_reaction[2,1]/v.wall_l for v in V], label = r'$P_y$ top')
# plt.plot(tt, [v.wall_reaction[3,1]/v.wall_l for v in V], label = r'$P_y$ bottom')
# plt.plot(tt, [v.wall_reaction[0,0]/v.wall_h for v in V], label = r'$P_x$ left')
# plt.plot(tt, [v.wall_reaction[1,0]/v.wall_h for v in V], label = r'$P_x$ right')

# Volume fraction
phi = [vol/(v.wall_l * v.wall_h) for v in V]
# plt.plot(tt, phi, label = r'$\phi$')

## pressure vs volume fraction plot
# plt.plot([np.abs(v.wall_reaction[3,1]/v.wall_l) for v in V], phi,  label = r'$|P_y|$ bottom')
# plt.plot([np.abs(v.wall_reaction[2,1]/v.wall_l) for v in V], phi,  label = r'$|P_y|$ top')
# plt.plot([np.abs(v.wall_reaction[0,0]/v.wall_h) for v in V], phi,  label = r'$|P_x|$ left')
# plt.plot([np.abs(v.wall_reaction[1,0]/v.wall_h) for v in V], phi,  label = r'$|P_x|$ right')

## avg pressure vs volume fraction
plt.plot([ (np.abs(v.wall_reaction[3,1]/v.wall_l) +
np.abs(v.wall_reaction[2,1]/v.wall_l) +
np.abs(v.wall_reaction[0,0]/v.wall_h) +
np.abs(v.wall_reaction[1,0]/v.wall_h))/4/9e8
    for v in V], phi,  label = r'$|p|$ avg')

## volume fraction vs pressure plot
# plt.plot(phi, [np.abs(v.wall_reaction[3,1]/v.wall_l) for v in V],  label = r'$|P_y|$ bottom')
# plt.plot(phi, [np.abs(v.wall_reaction[2,1]/v.wall_l) for v in V],  label = r'$|P_y|$ top')
# plt.plot(phi, [np.abs(v.wall_reaction[0,0]/v.wall_h) for v in V],  label = r'$|P_x|$ left')
# plt.plot(phi, [np.abs(v.wall_reaction[1,0]/v.wall_h) for v in V],  label = r'$|P_x|$ right')


## log plot
# plt.plot([np.log(np.abs(v.wall_reaction[3,1]/v.wall_l)) for v in V], phi,  label = r'$|P_y|$ bottom')
# plt.plot([np.log(np.abs(v.wall_reaction[2,1]/v.wall_l)) for v in V], phi,  label = r'$|P_y|$ top')
# plt.plot([np.log(np.abs(v.wall_reaction[0,0]/v.wall_h)) for v in V], phi,  label = r'$|P_x|$ left')
# plt.plot([np.log(np.abs(v.wall_reaction[1,0]/v.wall_h)) for v in V], phi,  label = r'$|P_x|$ right')
# plt.plot(tt, [v.wall_l * v.wall_h/vol for v in V], label = r'$\phi$ vs $F_y$ bottom')

plt.grid()
# plt.xlim(min(tt), max(tt))
# plt.autoscale(tight=True)
plt.gca().legend()
# plt.show()
plt.savefig('output/img/force_plot.png', dpi=300, bbox_inches='tight')

print('time taken ', time.time() - start)



