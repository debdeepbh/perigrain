import sys, os
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import load_setup
from read_sim_output import read_plotinfo

#######################################################################
# loc = 'output/hdf5/'
# plti = read_plotinfo(loc+'plotinfo.h5')
# dim = plti.dim

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
exp_b = load_setup.read_setup('data/hdf5/all.h5')
vol = exp_b.total_volume()
mass = exp_b.total_mass()
print('Total volume:', vol)
print('Total mass:', mass)
######################################################################
V = np.load('output/V.npy', allow_pickle=True)

# tt = np.array(range(plti.fc, plti.lc+1)) * plti.dt * plti.modulo * 1e6
tt = np.array(range(len(V))) +1

# Volume fraction
phi = [vol/(v.wall_l * v.wall_h) for v in V]
# plt.plot(tt, phi, label = r'$\phi$')

#######################################################################

### Total force 
# plt.plot(tt, [v.wall_reaction[2,1] for v in V], label = r'$F_y/V$ top')
# plt.plot(tt, [v.wall_reaction[3,1] for v in V], label = r'$F_y/V$ bottom')
# plt.plot(tt, [v.wall_reaction[0,0] for v in V], label = r'$F_x/V$ left')
# plt.plot(tt, [v.wall_reaction[1,0] for v in V], label = r'$F_x/V$ right')

### Pressure, average force
plt.plot(tt, [v.wall_reaction[2,1]/v.wall_l for v in V], label = r'$P_y$ top')
plt.plot(tt, [v.wall_reaction[3,1]/v.wall_l for v in V], label = r'$P_y$ bottom')
plt.plot(tt, [v.wall_reaction[0,0]/v.wall_h for v in V], label = r'$P_x$ left')
plt.plot(tt, [v.wall_reaction[1,0]/v.wall_h for v in V], label = r'$P_x$ right')

plt.grid()
# plt.xlim(min(tt), max(tt))
# plt.autoscale(tight=True)
plt.gca().legend()
# plt.show()
filename='output/img/avg_pressure_each_full.png'
print('Saving to', filename)
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

#######################################################################

## pressure vs volume fraction plot
# plt.plot([np.abs(v.wall_reaction[3,1]/v.wall_l) for v in V], phi,  label = r'$|P_y|$ bottom')
# plt.plot([np.abs(v.wall_reaction[2,1]/v.wall_l) for v in V], phi,  label = r'$|P_y|$ top')
# plt.plot([np.abs(v.wall_reaction[0,0]/v.wall_h) for v in V], phi,  label = r'$|P_x|$ left')
# plt.plot([np.abs(v.wall_reaction[1,0]/v.wall_h) for v in V], phi,  label = r'$|P_x|$ right')

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

## avg pressure vs volume fraction
plt.plot([ (np.abs(v.wall_reaction[3,1]/v.wall_l) +
np.abs(v.wall_reaction[2,1]/v.wall_l) +
np.abs(v.wall_reaction[0,0]/v.wall_h) +
np.abs(v.wall_reaction[1,0]/v.wall_h))/4
    for v in V], phi,  label = r'$|p|$ avg')
plt.xlabel('Avg pressure')
plt.ylabel('Volume fraction')


plt.grid()
# plt.xlim(min(tt), max(tt))
# plt.autoscale(tight=True)
plt.gca().legend()
# plt.show()
filename = 'output/img/vol_frac_avg_pres_full.png'
print('Saving to :', filename)
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()


#######################################################################

# let force on the bulk over time
plt.plot(tt, [v.f_sum[0]/v.wall_l for v in V],  label = r'$\sum F_x$')
plt.plot(tt, [v.f_sum[1]/v.wall_l for v in V],  label = r'$\sum F_y$')
plt.xlabel('Time')
plt.ylabel('Net force on bulk')


plt.grid()
# plt.xlim(min(tt), max(tt))
# plt.autoscale(tight=True)
plt.gca().legend()
# plt.show()
filename = 'output/img/net_force_on_bulk_full.png'
print('Saving to :', filename)
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()
