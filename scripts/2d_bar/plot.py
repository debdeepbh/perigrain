import sys, os
sys.path.append(os.getcwd())
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sys import argv

import load_setup
from read_sim_output import read_plotinfo

#######################################################################
# loc = 'output/hdf5/'
# plti = read_plotinfo(loc+'plotinfo.h5')
# dim = plti.dim

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

# override here
V = V[int(argv[1]):int(argv[2])]
l = float(argv[3])
s = float(argv[4])

# Load young's modulus
E = 0
with open('output/csv/bar_youngs.csv', 'r') as f:
    E = float(f.readline())

print('Youngs', E)
print('length of argv= ', len(argv))
print('length of V= ', len(V))


# tt = np.array(range(plti.fc, plti.lc+1)) * plti.dt * plti.modulo * 1e6
tt = np.array(range(len(V))) +1

# Volume fraction
phi = [vol/(v.wall_v * v.wall_h) for v in V]
# plt.plot(tt, phi, label = r'$\phi$')

#######################################################################

### Total force 
# plt.plot(tt, [v.wall_reaction[2,1] for v in V], label = r'$F_y/V$ top')
# plt.plot(tt, [v.wall_reaction[3,1] for v in V], label = r'$F_y/V$ bottom')
# plt.plot(tt, [v.wall_reaction[0,0] for v in V], label = r'$F_x/V$ left')
# plt.plot(tt, [v.wall_reaction[1,0] for v in V], label = r'$F_x/V$ right')

### Pressure, average force
# plt.plot(tt, [v.wall_reaction[2,1]/v.wall_h for v in V], label = r'$P_y$ top')
# plt.plot(tt, [v.wall_reaction[3,1]/v.wall_h for v in V], label = r'$P_y$ bottom')
# plt.plot(tt, [v.wall_reaction[0,0]/v.wall_v for v in V], label = r'$P_x$ left')

# ref length = 2l
# ref thickness = 2s
strain = np.array([(2*l-v.wall_h)/(2*l) for v in V])
# strain = np.array([(l-v.wall_h-1e-3/6)/l for v in V])
stress = np.array([v.wall_reaction[1,0]/(2*s) for v in V])


plt.plot(strain, stress, label = r'$\sigma_x$ right')

# material stress-strain curve
fixed = 10
plt.plot(strain, E * (strain - strain[fixed]) + stress[fixed], label = r'$\sigma_{mat}$')


plt.grid()
# plt.xlim(min(tt), max(tt))
# plt.autoscale(tight=True)
plt.gca().legend()
# plt.show()
filename = 'output/img/loading_curve.png'
print('Saving to :', filename)
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

#######################################################################
## slope of the curve
# sig_0 = stress[0]
# eps_0 = strain[0]
# # print(stress[1] - sig_0)
# # print(strain[1] - eps_0)
# # EE  = (stress[1] - sig_0 )/ (strain[1] - eps_0)
# print(stress)
# print(strain)
# running_E = np.array([(sig - sig_0)/(eps - eps_0) for (sig,eps) in zip(stress[1:],strain[1:])])
# E_diff = running_E - E

# # plt.plot(running_E)
# plt.plot(E_diff)

# plt.grid()
# # plt.xlim(min(tt), max(tt))
# # plt.autoscale(tight=True)
# plt.gca().legend()
# # plt.show()
# filename = 'output/img/E_diff.png'
# print('Saving to :', filename)
# plt.savefig(filename, dpi=300, bbox_inches='tight')
# plt.close()

#######################################################################
