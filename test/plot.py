import sys, os
sys.path.append(os.getcwd())

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sys import argv

import load_setup
from read_sim_output import read_plotinfo

######################################################################
V = np.load('test/V_n4.npy', allow_pickle=True)
V2 = np.load('test/V_plus0.2.npy', allow_pickle=True)

# override here
if len(argv) == 2:
    V = V[int(argv[1]):]
    V2 = V2[int(argv[1]):]
if len(argv) == 3:
    V = V[int(argv[1]):int(argv[2])]
    V2 = V2[int(argv[1]):int(argv[2])]

# tt = np.array(range(plti.fc, plti.lc+1)) * plti.dt * plti.modulo * 1e6
tt = np.array(range(len(V))) +1
tt2 = np.array(range(len(V2))) +1

# Volume fraction
# phi = [vol/(v.wall_v * v.wall_h) for v in V]
# plt.plot(tt, phi, label = r'$\phi$')

#######################################################################

### Total force 
# plt.plot(tt, [v.wall_reaction[2,1] for v in V], label = r'$F_y/V$ top')
# plt.plot(tt, [v.wall_reaction[3,1] for v in V], label = r'$F_y/V$ bottom')
# plt.plot(tt, [v.wall_reaction[0,0] for v in V], label = r'$F_x/V$ left')
# plt.plot(tt, [v.wall_reaction[1,0] for v in V], label = r'$F_x/V$ right')

### Pressure, average force
plt.plot(tt, [v.wall_reaction[2,1]/v.wall_h for v in V], label = r'$P_y$ top')
plt.plot(tt, [v.wall_reaction[3,1]/v.wall_h for v in V], label = r'$P_y$ bottom')
plt.plot(tt, [v.wall_reaction[0,0]/v.wall_v for v in V], label = r'$P_x$ left')
plt.plot(tt, [v.wall_reaction[1,0]/v.wall_v for v in V], label = r'$P_x$ right')

plt.plot(tt2, [v.wall_reaction[2,1]/v.wall_h for v in V2], linestyle='dashed', label = r'$P_y$ top')
plt.plot(tt2, [v.wall_reaction[3,1]/v.wall_h for v in V2], linestyle='dashed', label = r'$P_y$ bottom')
plt.plot(tt2, [v.wall_reaction[0,0]/v.wall_v for v in V2], linestyle='dashed', label = r'$P_x$ left')
plt.plot(tt2, [v.wall_reaction[1,0]/v.wall_v for v in V2], linestyle='dashed', label = r'$P_x$ right')

plt.grid()
# plt.xlim(min(tt), max(tt))
# plt.autoscale(tight=True)
plt.gca().legend()
# plt.show()
filename='test/avg_pressure_each.png'
print('Saving to', filename)
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

#######################################################################
### Top force
plt.plot(tt, [v.wall_reaction[2,1] for v in V], label = r'$F_y^{top}$ square')
# plt.plot(tt, [v.wall_reaction[3,1] for v in V], label = r'$P_y$ bottom')
# plt.plot(tt, [v.wall_reaction[0,0] for v in V], label = r'$P_x$ left')
# plt.plot(tt, [v.wall_reaction[1,0] for v in V], label = r'$P_x$ right')

plt.plot(tt2, [v.wall_reaction[2,1] for v in V2], linestyle='dashed', label = r'$F_y^{top}$ plus')
# plt.plot(tt2, [v.wall_reaction[3,1] for v in V2], linestyle='dashed', label = r'$P_y$ bottom')
# plt.plot(tt2, [v.wall_reaction[0,0] for v in V2], linestyle='dashed', label = r'$P_x$ left')
# plt.plot(tt2, [v.wall_reaction[1,0] for v in V2], linestyle='dashed', label = r'$P_x$ right')

plt.grid()
# plt.xlim(min(tt), max(tt))
# plt.autoscale(tight=True)
plt.gca().legend()
# plt.show()
filename='test/top_force.png'
print('Saving to', filename)
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

#######################################################################
### Top pressure
plt.plot(tt, [v.wall_reaction[2,1]/v.wall_h for v in V], label = r'$P_y^{top}$ square')
# plt.plot(tt, [v.wall_reaction[3,1] for v in V], label = r'$P_y$ bottom')
# plt.plot(tt, [v.wall_reaction[0,0] for v in V], label = r'$P_x$ left')
# plt.plot(tt, [v.wall_reaction[1,0] for v in V], label = r'$P_x$ right')

plt.plot(tt2, [v.wall_reaction[2,1]/v.wall_h for v in V2], linestyle='dashed', label = r'$P_y^{top}$ plus')
# plt.plot(tt2, [v.wall_reaction[3,1] for v in V2], linestyle='dashed', label = r'$P_y$ bottom')
# plt.plot(tt2, [v.wall_reaction[0,0] for v in V2], linestyle='dashed', label = r'$P_x$ left')
# plt.plot(tt2, [v.wall_reaction[1,0] for v in V2], linestyle='dashed', label = r'$P_x$ right')

plt.grid()
# plt.xlim(min(tt), max(tt))
# plt.autoscale(tight=True)
plt.gca().legend()
# plt.show()
filename='test/top_pressure.png'
print('Saving to', filename)
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

#######################################################################

### Total force 
# plt.plot(tt, [v.wall_reaction[2,1] for v in V], label = r'$F_y/V$ top')
# plt.plot(tt, [v.wall_reaction[3,1] for v in V], label = r'$F_y/V$ bottom')
# plt.plot(tt, [v.wall_reaction[0,0] for v in V], label = r'$F_x/V$ left')
# plt.plot(tt, [v.wall_reaction[1,0] for v in V], label = r'$F_x/V$ right')

### Pressure, average force
plt.plot(tt, [v.wall_reaction[2,1]/v.wall_h for v in V], label = r'$P_y$ top')
plt.plot(tt, [v.wall_reaction[3,1]/v.wall_h for v in V], label = r'$P_y$ bottom')
plt.plot(tt, [v.wall_reaction[0,0]/v.wall_v for v in V], label = r'$P_x$ left')
plt.plot(tt, [v.wall_reaction[1,0]/v.wall_v for v in V], label = r'$P_x$ right')

plt.plot(tt2, [v.wall_reaction[2,1]/v.wall_h for v in V2], linestyle='dashed', label = r'$P_y$ top')
plt.plot(tt2, [v.wall_reaction[3,1]/v.wall_h for v in V2], linestyle='dashed', label = r'$P_y$ bottom')
plt.plot(tt2, [v.wall_reaction[0,0]/v.wall_v for v in V2], linestyle='dashed', label = r'$P_x$ left')
plt.plot(tt2, [v.wall_reaction[1,0]/v.wall_v for v in V2], linestyle='dashed', label = r'$P_x$ right')

plt.grid()
# plt.xlim(min(tt), max(tt))
# plt.autoscale(tight=True)
plt.gca().legend()
# plt.show()
filename='test/avg_pressure_each.png'
print('Saving to', filename)
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

#######################################################################

### Pressure, average force
plt.plot(tt, [(v.wall_reaction[2,1] - v.wall_reaction[3,1])/v.wall_h for v in V], label = r'$P_y$ t-b')
plt.plot(tt, [(v.wall_reaction[1,0] - v.wall_reaction[0,0])/v.wall_v for v in V], label = r'$P_x$ l-r')
plt.plot(tt2, [(v.wall_reaction[2,1] - v.wall_reaction[3,1])/v.wall_h for v in V2], label = r'$P_y$ t-b', linestyle='dashed')
plt.plot(tt2, [(v.wall_reaction[1,0] - v.wall_reaction[0,0])/v.wall_v for v in V2], label = r'$P_x$ l-r', linestyle='dashed')

# plt.plot(tt2, [(v.wall_reaction[3,1] - v.wall_reaction[2,1])/v.wall_h for v in V2],  linestyle = 'dashed', label = r'$P_y$ b-t')
# plt.plot(tt2, [(v.wall_reaction[0,0] - v.wall_reaction[1,0] )/v.wall_v for v in V2], linestyle = 'dashed',  label = r'$P_x$ r-l')


plt.grid()
# plt.xlim(min(tt), max(tt))
# plt.autoscale(tight=True)
plt.gca().legend()
# plt.show()
filename='test/difference_pressure_each.png'
print('Saving to', filename)
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

#######################################################################

### Pressure, average force
plt.plot(tt, [( (np.abs(v.wall_reaction[2,1]) + np.abs(v.wall_reaction[3,1]))/v.wall_h + (np.abs(v.wall_reaction[1,0]) + np.abs(v.wall_reaction[0,0]) )/v.wall_v)/2 for v in V], label = r'$P_avg$ square')
plt.plot(tt2, [( (np.abs(v.wall_reaction[2,1]) + np.abs(v.wall_reaction[3,1]))/v.wall_h + (np.abs(v.wall_reaction[1,0]) + np.abs(v.wall_reaction[0,0]) )/v.wall_v)/2 for v in V2], linestyle='dashed', label = r'$P_avg$')



plt.grid()
# plt.xlim(min(tt), max(tt))
# plt.autoscale(tight=True)
plt.gca().legend()
# plt.show()
filename='test/press_avg_all.png'
print('Saving to', filename)
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

#######################################################################

### Deviatoric pressure
strain = [1-v.wall_v/V[0].wall_v for v in V]
strain2 = [1-v.wall_v/V2[0].wall_v for v in V2]


plt.plot(strain, [np.abs(v.wall_reaction[2,1]) / np.abs(v.wall_reaction[1,0] ) for v in V], label = r'$P_{deviatoric}$')
plt.plot(strain2, [np.abs(v.wall_reaction[2,1]) / np.abs(v.wall_reaction[1,0] ) for v in V2], label = r'$P_{deviatoric}$ +_{0.2}', linestyle='dashed')



plt.grid()
# plt.xlim(min(tt), max(tt))
# plt.autoscale(tight=True)
plt.gca().legend()
# plt.show()
filename='test/press_deviatoric.png'
print('Saving to', filename)
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

#######################################################################
# let force on the bulk over time
plt.plot(tt, [v.f_sum[0]/v.wall_h for v in V],  label = r'$\sum F_x$')
plt.plot(tt, [v.f_sum[1]/v.wall_h for v in V],  label = r'$\sum F_y$')
plt.xlabel('Time')
plt.ylabel('Net force on bulk')

plt.grid()
# plt.xlim(min(tt), max(tt))
# plt.autoscale(tight=True)
plt.gca().legend()
# plt.show()
filename = 'test/net_force_on_bulk.png'
print('Saving to :', filename)
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()
