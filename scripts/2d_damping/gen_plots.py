import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.append(os.getcwd())

from sys import argv

# last 3 args will be the output filename
t_i = len(argv)-4
# print(argv)

print('range: ', range(t_i))

d = []
labels = []
for i in range(t_i):
    print('loading file:', argv[i+1])
    d.append(np.genfromtxt(argv[i+1], delimiter=','))
    labels.append(os.path.splitext(os.path.basename(argv[i+1]))[0])
    
for i in range(t_i):
    plt.plot(d[i][:,0], d[i][:,1], label = r'$d_'+str(i)+'$')

plt.grid()
plt.autoscale(tight=True)
plt.gca().legend()
plt.xlabel(r'Time ($\mu$s)')
plt.ylabel(r'Displacement ($m$)')
# plt.show()
plt.savefig(argv[-3], dpi=300, bbox_inches='tight')
plt.close()

#######################################################################

for i in range(t_i):
    plt.plot(d[i][:,0], d[i][:,2], label = r'$d_'+str(i)+'$')

plt.grid()
# plt.autoscale(tight=True)
plt.gca().legend()
plt.xlabel(r'Time ($\mu$s)')
plt.ylabel(r'Velocity ($ms^{-1}$)')
# plt.show()
plt.savefig(argv[-2], dpi=300, bbox_inches='tight')
plt.close()

#######################################################################

from read_sim_output import movingaverage
for i in range(t_i):
    plt.plot(d[i][:,0], movingaverage(d[i][:,3], 10), label = r'$d_'+str(i)+'$')

plt.grid()
# plt.autoscale(tight=True)
plt.gca().legend()
plt.xlabel(r'Time ($\mu$s)')
plt.ylabel(r'Acceleration ($ms^{-2}$)')
# plt.show()
plt.savefig(argv[-1], dpi=300, bbox_inches='tight')
plt.close()
