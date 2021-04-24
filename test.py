import shape_dict
import pygmsh
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

from genmesh import genmesh

from exp_dict import get_incenter_mesh_loc

import meshio

import shape_dict

# shp = shape_dict.sphere_small_3d()
shp = shape_dict.disk_w_hole_3d()
# shp = shape_dict.sphere_unit_3d()

genmesh(P_bdry=[], meshsize=[], msh_file = shp.msh_file, do_plot = True, dimension = 3, dotsize = 20 )

# msh = genmesh(P_bdry = None, meshsize = None, msh_file = 'meshdata/peridem_mat.msh', do_plot = True, dotsize = 10 )

# P = np.array([
    # [0, 0],
    # [1, 0],
    # [1, 1],
    # [0.5, 1.5], #extra
    # [0, 1]
    # ])

# meshsize = 0.125/4

# Testing pygmsh
# with pygmsh.geo.Geometry() as geom:
    # polygon1 = geom.add_polygon(
        # P,
        # mesh_size= meshsize,
    # )
    # geom.add_physical(polygon1.surface, 'surface1')
    # mesh = geom.generate_mesh()

# # print(len(mesh.get_cells_type("triangle")))
# # print(len(mesh.cells[1][1]))
# # print(mesh.cells[1][1])

# print(mesh.cells[0][1])

# msh = get_incenter_mesh_loc(P, meshsize)
# msh = get_incenter_mesh_loc(P, meshsize, gen_bdry = False )
# msh = get_incenter_mesh_loc(P, meshsize, modify_nodal_circles= False, gen_bdry = False )

# print('total count: ', msh.count())
# print('polygon vol: ', msh.polygon_volume())
# print('occupied vol: ', msh.occupied_volume())
# print('packing raio: ', msh.packing_ratio())
# msh.info()
# msh.trim(min_rad = 0.005, max_rad = 0.01)
# msh.info()
# msh.plot(plot_edge = True)

## Optimization using gekko
# from gekko import GEKKO
# m = GEKKO(remote=False)
# m.options.SOLVER = 1
# x1,x2,x3,Z = m.Array(m.Var,4)
# m.Maximize(Z)
# m.Equation(x1+x2+x3==15)
# m.Equations([Z<=x1,Z<=x2,Z<=x3])
# m.solve()
# print('x1: ',x1.value[0])
# print('x2: ',x2.value[0])
# print('x3: ',x3.value[0])
# print('Z:  ',Z.value[0])

# from gekko import GEKKO
# m = GEKKO(remote=False)
# m.options.SOLVER = 1
# x1,x2,Z = m.Array(m.Var,3)
# pk1 = 0.7292786279623197
# pk2 = 0.1121364639219373
# m.Equation( Z <= ((x1 - pk1)**2 + (x2 - pk2)**2)**0.5 - 0.00001 ) 
# m.Equation( Z <= ((x1 - pk1*2)**2 + (x2 - pk2*2)**2)**0.5 - 0.00001 ) 
# m.solve()

# print(msh.range())
# print(msh.mean())
# msh.plot()


# meshsize = 1e-3/3
# dotsize = 15

# shape = shape_dict.plus()

# P_bdry = shape.P

# # mesh
# with pygmsh.geo.Geometry() as geom:
    # polygon1 = geom.add_polygon(
        # P_bdry,
        # mesh_size= meshsize,
    # )
    # # geom.add_physical(polygon1.surface, 'surface1')
    # mesh = geom.generate_mesh()

# Pos = mesh.points
# total_nodes = len(Pos)
# print(total_nodes)

# # Triangles
# T = mesh.cells[1][1]
# area = np.zeros((total_nodes, 1))



# plt.scatter(Pos[:,0], Pos[:,1], s = dotsize, marker = '.', linewidth = 0, cmap='viridis')

# # # text
# for i in range(0,len(Pos)):
    # plt.annotate(str(i), (Pos[i,0], Pos[i,1]))

# plt.axis('scaled')
# plt.show()


