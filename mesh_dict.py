import pygmsh
import numpy as np
import matplotlib.pyplot as plt

# import optimesh

# 3d plot
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits import mplot3d

import meshio

def genmesh(P_bdry, meshsize, msh_file = None, do_plot = True, dimension = 2, dotsize = 10, mesh_optimize=True):
    """Generate a mesh from given polygon
    :P_bdry: an array of boundary points
    :meshsize: 
    :returns: pos and vol
    """

    # mesh
    if msh_file is None:
        # print('Generating mesh from poly')
        # with pygmsh.geo.Geometry() as geom:
        with pygmsh.occ.Geometry() as geom:
            polygon1 = geom.add_polygon(
                P_bdry,
                mesh_size= meshsize,
            )
            geom.add_physical(polygon1.surface, 'surface1')
            mesh = geom.generate_mesh()
    else:
        print('Loading mesh from file: ', msh_file)
        mesh = meshio.read(msh_file)

    if mesh_optimize:
        # print('Optimizing mesh')
        # mesh = pygmsh.optimize(mesh, method="")
        # mesh = pygmsh.optimize(mesh, method="Netgen")
        # mesh = optimesh.optimize(mesh, "CVT (block-diagonal)", 1.0e-5, 100)
        pass


def test():
    filename = "meshdata/2d/test.msh"
    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(
            [
                [0.0, 0.0],
                [1.0, -0.2],
                [1.1, 1.2],
                [0.1, 0.7],
            ],
            mesh_size=0.1,
        )
        mesh = geom.generate_mesh()

    pygmsh.write(filename)
    return filename

