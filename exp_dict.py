import numpy as np
import copy
import h5py
import matplotlib.pyplot as plt
# to plot a collection of lines
from matplotlib.collections import LineCollection
from gekko import GEKKO

# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
from itertools import combinations
# from functools import partial
import time
from random import seed
from random import random

import shape_dict
import material_dict
from genmesh import genmesh


def single_bond_ext(ij_pair, pos, delta, remove_ncvx_bonds, interceptor):
    i = ij_pair[0]
    j = ij_pair[1]

    p_i = pos[i]
    p_j = pos[j]
    d = np.sqrt(np.sum(np.square(p_j - p_i))) #norm
    if (d <= delta):
        ##### remove nonconvex bonds
        if remove_ncvx_bonds:
            intersects = False
            for k in range(len(interceptor)):
                # print(k)
                A1 = interceptor[k][0:2]
                A2 = interceptor[k][2:4]
                if (lines_intersect(A1, A2, p_i, p_j)):
                    intersects = True
                    break
            if not intersects:
                return [i,j]
            # else:
                # return None
        else:
            return [i,j]
    # else:
        # return None

def lines_intersect(A1, A2, A3, A4):
    """ Whether the lines A1-A2 and A3-A4 intersect
    :returns: TODO
    : Ai = [Ai_x, Ai_y] is an np.array
    """

    A = np.array([A1, A2, A3, A4])
    x= A[:,0]
    y= A[:,1]

    M = np.array([ [x[0] - x[1], x[3] - x[2]], [y[0] - y[1], y[3] - y[2]] ])
    b = np.array( [ x[3] - x[1], y[3] - y[1] ] )

    if (np.linalg.det(M) == 0):
        return False
    else:
        sols = np.linalg.solve(M, b)
        if ((sols[0] < 1) and (sols[0] > 0) and (sols[1] < 1) and (sols[1] > 0)):
            return True
        else:
            return False

def get_nbd_vertex_elem(T, total_vertices):
    """ From the element (facet) array, compute the facet-neighborhood array of the vertices

    :T: vertex indices of all elements
    :returns: neighboring facet indices of all vertices

    """

    full_list = []

    for i in range(total_vertices):
        # vertex i won't appear more than once in a single row
        row_list = np.where(T == i)[0]
        full_list.append(row_list.tolist())

    return full_list
    

class Particle(object):
    """ A discretized particle with nodes 
    Convention: All member variables are made accessible in the first level to make it compatible with other functions, i.e. self.pos instead of self.mesh.pos
    """
    # def __init__(self, shape, meshsize, material):
    def __init__(self, mesh, shape, material, nbdarr_in_parallel=True):
        self.shape = shape


        # generate mesh
        # mesh = genmesh(shape.P, meshsize)
        # self.mesh = mesh
        #redundant assignment.  Should be self.mesh.pos
        #Keeping it for compatibility with other functions.

        self.pos = mesh.pos
        self.vol = mesh.vol


        # boundary nodes
        self.edge_nodes = mesh.bdry_nodes

        # default nonlocal boundary nodes: all
        self.nonlocal_bdry_nodes = range(len(self.pos))

        # print('Computing boundary nodes: ')
        # print('With hard-coded contact radius: ')
        # c_R = 1e-3/5
        # temp = []
        # for p in range(len(self.edge_nodes)):
            # e_node = self.edge_nodes[p]
            # pos_node = self.pos[e_node]
            # for q in range(len(self.pos)):
                # pos_other = self.pos[q]
                # d = np.sqrt(np.sum(np.square(pos_node - pos_other))) #norm
                # # if (d <= self.contact.contact_radius):
                # if (d <= c_R*1.1):
                    # temp.append(q)
        # self.nonlocal_bdry_nodes = list(set(temp))

        self.bdry_edges = mesh.bdry_edges

        # dimension
        total_nodes = len(self.pos);
        self.dim = len(self.pos[0])
        # print('dimension = ', self.dim)

        # default initial data
        self.disp = np.zeros((total_nodes, self.dim));
        self.vel = np.zeros((total_nodes, self.dim));
        self.acc = np.zeros((total_nodes, self.dim));
        self.extforce = np.zeros((total_nodes, self.dim));

        self.clamped_nodes = []

        self.material = material

        # torque info
        self.torque_axis = 2
        self.torque_val = 0

        # extra properties
        self.movable = 1
        self.breakable = 1
        self.stoppable = 1

        # neighborhood array generation
        # This is connectivity: a list of edges that are connected
        # print('Computing neighborhood connectivity.')
        self.NArr = []
        nci = self.shape.nonconvex_interceptor
        # print('nci', nci)
        remove_ncvx_bonds = False
        interceptor = []
        if (nci and self.dim ==2):
            remove_ncvx_bonds = True
            interceptor = nci.all()
            # print('(Will remove nonconvex bonds)')


        # start = time.time()

        def single_bond(self, ij_pair):
            i = ij_pair[0]
            j = ij_pair[1]

            p_i = self.pos[i]
            p_j = self.pos[j]
            d = np.sqrt(np.sum(np.square(p_j - p_i))) #norm
            if (d <= self.material.delta):
                ##### remove nonconvex bonds
                if remove_ncvx_bonds:
                    intersects = False
                    for k in range(len(interceptor)):
                        # print(k)
                        A1 = interceptor[k][0:2]
                        A2 = interceptor[k][2:4]
                        if (lines_intersect(A1, A2, p_i, p_j)):
                            intersects = True
                            break
                    if not intersects:
                        return [i,j]
                    # else:
                        # return None
                else:
                    return [i,j]
            # else:
                # return None

        # one parameter function to be used by pool
        def oneparam_f_bond(ij):
            return single_bond(self,ij)
            ## Using an external function to call for parallelization, otherwise, I get error 
            # return single_bond_ext(ij, self.pos, self.material.delta, remove_ncvx_bonds, interceptor)

        # print('Will generate nbdarr')

        if nbdarr_in_parallel:
            ## parallel attempt
            a_pool = Pool()
            all_bonds = a_pool.map(oneparam_f_bond, combinations(range(total_nodes),2)) 
            # print(all_bonds)
        else:
            ## Serial version

            # for ij in list(combinations(range(total_nodes), 2)):
            # print(out_bonds(self, ij))

            all_bonds = []
            for ij in list(combinations(range(total_nodes), 2)):
                all_bonds.append( oneparam_f_bond(ij))

        # remove all None
        self.NArr = np.array([i for i in all_bonds if i is not None])

        # print('Done generating nbdarr')


        # else:
            # ## Serial version
            # for i in range(total_nodes):
                # p_i = self.pos[i]
                # for j in range(i+1, total_nodes):
                    # # if (j != i):
                        # p_j = self.pos[j]
                        # d = np.sqrt(np.sum(np.square(p_j - p_i))) #norm
                        # if (d <= self.material.delta):
                            # ##### remove nonconvex bonds
                            # if remove_ncvx_bonds:
                                # intersects = False
                                # for k in range(len(interceptor)):
                                    # # print(k)
                                    # A1 = interceptor[k][0:2]
                                    # A2 = interceptor[k][2:4]
                                    # if (lines_intersect(A1, A2, p_i, p_j)):
                                        # intersects = True
                                        # break

                                # if not intersects:
                                    # self.NArr.append([i, j])
            # self.NArr  = np.array(self.NArr)

            # ## remove nonconvex bonds
            # if self.dim==2:
                # # print(self.shape.nonconvex_interceptor)
                # if self.shape.nonconvex_interceptor is not None:
                    # interceptor = self.shape.nonconvex_interceptor.all()
                    # print('Removing nonconvex bonds')
                    # to_delete = []
                    # for j in range(len(interceptor)):
                        # A1 = interceptor[j][0:2]
                        # A2 = interceptor[j][2:4]
                        # for i in range(len(self.NArr)):
                            # edge = self.NArr[i]
                            # A3 = self.pos[edge[0]]
                            # A4 = self.pos[edge[1]]

                            # if lines_intersect(A1, A2, A3, A4):
                                # # print('non-convex bond: ', edge)
                                # to_delete.append(i)
                                # # print(len(self.NArr))
                
                    # # delete at once. delete is not in-place
                    # self.NArr = np.delete(self.NArr, to_delete, 0)
                    # pass
            # else:
                # pass
                # # 3d code for nonconvex bond removal goes here


        # print(self.NArr)
        # print('time taken ', time.time() - start)


                
    def gen_nonlocal_boundry_nodes(self, contact_radius, scaled = 1):
        """ Generate nonlocal boundary nodes
        :contact_radius: 
        """
        # print('Computing nonlocal boundary nodes: ', end = ' ', flush = True)
        if contact_radius is not None:
            temp = []
            for p in range(len(self.edge_nodes)):
                e_node = self.edge_nodes[p]
                pos_node = self.pos[e_node]
                for q in range(len(self.pos)):
                    pos_other = self.pos[q]
                    d = np.sqrt(np.sum(np.square(pos_node - pos_other))) #norm
                    # if (d <= self.contact.contact_radius):
                    if (d <= contact_radius*scaled):
                        temp.append(q)
            self.nonlocal_bdry_nodes = list(set(temp))
        # print(len(self.nonlocal_bdry_nodes))


    def rotate(self, angle):
        """ Rotate by angle in the clockwise direction
        Only for 2D particles
        """
        if self.dim!=2:
            print('Wrong dimension for rotation. Need 2. Use rotate3d for 3D particle.')
        Rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)] ])
        self.pos = self.pos @ Rot   # @ is matrix multiplication, so is a.dot(b) 

    def rotate3d(self, axis, angle):
        """ Rotate by angle in the counterclockwise direction about an axis
        Only for 3d particles
        """
        if axis=='x':
            Rot = np.array([[1,0,0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)] ])
        if axis=='y':
            Rot = np.array([[np.cos(angle),0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)] ])
        if axis=='z':
            Rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1] ])

        self.pos = self.pos @ Rot   # @ is matrix multiplication, so is a.dot(b) 

    def trasnform_bymat(self, matrix):
        """ Transform using a matrix
        """
        # Rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)] ])
        self.pos = self.pos @ matrix   # @ is matrix multiplication, so is a.dot(b) 
        print('Caution: did not transform the volume to maintain the density.')

    def shift(self, shift):
        self.pos += shift

    def scale(self, scale):
        self.pos *= scale
        # update the volume
        self.vol *= (scale**self.dim)

        # delta and cnot are modified since the neighbors are not being recomputed

class ShapeList(object):
    """Contains a list of instances of Shape class, and other info """
    def __init__(self):
        self.shape_list = []
        self.count_list = []
        self.meshsize_list = []
        self.material_list = []

        # self.msh_file_list = []

    def append(self, shape, count, meshsize, material, plot_shape = True):
    # def append(self, shape, count, meshsize, material, msh_file= None):
        """TODO: Docstring for append.

        :shape: TODO
        :count: TODO
        :meshsize: TODO
        :material: TODO
        :returns: TODO

        """
        self.shape_list.append(shape)
        self.count_list.append(count)
        self.meshsize_list.append(meshsize)
        self.material_list.append(material)



    def generate_mesh(self, dimension = 2, contact_radius = None, plot_mesh = True, plot_node_text=False, plot_shape = True, shapes_in_parallel=False):
        """Returns a rank-2 array of particle meshes
        : shapes_in_parallel: if set to true, it computes the particle properties in parallel for each shape. Otherwise, the NbdArr and boundary node computation happens in parallel for each node. 
        Outermost parallelization is most preferable. So, for multiple shapes/particles with small number of nodes requires this option to be true. On the other hands, for single particle with many nodes, setting this to false works best.
        :returns: particles[sh][i] for shape sh count i
        """

        # if the shapes are computed in parallel, do not compute NbdArr etc in parallel
        # outermost parallelism
        nbdarr_in_parallel  = not shapes_in_parallel

        print('Shape:', end=' ', flush=True)

        # Helper function
        def gen_particle(sh):
            print(sh, end=' ', flush=True)
                # print(i, end = ' ', flush=True)
            shape = self.shape_list[sh]
            if shape.msh_file is None:
                mesh = genmesh(P_bdry=shape.P, pygmsh_geom=shape.pygmsh_geom, meshsize=self.meshsize_list[sh], dimension = dimension)

                # print(shape.P)
                if (plot_shape and (shape.P is not None)):
                    shape.plot(bdry_arrow=True, extended_bdry=True, angle_bisector=True)
                if plot_mesh:
                    mesh.plot(dotsize=10, plot_node_text=plot_node_text, highlight_bdry_nodes=True)
            else:
                # when the .msh is specified
                # mesh = genmesh(P_bdry=None, meshsize=None, msh_file=shape.msh_file, dimension = dimension, do_plot = plot_mesh)
                mesh = genmesh(P_bdry=None, meshsize=None, msh_file=shape.msh_file, dimension = dimension)
                if plot_mesh:
                    mesh.plot(dotsize=10, plot_node_text=plot_node_text, highlight_bdry_nodes=True)



            # print('total mesh volume: ', np.sum(mesh.vol))

            PP = Particle(mesh=mesh, shape=self.shape_list[sh], material=self.material_list[sh], nbdarr_in_parallel=nbdarr_in_parallel)

            # print('Done generating particles')

            # generate nonlocal boundary nodes
            PP.gen_nonlocal_boundry_nodes(contact_radius)

            particles_sh = []
            for i in range(self.count_list[sh]):
                # shallow copy (=) refers to the same object, instead of creating a new one
                this_PP = copy.deepcopy(PP)
                particles_sh.append(this_PP)

            # particles.append(particles_sh)
            return particles_sh


        start_sh = time.time()
        if shapes_in_parallel:
            # parallel attempt
            shape_pool = Pool()
            particles = shape_pool.map(gen_particle, range(len(self.shape_list))) 
        else:
            # serial version
            particles = []
            for sh in range(len(self.shape_list)):
                particles.append(gen_particle(sh))

        print('\n')
        print('time taken to generate all shapes', time.time() - start_sh)

        
        return particles

        
# def generate_particles(shapes, shape_count, meshsize, material_list):
    # """ Generate Particle related to each shape
    # :shapes: a list of instances of the shape class generated from shape_dict.py
    # :shape_count: how many members of each shape to create
    # :returns: 2d array of particles, particles[shape][count]
    # """
    # particles = []
    # for sh in range(len(shapes)):
        # print('Shape', sh)
        # PP = Particle(shapes[sh], meshsize, material_list[sh])

        # particles_sh = []
        # for i in range(shape_count[sh]):
            # # shallow copy (=) refers to the same object, instead of creating a new one
            # this_PP = copy.deepcopy(PP)
            # particles_sh.append(this_PP)

        # particles.append(particles_sh)
    
    # return particles


class Wall(object):
    """ Geometric wall """
    def __init__(self, allow = 0, left = None, right = None, top = None, bottom = None ):
        self.allow  = allow
        self.left        = left
        self.right       = right
        self.top         = top 
        self.bottom      = bottom

        self.reaction = None

    def get_lrtp(self):
        """ Returns the wall dimensions in an array
        :returns: TODO

        """
        return np.array([self.left, self.right, self.top, self.bottom])

    def set_size(self, ss):
        """ Reads the wall size from an array
        """
        if (len(ss) !=4):
            print('Error: input array dimension is not 4.')
        self.left        = ss[0]
        self.right       = ss[1]
        self.top         = ss[2]
        self.bottom      = ss[3]

    def get_lines(self):
        """ List of lines that are wall boundaries
        To feed to the collection
        """
        a = [self.left, self.bottom]
        b = [self.right, self.bottom]
        c = [self.right, self.top]
        d = [self.left, self.top]

        return [ [a, b], [b,c], [c,d], [d,a] ]

    def print(self):
        print('allow =  ', self.allow)
        print('left: ', self.left)
        print('right: ', self.right)
        print('top: ', self.top)
        print('bottom: ', self.bottom)

    def get_h(self):
        """get the horizontal length of the wall
        """
        return self.right - self.left
    def get_v(self):
        """get the vertical height of the wall
        """
        return self.top - self.bottom
    def wall_vol(self):
        """returns the wall volume
        """
        return self.get_v() * self.get_h()

class Wall3d(object):
    """ Geometric wall """
    def __init__(self, allow = 0, x_min = None, y_min = None, z_min = None, x_max = None, y_max = None, z_max = None):
        self.allow  = allow
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max

    def get_lrtp(self):
        """ Returns the wall dimensions in an array
        :returns: TODO

        """
        return np.array([self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max])

    def set_size(self, ss):
        """ Reads the wall size from an array
        """
        if (len(ss) !=6):
            print('Error: input array dimension is not 4.')

        self.x_min = ss[0] 
        self.y_min = ss[1]
        self.z_min = ss[2]
        self.x_max = ss[3]
        self.y_max = ss[4]
        self.z_max = ss[5]





    def get_faces(self):

        # list of vertices
        Z = np.array([
            [self.x_min, self.y_min, self.z_min],
            [self.x_max, self.y_min, self.z_min],
            [self.x_max, self.y_max, self.z_min],
            [self.x_min, self.y_max, self.z_min],
            [self.x_min, self.y_min, self.z_max],
            [self.x_max, self.y_min, self.z_max],
            [self.x_max, self.y_max, self.z_max],
            [self.x_min, self.y_max, self.z_max]
            ])

        # list of faces
        faces = [
             [Z[0],Z[1],Z[2],Z[3]],
             [Z[4],Z[5],Z[6],Z[7]], 
             [Z[0],Z[1],Z[5],Z[4]], 
             [Z[2],Z[3],Z[7],Z[6]], 
             [Z[1],Z[2],Z[6],Z[5]],
             [Z[4],Z[7],Z[3],Z[0]]
             ]

        return faces
    
    def get_lines(self):
        # list of vertices
        Z = np.array([
            [self.x_min, self.y_min, self.z_min],
            [self.x_max, self.y_min, self.z_min],
            [self.x_max, self.y_max, self.z_min],
            [self.x_min, self.y_max, self.z_min],
            [self.x_min, self.y_min, self.z_max],
            [self.x_max, self.y_min, self.z_max],
            [self.x_max, self.y_max, self.z_max],
            [self.x_min, self.y_max, self.z_max]
            ])

        # list of faces
        faces = [
             [Z[0],Z[1],Z[2],Z[3]],
             [Z[4],Z[5],Z[6],Z[7]], 
             [Z[0],Z[1],Z[5],Z[4]], 
             [Z[2],Z[3],Z[7],Z[6]], 
             [Z[1],Z[2],Z[6],Z[5]],
             [Z[4],Z[7],Z[3],Z[0]]
             ]

        lines = np.array([
            [Z[0], Z[1]],
            [Z[1], Z[2]],
            [Z[2], Z[3]],
            [Z[3], Z[0]],

            [Z[4], Z[5]],
            [Z[5], Z[6]],
            [Z[6], Z[7]],
            [Z[7], Z[4]],

            [Z[0], Z[4]],
            [Z[1], Z[5]],
            [Z[2], Z[6]],
            [Z[3], Z[7]]
            ])

        return lines



    def print(self):
        print('allow =  ', self.allow)
        print('x_min: ', self.x_min)
        print('y_min: ', self.y_min)
        print('z_min: ', self.z_min)
        print('x_max: ', self.x_max)
        print('y_max: ', self.y_max)
        print('z_max: ', self.z_max)
                
class Contact(object):
    """Pairwise contact properties"""
    def __init__(self, contact_radius, normal_stiffness, damping_ratio, friction_coefficient):
        self.contact_radius         = contact_radius
        self.normal_stiffness        = normal_stiffness
        self.damping_ratio           = damping_ratio
        self.friction_coefficient    = friction_coefficient 

    def print(self):
        """print info

        :f: TODO
        :returns: TODO

        """
        print('contact_radius: ', self.contact_radius)
        print('normal_stiffness: ', self.normal_stiffness)
        print('damping_ratio: ', self.damping_ratio)
        print('friction_coefficient: ', self.friction_coefficient)
        

class Experiment(object):

    """Experiment setup"""

    def __init__(self, particles, wall, contact):
        """Collection of all

        :particles: TODO
        :wall: TODO
        :contact: TODO

        """
        self.particles = particles
        self.wall = wall
        self.contact = contact

        
        ## compute the boundary nodes, for each particle count
        # print('Computing boundary nodes: ')
        # for sh in range(len(self.particles)):
            # count = len(self.particles[sh])
            # for i in range(count):
                # particle = self.particles[sh][i]
                # temp = []
                # for p in range(len(particle.edge_nodes)):
                    # e_node = particle.edge_nodes[p]
                    # pos_node = particle.pos[e_node]
                    # for q in range(len(particle.pos)):
                        # pos_other = particle.pos[q]
                        # d = np.sqrt(np.sum(np.square(pos_node - pos_other))) #norm
                        # # if (d <= self.contact.contact_radius):
                        # if (d <= self.contact.contact_radius*1.1):
                            # temp.append(q)
                # particle.nonlocal_bdry_nodes = list(set(temp))
                ## particle.nonlocal_bdry_nodes = list(set(range(len(particle.pos))))

        
                # print('edge nodes: ', len(particle.edge_nodes))
                # print('bdry nodes: ', len(particle.nonlocal_bdry_nodes))
                # print(i, [len(particle.edge_nodes), len(particle.nonlocal_bdry_nodes)], 
                # print(i, end = ' ', flush=True)

        # print('')



    def save(self, filename):
        """Save experiment setup to file

        :filename: filename with path
        :returns: TODO

        """
        print('Saving univ particle: ')
        with h5py.File(filename, "w") as f:
            # universal index starts at 1 to be compatible with matlab
            j = 1

            for sh in range(len(self.particles)):
                count = len(self.particles[sh])
                for i in range(count):
                    particle = self.particles[sh][i]
                    p_ind = ('P_%05d' % j)
                    f.create_dataset(p_ind + '/Pos', data=particle.pos)
                    f.create_dataset(p_ind + '/Vol', data=particle.vol)
                    f.create_dataset(p_ind + '/Connectivity', data=particle.NArr)
                    # needs to be a column matrix
                    f.create_dataset(p_ind + '/bdry_nodes', data=np.array([particle.nonlocal_bdry_nodes]).transpose())
                    f.create_dataset(p_ind + '/bdry_edges', data=np.array(particle.bdry_edges))

                    f.create_dataset(p_ind + '/clamped_nodes', data=np.array(particle.clamped_nodes))

                    # Initial conditions
                    f.create_dataset(p_ind + '/disp', data=particle.disp)
                    f.create_dataset(p_ind + '/vel', data=particle.vel)
                    f.create_dataset(p_ind + '/acc', data=particle.acc)
                    f.create_dataset(p_ind + '/extforce', data=particle.extforce)

                    # Material properties
                    f.create_dataset(p_ind + '/delta', data=[[particle.material.delta]])
                    f.create_dataset(p_ind + '/rho', data= [[particle.material.rho]])
                    f.create_dataset(p_ind + '/cnot', data= [[particle.material.cnot]])
                    f.create_dataset(p_ind + '/snot', data= [[particle.material.snot]])

                    # torque info
                    f.create_dataset(p_ind + '/torque_axis', data= [[particle.torque_axis]])
                    f.create_dataset(p_ind + '/torque_val', data= [[particle.torque_val]])

                    # extra properties
                    f.create_dataset(p_ind + '/movable', data= [[particle.movable]])
                    f.create_dataset(p_ind + '/breakable', data= [[particle.breakable]])
                    f.create_dataset(p_ind + '/stoppable', data= [[particle.stoppable]])

                    print(j, end = ' ', flush=True)
                    j = j+1

            print('\n')

            # making all scalars a rank-2 data (matrix) to make it compatible with matlab
            f.create_dataset('total_particles_univ', data=[[j-1]])

            # contact info
            f.create_dataset('pairwise/contact_radius', data=[[self.contact.contact_radius]])
            f.create_dataset('pairwise/normal_stiffness', data=[[self.contact.normal_stiffness]])
            f.create_dataset('pairwise/damping_ratio', data=[[self.contact.damping_ratio]])
            f.create_dataset('pairwise/friction_coefficient', data=[[self.contact.friction_coefficient]])

            # wall info
            f.create_dataset('wall/allow_wall', data = [[self.wall.allow]])
            f.create_dataset('wall/geom_wall_info', data = np.array([self.wall.get_lrtp()]).transpose() )


        
    # [Deprecated] Use  load_setup.plot() instead
    def plot(self, plot_delta = True, plot_contact_rad = True, plot_bonds = False, plot_bdry_nodes = False, plot_bdry_edges= True, plot_wall = True, dotsize = 10, linewidth = 0.8):
        """TODO: Docstring for plot.

        :plot_delta: TODO
        :plot_bonds: TODO
        :dotsize: TODO
        :linewidth: TODO
        :returns: TODO

        """

        print('Plotting the experiment setup.')

        # close existing plots
        plt.close()


        print('Univ particle: ')
        for sh in range(len(self.particles)):
            count = len(self.particles[sh])
            for i in range(count):
                # Plot the mesh
                Pos = self.particles[sh][i].pos
                plt.scatter(Pos[:,0], Pos[:,1], s = dotsize, marker = '.', linewidth = 0, cmap='viridis')

                print(i, end = ' ', flush=True)
        print('Done with shapes.\n')

        if plot_bdry_nodes:
            print('Plotting boundary nodes of particles')
            for sh in range(len(self.particles)):
                count = len(self.particles[sh])
                for i in range(count):
                    # Plot the mesh
                    Pos = self.particles[sh][i].pos
                    bd = self.particles[sh][i].nonlocal_bdry_nodes
                    plt.scatter(Pos[bd,0], Pos[bd,1], s = dotsize*2, marker = '.', linewidth = 0, cmap='viridis')

        if plot_bdry_edges:
            print('Plotting boundary edges of particles')
            for sh in range(len(self.particles)):
                count = len(self.particles[sh])
                for i in range(count):
                    # Plot the mesh
                    Pos = self.particles[sh][i].pos
                    bde = self.particles[sh][i].bdry_edges
                    plt.scatter(Pos[bde,0], Pos[bde,1], s = dotsize*2, marker = '.', linewidth = 0, cmap='viridis')
                    X12 = Pos[:,0][bde]
                    Y12 = Pos[:,1][bde]
                    plt.plot(X12, Y12, 'k-', linewidth = 0.3 )

        
        if plot_bonds:
            print('Plotting bonds of particles')
            for sh in range(len(self.particles)):
                count = len(self.particles[sh])
                for i in range(count):
                    Part = self.particles[sh][i]
                    V1 = Part.NArr[:,0]
                    V2 = Part.NArr[:,1]

                    X12 = [Part.pos[V1,0], Part.pos[V2,0]]
                    Y12 = [Part.pos[V1,1], Part.pos[V2,1]]
                    plt.plot(X12, Y12, 'b-', linewidth = 0.3 )
        
        if plot_delta:
            print('Plotting peridynamic horizon delta')
            sh = 0
            i = 0
            node = 0

            px = self.particles[sh][i].pos[node][0]
            py = self.particles[sh][i].pos[node][1]

            delta = self.particles[sh][i].material.delta

            t = np.linspace(0, 2*np.pi, num = 50)
            plt.plot( px+ delta * np.cos(t), py + delta * np.sin(t), linewidth = 0.5)

        if plot_contact_rad:
            print('Plotting contact radius')
            sh = 0
            i = 0
            node = 0

            px = self.particles[sh][i].pos[node][0]
            py = self.particles[sh][i].pos[node][1]

            delta = self.particles[sh][i].material.delta

            t = np.linspace(0, 2*np.pi, num = 50)
            plt.plot( px+ self.contact.contact_radius * np.cos(t), py + self.contact.contact_radius * np.sin(t), linewidth = 0.5)

        if plot_wall:
            if self.wall.allow == 1:
                # P = np.array([
                    # [wall_left, wall_bottom],
                    # [wall_right, wall_bottom],
                    # [wall_right, wall_top],
                    # [wall_left, wall_top]
                    # ])
                x01 = np.array([
                    [self.wall.left, self.wall.right],
                    [self.wall.right, self.wall.right],
                    [self.wall.right, self.wall.left],
                    [self.wall.left, self.wall.left],
                    ])
                y01 = np.array([
                    [self.wall.bottom, self.wall.bottom],
                    [self.wall.bottom, self.wall.top],
                    [self.wall.top, self.wall.top],
                    [self.wall.top, self.wall.bottom]
                    ])
                plt.plot(x01, y01, 'b-')


        plt.axis('scaled')

        # save the image
        plt.savefig('setup.png', dpi=300, bbox_inches='tight')

        plt.show()


class IncenterNodes(object):
    """Incenter and radius of incircles"""
    def __init__(self, incenter, incircle_rad, mesh):
        self.incenter = incenter
        self.incircle_rad = incircle_rad
        self.mesh = mesh

    def dim(self):
        # return len(self.mesh.pos[0])
        return len(self.incenter[0])

    def count(self):
        return len(self.incircle_rad)

    def range(self):
        """min and max of the radius
        """
        return [np.amin(self.incircle_rad), np.amax(self.incircle_rad)]

    def mean(self):
        """ mean 
        """
        return np.mean(self.incircle_rad)

    def std(self):
        """standard deviation
        """
        return np.std(self.incircle_rad)

    def total_volume(self):
        """Total volume of all the bounding polygon
        """
        return np.sum(self.mesh.vol)

    def occupied_volume(self):
        """Total volume of all the circles combined
        :returns: TODO

        """
        dimension = self.dim()
        if (dimension ==2):
            return np.sum(np.pi * self.incircle_rad**2)
        else:
            return np.sum(4 * np.pi/3 * self.incircle_rad**3)

    def packing_ratio(self):
        """Ratio of occupied_volume to total polygon volume
        """
        return self.occupied_volume() / self.total_volume()

    def info(self):
        """Prints info about the arrangement
        """
        print('Count: ', self.count())
        print('Range: ', self.range())
        print('Mean: ', self.mean())
        print('Standard deviation: ', self.std())
        print('Packing ratio: ', self.packing_ratio())


        


    ## Delete beyond min and max or create two methods for min and max
    def trim(self, min_rad = None, max_rad = None):
        """Drop elements that are beyond min and max radius
        :min_rad: TODO
        :max_rad: TODO
        """

        # Todo: merge two operations into a single for loop
        if min_rad is not None:
            del_list = []
            for i in range(len(self.incircle_rad)):
                if ( self.incircle_rad[i] < min_rad ):
                    del_list.append(i)
            self.incenter = np.delete(self.incenter, del_list, 0)
            self.incircle_rad = np.delete(self.incircle_rad, del_list, 0)

        if max_rad is not None:
            del_list = []
            for i in range(len(self.incircle_rad)):
                if ( self.incircle_rad[i] > max_rad ):
                    del_list.append(i)
            self.incenter = np.delete(self.incenter, del_list, 0)
            self.incircle_rad = np.delete(self.incircle_rad, del_list, 0)

    def plot(self, do_plot = True, plot_edge = True, plot_mesh = True, remove_axes = True, save_file = None, plot_circles=True):
        """TODO: Docstring for plot.
        :line: plot lines
        """

        # compute the dimension
        dimension = self.dim()

        if plot_circles:
            if (dimension==2):
                for i in range(len(self.incircle_rad)):
                    t = np.linspace(0, 2*np.pi, num = 50)
                    plt.plot( self.incenter[i,0] + self.incircle_rad[i] * np.cos(t), self.incenter[i,1] + self.incircle_rad[i] * np.sin(t), 'k-', linewidth = 0.5)
            else:
                pass


        if plot_mesh:
            if (dimension ==2):
                edges = self.mesh.get_edges()
                V1 = edges[:,0]
                V2 = edges[:,1]

                P1 = self.mesh.pos[V1]
                P2 = self.mesh.pos[V2]
                ls =  [ [p1, p2] for p1, p2 in zip(P1,P2)] 

                lc = LineCollection(ls, linewidths=0.5, colors='b')
                plt.gca().add_collection(lc)
            else:
                pass

        if plot_edge:
            if (dimension == 2):
                P1 = self.mesh.pos[self.mesh.bdry_edges[:,0]]
                P2 = self.mesh.pos[self.mesh.bdry_edges[:,1]]
                ls =  [ [p1, p2] for p1, p2 in zip(P1,P2)] 

                lc = LineCollection(ls, linewidths=0.5, colors='r')
                plt.gca().add_collection(lc)
            else:
                pass

        if remove_axes:
            if (dimension == 2):
                plt.gca().set_axis_off()
            else:
                pass

        # adjust plot properties
        if (dimension ==2):
            # scale the axes
            plt.axis('scaled')
        else:
            pass
        

        if save_file:
            # save the image
            plt.savefig(save_file, dpi=300, bbox_inches='tight')

        if do_plot:
            if (dimension ==2):
                plt.show()
            else:
                print('Plotting 3D arrangement not implemented yet')

        
def get_unif_mesh_loc(x_range, y_range, num_x=1, num_y=1):
    """ Generate uniformly distributed arrangement
    """

    x_size =  (x_range[1] - x_range[0])/num_x
    y_size =  (y_range[1] - y_range[0])/num_y

    incircle_rad = np.array([0] * (num_x*num_y)) + min(x_size, y_size)/2

    incenter = []
    for i in range(num_y):
        for j in range(num_x):
            incenter.append( [x_range[0]+(j+1/2)*x_size, y_range[0]+(i+1/2)*y_size ])

    incenter = np.array(incenter)
    incircle_rad = np.array(incircle_rad)

    return IncenterNodes(incenter, incircle_rad, None)


def get_incenter_mesh_loc(P, meshsize, dimension = 2, msh_file = None, modify_nodal_circles = True, gen_bdry = True, min_rad = None, max_rad = None):
    """Given a polygon, generate particle locations and radii that fit within the polygon wall via the computation of the incenter

    :P: The vertices of a polygon, list
    :meshsize: 
    :min_rad: drop any radius below this
    :max_rad: drop any radius above this
    :returns: a class that contains the location of the centers, and the radii
    """
    # generate mesh
    # mesh = genmesh(P, meshsize, msh_file = msh_file, do_plot = False, dimension = dimension)
    mesh = genmesh(P, meshsize, msh_file = msh_file, dimension = dimension)


    if (dimension ==2):
        # vertices
        v_a = mesh.pos[mesh.T[:,0]]
        v_b = mesh.pos[mesh.T[:,1]]
        v_c = mesh.pos[mesh.T[:,2]]

        ab = v_b - v_a
        ac = v_c - v_a
        bc = v_c - v_b

        # lengths of the sides
        l_a = np.sqrt(np.sum( (bc)**2, axis = 1))[:, None]
        l_b = np.sqrt(np.sum( (ac)**2, axis = 1))[:, None]
        l_c = np.sqrt(np.sum( (ab)**2, axis = 1))[:, None]

        sum_l = l_a + l_b + l_c

        # print('l_a :', l_a.shape)
        # print('v_a :', v_a.shape)
        
        # incenter
        incenter = (l_a * v_a + l_b * v_b + l_c * v_c) / sum_l

        # print('incenter :', incenter.shape)
        # time.sleep(5.5)


        # area of the triangles
        T_area = 0.25 * np.sqrt(sum_l * (l_a-l_b+l_c) * (l_b-l_c+l_a) * (l_c-l_a+l_b) )

        ## another approach for area of the triangles
        # u_x = mesh.pos[mesh.T[:,1], 0] - mesh.pos[mesh.T[:,0], 0]
        # u_y = mesh.pos[mesh.T[:,1], 1] - mesh.pos[mesh.T[:,0], 1]
        # v_x = mesh.pos[mesh.T[:,2], 0] - mesh.pos[mesh.T[:,0], 0]
        # v_y = mesh.pos[mesh.T[:,2], 1] - mesh.pos[mesh.T[:,0], 1]
        # cp = 0.5 * abs(u_x * v_y - u_y * v_x)
        # print(T_area - np.array([cp]).transpose())

        # incircle radius
        incircle_rad = 2 * T_area / sum_l
    else:
        # dim = 3

        print('Total tetrahedrons: ', len(mesh.T))

        x1 = mesh.pos[mesh.T[:, 0], 0]
        y1 = mesh.pos[mesh.T[:, 0], 1]
        z1 = mesh.pos[mesh.T[:, 0], 2]

        x2 = mesh.pos[mesh.T[:, 1], 0]
        y2 = mesh.pos[mesh.T[:, 1], 1]
        z2 = mesh.pos[mesh.T[:, 1], 2]

        x3 = mesh.pos[mesh.T[:, 2], 0]
        y3 = mesh.pos[mesh.T[:, 2], 1]
        z3 = mesh.pos[mesh.T[:, 2], 2]

        x4 = mesh.pos[mesh.T[:, 3], 0]
        y4 = mesh.pos[mesh.T[:, 3], 1]
        z4 = mesh.pos[mesh.T[:, 3], 2]

        T_vol = np.abs( 1/6*( (x4-x1) * ((y2-y1)*(z3-z1)-(z2-z1)*(y3-y1)) + (y4-y1) * ((z2-z1)*(x3-x1)-(x2-x1)*(z3-z1)) + (z4-z1) * ((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1)) ) )

        alpha = (T_vol * 6)[:, None]

        # print('alpha shape: ', alpha.shape)

        # copied from sage
        denom = np.sqrt(((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))**2 + ((x1 - x3)*(z1 - z2) - (x1 - x2)*(z1 - z3))**2 + ((y1 - y3)*(z1 - z2) - (y1 - y2)*(z1 - z3))**2) + np.sqrt(((x1 - x4)*(y1 - y2) - (x1 - x2)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z2) - (x1 - x2)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z2) - (y1 - y2)*(z1 - z4))**2) + np.sqrt(((x1 - x4)*(y1 - y3) - (x1 - x3)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z3) - (x1 - x3)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z3) - (y1 - y3)*(z1 - z4))**2) + np.sqrt(((x2 - x4)*(y2 - y3) - (x2 - x3)*(y2 - y4))**2 + ((x2 - x4)*(z2 - z3) - (x2 - x3)*(z2 - z4))**2 + ((y2 - y4)*(z2 - z3) - (y2 - y3)*(z2 - z4))**2)
        
        # convert to (n,1) array from (n,)
        denom  = denom[:,None]

        # print('denom shape: ', denom.shape)

        numer = np.zeros((len(denom), 3))

        # print('numer shape: ', numer.shape)

        numer[:,0] = np.sqrt(((x2 - x4)*(y2 - y3) - (x2 - x3)*(y2 - y4))**2 + ((x2 - x4)*(z2 - z3) - (x2 - x3)*(z2 - z4))**2 + ((y2 - y4)*(z2 - z3) - (y2 - y3)*(z2 - z4))**2)*x1 + np.sqrt(((x1 - x4)*(y1 - y3) - (x1 - x3)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z3) - (x1 - x3)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z3) - (y1 - y3)*(z1 - z4))**2)*x2 + np.sqrt(((x1 - x4)*(y1 - y2) - (x1 - x2)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z2) - (x1 - x2)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z2) - (y1 - y2)*(z1 - z4))**2)*x3 + np.sqrt(((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))**2 + ((x1 - x3)*(z1 - z2) - (x1 - x2)*(z1 - z3))**2 + ((y1 - y3)*(z1 - z2) - (y1 - y2)*(z1 - z3))**2)*x4

        numer[:,1] = np.sqrt(((x2 - x4)*(y2 - y3) - (x2 - x3)*(y2 - y4))**2 + ((x2 - x4)*(z2 - z3) - (x2 - x3)*(z2 - z4))**2 + ((y2 - y4)*(z2 - z3) - (y2 - y3)*(z2 - z4))**2)*y1 + np.sqrt(((x1 - x4)*(y1 - y3) - (x1 - x3)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z3) - (x1 - x3)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z3) - (y1 - y3)*(z1 - z4))**2)*y2 + np.sqrt(((x1 - x4)*(y1 - y2) - (x1 - x2)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z2) - (x1 - x2)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z2) - (y1 - y2)*(z1 - z4))**2)*y3 + np.sqrt(((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))**2 + ((x1 - x3)*(z1 - z2) - (x1 - x2)*(z1 - z3))**2 + ((y1 - y3)*(z1 - z2) - (y1 - y2)*(z1 - z3))**2)*y4

        numer[:,2] = np.sqrt(((x2 - x4)*(y2 - y3) - (x2 - x3)*(y2 - y4))**2 + ((x2 - x4)*(z2 - z3) - (x2 - x3)*(z2 - z4))**2 + ((y2 - y4)*(z2 - z3) - (y2 - y3)*(z2 - z4))**2)*z1 + np.sqrt(((x1 - x4)*(y1 - y3) - (x1 - x3)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z3) - (x1 - x3)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z3) - (y1 - y3)*(z1 - z4))**2)*z2 + np.sqrt(((x1 - x4)*(y1 - y2) - (x1 - x2)*(y1 - y4))**2 + ((x1 - x4)*(z1 - z2) - (x1 - x2)*(z1 - z4))**2 + ((y1 - y4)*(z1 - z2) - (y1 - y2)*(z1 - z4))**2)*z3 + np.sqrt(((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))**2 + ((x1 - x3)*(z1 - z2) - (x1 - x2)*(z1 - z3))**2 + ((y1 - y3)*(z1 - z2) - (y1 - y2)*(z1 - z3))**2)*z4

        incenter = numer / denom
        incircle_rad = alpha / denom


    ## Add entries for nodes, exclude the boundary nodes
    # nodal_rads = []
    # nodal_centers = []
    # neighboring triangles of each node
    nbd_node_tri = get_nbd_vertex_elem(mesh.T, len(mesh.pos))

#######################################################################
# Parallel attempt

#######################################################################
    
    if modify_nodal_circles:
        m_arr = []
        for i in range(len(mesh.pos)):
            nbd_elems = nbd_node_tri[i]
            max_fitting_rad = np.amin(np.sqrt(np.sum( (incenter[nbd_elems] - mesh.pos[i])**2, axis = 1)) - incircle_rad[nbd_elems].transpose())
            if i not in set(mesh.bdry_nodes):
                # gekko stuff
                m = GEKKO(remote=False)
                m.options.SOLVER = 1
                m.options.DIAGLEVEL = 0
                ## solving maximin problem
                x1,x2,Z = m.Array(m.Var,3)
                # x1, x2, Z = [m.Var() for i in range(3)]
                x1.value = mesh.pos[i][0]
                x2.value = mesh.pos[i][1]
                m.Maximize(Z)
                # m.Equation(x1+x2+x3==15)
                m.Equation( x1 >= np.min( incenter[nbd_elems][:,0] ) )
                m.Equation( x1 <= np.max( incenter[nbd_elems][:,0] ) )
                m.Equation( x2 >= np.min( incenter[nbd_elems][:,1] ) )
                m.Equation( x2 <= np.max( incenter[nbd_elems][:,1] ) )
                for k in range(len(nbd_elems)):
                    nbd_id = nbd_elems[k]
                    rk = incircle_rad[nbd_id][0]
                    pk1 = incenter[nbd_id][0]
                    pk2 = incenter[nbd_id][1]
                    m.Equation( Z <= ( (x1-pk1)**2.0 + (x2-pk2)**2.0 )**0.5 - rk )
                    m.Equation( ( (x1-pk1)**2.0 + (x2-pk2)**2.0 )**0.5 >= rk )
                # m.solve()
                m_arr.append(m)

                ## add to the list
                # nodal_centers.append(np.array([x1.value[0], x2.value[0]]))
                # nodal_rads.append(Z.value[0])

                ## add to the list: the Original node
                # nodal_rads.append(max_fitting_rad)
                # nodal_centers.append(mesh.pos[i])
            else:
                if gen_bdry:
                    # gekko stuff
                    m = GEKKO(remote=False)
                    m.options.SOLVER = 1
                    m.options.DIAGLEVEL = 0
                    ## solving maximin problem
                    x1,x2,Z = m.Array(m.Var,3)
                    m.Maximize(Z)
                    for k in range(len(nbd_elems)):
                        nbd_id = nbd_elems[k]
                        rk = incircle_rad[nbd_id][0]
                        pk1 = incenter[nbd_id][0]
                        pk2 = incenter[nbd_id][1]
                        m.Equation( Z <= ( (x1-pk1)**2.0 + (x2-pk2)**2.0 )**0.5 - rk )
                        m.Equation( ( (x1-pk1)**2.0 + (x2-pk2)**2.0 )**0.5 >= rk )

                    ## get boundary edges associated with the boundary nodes
                    for e in range(len(mesh.bdry_edges)):
                        edge = mesh.bdry_edges[e]
                        if i in set(edge):
                            e2 = mesh.pos[edge[1]]
                            e1 = mesh.pos[edge[0]]

                            u = e2 - e1
                            u_norm = (u[0]**2 + u[1]**2)**0.5

                            # cross_prod = np.abs( u[0]*(x2 - e2) - u[1]*(x1 - e1) )
                            # normal distance to the edge: h = |uxv|/|u|, u=e2-e1, v=x-e1
                            m.Equation( Z <= np.abs( u[0]*(x2 - e1[1]) - u[1]*(x1 - e1[0]) ) / u_norm)

                    # initial guess, the centroid
                    all_x1 = np.append(incenter[nbd_elems][:,0], mesh.pos[i][0])
                    all_x2 = np.append(incenter[nbd_elems][:,1], mesh.pos[i][1])

                    x1.value = np.mean(all_x1)
                    x2.value = np.mean(all_x2)

                    # generic bounds
                    m.Equation( x1 >= np.min(all_x1) )
                    m.Equation( x1 <= np.max(all_x1) )
                    m.Equation( x2 >= np.min(all_x2) )
                    m.Equation( x2 <= np.max(all_x2) )
                    # m.solve()
                    m_arr.append( m )

                    ## add to the list
                    # nodal_centers.append(np.array([x1.value[0], x2.value[0]]))
                    # nodal_rads.append(Z.value[0])


        # print(m_arr)
        start = time.time()

        ## Solve all at once in parallel
        a_pool = Pool()
        # cl_solved_arr = a_pool.map(solve_par, m_arr)
        print('Modifying nodal circles in parallel.')
        cl_solved_arr = np.array(a_pool.map(solve_par, m_arr))
        print('time taken ', time.time() - start)

        nodal_centers = cl_solved_arr[:,0:2]
        nodal_rads = cl_solved_arr[:,2]

        ## Add to the main array
        incircle_rad = np.append(incircle_rad, np.array([nodal_rads]).transpose(), axis = 0)
        incenter = np.append(incenter, nodal_centers, axis = 0)

    else:   # if there is no modification to the nodal circles
        nodal_rads = []
        nodal_centers = []
        for i in range(len(mesh.pos)):
            if i not in set(mesh.bdry_nodes):
                nbd_elems = nbd_node_tri[i]
                max_fitting_rad = np.amin(np.sqrt(np.sum( (incenter[nbd_elems] - mesh.pos[i])**2, axis = 1)) - incircle_rad[nbd_elems].transpose())

                # add to the list: the Original node
                nodal_rads.append(max_fitting_rad)
                nodal_centers.append(mesh.pos[i])

        print('Adding for each node: ', len(nodal_rads))

        # append to the main array
        incircle_rad = np.append(incircle_rad, np.array([nodal_rads]).transpose(), axis = 0)
        incenter = np.append(incenter, nodal_centers, axis = 0)


    return IncenterNodes(incenter, incircle_rad, mesh)


def solve_par(m):
    """ Solving gekko optimization problem (to be called in parallel)
    """
    m.solve(disp=False)
    # print(m._variables)
    ## outputs a row vector, rank 1
    solved_var = np.array(m._variables).transpose()[0]
    # print(solved_var)
    return solved_var
    
    # nodal_centers.append(np.array([solved_var[0][0], cl_solved_arr[i][1][0]]))
    # nodal_rads.append(cl_solved_arr[i][2][0])
    # return m._variables
    # return np.array([m_arr[i].x1.value[0], m_arr[i].x2.value[0], m_arr[i].Z.value[0]])

#######################################################################

def own_weight_float_force():
    """ Particle to collapse under its own weight
    """

    delta = 1e-3
    meshsize = 1e-3/3
    contact_radius = 1e-3/3;

    SL = ShapeList()

    # SL.append(shape=shape_dict.pacman(), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
    # SL.append(shape=shape_dict.small_disk(), count=1, meshsize=meshsize, material=material_dict.peridem(delta))

    bulk_l = 7e-3
    # clamp_l = 2e-3
    clamp_l = 4e-3

    bulk_s = 3e-3
    clamp_s = 0.5e-3/2

    # SL.append(shape=shape_dict.plank(l=bulk_l, s=bulk_s), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, Gnot_scale=0.5))
    SL.append(shape=shape_dict.plank_wedge(l=bulk_l, s=bulk_s, w_loc_ratio=0.5, w_thickness_ratio=0.01, w_depth_ratio=1), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, Gnot_scale=0.5))

    SL.append(shape=shape_dict.plank(l=clamp_l, s=clamp_s), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
    # SL.append(shape=shape_dict.plank(l=clamp_l, s=clamp_s), count=3, meshsize=meshsize, material=material_dict.peridem(delta))

    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)
    # particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=True, plot_shape=False)

    # apply transformation
    # particles[0][0].rotate(-np.pi/2)
    # particles[1][0].shift([0, -3e-3])

    particles[1][0].shift([-(bulk_l - clamp_l), +(bulk_s+clamp_s)+contact_radius/2 ])
    particles[1][1].shift([-(bulk_l - clamp_l), -(bulk_s+clamp_s)-contact_radius/2 ])
    
    # one that pushes
    # particles[1][2].shift([-(bulk_l + clamp_l), 0 ])


    # Initial data
    # particles[0][0].vel += [0, -20]
    # particles[0][0].vel += [0, -2]
    particles[0][0].acc += [0, -5e4]
    particles[0][0].extforce += [0, -5e4 * particles[0][0].material.rho]

    # force in the x-direction to simulate flow
    # particles[0][0].acc += [5e-4, 0]
    # particles[0][0].extforce += [5e4 * particles[0][0].material.rho, 0]


    # particles[0][0].acc += [0, -5e4]
    # particles[0][0].extforce += [0, -5e4 * particles[0][0].material.rho]

    # plank does not move
    particles[1][0].movable = 0
    particles[1][1].movable = 0

    # one that pushes
    # particles[1][2].acc += [5e4, 0]
    # particles[1][2].extforce += [5e4 * particles[0][0].material.rho, 0]


    # wall info
    wall_left   = -25e-3
    wall_right  = 25e-3
    wall_top    = 15e-3
    wall_bottom = -15e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)


    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9


    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)


    return Experiment(particles, wall, contact)

def own_weight():
    """ Particle to collapse under its own weight
    """

    delta = 1e-3
    meshsize = 1e-3/2
    contact_radius = 1e-3/3;

    SL = ShapeList()

    # SL.append(shape=shape_dict.pacman(), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
    # SL.append(shape=shape_dict.small_disk(), count=1, meshsize=meshsize, material=material_dict.peridem(delta))

    bulk_l = 7e-3
    # clamp_l = 2e-3
    clamp_l = 6e-3

    bulk_s = 5e-3
    clamp_s = 0.5e-3/2

    # SL.append(shape=shape_dict.plank(l=bulk_l, s=bulk_s), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, Gnot_scale=0.5))
    SL.append(shape=shape_dict.plank_wedge(l=bulk_l, s=bulk_s, w_loc_ratio=0.5, w_thickness_ratio=0.02, w_depth_ratio=1.5), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    # SL.append(shape=shape_dict.ice_wedge(l=bulk_l, s=bulk_s, extra_ratio=1.2, w_loc_ratio=0.1, w_thickness_ratio=0.02, w_depth_ratio=1.5), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, Gnot_scale=0.5, rho_scale=1))


    # SL.append(shape=shape_dict.plank(l=clamp_l, s=clamp_s), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
    # SL.append(shape=shape_dict.plank(l=clamp_l, s=clamp_s), count=3, meshsize=meshsize, material=material_dict.peridem(delta))

    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=True)
    # particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=True, plot_shape=False)

    # apply transformation
    # particles[0][0].rotate(-np.pi/2)
    # particles[1][0].shift([0, -3e-3])

    # particles[1][0].shift([-(bulk_l - clamp_l), +(bulk_s+clamp_s)+contact_radius/2 ])
    # particles[1][1].shift([-(bulk_l - clamp_l), -(bulk_s+clamp_s)-contact_radius/2 ])

    
    # one that pushes
    # particles[1][2].shift([-(bulk_l + clamp_l), 0 ])


    # Initial data
    # particles[0][0].vel += [0, -20]
    # particles[0][0].vel += [0, -2]

    particles[0][0].acc += [0, -5e4]
    particles[0][0].extforce += [0, -5e4 * particles[0][0].material.rho]

    # force in the x-direction to simulate flow
    part = particles[0][0]
    for i in range(len(part.pos)):
        right_edge = bulk_l
        if np.abs(part.pos[i][0] - right_edge) < delta :
            acc_val = 5e4
            # acc_val = 5e5
            # height-dependent force, plus if above, minus if below
            # acc_val = part.pos[i][1]/bulk_s * 5e4

            part.acc[i] += [acc_val, 0]
            part.extforce[i] += [acc_val * part.material.rho, 0]

        # clamped nodes
        left_edge = -bulk_l
        if np.abs(part.pos[i][0] - left_edge) < delta :
            part.clamped_nodes.append(i)






    # plank does not move
    # particles[1][0].movable = 0
    # particles[1][1].movable = 0



    # wall info
    wall_left   = -25e-3
    wall_right  = 25e-3
    wall_top    = 15e-3
    wall_bottom = -15e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)


    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9


    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)


    return Experiment(particles, wall, contact)

def flow():
    """ elastic bar for stress strain curve
    """

    delta = 1e-3
    meshsize = 1e-3/2
    contact_radius = 1e-3/3;

    SL = ShapeList()

    # l= 20e-3
    l= 10e-3
    s = 5e-3

    # SL.append(shape=shape_dict.line_1d(), count=1, meshsize=meshsize, material=material_dict.peridem_1d_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    # SL.append(shape=shape_dict.tie(l=10e-3, s=0.25e-3), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    SL.append(shape=shape_dict.plank(l=l, s=s), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=50))

    SL.material_list[0].print()

    # particles = SL.generate_mesh(dimension = 1, contact_radius = contact_radius, plot_mesh=True, plot_shape=True)
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)

    # apply transformation
    # particles[0][0].rotate(-np.pi/2)
    # particles[1][0].shift([0, -3e-3])

    # particles[0][1].shift([0, -1e-3])

    # Initial data
    # particles[0][0].vel += [0, -20]
    # particles[0][0].vel += [0, -2]


    # force in the horizontal direction
    F_val = 5e4
    particles[0][0].acc += [F_val, 0]
    particles[0][0].extforce += [F_val * particles[0][0].material.rho, 0]

    # clamped nodes
    part = particles[0][0]

    # for i in range(len(part.pos)):
        # # clamp the top and bottom
        # if np.abs(part.pos[i][1] - (-s))< 1e-5:
            # part.clamped_nodes.append(i)
        # if np.abs(part.pos[i][1] - (s))< 1e-5:
            # part.clamped_nodes.append(i)
        # # apply force on the right
        # if np.abs(part.pos[i][0] - (l))< 1e-5:
            # part.extforce[i] += [-5e6 * part.material.rho, 0]

    # particles[0][0].vol[0] *= 2
    # particles[0][0].clamped_nodes.append(3)
    # particles[0][0].clamped_nodes.append(81)


    # wall info
    wall_left   = -2*l
    wall_right  = 2*l
    wall_top    = s 
    wall_bottom = -s
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9

    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    return Experiment(particles, wall, contact)

def bar():
    """ elastic bar for stress strain curve
    """

    delta = 1e-3
    meshsize = 1e-3/2
    contact_radius = 1e-3/3;

    SL = ShapeList()

    # l= 20e-3
    l= 10e-3
    s = 2e-3

    # SL.append(shape=shape_dict.line_1d(), count=1, meshsize=meshsize, material=material_dict.peridem_1d_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    # SL.append(shape=shape_dict.tie(l=10e-3, s=0.25e-3), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    # SL.append(shape=shape_dict.plank(l=l, s=s), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, G_scale=1, K_scale=0.5, Gnot_scale=0.1, rho_scale=50))
    SL.append(shape=shape_dict.plank(l=l, s=s), count=1, meshsize=meshsize, material=material_dict.sodalime_similar_to_peridem(delta, E_scale=0.5, Gnot_scale=0.1, rho_scale=1))

    SL.material_list[0].print()

    # particles = SL.generate_mesh(dimension = 1, contact_radius = contact_radius, plot_mesh=True, plot_shape=True)
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)

    # apply transformation
    # particles[0][0].rotate(-np.pi/2)
    # particles[1][0].shift([0, -3e-3])

    # particles[0][1].shift([0, -1e-3])

    # Initial data
    # particles[0][0].vel += [0, -20]
    # particles[0][0].vel += [0, -2]

    # particles[0][0].acc += [0, -5e4]
    # particles[0][0].extforce += [0, -5e4 * particles[0][0].material.rho]

    # clamped nodes
    part = particles[0][0]

    # part.clamped_nodes.append(0)

    # F_acc = 5e6

    ## To observe the stress-strain curve, we specify Force and compute the force density and initial acceleration from it (as opposed to specifying acceleration)
    ## That way, for any rho, the elongation of the rod will be the same
    Force = 1e4

    # clamp the left and stretch the right
    for i in range(len(part.pos)):
        # clamp the left
        if np.abs(part.pos[i][0] - (-l))< meshsize/2:
            part.clamped_nodes.append(i)

        # apply force on the right
        # Use with (in config)
        ## gradient_extforce = 1
        ## extforce_maxstep = 20000

        if np.abs(part.pos[i][0] - (l))< meshsize/2:
            # part.acc[i] += [F_acc, 0]
            part.acc[i] += [Force/(part.vol[i][0] * part.material.rho), 0]
            # force density
            # part.extforce[i] += [F_acc * part.material.rho, 0]
            part.extforce[i] += [Force/part.vol[i][0], 0]

    # particles[0][0].vol[0] *= 2
    # particles[0][0].clamped_nodes.append(3)
    # particles[0][0].clamped_nodes.append(81)


    # wall info
    # wall_left   = -l - contact_radius*1.5
    # wall_right  = l + contact_radius*1.5
    wall_left   = -2*l
    wall_right  = 2*l
    wall_top    = 2*l
    wall_bottom = -2*l
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9

    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    return Experiment(particles, wall, contact)

def worm():
    """ Pendulum hanging
    """

    delta = 1e-3
    meshsize = 1e-3/2
    contact_radius = 1e-3/3;

    SL = ShapeList()

    # l= 20e-3
    l= 10e-3
    s = 0.25e-3

    # SL.append(shape=shape_dict.line_1d(), count=1, meshsize=meshsize, material=material_dict.peridem_1d_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    # SL.append(shape=shape_dict.tie(l=10e-3, s=0.25e-3), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    SL.append(shape=shape_dict.plank(l=l, s=s), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=50))

    # particles = SL.generate_mesh(dimension = 1, contact_radius = contact_radius, plot_mesh=True, plot_shape=True)
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)

    # apply transformation
    # particles[0][0].rotate(-np.pi/2)
    # particles[1][0].shift([0, -3e-3])

    # particles[0][1].shift([0, -1e-3])

    # Initial data
    # particles[0][0].vel += [0, -20]
    # particles[0][0].vel += [0, -2]

    particles[0][0].acc += [0, -5e4]
    particles[0][0].extforce += [0, -5e4 * particles[0][0].material.rho]

    # clamped nodes
    part = particles[0][0]

    # part.clamped_nodes.append(0)

    for i in range(len(part.pos)):
        if np.abs(part.pos[i][0] - 10e-3)< 1e-3:
            part.extforce[i] += [-5e5 * part.material.rho, 0]

        # if np.abs(part.pos[i][0] + 10e-3)< 1e-3:
            # part.extforce[i] += [+5e6 * part.material.rho, 0]
    # particles[0][0].vol[0] *= 2
    # particles[0][0].clamped_nodes.append(3)
    # particles[0][0].clamped_nodes.append(81)


    # wall info
    wall_left   = -50e-3
    wall_right  = 50e-3
    # wall_top    = 50e-3
    # wall_bottom = -50e-3
    wall_top    = 10e-3
    wall_bottom = -0e-3 - contact_radius*2
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9

    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    return Experiment(particles, wall, contact)

def pendulum():
    """ Pendulum hanging
    """

    delta = 1e-3
    meshsize = 1e-3/2
    contact_radius = 1e-3/3;

    SL = ShapeList()


    # SL.append(shape=shape_dict.line_1d(), count=1, meshsize=meshsize, material=material_dict.peridem_1d_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    # SL.append(shape=shape_dict.tie(l=10e-3, s=0.25e-3), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))
    SL.append(shape=shape_dict.plank(l=10e-3, s=0.25e-3), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, G_scale=0.5, K_scale=0.5, Gnot_scale=0.1, rho_scale=1))

    # particles = SL.generate_mesh(dimension = 1, contact_radius = contact_radius, plot_mesh=True, plot_shape=True)
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=True, plot_shape=False, plot_node_text=True, shapes_in_parallel=False)

    # apply transformation
    # particles[0][0].rotate(-np.pi/2)
    # particles[1][0].shift([0, -3e-3])

    # particles[0][1].shift([0, -1e-3])

    # Initial data
    # particles[0][0].vel += [0, -20]
    # particles[0][0].vel += [0, -2]

    particles[0][0].acc += [0, -5e4]
    particles[0][0].extforce += [0, -5e4 * particles[0][0].material.rho]

    # clamped nodes
    particles[0][0].clamped_nodes.append(0)
    # particles[0][0].clamped_nodes.append(3)
    # particles[0][0].vol[0] *= 2
    # particles[0][0].clamped_nodes.append(3)
    # particles[0][0].clamped_nodes.append(81)

    # wall info
    wall_left   = -40e-3
    wall_right  = 20e-3
    wall_top    = 10e-3
    wall_bottom = -25e-3
    # wall_top    = 5e-3
    # wall_bottom = -1e-3 - contact_radius
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9

    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    return Experiment(particles, wall, contact)

def particle_plank():
    """ Two particles colliding 
    """

    delta = 1e-3
    meshsize = 1e-3/5
    contact_radius = 1e-3/3;

    SL = ShapeList()

    # SL.append(shape=shape_dict.pacman(), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
    SL.append(shape=shape_dict.small_disk(), count=1, meshsize=meshsize, material=material_dict.peridem(delta))
    SL.append(shape=shape_dict.plank(l=2e-3, s=0.5e-3), count=1, meshsize=meshsize, material=material_dict.peridem(delta))

    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)
    # particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=True, plot_shape=False)

    # apply transformation
    # particles[0][0].rotate(-np.pi/2)
    particles[1][0].shift([0, -3e-3])

    # Initial data
    # particles[0][0].vel += [0, -20]
    # particles[0][0].vel += [0, -2]
    particles[0][0].acc += [0, -5e4]
    particles[0][0].extforce += [0, -5e4 * particles[0][0].material.rho]

    # plank does not move
    particles[1][0].movable = 0

    # wall info
    wall_left   = -2.5e-3
    wall_right  = 2.5e-3
    wall_top    = 2e-3
    wall_bottom = -5e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)


    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9


    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)


    return Experiment(particles, wall, contact)

def particle_confined():
    """ Two particles colliding 
    """

    delta = 1e-3
    meshsize = 1e-3/8
    contact_radius = 1e-3/3;

    SL = ShapeList()

    SL.append(shape=shape_dict.small_disk(steps=20), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, K_scale=0.6, G_scale=0.6, Gnot_scale=0.05))
    # SL.append(shape=shape_dict.plank(l=1e-3,s=1e-3), count=1, meshsize=meshsize, material=material_dict.peridem(delta))
    # SL.append(shape=shape_dict.pacman(), count=2, meshsize=meshsize, material=material_dict.peridem(delta))

    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=True, plot_shape=False)

    # apply transformation
    # particles[0][0].rotate(np.pi/2)
    # particles[0][0].shift([0, 3e-3])

    # Initial data
    # particles[0][0].vel += [0, -20]
    # particles[0][0].vel += [0, -2]
    # particles[0][0].acc += [0, -5e4]
    # particles[0][0].extforce += [0, -5e4 * particles[0][0].material.rho]

    # wall info
    wall_left   = -1.4e-3
    wall_right  = 1.4e-3
    wall_top    = 1.4e-3
    wall_bottom = -1.4e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)


    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9


    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)


    return Experiment(particles, wall, contact)

def wall_particle():
    """ particles colliding with the wall
    """

    delta = 1e-3
    meshsize = 1e-3/5
    contact_radius = 1e-3/3;

    SL = ShapeList()

    SL.append(shape=shape_dict.small_disk(steps=20), count=1, meshsize=meshsize, material=material_dict.peridem(delta))
    # SL.append(shape=shape_dict.polygon_inscribed(sides=6), count=1, meshsize=meshsize, material=material_dict.peridem(delta))
    # SL.append(shape=shape_dict.pacman(), count=2, meshsize=meshsize, material=material_dict.peridem(delta))

    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=True, plot_shape=False)

    # apply transformation
    particles[0][0].rotate(np.pi/2)
    # particles[0][0].shift([0, 3e-3])

    # Initial data
    # particles[0][0].vel += [0, -20]
    # particles[0][0].vel += [0, -2]
    particles[0][0].acc += [0, -5e4]
    particles[0][0].extforce += [0, -5e4 * particles[0][0].material.rho]

    # wall info
    wall_left   = -4e-3
    wall_right  = 4e-3
    wall_top    = 5e-3
    wall_bottom = -2e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)


    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.9

    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)


    return Experiment(particles, wall, contact)

def collision():
    """ Two particles colliding 
    """

    delta = 1e-3
    meshsize = 1e-3/8
    contact_radius = 1e-3/3;

    ## shape: 0
    # shapes.append(shape_dict.small_disk())
    # shapes.append(shape_dict.plus())

    SL = ShapeList()

    # SL.append(shape=shape_dict.test(), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
    SL.append(shape=shape_dict.perturbed_disk(), count=2, meshsize=meshsize, material=material_dict.peridem(delta))

    # SL.append(shape=shape_dict.small_disk(), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
    # SL.append(shape=shape_dict.ring_segment(scaling=1.5e-3, steps=3, angle=np.pi, inner_rad=0.75), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
    # SL.append(shape=shape_dict.pacman(), count=2, meshsize=meshsize, material=material_dict.peridem(delta))

    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=True, plot_shape=True)

    # apply transformation
    particles[0][0].rotate(np.pi/2)
    particles[0][1].rotate(-np.pi/2)
    particles[0][1].shift([0, 3e-3])

    ## For ring_segment
    # particles[0][0].rotate(-np.pi/2)
    # particles[0][1].rotate(np.pi/2)
    # particles[0][1].shift([1.1e-3, 3e-3])

    # Initial data
    particles[0][1].vel += [0, -20]
    # particles[0][1].acc += [0, -5e4]
    # particles[0][1].extforce += [0, -5e4 * particles[0][1].material.rho]

    particles[0][0].stoppable =0

    # wall info
    wall_left   = -6e-3
    wall_right  = 6e-3
    wall_top    = 6e-3
    wall_bottom = -6e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)


    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.8


    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)


    return Experiment(particles, wall, contact)

def collision_pacman():
    """ Two particles colliding 
    """

    delta = 1e-3
    meshsize = 1e-3/6
    contact_radius = 1e-3/3

    pacman_angle = np.pi/2
    ## shape: 0
    # shapes.append(shape_dict.small_disk())
    # shapes.append(shape_dict.plus())

    SL = ShapeList()

    # SL.append(shape=shape_dict.small_disk(), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
    # SL.append(shape=shape_dict.pacman(angle = pacman_angle), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
    # SL.append(shape=shape_dict.plus(), count=2, meshsize=meshsize, material=material_dict.peridem_softer_1(delta))
    # SL.append(shape=shape_dict.pacman(angle = pacman_angle), count=2, meshsize=meshsize, material=material_dict.peridem_softer_1(delta))
    # SL.append(shape=shape_dict.reverse_disk(), count=2, meshsize=meshsize, material=material_dict.peridem_softer_1(delta))
    SL.append(shape=shape_dict.wheel_annulus(scaling=1e-3, inner_circle_ratio=0.5,  meshsize=meshsize/2, nci_steps=20), count=2, meshsize=meshsize, material=material_dict.peridem_softer_1(delta))
    # SL.append(shape=shape_dict.box(), count=2, meshsize=meshsize, material=material_dict.peridem_softer_1(delta))
    # SL.append(shape=shape_dict.box_notch(), count=2, meshsize=meshsize, material=material_dict.peridem_softer_1(delta))
    # SL.append(shape=shape_dict.box_notch_2(), count=2, meshsize=meshsize, material=material_dict.peridem_softer_1(delta))
    # SL.append(shape=shape_dict.box_notch_3(), count=2, meshsize=meshsize, material=material_dict.peridem_softer_1(delta))
    # SL.append(shape=shape_dict.ring_segment(), count=2, meshsize=meshsize, material=material_dict.peridem(delta))

    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius)

    # apply transformation
    particles[0][0].rotate(-pacman_angle/2)
    particles[0][1].rotate(-pacman_angle/2 - 2*np.pi/2)
    particles[0][1].shift([1e-3+contact_radius*1.001, 3e-3])

    # Initial data
    particles[0][1].vel += [0, -25]
    # particles[0][1].acc += [0, -5e4]
    # particles[0][1].extforce += [0, -5e4 * particles[0][1].material.rho]

    # wall info
    wall_left   = -6e-3
    wall_right  = 6e-3
    wall_top    = 6e-3
    wall_bottom = -6e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.8


    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)


    return Experiment(particles, wall, contact)

def fixed_prenotch():
    """  particle with prenotch at rest to check if it self-explodes
    """

    delta = 1e-3
    meshsize = 1e-3/5
    contact_radius = 1e-3/3;

    ## shape: 0
    # shapes.append(shape_dict.small_disk())
    # shapes.append(shape_dict.plus())

    SL = ShapeList()

    SL.append(shape=shape_dict.box_prenotch(l=2e-3, s=2e-3, a=0.4e-3), count=1, meshsize=meshsize, material=material_dict.sodalime_similar_to_peridem(delta, Gnot_scale=0.4))

    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=True)


    # Initial data
    particles[0][0].vel += [0, -20]
    # particles[0][1].vel += [0, 20]

    # particles[0][1].acc += [0, -5e4]
    # particles[0][1].extforce += [0, -5e4 * particles[0][1].material.rho]

    # wall info
    wall_left   = -8e-3
    wall_right  = 8e-3
    wall_top    = 8e-3
    wall_bottom = -8e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)


    # contact properties
    # SL.append(shape=shape_dict.test(), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.9
    friction_coefficient = 0.8


    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)


    return Experiment(particles, wall, contact)
def crack_prenotch():
    """ crack formation in particle with prenotch
    """

    delta = 1e-3
    meshsize = 1e-3/5
    contact_radius = 1e-3/3

    SL = ShapeList()

    SL.append(shape=shape_dict.plank(l=2e-3, s=0.5e-3), count=2, meshsize=meshsize, material=material_dict.peridem(delta))
    SL.append(shape=shape_dict.box_prenotch(l=2e-3, s=2e-3, a=0.4e-3), count=1, meshsize=meshsize, material=material_dict.sodalime_similar_to_peridem(delta, Gnot_scale=0.1))

    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=True)

    # apply transformation
    # particles[0][0].rotate(np.pi/2)
    # particles[0][1].rotate(-np.pi/2)
    particles[0][0].shift([0, 3e-3])
    particles[0][1].shift([0, -3e-3])

    # Initial data
    particles[0][0].vel += [0, -20]
    particles[0][1].vel += [0, 20]

    particles[0][0].breakable = 0 
    particles[0][1].breakable = 0
    
    # particles[0][1].acc += [0, -5e4]
    # particles[0][1].extforce += [0, -5e4 * particles[0][1].material.rho]

    # wall info
    wall_left   = -8e-3
    wall_right  = 8e-3
    wall_top    = 8e-3
    wall_bottom = -8e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # contact properties
    # normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    ## Debug
    # normal_stiffness = particles[1][0].material.cnot/particles[1][0].material.delta
    # Latest
    normal_stiffness = particles[1][0].material.cnot/contact_radius
    damping_ratio = 0.9
    friction_coefficient = 0.8

    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    return Experiment(particles, wall, contact)

def hopper_2():

    """ Hopper flow
    Caution: parallelism crashes. Why??
    """

    delta = 1e-3
    meshsize = delta/3
    contact_radius = delta/5;

    # at least this much space to leave between particles. Will get halved while selecting max radius
    min_particle_spacing = contact_radius*1.001

    # l= 20e-3
    l= 10e-3
    s = 2e-3

    # wall info
    L = 2*l + s
    wall_left   = -L 
    wall_right  = L 
    wall_top    = L
    wall_bottom = -L
    # wall_bottom = -5e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # particle generation boundary
    c = min_particle_spacing/2
    P = np.array([
        # [wall_left + c, wall_bottom + c ],
        [wall_left + c, 0+s+ c ],
        [wall_right - c, 0+s + c],
        [wall_right - c, wall_top - c],
        [wall_left + c, wall_top - c]
        ])

    # particle generation boundary
    # P = np.array([
        # [wall_left, wall_bottom],
        # [wall_right, wall_bottom],
        # [wall_right, wall_top],
        # [wall_left, wall_top]
        # ])

    ## Generate particle arrangement
    ## particle generation spacing
    P_meshsize = 3.5e-3 
    msh = get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= True, gen_bdry = True )
    # msh = get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= False, gen_bdry = False )
    # msh = get_incenter_mesh_loc(P, P_meshsize, msh_file='meshdata/closepacked_mat.msh', modify_nodal_circles= True, gen_bdry = False )
    # msh = get_incenter_mesh_loc(P, P_meshsize, msh_file='meshdata/closepacked_mat.msh', modify_nodal_circles= False, gen_bdry = False )
    # print('total count: ', msh.count())
    # print('total vol: ', msh.total_volume())
    # print('occupied vol: ', msh.occupied_volume())
    # print('packing raio: ', msh.packing_ratio())
    # msh.info()
    # msh.trim(min_rad = 0.005 + min_particle_spacing/2, max_rad = 0.01 + min_particle_spacing/2)
    # msh.trim(min_rad = 0.8 * 1e-3 + min_particle_spacing/2)
    msh.info()
    msh.plot(plot_edge = True)
    # reduce radius to avoid contact
    msh.incircle_rad -= min_particle_spacing/2
    msh.info()
    msh.plot(plot_edge = True)
    # radius of base object is 1e-3
    # scaling_list = msh.incircle_rad/1e-3


    ## Uniform location and scaling
    # num_x = 4
    # num_y = 3
    # x_range = [wall_left + c, wall_right-c]
    # y_range = [0+s+c, wall_top-c]
    # msh = get_unif_mesh_loc(x_range, y_range, num_x=num_x, num_y=num_y)
    # msh.plot(plot_edge=False, plot_mesh=False)
    # for i in range(msh.count()):
        # if i%num_x >= num_x/2:
            # msh.incircle_rad[i]=2e-3
        # else:
            # msh.incircle_rad[i]=1e-3



    # Create a list of shapes
    SL = ShapeList()
    # shape = shape_dict.load('meshdata/peridem_mat.msh')
    # SL.append(shape=shape_dict.small_disk(steps = 4), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))
    # SL.append(shape=shape_dict.small_disk(), count=msh.count(), meshsize=meshsize, material=material_dict.peridem_deformable(delta, rho_scale=5e2))
    # SL.append(shape=shape_dict.plus_inscribed(notch_dist=0.2), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))
    
    # SL.append(shape=shape_dict.perturbed_disk(steps=16), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))

    # for i in range(msh.count()):
    for i in range(msh.count()):

        sc = msh.incircle_rad[i]
        # delta_sc = delta
        delta_sc = sc/2

        SL.append(shape=shape_dict.perturbed_disk(steps=16, scaling=sc), count=1, meshsize=meshsize, material=material_dict.sodalime_similar_to_peridem(delta_sc))

    # planks
    # SL.append(shape=shape_dict.plank(l=l, s=s), count=2, meshsize=meshsize, material=material_dict.sodalime_similar_to_peridem(delta))

    # generate the mesh for each shape
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False, shapes_in_parallel=True)

    ## Apply transformation
    seed(1)
    # for i in range(msh.count()):
    for i in range(msh.count()):
        # particles[0][i].trasnform_bymat(np.array([ [0.8, 0], [0, 1] ]))
        # particles[0][i].trasnform_bymat(np.array([ [0.5 + (random() * (1 - 0.5)), 0], [0, 1] ]))

        ## generate random rotation
        particles[i][0].rotate(0 + (random() * (2*np.pi - 0)) )

        ## Debug, use uniform arrangement
        # particles[0][i].shift(msh.incenter[i])
        particles[i][0].shift(msh.incenter[i])

    print('len: ', len(particles))
    
    plank_ind = msh.count()
    # particles[plank_ind][0].shift([ -(l+s), 0])
    # particles[plank_ind][1].shift([ (l+s), 0])

    # particles[plank_ind][0].movable = 0
    # particles[plank_ind][1].movable = 0

    g_val = -5e4

    # Initial data
    # particles[0][1].vel += [0, -20]
    # particles[0][1].acc += [0, -5e4]
    # particles[0][1].extforce += [0, -5e4 * particles[0][1].material.rho]
    for i in range(msh.count()):
        print('i', i)
        
        # particles[0][i].acc += [0, -5e3]
        # particles[0][i].extforce += [0, -5e3 * particles[0][i].material.rho]

        particles[i][0].acc += [0, g_val]
        particles[i][0].extforce += [0, g_val * particles[i][0].material.rho]

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.8
    friction_coefficient = 0.8
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    # contact.print()
    # material_dict.peridem(delta).print()

    return Experiment(particles, wall, contact)

def hopper():
    """ A bulk of particles, locations are generated from a mesh
    Each particle mesh is generated using scaling factor, hence the neighborhood and boundary nodes are computed separately
    """

    delta = 1e-3
    meshsize = delta/3
    contact_radius = delta/5

    # at least this much space to leave between particles. Will get halved while selecting max radius
    min_particle_spacing = contact_radius*1.001

    l= 6e-3
    s = 0.5e-3
    gap = 2e-3

    # plank_angle = 0
    plank_angle = np.pi/6
    angle_shift = l * ( 1 - np.cos(plank_angle))

    y_plank = 0e-3
    # y_plank = -4e-3

    # wall info
    L = 2*l + gap/2
    wall_left   = -L 
    wall_right  = L 
    wall_top    = L * 1
    wall_bottom = -L * 1
    # wall_bottom = -5e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # particle generation boundary
    c = min_particle_spacing/2
    P = np.array([
        # [wall_left + c, wall_bottom + c ],
        [wall_left + c, y_plank+s+ c + l*np.sin(plank_angle)  ],
        [wall_left + c + 2*l - angle_shift, y_plank+s+ c - l*np.sin(plank_angle)  ],
        [wall_right - c - 2*l + angle_shift, y_plank+s + c - l*np.sin(plank_angle)  ],
        [wall_right - c, y_plank+s + c + l*np.sin(plank_angle)  ],
        [wall_right - c, wall_top - c],
        [wall_left + c, wall_top - c]
        ])

    # particle generation boundary
    # P = np.array([
        # [wall_left, wall_bottom],
        # [wall_right, wall_bottom],
        # [wall_right, wall_top],
        # [wall_left, wall_top]
        # ])

    ## Generate particle arrangement
    ## particle generation spacing
    P_meshsize = 3.5e-3
    msh = get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= True, gen_bdry = False )
    # reduce radius to avoid contact
    msh.incircle_rad -= min_particle_spacing/2
    msh.info()
    msh.plot(plot_edge = True)

    # Create a list of shapes
    SL = ShapeList()

    # append each shape with own scaling
    for i in range(msh.count()):
        # delta_mod = msh.incircle_rad[i]/2
        delta_mod = delta
        seed_i = i

        SL.append(shape=shape_dict.perturbed_disk(steps=16, scaling=msh.incircle_rad[i], seed=seed_i), count=1, meshsize=meshsize, material=material_dict.sodalime_similar_to_peridem(delta_mod))

    delta_plank = delta/2
    SL.append(shape=shape_dict.plank(l=l, s=s), count=2, meshsize=meshsize, material=material_dict.sodalime_similar_to_peridem(delta_plank))

    # generate the mesh for each shape
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False, shapes_in_parallel=True)

    ## Apply transformation
    seed(1)
    g_val = -5e4
    for i in range(msh.count()):
        ## scaling is done while generating the mesh
        ## generate random rotation
        particles[i][0].rotate(0 + (random() * (2*np.pi - 0)) )
        particles[i][0].shift(msh.incenter[i])

        # Initial data
        # particles[0][1].vel += [0, -20]
        particles[i][0].acc += [0, g_val]
        particles[i][0].extforce += [0, g_val * particles[i][0].material.rho]

    # position the planks
    plank_ind = msh.count()
    particles[plank_ind][0].rotate(plank_angle)
    particles[plank_ind][1].rotate(-plank_angle)

    particles[plank_ind][0].shift([ -(l+gap/2+angle_shift), y_plank])
    particles[plank_ind][1].shift([ (l+gap/2+angle_shift), y_plank])
    particles[plank_ind][0].movable = 0
    particles[plank_ind][1].movable = 0

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.8
    friction_coefficient = 0.8
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    return Experiment(particles, wall, contact)

def coffee_diffmesh():
    """ A bulk of particles, locations are generated from a mesh
    are ground up by a coffee grinder
    """

    # delta = 1e-3
    # delta = 10e-3
    delta = 8e-3
    meshsize = delta/5
    contact_radius = delta/4

    # blade_l = 5e-3
    # blade_s = 0.5e-3
    blade_l = 70e-3
    blade_s = 5e-3

    # particle toughness
    Gnot_scale = 0.7
    # rho_scale = 0.6
    # attempt to modify further
    rho_scale = 0.7

    # at least this much space to leave between particles. Will get halved while selecting max radius
    min_particle_spacing = contact_radius*1.001

    # wall info
    # cl = 6e-3
    cl = 100e-3 # 10 cm

    wall_left   = -cl
    wall_right  = cl
    wall_top    = cl
    wall_bottom = -cl
    # wall_bottom = -5e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # particle generation boundary
    c = min_particle_spacing/2
    P = np.array([
        [wall_left + c, wall_bottom + c ],
        [wall_right - c, wall_bottom + c],
        # [wall_right - c, wall_top - c],
        # [wall_left + c, wall_top - c]
        ## only bottom half
        [wall_right - c, 0 - blade_s - c],
        [wall_left + c, 0-blade_s - c]

        ])


    # Create a list of shapes
    SL = ShapeList()

    # grinder blade
    SL.append(shape=shape_dict.plank(l=blade_l, s=blade_s) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))

    add_particles = 1

    if add_particles:
        # particle generation spacing
        # P_meshsize = 3.5e-3
        P_meshsize = 35e-3 # cm order
        msh = get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= True, gen_bdry = False )

        msh.trim(min_rad = 6e-3 + min_particle_spacing/2)

        # reduce radius to avoid contact
        msh.incircle_rad -= min_particle_spacing/2
        msh.info()
        msh.plot(plot_edge = True)
        ###########################
        # particles
        # append each shape with own scaling
        for i in range(msh.count()):
            SL.append(shape=shape_dict.perturbed_disk(steps=16, scaling=msh.incircle_rad[i]), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, rho_scale=rho_scale, Gnot_scale=Gnot_scale))

    # generate the mesh for each shape
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False, shapes_in_parallel=False)

    if add_particles:
        ## Apply transformation
        seed(1)
        g_val = -5e4
        for i in range(msh.count()):
            ## scaling is done while generating the mesh
            ## generate random rotation
            particles[i+1][0].rotate(0 + (random() * (2*np.pi - 0)) )
            particles[i+1][0].shift(msh.incenter[i])

            # Initial data
            # particles[0][1].vel += [0, -20]
            # particles[i][0].acc += [0, g_val]
            # particles[i][0].extforce += [0, g_val * particles[i][0].material.rho]
        
        ###########################


    ## applying torque
    blade = particles[0][0]
    blade.breakable = 0
    blade.stoppable = 0

    # 1000 RPM (pretty high for burr coffee grinder) = 1000 * 2 * pi / 60 (rad/s) ~ 104 rad/s
    # v_val = 1000
    v_val = 600

    perp = np.array([[0, -1], [1, 0]])

    # r = np.sqrt( np.sum(blade.pos[0]**2) )
    # u_dir = blade.pos[0] / r
    # t_dir = perp @ u_dir 

    # print(blade.vel[0])
    # print(u_dir)
    # print(t_dir)


    for j in range(len(blade.pos)):
        r = np.sqrt( np.sum(blade.pos[j]**2) )
        u_dir = blade.pos[j] / r
        t_dir = perp @ u_dir
        blade.vel[j] += v_val * r * t_dir

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.8
    friction_coefficient = 0.8
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    return Experiment(particles, wall, contact)

def wheel():
    """ Wheel on flat surface
    """

    # whether or not to add particles
    add_particles = 0
    wheel_rad = 3e-3

    # delta = 1e-3
    # delta = 10e-3
    delta = 1e-3
    meshsize = delta/2
    contact_radius = delta/4

    # blade_l = 5e-3
    # blade_s = 0.5e-3
    blade_l = 70e-3
    blade_s = 5e-3

    # particle toughness
    Gnot_scale = 0.7
    # rho_scale = 0.6
    # attempt to modify further
    rho_scale = 0.7

    # at least this much space to leave between particles. Will get halved while selecting max radius
    min_particle_spacing = contact_radius*1.001

    # wall info
    # cl = 6e-3
    cl = 10e-3 # 10 cm

    wall_left   = -cl
    wall_right  = cl
    wall_top    = cl/2
    wall_bottom = -cl/2
    # wall_bottom = -5e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # particle generation boundary
    c = min_particle_spacing/2
    P = np.array([
        [wall_left + c, wall_bottom + c ],
        [wall_right - c, wall_bottom + c],
        # [wall_right - c, wall_top - c],
        # [wall_left + c, wall_top - c]
        ## only bottom half
        [wall_right - c, 0 - blade_s - c],
        [wall_left + c, 0-blade_s - c]

        ])


    # Create a list of shapes
    SL = ShapeList()

    # wheel
    # SL.append(shape=shape_dict.small_disk(scaling=wheel_rad) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    SL.append(shape=shape_dict.small_disk(scaling=wheel_rad) , count=2, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.plank(l=wheel_rad, s=wheel_rad) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))


    if add_particles:
        # particle generation spacing
        # P_meshsize = 3.5e-3
        P_meshsize = 35e-3 # cm order
        msh = get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= True, gen_bdry = False )

        msh.trim(min_rad = 6e-3 + min_particle_spacing/2)

        # reduce radius to avoid contact
        msh.incircle_rad -= min_particle_spacing/2
        msh.info()
        msh.plot(plot_edge = True)
        ###########################
        # particles
        # append each shape with own scaling
        for i in range(msh.count()):
            SL.append(shape=shape_dict.perturbed_disk(steps=16, scaling=msh.incircle_rad[i]), count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta, rho_scale=rho_scale, Gnot_scale=Gnot_scale))

    # generate the mesh for each shape
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False, shapes_in_parallel=False)

    # wheel location
    move_diag = 0e-3
    particles[0][0].shift( [move_diag+wall_left+wheel_rad+contact_radius*2, move_diag+wall_bottom + wheel_rad + contact_radius])

    # wheel-2
    move_diag = 0e-3
    particles[0][1].shift( [wall_right-wheel_rad-move_diag-contact_radius*2, move_diag+wall_bottom + wheel_rad + contact_radius])

    ## applying torque
    wheel = particles[0][0]
    wheel.breakable = 0
    wheel.stoppable = 1

    # 1000 RPM (pretty high for burr coffee grinder) = 1000 * 2 * pi / 60 (rad/s) ~ 104 rad/s
    # v_val = 1000
    v_val = -600

    perp = np.array([[0, -1], [1, 0]])

    # r = np.sqrt( np.sum(blade.pos[0]**2) )
    # u_dir = blade.pos[0] / r
    # t_dir = perp @ u_dir 

    # print(blade.vel[0])
    # print(u_dir)
    # print(t_dir)

    #gravity
    g_val = -1e3

    centroid = np.mean(wheel.pos + wheel.disp, axis=0)
    print('centroid', centroid)

    # initial angular velcity
    # for j in range(len(wheel.pos)):
        # r_vec = (wheel.pos[j] - centroid)
        # r = np.sqrt( np.sum(r_vec**2) )
        # u_dir = r_vec / r
        # t_dir = perp @ u_dir
        # wheel.vel[j] += v_val * r * t_dir

    #torque
    wheel.torque_axis = 2
    wheel.torque_val = -1e6
    # gravity
    wheel.extforce += [0, g_val * wheel.material.rho]


    # wheel-2 torque
    particles[0][1].torque_axis = 2
    particles[0][1].torque_val = 1e6
    # gravity
    particles[0][1].extforce += [0, g_val * wheel.material.rho]

    # if add_particles:
        # ## Apply transformation
        # seed(1)
        # g_val = -5e4
        # for i in range(msh.count()):
            # ## scaling is done while generating the mesh
            # ## generate random rotation
            # particles[i+1][0].rotate(0 + (random() * (2*np.pi - 0)) )
            # particles[i+1][0].shift(msh.incenter[i])

            # Initial data
            # particles[0][1].vel += [0, -20]
            # particles[i][0].acc += [0, g_val]
            # particles[i][0].extforce += [0, g_val * particles[i][0].material.rho]
        
        ###########################

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.8
    friction_coefficient = 0.8
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    return Experiment(particles, wall, contact)

def wheel_on_inclined():
    """ Wheel on inclined plane
    """

    # whether or not to add particles
    wheel_rad = 3e-3

    # delta = 1e-3
    # delta = 10e-3
    delta = 1e-3
    # meshsize = delta/2
    meshsize = delta/3
    contact_radius = delta/4

    # blade_l = 5e-3
    # blade_s = 0.5e-3
    blade_l = 20e-3
    blade_s = 0.5e-3
    blade_angle = -np.pi/10

    # particle toughness
    Gnot_scale = 0.7
    # rho_scale = 0.6
    # attempt to modify further
    rho_scale = 0.7

    # at least this much space to leave between particles. Will get halved while selecting max radius
    min_particle_spacing = contact_radius*1.001

    # wall info
    # cl = 6e-3
    # cl = 10e-3 # 10 cm
    cl = blade_l*1.5

    wall_left   = -cl
    wall_right  = cl
    wall_top    = cl
    wall_bottom = -cl
    # wall_bottom = -5e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # particle generation boundary
    c = min_particle_spacing/2
    P = np.array([
        [wall_left + c, wall_bottom + c ],
        [wall_right - c, wall_bottom + c],
        # [wall_right - c, wall_top - c],
        # [wall_left + c, wall_top - c]
        ## only bottom half
        [wall_right - c, 0 - blade_s - c],
        [wall_left + c, 0-blade_s - c]

        ])


    # Create a list of shapes
    SL = ShapeList()

    # wheel
    # SL.append(shape=shape_dict.small_disk(scaling=wheel_rad) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.small_disk(scaling=wheel_rad) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.pygmsh_geom_test(scaling=wheel_rad, meshsize=meshsize) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.gmsh_test(scaling=wheel_rad, meshsize=meshsize/2) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    SL.append(shape=shape_dict.wheel_annulus(scaling=wheel_rad, inner_circle_ratio=0.7, meshsize=meshsize, filename_suffix='00') , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.annulus() , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    # SL.append(shape=shape_dict.wheel_ring(scaling=2e-3) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))
    SL.append(shape=shape_dict.plank(l=blade_l, s=blade_s) , count=1, meshsize=meshsize, material=material_dict.peridem_deformable(delta))


    # generate the mesh for each shape
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False, shapes_in_parallel=False)

    print('Here')

    # plank, rotation
    particles[1][0].rotate(blade_angle)
    particles[1][0].movable =0
    particles[1][0].stoppable =0

    # wheel location
    l_offset = -2e-3
    s_offset = 0e-3

    radial_dir = np.array([-np.cos(-blade_angle) , -np.sin(-blade_angle)])
    tang_dir = np.array([-np.sin(-blade_angle) , np.cos(-blade_angle)])
    shift_vec = (blade_l + l_offset)  * radial_dir + (blade_s + wheel_rad + contact_radius+s_offset) * tang_dir
    # shift_vec = blade_l  * tang_dir
    print('shift_val', shift_vec)
    particles[0][0].shift( shift_vec)


    ## applying torque
    wheel = particles[0][0]
    wheel.breakable = 0
    wheel.stoppable = 1
    #torque
    wheel.torque_val = -5e5
    #gravity
    g_val = -1e3
    wheel.extforce += [0, g_val * wheel.material.rho]

    # initial angular velcity
    # 1000 RPM (pretty high for burr coffee grinder) = 1000 * 2 * pi / 60 (rad/s) ~ 104 rad/s
    # v_val = 1000
    v_val = -600
    perp = np.array([[0, -1], [1, 0]])
    centroid = np.mean(wheel.pos + wheel.disp, axis=0)
    print('centroid', centroid)
    for j in range(len(wheel.pos)):
        r_vec = (wheel.pos[j] - centroid)
        r = np.sqrt( np.sum(r_vec**2) )
        u_dir = r_vec / r
        t_dir = perp @ u_dir
        wheel.vel[j] += v_val * r * t_dir


    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.8
    friction_coefficient = 0.8
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    return Experiment(particles, wall, contact)

def bulk_generated_diffmesh():
    """ A bulk of particles, locations are generated from a mesh
    Each particle mesh is generated using scaling factor, hence the neighborhood and boundary nodes are computed separately
    """

    delta = 1e-3
    meshsize = 1e-3/5
    contact_radius = delta/5

    # at least this much space to leave between particles. Will get halved while selecting max radius
    min_particle_spacing = contact_radius*1.001

    # wall info
    wall_left   = -10e-3
    wall_right  = 10e-3
    wall_top    = 10e-3
    wall_bottom = 0e-3
    # wall_bottom = -5e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # particle generation boundary
    c = min_particle_spacing/2
    P = np.array([
        [wall_left + c, wall_bottom + c ],
        [wall_right - c, wall_bottom + c],
        [wall_right - c, wall_top - c],
        [wall_left + c, wall_top - c]
        ])

    # particle generation spacing
    P_meshsize = 3.5e-3
    msh = get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= True, gen_bdry = True )
    # reduce radius to avoid contact
    msh.incircle_rad -= min_particle_spacing/2
    msh.info()
    msh.plot(plot_edge = True)

    # Create a list of shapes
    SL = ShapeList()

    # append each shape with own scaling
    for i in range(msh.count()):
        SL.append(shape=shape_dict.perturbed_disk(steps=16, scaling=msh.incircle_rad[i]), count=1, meshsize=meshsize, material=material_dict.peridem(delta))

    # generate the mesh for each shape
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False, shapes_in_parallel=False)

    ## Apply transformation
    seed(1)
    g_val = -5e4
    for i in range(msh.count()):
        ## scaling is done while generating the mesh
        ## generate random rotation
        particles[i][0].rotate(0 + (random() * (2*np.pi - 0)) )
        particles[i][0].shift(msh.incenter[i])

        # Initial data
        # particles[0][1].vel += [0, -20]
        particles[i][0].acc += [0, g_val]
        particles[i][0].extforce += [0, g_val * particles[i][0].material.rho]

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.8
    friction_coefficient = 0.8
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    return Experiment(particles, wall, contact)

def bulk_generated():
    """ A bulk of particles, locations are generated from a mesh
    """

    delta = 1e-3
    meshsize = 1e-3/5
    contact_radius = delta/5;

    # at least this much space to leave between particles. Will get halved while selecting max radius
    min_particle_spacing = contact_radius*1.001

    # wall info
    wall_left   = -10e-3
    wall_right  = 10e-3
    wall_top    = 10e-3
    wall_bottom = 0e-3
    # wall_bottom = -5e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)


    ## shape: 0
    # shapes.append(shape_dict.small_disk())
    # shapes.append(shape_dict.plus())

    # Material property
    # material_list.append(material_dict.peridem(delta))

    # time.sleep(5.5)
    

    # particle generation boundary
    c = min_particle_spacing/2
    P = np.array([
        [wall_left + c, wall_bottom + c ],
        [wall_right - c, wall_bottom + c],
        [wall_right - c, wall_top - c],
        [wall_left + c, wall_top - c]
        ])

    # particle generation boundary
    # P = np.array([
        # [wall_left, wall_bottom],
        # [wall_right, wall_bottom],
        # [wall_right, wall_top],
        # [wall_left, wall_top]
        # ])
    # particle generation spacing
    P_meshsize = 3.5e-3
    msh = get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= True, gen_bdry = True )
    # msh = get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= False, gen_bdry = False )
    # msh = get_incenter_mesh_loc(P, P_meshsize, msh_file='meshdata/closepacked_mat.msh', modify_nodal_circles= True, gen_bdry = False )
    # msh = get_incenter_mesh_loc(P, P_meshsize, msh_file='meshdata/closepacked_mat.msh', modify_nodal_circles= False, gen_bdry = False )

    # print('total count: ', msh.count())
    # print('total vol: ', msh.total_volume())
    # print('occupied vol: ', msh.occupied_volume())
    # print('packing raio: ', msh.packing_ratio())
    # msh.info()
    # msh.trim(min_rad = 0.005 + min_particle_spacing/2, max_rad = 0.01 + min_particle_spacing/2)
    # msh.trim(min_rad = 0.8 * 1e-3 + min_particle_spacing/2)
    msh.info()
    msh.plot(plot_edge = True)

    # reduce radius to avoid contact
    msh.incircle_rad -= min_particle_spacing/2
    msh.info()
    msh.plot(plot_edge = True)
    # radius of base object is 1e-3
    scaling_list = msh.incircle_rad/1e-3

    # Create a list of shapes
    SL = ShapeList()
    # shape = shape_dict.load('meshdata/peridem_mat.msh')
    # SL.append(shape=shape_dict.small_disk(steps = 4), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))
    # SL.append(shape=shape_dict.small_disk(), count=msh.count(), meshsize=meshsize, material=material_dict.peridem_deformable(delta, rho_scale=5e2))
    # SL.append(shape=shape_dict.plus_inscribed(notch_dist=0.2), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))
    SL.append(shape=shape_dict.perturbed_disk(steps=16), count=msh.count(), meshsize=meshsize, material=material_dict.peridem(delta))

    # generate the mesh for each shape
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius)

    ## Apply transformation
    seed(1)
    for i in range(msh.count()):
        particles[0][i].scale(scaling_list[i])

        # particles[0][i].trasnform_bymat(np.array([ [0.8, 0], [0, 1] ]))
        # particles[0][i].trasnform_bymat(np.array([ [0.5 + (random() * (1 - 0.5)), 0], [0, 1] ]))

        ## generate random rotation
        particles[0][i].rotate(0 + (random() * (2*np.pi - 0)) )
        particles[0][i].shift(msh.incenter[i])


    # Initial data
    # particles[0][1].vel += [0, -20]
    # particles[0][1].acc += [0, -5e4]
    # particles[0][1].extforce += [0, -5e4 * particles[0][1].material.rho]
    for i in range(msh.count()):
        # particles[0][i].acc += [0, -5e3]
        # particles[0][i].extforce += [0, -5e3 * particles[0][i].material.rho]

        particles[0][i].acc += [0, -10]
        particles[0][i].extforce += [0, -10 * particles[0][i].material.rho]
        pass

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.8
    friction_coefficient = 0.8
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    # contact.print()
    # material_dict.peridem(delta).print()

    return Experiment(particles, wall, contact)

def bulk_generated_mixed():
    """ A bulk of particles, locations are generated from a mesh
    """

    delta = 1e-3
    meshsize = 1e-3/3
    contact_radius = delta/5;

    # at least this much space to leave between particles. Will get halved while selecting max radius
    min_particle_spacing = contact_radius*1.001

    # wall info
    wall_left   = -10e-3
    wall_right  = 10e-3
    wall_top    = 10e-3
    wall_bottom = 0e-3
    # wall_bottom = -5e-3
    wall = Wall(1, wall_left, wall_right, wall_top, wall_bottom)

    # particle generation boundary
    c = min_particle_spacing/2
    P = np.array([
        [wall_left + c, wall_bottom + c ],
        [wall_right - c, wall_bottom + c],
        [wall_right - c, wall_top - c],
        [wall_left + c, wall_top - c]
        ])
    # particle generation spacing
    P_meshsize = 3.5e-3
    # P_meshsize = 3.5e-3 *10
    msh = get_incenter_mesh_loc(P, P_meshsize, modify_nodal_circles= True, gen_bdry = True )
    # reduce radius to avoid contact
    msh.incircle_rad -= min_particle_spacing/2
    msh.info()
    msh.plot(plot_edge = True)
    # radius of base object is 1e-3
    scaling_list = msh.incircle_rad/1e-3

    # Create a list of shapes
    SL = ShapeList()

    shape_array = []


    ## specify the shapes
    for sh in range(5):
        shape_array.append(shape_dict.perturbed_disk(seed=sh, steps=20))

    ## specify the shapes
    # shape_array.append(shape_dict.small_disk())
    # shape_array.append(shape_dict.plus_inscribed())
    # shape_array.append(shape_dict.small_disk(steps=4))
    # shape_array.append(shape_dict.small_disk(steps=6))


    shapes = len(shape_array)
    # generate the count of each shape
    num_each = int(msh.count()/shapes)
    # remainder, could be zero
    sh_mod = msh.count() % shapes
    counts = []
    for sh in range(shapes):
        if sh < sh_mod:
            counts.append(num_each + 1)
        else:
            counts.append(num_each)
    print('Count of each shape', counts)

    # append shapes based on count
    for sh in range(shapes):
        SL.append(shape=shape_array[sh], count=counts[sh], meshsize=meshsize, material=material_dict.peridem(delta))

    # generate the mesh for each shape
    particles = SL.generate_mesh(dimension = 2, contact_radius = contact_radius, plot_mesh=False, plot_shape=False)

    np.random.seed(1)
    seed(1)

    # generate a random permutation to distribute the shapes
    rnd_perm = np.random.permutation(msh.count())

    ## Apply transformation
    # absolute (linear) index
    abs_idx = 0
    for sh in range(shapes):
        for i in range(counts[sh]):
            particles[sh][i].scale(scaling_list[rnd_perm[abs_idx]])
            # particles[0][i].trasnform_bymat(np.array([ [0.8, 0], [0, 1] ]))
            # particles[0][i].trasnform_bymat(np.array([ [0.5 + (random() * (1 - 0.5)), 0], [0, 1] ]))
            ## generate random rotation
            particles[sh][i].rotate(0 + (random() * (2*np.pi - 0)) )
            particles[sh][i].shift(msh.incenter[rnd_perm[abs_idx]])

            # increment absolute index
            abs_idx += 1

        print('end count of this shape')


    # Initial data
    for sh in range(shapes):
        for i in range(counts[sh]):
            # particles[sh][1].vel += [0, -20]
            particles[sh][i].acc += [0, -5e3]
            particles[sh][i].extforce += [0, -5e3 * particles[0][i].material.rho]

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,4));
    damping_ratio = 0.8
    friction_coefficient = 0.8
    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    # contact.print()
    # material_dict.peridem(delta).print()

    return Experiment(particles, wall, contact)

def collision_3d_bad():
    """ Caution: bad msh included: Two particles colliding in 3D
    """

    delta = 1e-3
    meshsize = None # set in .geo and generated .msh
    contact_radius = 1e-3/5;


    SL = ShapeList()

    ## shape: 0
    SL.append(shape=shape_dict.sphere_small_3d(), count=1, meshsize=meshsize, material=material_dict.peridem_3d(delta)) 
    SL.append(shape=shape_dict.sphere_small_3d_bad(), count=1, meshsize=meshsize, material=material_dict.peridem_3d(delta)) # bad

    # material_dict.peridem_3d(delta).print()

    particles = SL.generate_mesh(dimension = 3, contact_radius=contact_radius)

    # apply transformation
    particles[0][0].rotate3d('z', np.pi/2)
    particles[1][0].rotate3d('z', -np.pi/2)
    particles[1][0].shift([0, 0, 3e-3])
    # particles[0][1].shift([0, 0, 2.6e-3])

    # Initial data
    particles[1][0].vel += [0, 0, -20]
    # particles[0][1].acc += [0, 0, -16e4]
    # particles[0][1].extforce += [0, 0, -16e4 * particles[0][1].material.rho]

    # wall info

    x_min = -5e-3
    y_min = -5e-3
    z_min = -5e-3
    x_max = 5e-3
    y_max = 5e-3
    z_max = 5e-3
    wall = Wall3d(1, x_min, y_min, z_min, x_max, y_max, z_max)
    # wall = Wall3d(0)

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,5));
    damping_ratio = 0.8
    friction_coefficient = 0.8

    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    return Experiment(particles, wall, contact)

def collision_3d():
    """ Two particles colliding in 3D
    """

    delta = 1e-3
    meshsize = None # set in .geo and generated .msh
    # contact_radius = 1e-3/3;
    contact_radius = 1e-3/(2.5);    # conserves momentum better (than delta/3)

    SL = ShapeList()

    ## shape: 0
    # SL.append(shape=shape_dict.sphere_small_3d_mat_bad(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))   # bad msh file
    # SL.append(shape=shape_dict.sphere_small_3d_mat(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))

    SL.append(shape=shape_dict.sphere_small_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))
    # SL.append(shape=shape_dict.disk_w_hole_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))
    # SL.append(shape=shape_dict.plus_small_3d(), count=2, meshsize=meshsize, material=material_dict.peridem_3d(delta))

    mat = material_dict.peridem_3d(delta)
    mat.print()

    particles = SL.generate_mesh(dimension = 3, contact_radius=contact_radius)

    # apply transformation
    particles[0][0].rotate3d('z', np.pi/2)
    particles[0][1].rotate3d('z', -np.pi/2)
    particles[0][1].shift([0, 0, 3e-3])
    # particles[0][1].shift([0, 0, 2.6e-3])

    # Initial data
    particles[0][1].vel += [0, 0, -20]
    # particles[0][1].acc += [0, 0, -16e4]
    # particles[0][1].extforce += [0, 0, -16e4 * particles[0][1].material.rho]

    # wall info

    x_min = -5e-3
    y_min = -5e-3
    z_min = -5e-3
    x_max = 5e-3
    y_max = 5e-3
    z_max = 5e-3
    wall = Wall3d(1, x_min, y_min, z_min, x_max, y_max, z_max)
    # wall = Wall3d(0)

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,5));
    # normal_stiffness = 15 * mat.E /( np.pi * np.power(delta,5) * (1 - 2*mat.nu));

    damping_ratio = 0.8
    friction_coefficient = 0.8

    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    return Experiment(particles, wall, contact)


def bulk_generated_3d():
    """ bulk particle generation for 3D region from tetrahedrons
    """

    delta = 1e-3
    meshsize = None # set in .geo and generated .msh
    # contact_radius = 1e-3/3;
    contact_radius = 1e-3/(2.5)    # conserves momentum better (than delta/3)

    # at least this much space to leave between particles. Will get halved while selecting max radius
    min_particle_spacing = contact_radius*1.001

    # wall info
    x_min = -20e-3
    y_min = -20e-3
    z_min = -20e-3
    x_max = 20e-3
    y_max = 20e-3
    z_max = 20e-3
    wall = Wall3d(1, x_min, y_min, z_min, x_max, y_max, z_max)


    ####
    # Bulk generation

    # particle generation spacing
    msh = get_incenter_mesh_loc(P = None, meshsize=None, dimension = 3, msh_file='meshdata/3d/closepacked_cube.msh', modify_nodal_circles= False, gen_bdry = False )

    # print('total count: ', msh.count())
    # print('total vol: ', msh.total_volume())
    # print('occupied vol: ', msh.occupied_volume())
    # print('packing raio: ', msh.packing_ratio())
    msh.info()
    # msh.trim(min_rad = 0.005, max_rad = 0.01)

    # Adding min_particle_spacing/2 here ensures that later, after reducing their radii by min_particle_spacing/2 produces the same range (and max, min, mean) of particle radii as set here
    msh.trim(min_rad = 0.8*1e-3 + min_particle_spacing/2, max_rad = 1.2*1e-3 + min_particle_spacing/2)

    msh.info()
    msh.plot(plot_edge = True)

    # reduce radius to avoid contact
    msh.incircle_rad -= min_particle_spacing/2
    msh.info()
    msh.plot(plot_edge = True)
    # radius of base object is 1e-3
    scaling_list = msh.incircle_rad/1e-3

    ## Create a list of shapes
    SL = ShapeList()

    ## shape: 0
    SL.append(shape=shape_dict.sphere_small_3d(), count=msh.count(), meshsize=meshsize, material=material_dict.peridem_3d(delta))

    mat = material_dict.peridem_3d(delta)
    mat.print()

    particles = SL.generate_mesh(dimension = 3, contact_radius=contact_radius)

    ## apply transformation
    #particles[0][0].rotate3d('z', np.pi/2)
    #particles[0][1].rotate3d('z', -np.pi/2)
    #particles[0][1].shift([0, 0, 3e-3])
    ## particles[0][1].shift([0, 0, 2.6e-3])

    ## Apply transformation
    seed(1)
    for i in range(msh.count()):
        particles[0][i].scale(scaling_list[i])

        # particles[0][i].trasnform_bymat(np.array([ [0.8, 0], [0, 1] ]))
        # particles[0][i].trasnform_bymat(np.array([ [0.5 + (random() * (1 - 0.5)), 0], [0, 1] ]))

        ## generate random rotation
        # particles[0][i].rotate(0 + (random() * (2*np.pi - 0)) )
        particles[0][i].shift(msh.incenter[i])

    ## Initial data
    for i in range(msh.count()):
        particles[0][i].acc += [0, 0, -5e3]
        particles[0][i].extforce += [0, 0, -5e3 * particles[0][i].material.rho]

    ## wall = Wall3d(0)

    # contact properties
    normal_stiffness = 18 * material_dict.peridem(delta).bulk_modulus /( np.pi * np.power(delta,5));
    # normal_stiffness = 15 * mat.E /( np.pi * np.power(delta,5) * (1 - 2*mat.nu));

    damping_ratio = 0.8
    friction_coefficient = 0.8

    contact  = Contact(contact_radius, normal_stiffness, damping_ratio, friction_coefficient)

    return Experiment(particles, wall, contact)

