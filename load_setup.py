
import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use('Agg')

# to plot a collection of lines
from matplotlib.collections import LineCollection

# 3d plot
from mpl_toolkits.mplot3d import Axes3D

# plotting 3d polygons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import h5py
import numpy as np

# from multiprocessing import Pool

## regex matching
import re

from exp_dict import Contact, Wall, Wall3d

dotsize = 10

class Particle_brief(object):
    """ Create a particle with brief info
    : pos vs CurrPos depends on the usage
    """
    def __init__(self, pos = None, CurrPos = None, q = None, vol = None, connectivity = None, nonlocal_bdry_nodes = None, bdry_edges = None, disp = None, vel = None, acc = None, extforce = None, force = None, delta = None, rho = None, cnot = None, snot = None):

        self.pos = pos
        # current position
        self.CurrPos = CurrPos
        self.vol = vol
        self.connectivity = connectivity
        self.nonlocal_bdry_nodes = nonlocal_bdry_nodes
        self.bdry_edges = bdry_edges


        self.disp = disp
        self.vel = vel
        self.acc = acc
        self.extforce = extforce
        # internal force
        self.force = force
        # Quantity to plot with scatterplot, scalar and will be populated before plotting
        self.q = q

        self.delta = delta
        self.rho = rho
        self.cnot = cnot
        self.snot = snot

class Experiment_brief(object):
    """A collection of universal particle array, wall, and contact """
    def __init__(self, PArr, contact, wall):
        self.PArr = PArr
        self.contact = contact 
        self.wall = wall


    def total_volume(self):
        """compute the total volume of all the particles
        :returns: scalar
        """
        mPArr = self.PArr
        vol = 0
        for i in range(len(mPArr)):
            this_vol =  np.sum(mPArr[i].vol)
            vol += this_vol

        return vol

    def total_mass(self):
        """compute the total volume of all the particles
        :returns: scalar
        """
        mPArr = self.PArr
        mass = 0
        for i in range(len(mPArr)):
            mass += mPArr[i].rho * np.sum(mPArr[i].vol)

        return mass



    def plot(self, by_CurrPos = False, plot_scatter = True, plot_delta = True, plot_contact_rad = True, plot_bonds = False, plot_bdry_nodes = False, plot_bdry_edges= True, edge_alpha = 0.2, plot_wall = True, wall_linewidth = 1, plot_wall_faces = False, wall_color = 'cyan', wall_alpha = 0.1, camera_angle = None, do_plot = True, do_save = True, save_filename = 'setup.png', dotsize = 10, plot_vol = False, linewidth = 0.3, limits = None, remove_axes=False, grid  = False, colorbar=True, colorlim=None):
        """TODO: Docstring for plot.

        :plot_delta: TODO
        :plot_bonds: TODO
        :dotsize: TODO
        :linewidth: TODO
        :returns: TODO

        """

        # close existing plots
        plt.close()

        # print('Plotting the experiment setup.')

        dim = len(self.PArr[0].pos[0])
        # print('Dimension: ', dim)


        if dim ==3:
            fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax = fig.gca(projection='3d')
            ax = Axes3D(fig)
        else:
            ax = plt.gca()


        # print('Plotting particle nodes: ')
        for i, P_b in enumerate(self.PArr):
            # Plot the mesh
            if by_CurrPos:
                Pos = P_b.CurrPos
            else:
                Pos = P_b.pos

            # print(P_b.pos)
            # print(P_b.vel)

            # specifying `c=q` dramatically reduces the plotting time
            if P_b.q is None:
                # print('Setting to zero')
                colors = np.zeros(len(Pos))
            else:
                colors = P_b.q

            if plot_scatter:

                if plot_vol:
                    M = np.max(P_b.vol)
                    d_sz = P_b.vol/M*dotsize
                else:
                    d_sz = dotsize

                if dim ==2:
                    plt.scatter(Pos[:,0], Pos[:,1], c = colors, s = d_sz, marker = '.', linewidth = 0, cmap='viridis')
                else:
                    ax.scatter(Pos[:,0], Pos[:,1], Pos[:,2], c = colors, s = d_sz, marker = '.', linewidth = 0, cmap='viridis')

            if plot_bdry_nodes:
                bd = P_b.nonlocal_bdry_nodes
                # time.sleep(5.5)
                if dim==2:
                    plt.scatter(Pos[bd,0], Pos[bd,1], s = dotsize*1.1, marker = '.', linewidth = 0, cmap='viridis')
                else:
                    ax.scatter(Pos[bd,0], Pos[bd,1], Pos[bd,2], s = dotsize*1.1, marker = '.', linewidth = 0, cmap='viridis')


            if plot_bdry_edges:
                bde = P_b.bdry_edges
                if dim==2:
                    X12 = Pos[:,0][bde]
                    Y12 = Pos[:,1][bde]
                    plt.plot(X12, Y12, 'k-', linewidth = linewidth, alpha = edge_alpha )
                else:
                    # ax.plot_trisurf(Pos[:,0], Pos[:,1], Pos[:,2], triangles=P_b.bdry_edges, cmap=plt.cm.Spectral, alpha=0.5)
                    ax.plot_trisurf(Pos[:,0], Pos[:,1], Pos[:,2], triangles=P_b.bdry_edges, alpha=edge_alpha)

            if plot_bonds:

                V1 = P_b.connectivity[:,0]
                V2 = P_b.connectivity[:,1]

                P1 = Pos[V1]
                P2 = Pos[V2]
                # print(P1)
                ls =  [ [p1, p2] for p1, p2 in zip(P1,P2)] 

                if dim==2:
                    # for j in range(len(P_b.connectivity)):
                        # v1 = V1[j]
                        # v2 = V2[j]
                        # P1 = Pos[v1]
                        # P2 = Pos[v2]
                        # plt.plot([P1[0], P2[0]], [P1[1], P2[1]], 'b-', linewidth = linewidth )
                    # Plot fast using collections
                    lc = LineCollection(ls, linewidths=linewidth, colors='b')
                    # ax = plt.gca()
                    ax.add_collection(lc)
                else:
                    lc = Line3DCollection(ls, linewidths=linewidth, colors='b')
                    ax.add_collection(lc)


            # print(i, end = ' ', flush=True)


        if plot_delta:
            i = 0
            node = 0

            px = self.PArr[i].pos[node][0]
            py = self.PArr[i].pos[node][1]

            delta = P_b.delta

            t = np.linspace(0, 2*np.pi, num = 50)
            if dim==2:
                plt.plot( px+ delta * np.cos(t), py + delta * np.sin(t), linewidth = linewidth)
            else:
                pz = self.PArr[i].pos[node][2]
                # print('Implement more general delta: sphere')
                ax.plot( px+ delta * np.cos(t), py + delta * np.sin(t), pz, linewidth = linewidth)
        

        if plot_contact_rad:
            i = 0
            node = 1

            if by_CurrPos:
                px = self.PArr[i].CurrPos[node][0]
                py = self.PArr[i].CurrPos[node][1]
            else:
                px = self.PArr[i].pos[node][0]
                py = self.PArr[i].pos[node][1]

            t = np.linspace(0, 2*np.pi, num = 50)
            if dim==2:
                plt.plot( px+ self.contact.contact_radius * np.cos(t), py + self.contact.contact_radius * np.sin(t), linewidth = linewidth)
            else:
                pz = self.PArr[i].pos[node][2]
                ## draw circle
                # ax.plot( px+ self.contact.contact_radius * np.cos(t), py + self.contact.contact_radius * np.sin(t), pz, linewidth = linewidth)

                # draw sphere
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = np.cos(u)*np.sin(v)
                y = np.sin(u)*np.sin(v)
                z = np.cos(v)
                ax.plot_wireframe(px + self.contact.contact_radius*x, py + self.contact.contact_radius*y, pz + self.contact.contact_radius*z, color="r", linewidth = linewidth)

        if plot_wall:
            if self.wall.allow == 1:
                if dim ==2:
                    # print('Plotting wall in 2d')
                    ls = self.wall.get_lines()
                    lc = LineCollection(ls, linewidths=wall_linewidth, colors='b', alpha = wall_alpha)
                    # ax = plt.gca()
                    ax.add_collection(lc)

                else:
                    if plot_wall_faces:
                        ls = self.wall.get_faces()
                        ax.add_collection3d(Poly3DCollection(ls, facecolors=wall_color, linewidths=linewidth, edgecolors='r', alpha=wall_alpha))
                    else:
                        ls = self.wall.get_lines()
                        lc = Line3DCollection(ls, linewidths=wall_linewidth, colors='b', alpha=wall_alpha)
                        ax.add_collection(lc)


        ## Specify xyz axes limits
        # if limits are not specified
        if limits is None:
            # if wall exits, choose the wall boundary as default limits
            if self.wall.allow:
                # print('Choosing the plot boundary to be the wall.')
                ws = self.wall.get_lrtp()
                if len(ws) == 4:
                    # dim = 2
                    extents = np.array([ [self.wall.left, self.wall.right], [self.wall.bottom, self.wall.top] ])
                else:
                    # dim = 3
                    extents = np.array([ [self.wall.x_min, self.wall.x_max], [self.wall.y_min, self.wall.y_max], [self.wall.z_min, self.wall.z_max] ])

            # if wall does not exist, determine from the data
            else:
                extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        else:
            # print('Limits are specified')
            extents = np.array(limits)

        # fix the axes
        if (dim == 3):
            # print(extents)
            sz = extents[:,1] - extents[:,0]
            centers = np.mean(extents, axis=1)
            maxsize = max(abs(sz))
            r = maxsize/2
            for ctr, direc in zip(centers, 'xyz'):
                getattr(ax, 'set_{}lim'.format(direc))(ctr - r, ctr + r)

            # XYZlim = [-3e-3, 3e-3]
            # ax.set_xlim3d(XYZlim)
            # ax.set_ylim3d(XYZlim)
            # ax.set_zlim3d(XYZlim)
            # ax.set_aspect('equal')
            ax.set_box_aspect((1, 1, 1))
        else:
            ## Note: plt.axis('scaled') goes _before_ setting plt.xlim() and plt.ylim()
            ## for the boundaries to remain constant
            plt.axis('scaled')
            plt.xlim(extents[0][0], extents[0][1])
            plt.ylim(extents[1][0], extents[1][1])


        ## Plot properties
        # if dim==2:
            # ax = plt.gca()

        # turn grid on or off
        ax.grid(grid)

        # camera angle
        if camera_angle is not None:
            if (dim ==3):
                # print('dim, cam angle', dim,  camera_angle)
                ax.view_init(elev = camera_angle[0], azim = camera_angle[1])
            else:
                pass
                # print('dim, cam angle', dim,  camera_angle)

        if remove_axes:
            ax.set_axis_off()

        if colorlim is not None:
            plt.clim(colorlim[0], colorlim[1])
        if colorbar:
            if dim==2:
                plt.colorbar()
            else:
                print('colorbar in 3d not implemented yet.')
                pass


        # # save the image
        if do_save:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            # plt.savefig(save_filename, dpi=200, bbox_inches='tight')
            # plt.savefig(save_filename, dpi=300)


        if do_plot:
            plt.show()

        # print('')

            

def read_setup(filename, matlab_data = False, print_output=True):

    """ Loads all the data from the hdf5 file that contains the experiment setup
    :filename: path/all.h5
    :returns: Experiment_brief, the universal Particle_brief array is linearized in shape and has less info for plotting
    """

    if print_output:
        print('Reading file: ', filename)
    # an array of Particle_brief
    PArr= []

    f = h5py.File(filename, "r")
    for name in f:
        if re.match(r'P_[0-9]+', name):
            if matlab_data:
                P = Particle_brief( 
                        pos          = np.array(f[name+'/Pos']),
                        vol          = np.array(f[name+'/Vol']),

                        connectivity = np.array(f[name+'/Connectivity'].astype(int)),
                        nonlocal_bdry_nodes   = np.array(f[name+'/bdry_nodes']).astype(int),
                        # bdry_edges   = np.array(f[name+'/bdry_edges']),

                        disp         = np.array(f[name+'/disp']),
                        vel          = np.array(f[name+'/vel']),
                        acc          = np.array(f[name+'/acc']),
                        extforce     = np.array(f[name+'/extforce']),

                        delta        = np.array(f[name+'/delta'])[0][0],
                        rho          = np.array(f[name+'/rho'])[0][0],
                        cnot         = np.array(f[name+'/cnot'])[0][0],
                        snot         = np.array(f[name+'/snot'])[0][0],
                        )
            else:
                P = Particle_brief( 
                        pos          = np.array(f[name+'/Pos']),
                        vol          = np.array(f[name+'/Vol']),
                        connectivity = np.array(f[name+'/Connectivity']),
                        nonlocal_bdry_nodes   = np.array(f[name+'/bdry_nodes']),
                        bdry_edges   = np.array(f[name+'/bdry_edges']),

                        disp         = np.array(f[name+'/disp']),
                        vel          = np.array(f[name+'/vel']),
                        acc          = np.array(f[name+'/acc']),
                        extforce     = np.array(f[name+'/extforce']),

                        delta        = np.array(f[name+'/delta'])[0][0],
                        rho          = np.array(f[name+'/rho'])[0][0],
                        cnot         = np.array(f[name+'/cnot'])[0][0],
                        snot         = np.array(f[name+'/snot'])[0][0],
                        )

            # print('check if the same instance of class: P = ', P)
            PArr.append(P)

        elif (name == 'pairwise'):
            contact = Contact(
                    contact_radius = np.array(f[name+'/contact_radius'])[0][0],
                    normal_stiffness = np.array(f[name+'/normal_stiffness'])[0][0],
                    damping_ratio = np.array(f[name+'/damping_ratio'])[0][0],
                    friction_coefficient = np.array(f[name+'/friction_coefficient'])[0][0],
                    )
            if print_output:
                contact.print()
                # making all scalars a rank-2 data (matrix) to make it compatible with matlab
                # f.create_dataset('total_particles_univ', data=[[j-1]])

                # # contact info
                # f.create_dataset('pairwise/contact_radius', data=[[self.contact.contact_radius]])
                # f.create_dataset('pairwise/normal_stiffness', data=[[self.contact.normal_stiffness]])
                # f.create_dataset('pairwise/damping_ratio', data=[[self.contact.damping_ratio]])
                # f.create_dataset('pairwise/friction_coefficient', data=[[self.contact.friction_coefficient]])
        elif (name == 'wall'):
            if print_output:
                print('wall: ')
            # convert hdf data to np-array to treat like matrices
            get_allow = np.array(f[name+'/allow_wall'])
            allow = get_allow[0][0]
            if allow == 1:
                # wall = Wall(
                        # allow = 1,
                        # left   =np.array(f[name+'/geom_wall_info'])[0][0],
                        # right  =np.array(f[name+'/geom_wall_info'])[1][0],
                        # top    =np.array(f[name+'/geom_wall_info'])[2][0],
                        # bottom =np.array(f[name+'/geom_wall_info'])[3][0]
                        # )

                wall_size = np.array(f[name+'/geom_wall_info'])[:,0]

                if len(wall_size) == 4:
                    # this means dimension is 4
                    wall = Wall(1)
                    wall.set_size(wall_size)
                else:
                    wall = Wall3d(1)
                    wall.set_size(wall_size)
            else:
                wall  = Wall(allow = 0)
                # get_lrtp(): return np.array([self.left, self.right, self.top, self.bottom])
                # # wall info
                # f.create_dataset('wall/allow_wall', data = [[self.wall.allow]])
                # f.create_dataset('wall/geom_wall_info', data = np.array([self.wall.get_lrtp()]).transpose() )
            if print_output:
                wall.print()
            

        elif (name == 'total_particles_univ'):
            tot_p = np.array(f[name])[0][0]
        else:
            print('Leftover header: ', name)

    if print_output:
        print('total particles claimed: ', tot_p)
        print('total particles read: ', len(PArr))


    return Experiment_brief(PArr = PArr, contact = contact, wall = wall)



