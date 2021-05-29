import pygmsh
import meshio
import numpy as np

import gmsh
import sys

class Shape:
    """Returns the boundary nodes and the nonconvex_interceptor"""
    def __init__(self, P, pygmsh_geom=None, nonconvex_interceptor = [], msh_file = None):
        self.P = P
        self.pygmsh_geom = pygmsh_geom
        self.nonconvex_interceptor = nonconvex_interceptor
        self.msh_file = msh_file

class NonConvexInterceptor(object):
    """ A set of lines that intersect with bonds that extend beyond the domain"""
    def __init__(self, obt_bisec, ext_bdry, unit_normal, l_dir):
        self.obt_bisec = obt_bisec
        self.ext_bdry = ext_bdry
        self.unit_normal = unit_normal
        self.l_dir = l_dir

        self.use = 'all'
        
    def all(self):
        """all lines combined
        :returns: TODO

        """
        if (self.use == 'all'):
            return np.append(self.obt_bisec, self.ext_bdry, axis = 0)
        elif (self.use == 'bisec'):
            return self.obt_bisec
        elif (self.use == 'bdry'):
            return self.ext_bdry
        else:
            print('Wrong nonconvex_intercetor.use string given.')

def gen_nonconvex_interceptors(P, extension = 5e-5 ):
    """
    :P: A counterclockwise oriented polygon
    :returns: nx4 array, each row contains the endpoint of each interceptor

    """
    n = len(P)
    l = np.zeros((n,2))
    for i in range(n):
        l[i] = P[(i+1)%n] - P[i%n] 

    # max of absolute value for shape boundary rectangle
    P_max = np.max(np.sqrt(P**2), axis = 0)
    # print('P_max: ', P_max)

    P_ext = np.zeros((n,2))
    unit_normal = np.zeros((n,2))
    l_dir = np.zeros((n,2))

    obt_bisec = []
    
    for i in range(n):
        cross_prod = np.cross(l[i], l[(i+1)%n])

        # angle_between = np.arcsin(-cross_prod/np.linalg.norm(l[i])/np.linalg.norm(l[(i+1)%n]))
        # print('nonconvex_angle in degree: ', angle_between * 180/np.pi)

        # if(cross_prod < 0):
        # one vertex of the interceptor
        A0 = P[(i+1)%n]
        # flip the inward tangent
        l0_u = -l[i]
        l0_u /= np.linalg.norm(l0_u)
        # store
        l_dir[i] = -l0_u
        l1_u = l[(i+1)%n]
        l1_u /= np.linalg.norm(l1_u)
        # print('l0, l1:', l0_u, l1_u)
        v = (l0_u + l1_u)/2 
        unit_normal[(i+1)%n] = v

        if (np.linalg.norm(v) == 0):
            # print('yes, the norm is zero')
            # clockwise 90 degree rotation of l0_u or l1_u, (-y, x)
            v[0] =  -l0_u[1]
            v[1] = l0_u[0]
        # print('i = ', i, ' v = ', v)
            
        v /= np.linalg.norm(v)

        if (cross_prod>0):
            P_ext[(i+1)%n] = A0 - extension * v
        else:
            P_ext[(i+1)%n] = A0 + extension * v
        

        if(cross_prod < 0):
            t_list = []
            # find the intersections with other lines (excluding l(i), l(i+1))
            for j in range(n):
                if j not in [i, j+1]:
                    mat = np.c_[ v, P[j]-P[(j+1)%n] ]
                    b = P[j] - A0

                    if (np.linalg.det(mat)):
                        ts = np.linalg.solve(mat, b)
                        # print('solution for i=', i, ' j=', j, ': ', ts)
                        t = ts[0]
                        s = ts[1]

                        if ( ((s >=0)and(s<=1)) and (t > 0) ):
                            t_list.append(t)
                            # print('Self-intersection detected: ', j)

            if t_list:
                t_min = np.min(np.array(t_list))
                A1 = A0 + t_min * v
                # print('Found self-intersection: i=', i, ';from ', A0, ' to ', A1)
            else:
                # print('No self-intersection. Using maximum for i=', i)
                t_bd = np.linalg.norm(A0) + P_max
                # print('v = ', v)
                # print('max = ', t_bd)
                A1 = A0 + t_bd * v
                # print('No intersection with self-boundary. A1 = ', A1)

            obt_bisec.append( [A0[0], A0[1], A1[0], A1[1]] )

    B_ext = np.append(P_ext, np.roll(P_ext,-1, axis = 0), axis = 1)

    # out = np.append(np.array(nonconvex_interceptor), B_ext, axis = 0)
    # return out
    return NonConvexInterceptor(obt_bisec=np.array(obt_bisec), ext_bdry=B_ext, unit_normal=unit_normal, l_dir=l_dir)



def line_1d():
    return Shape(P=None, nonconvex_interceptor=None, msh_file='meshdata/1d/line.msh')


def plank(l=5e-3, s=1e-3):
    P = np.array([
        [-l,-s],
        [l,-s],
        [l,s],
        [-l,s]
        ])
    return Shape(P)

def tie(l=5e-3, s=1e-3):
    P = np.array([
        [-l,-s],
        [l,-s],
        [l,s],
        # [-l,s]
        ])
    return Shape(P)

def plank_wedge(l=5e-3, s=1e-3, w_loc_ratio=0.25, w_thickness_ratio=0.05, w_depth_ratio=1):
    P = np.array([
        [-l,-s],
        [l,-s],
        [l,s],
        [l*(1-w_loc_ratio)+l*(w_thickness_ratio), s],
        [l*(1-w_loc_ratio), s*(1-w_depth_ratio)],
        [l*(1-w_loc_ratio)-l*(w_thickness_ratio), s],
        [-l,s]
        ])

    nci = gen_nonconvex_interceptors(P)
    nci.use = 'bisec'
    # print(nci)
    return Shape(P, nci)

def ice_wedge(l=5e-3, s=1e-3, extra_ratio=1.2, w_loc_ratio=0.25, w_thickness_ratio=0.05, w_depth_ratio=1):
    P = np.array([
        [-l,-s],
        [l,-s],
        # [l,s],
        [l*extra_ratio, s],
        [l*(1-w_loc_ratio)+l*(w_thickness_ratio), s],
        [l*(1-w_loc_ratio), s*(1-w_depth_ratio)],
        [l*(1-w_loc_ratio)-l*(w_thickness_ratio), s],
        [-l,s]
        ])

    nci = gen_nonconvex_interceptors(P)
    nci.use = 'bisec'
    # print(nci)
    return Shape(P, nci)

def box():
    scaling = 1e-3
    P = scaling * np.array([
        [0,0],
        [1,0],
        [1,1],
        [0,1]
        ])
    return Shape(P)

def box_notch():
    scaling = 1e-3
    P = scaling * np.array([
        [0,0],
        [1,0],
        [1,0.5],
        [0.5,0.5],
        [0.5,1],
        # [1,1],
        [0,1]
        ])
    nci = gen_nonconvex_interceptors(P)
    nci.use = 'bisec'
    # print(nci)
    return Shape(P, nci)

def box_prenotch(l=5e-3, s=1e-3, a=0.1e-3):
    # a: slit length
    P = np.array([
        [-l,-s],
        [l,-s],
        [l,-a/2],
        [0,-a/2],
        [-a/2,0],
        [0,a/2],
        [l,a/2],
        [l,s],
        [-l,s]
        ])
    nci = gen_nonconvex_interceptors(P)
    nci.use = 'bisec'
    # print(nci)
    return Shape(P, nci)

def box_notch_2():
    scaling = 1e-3
    P = scaling * np.array([
        [0,0],
        [1,0],
        [1,0.5],
        [0.5,0.5],
        [0.5,0.75],
        [1,0.75],
        [1,1],
        [0.5,1],
        # [1,1],
        [0,1]
        ])
    nci = gen_nonconvex_interceptors(P)
    nci.use = 'all'
    # print(nci)
    return Shape(P, nci)

def box_notch_3():
    scaling = 1e-3
    P = scaling * np.array([
        [0,0],
        [1,0],
        [1,0.5],
        [0.5,0.5],
        # [0.3,0.6],
        [0.3,0.3],
        [0.2,0.6],
        [1,0.7],
        [1,1],
        # intentional defect, parallel line
        # [0.5,1],
        # [1,1],
        [0,1]
        ])
    nci = gen_nonconvex_interceptors(P)
    nci.use = 'all'

    # print(nci)
    return Shape(P, nci)

def vase(l=4e-3, s=1e-3, amp=0.5e-3, steps = 20, n_pi=4, phase_1=np.pi/2, phase_2=np.pi/2):

    xx = np.linspace(-l, l, num=steps)
    P = np.c_[xx, -s + amp*np.sin(xx*n_pi*np.pi/l + phase_1)]
    xx_rev = np.linspace(l, -l, num=steps)
    P_rev = np.c_[xx_rev, s + amp*np.sin(xx*n_pi*np.pi/l + phase_2)]
    P = np.append(P, P_rev, axis=0)

    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'
    return Shape(P, nonconvex_interceptor)


def small_disk(scaling = 1e-3, steps = 20):
    angles = np.linspace(0, 2* np.pi, num = steps, endpoint = False)
    P = np.array([np.cos(angles), np.sin(angles)])
    # return column matrices
    P = scaling * P.transpose()
    return Shape(P)


def perturbed_disk(steps = 20, seed=5, scaling=1e-3, std=0.2, angle_drift_amp = 0, angle_drift_std_ratio = 0.25):

    """ 
    :angle_drift_amp: If you want the particle to protrude outward, make sure (scaling + angle_drift_amp) <= radius of bounding circle
    If you want a dent in the particle, use negative value of angle_drift_amp


    """

    np.random.seed(seed)

    angles = np.linspace(0, 2* np.pi, num = steps, endpoint = False)

    # rads = 0 + (np.random.rnd(steps, 1) * (2*np.pi - 0))

    xx = np.linspace(0, steps, num=steps)
    var = steps * angle_drift_std_ratio
    yy = angle_drift_amp * np.exp( - (xx - np.floor(steps/2))**2/var )

    # to ensure the particle is within a bounding circle of radius `scaling`
    # rad_scales =  (np.ones((1,steps)) - angle_drift_amp) + yy
    rad_scales =  (np.ones((1,steps))) + yy

    # uniform sample from [0, scaling*std]
    rands = np.random.rand(1,steps)
    # print(rands * std)
    # rads = scaling * (1-std) + (np.random.rand(1,steps) * scaling * std)
    rads = scaling * (1-std) * rad_scales + (rands * scaling * std)

    # normal sample
    # random_l = np.random.normal(loc = std * scaling, scale=scaling*std/2, size = (1,steps))
    # print(random_l)
    # rads = scaling * (1-std) + random_l

    P = rads * np.array([np.cos(angles), np.sin(angles)])
    # return column matrices
    P =  P.transpose()

    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'

    return Shape(P, nonconvex_interceptor)

def polygon_inscribed(sides = 5, scaling = 1e-3):
    angles = np.linspace(0, 2* np.pi, num = sides, endpoint = False)
    P = np.array([np.cos(angles), np.sin(angles)])
    # return column matrices
    P = scaling * P.transpose()
    return Shape(P)

def pacman(angle = np.pi/2):
    scaling = 1e-3;
    steps = 20;
    angles = np.linspace(angle/2, 2* np.pi - angle/2, num = steps, endpoint = True)
    P = np.array([np.cos(angles), np.sin(angles)])
    # print(P)
    P = np.append(P, np.array([[0], [0]]), axis = 1)
    # return column matrices
    P = scaling * P.transpose()

    # % lines that intercepts non-convex bonds [A1,A2]
    # nonconvex_interceptor = scaling * np.array([
        # [0,0, 1,0]
    # ])
    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'
    # print('pacman nonconvex_interceptor: ', nonconvex_interceptor)
    return Shape(P, nonconvex_interceptor)

def ring_segment(scaling = 1e-3, steps=5, angle = np.pi/2, inner_rad=0.5):
    angles = np.linspace(angle/2, 2* np.pi - angle/2, num = steps, endpoint = True)
    P = np.array([np.cos(angles), np.sin(angles)])

    angles_rev = np.linspace(2* np.pi - angle/2, angle/2, num = steps, endpoint = True)
    Q = inner_rad*np.array([np.cos(angles_rev), np.sin(angles_rev)])

    P = np.append(P, Q, axis = 1)

    # return column matrices
    P = scaling * P.transpose()

    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'
    return Shape(P, nonconvex_interceptor)

def plus(ratio = 0.25):
    scaling = 1e-3;

    # ratio = 0.25
    long = 1
    short = ratio * long

    P = scaling * np.array([ [short, -short],
            [long, -short],
            [long, short],
            [short, short],
            [short, long],
            [-short, long],
            [-short, short],
            [-long, short],
            [-long, -short],
            [-short, -short],
            [-short, -long],
            [short, -long]
            ])

    # % lines that intercepts non-convex bonds [A1,A2]
    # nonconvex_interceptor = scaling * np.array([ [short, -short, long, -long],
            # [short, short, long, long],
            # [-short, short, -long, long],
            # [-short, -short, -long, -long],
            # ])
    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'

    return Shape(P, nonconvex_interceptor)

def plus_inscribed(notch_dist = 0.25, scaling = 1e-3):
    """ a plus with maximum possible lenth while being inscribed in a disk of radius 1e-3
    The maximum possible thickness is 
    : notch_dist: the distance from the center of disk to the inner notch, betwen 0 and 1 
    """

    short = notch_dist/np.sqrt(2)
    long = np.sqrt(1 - short**2)

    P = scaling * np.array([ [short, -short],
            [long, -short],
            [long, short],
            [short, short],
            [short, long],
            [-short, long],
            [-short, short],
            [-long, short],
            [-long, -short],
            [-short, -short],
            [-short, -long],
            [short, -long]
            ])

    # % lines that intercepts non-convex bonds [A1,A2]
    # nonconvex_interceptor = scaling * np.array([ [short, -short, long, -long],
            # [short, short, long, long],
            # [-short, short, -long, long],
            # [-short, -short, -long, -long],
            # ])

    nonconvex_interceptor = gen_nonconvex_interceptors(P)
    nonconvex_interceptor.use = 'bisec'
    return Shape(P, nonconvex_interceptor)

def small_disk_fromfile():
    return Shape(P=[], nonconvex_interceptor=[], msh_file='meshdata/peridem_mat.msh')

def box_wo_circ():
    msh_file='meshdata/2d/box_wo_circ.msh'
    return Shape(P=[], nonconvex_interceptor=[], msh_file=msh_file)

def test():
    filename = "meshdata/2d/test.msh"
    with pygmsh.occ.Geometry() as geom:

        P_bdry=[
            [0.0, 0.0],
            [1.0, -0.2],
            [1.1, 1.2],
            [0.1, 0.7],
        ]
        meshsize = 0.1
        # geom.add_polygon(
            # [
                # [0.0, 0.0],
                # [1.0, -0.2],
                # [1.1, 1.2],
                # [0.1, 0.7],
            # ],
            # mesh_size=0.1,
        # )
        # mesh = geom.generate_mesh()

        polygon1 = geom.add_polygon(
            P_bdry,
            mesh_size= meshsize,
        )
        geom.add_physical(polygon1.surface, "surface1")
        mesh = geom.generate_mesh()
        meshio.write(filename, mesh, file_format="gmsh")

    return Shape(P=[], nonconvex_interceptor=[], msh_file=filename)

def pygmsh_geom_test_works(scaling=1e-3, meshsize = 0.5e-3):
    P_bdry= scaling * np.array([
        [0.0, 0.0],
        [1.0, -0.2],
        [1.1, 1.2],
        [0.1, 0.7],
    ])
    msh_file = 'meshdata/geom_test.msh'
    with pygmsh.occ.Geometry() as geom:
        polygon1 = geom.add_polygon(
            P_bdry,
            mesh_size= meshsize,
        )
        geom.add_physical(polygon1.surface, 'surface1')
        mesh = geom.generate_mesh()
        print(mesh)
        # mesh.write(msh_file)
        pygmsh.write(msh_file)
        print('saved')

    return Shape(P=[], nonconvex_interceptor=[], msh_file=msh_file)

def pygmsh_geom_test_also_works(scaling=1e-3, meshsize = 0.5e-3):
    P_bdry= scaling * np.array([
        [0.0, 0.0],
        [1.0, -0.2],
        [1.1, 1.2],
        [0.1, 0.7],
    ])
    # Initialize empty geometry using the build in kernel in GMSH
    geometry = pygmsh.geo.Geometry()
    # Fetch model we would like to add data to
    model = geometry.__enter__()
    # add polygon
    polygon1 = model.add_polygon(
        P_bdry,
        mesh_size= meshsize,
    )
    return Shape(P=[], nonconvex_interceptor=[], pygmsh_geom=geometry)

def pygmsh_geom_test_spline(scaling=5e-3, meshsize = 0.5e-3):
    # Initialize empty geometry using the build in kernel in GMSH
    geometry = pygmsh.geo.Geometry()
    # Fetch model we would like to add data to
    model = geometry.__enter__()

    # circle1 = model.add_circle([0,0], radius=scaling, mesh_size=meshsize)
    # circle2 = model.add_circle([0,0], radius=scaling/2, mesh_size=meshsize)

    # loop1 = model.add_curve_loop(circle1.curve_loop)
    # loop2 = model.add_curve_loop([circle2])
    # loop1 = model.add_curve_loop(circle1)
    # loop2 = model.add_curve_loop(circle2)

    # plane_surface = model.add_plane_surface(circle1.curve_loop, holes=[circle2.curve_loop])

    print('geom deon')
    # model.add_physical([circle1, circle2], "surface1")
    # model.add_plane_surface([circle1.curve_loop, circle2.curve_loop], meshsize)
    # model.add_plane_surface(circle1.curve_loop)


    lcar = meshsize
    p1 = model.add_point([0.0*scaling, 0.0*scaling], lcar)
    p2 = model.add_point([1.0*scaling, 0.0*scaling], lcar)
    p3 = model.add_point([1.0*scaling, 0.5*scaling], lcar)
    p4 = model.add_point([1.0*scaling, 1.0*scaling], lcar)
    s1 = model.add_bspline([p1, p2, p3, p4])

    p2 = model.add_point([0.0*scaling, 1.0*scaling], lcar)
    p3 = model.add_point([0.5*scaling, 1.0*scaling], lcar)
    s2 = model.add_spline([p4, p3, p2, p1])

    ll = model.add_curve_loop([s1, s2])
    pl = model.add_plane_surface(ll)

    return Shape(P=[], nonconvex_interceptor=[], pygmsh_geom=geometry)

def pygmsh_geom_test(scaling=5e-3, meshsize = 0.5e-3):
    msh_file = 'meshdata/geom_test.msh'
    with pygmsh.geo.Geometry() as geom:
        circle1 = geom.add_circle([0,0], radius=scaling, mesh_size=meshsize)
        circle2 = geom.add_circle([0,0], radius=scaling/2, mesh_size=meshsize)

        geom.boolean_difference(circle1, [circle2])
        # geom.boolean_difference(circle1, circle2)
        # geom.add_physical(polygon1.surface, 'surface1')
        mesh = geom.generate_mesh()
        print(mesh)
        # mesh.write(msh_file)
        pygmsh.write(msh_file)
        print('saved')
    
    return Shape(P=[], nonconvex_interceptor=[], msh_file=msh_file)


def annulus():
    return Shape(P=[], nonconvex_interceptor=[], msh_file='meshdata/2d/annulus.msh')

def wheel_annulus(scaling=1e-3, meshsize=1e-3, inner_circle_ratio=0.7):
    """
    :returns: TODO

    """
    msh_file = 'meshdata/msh_test.msh'
    gmsh.initialize()
    print('done')
    # - the first 3 arguments are the point coordinates (x, y, z)
    # - the next (optional) argument is the target mesh size close to the point
    # - the last (optional) argument is the point tag (a stricly positive integer
    #   that uniquely identifies the point)
    # gmsh.model.occ.addPoint(0, 0, 0, meshsize, 1)
    gmsh.model.occ.addCircle(0, 0, 0, scaling, 1)
    gmsh.model.occ.addCircle(0, 0, 0, scaling*inner_circle_ratio, 2)
    gmsh.model.occ.addCurveLoop([1], 1)
    gmsh.model.occ.addCurveLoop([2], 2)
    gmsh.model.occ.addPlaneSurface([1, 2], 1)

    # gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);

    # gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5);

    # obligatory before generating the mesh
    gmsh.model.occ.synchronize()
    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)
    # save to file
    gmsh.write(msh_file)
    # if '-nopopup' not in sys.argv:
    # gmsh.fltk.run()

    return Shape(P=[], nonconvex_interceptor=[], msh_file=msh_file)

def gmsh_test(scaling=1e-3, meshsize=1e-3):
    """
    :returns: TODO

    """
    msh_file = 'meshdata/msh_test.msh'
    gmsh.initialize()
    print('done')
    # - the first 3 arguments are the point coordinates (x, y, z)
    # - the next (optional) argument is the target mesh size close to the point
    # - the last (optional) argument is the point tag (a stricly positive integer
    #   that uniquely identifies the point)
    # gmsh.model.occ.addPoint(0, 0, 0, meshsize, 1)
    gmsh.model.occ.addCircle(0, 0, 0, scaling, 1)
    gmsh.model.occ.addCircle(0, 0, 0, scaling/2, 2)
    gmsh.model.occ.addCurveLoop([1], 1)
    gmsh.model.occ.addCurveLoop([2], 2)
    gmsh.model.occ.addPlaneSurface([1, 2], 1)

    # gmsh.option.setNumber("Mesh.Algorithm", 6);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", meshsize);
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", meshsize);

    # gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.5);

    # obligatory before generating the mesh
    gmsh.model.occ.synchronize()
    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)
    # save to file
    gmsh.write(msh_file)
    # if '-nopopup' not in sys.argv:
    # gmsh.fltk.run()

    return Shape(P=[], nonconvex_interceptor=[], msh_file=msh_file)
#######################################################################
#                                 3D                                  #
#######################################################################


def sphere_small_3d():
    return Shape(P=[], nonconvex_interceptor=[], msh_file='meshdata/3d/3d_sphere_small.msh')


def sphere_small_3d_bad():
    print("Caution: bad msh because the .geo has a point in the center")
    return Shape(P=[], nonconvex_interceptor=[], msh_file='meshdata/3d/3d_sphere_small_bad.msh')

def sphere_small_3d_mat():
    # Copied from mperidem
    return Shape(P=[], nonconvex_interceptor=[], msh_file='meshdata/3d/3d_sphere_small_mat.msh')
def sphere_small_3d_mat_bad():
    # Copied from mperidem
    print("Caution: bad msh file. This one should cause the simulation to fail after contact.")
    return Shape(P=[], nonconvex_interceptor=[], msh_file='meshdata/3d/3d_sphere_small_mat_bad.msh')

def disk_w_hole_3d():
    return Shape(P=[], nonconvex_interceptor=[], msh_file='meshdata/3d/disk_w_hole_small.msh')

def sphere_unit_3d():
    return Shape(P=[], nonconvex_interceptor=[], msh_file='meshdata/3d/3d_sphere_unit.msh')

def plus_small_3d():
    return Shape(P=[], nonconvex_interceptor=[], msh_file='meshdata/3d/3d_plus_small.msh')
