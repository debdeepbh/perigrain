import copy
import h5py
import numpy as np
## regex matching
import re
import os.path

from exp_dict import Wall, Wall3d

def total_neighbors(conn, N):
    """Compute the total number of neighbors from connectivity data
    :conn: connectivity matrix, mx2, m=total intact bonds
    :N: total number of nodes
    :returns: TODO
    """
    deg = np.zeros(N)
    for i in range(len(conn)):
        deg[conn[i][0]] += 1
        deg[conn[i][1]] += 1

    return deg

class PlotInfo(object):
    """reading plotinfo for simulation related info"""
    def __init__(self, fc, lc, dt, modulo, dim):
        self.fc = fc
        self.lc = lc
        self.dt = dt
        self.modulo = modulo
        self.dim = dim
    def print(self):
        print('first counter = ', self.fc)
        print('last counter = ', self.lc)
        print('dt = ', self.dt)
        print('modulo = ', self.modulo)
        print('dim = ', self.dim)

def read_plotinfo(plotinfo_file, read_dim=False, quiet=False):
    if quiet:
        print('Reading file', plotinfo_file)
    p = h5py.File(plotinfo_file, "r")

    first_counter = int(p['f_l_counter'][0])
    last_counter = int(p['f_l_counter'][1])
    dt = float(p['dt'][0])
    modulo = int(p['modulo'][0])
    dim = int(p['dimension'][0])


    return PlotInfo(fc = first_counter, lc = last_counter, dt = dt, modulo = modulo, dim = dim)

def read_wallinfo(wallinfo_file):
    # print('Reading file', wallinfo_file)
    p = h5py.File(wallinfo_file, "r")
    return np.array(p['wall_info'])

def update_wallinfo(wall_filename, wall):
    """ update wall info from filename, if exits
    """
    # load wall info
    if os.path.isfile(wall_filename):
        p = h5py.File(wall_filename, "r")
        wi = np.array(p['wall_info'])[:,0]
        # wi = read_wallinfo(wall_filename)[:,0]
        if (len(wi)==4):
            # print(wi)
            wall.left        = wi[0]
            wall.right       = wi[1]
            wall.top         = wi[2] 
            wall.bottom      = wi[3]
        else:
            wall.x_min = wi[0]
            wall.y_min = wi[1]
            wall.z_min = wi[2]
            wall.x_max = wi[3]
            wall.y_max = wi[4]
            wall.z_max = wi[5]

        wall.reaction = np.array(p['reaction'])

def read_run_time(run_time_file):
    print('Reading file', run_time_file)
    p = h5py.File(run_time_file, "r")

    run_time = np.array(p['run_time'])
    t_ind = np.array(p['t_ind'])

    return np.c_[t_ind, run_time]

def populate_current(filename, exp_b, q = None, read_CurrPos=True, read_vel=False, read_acc=False, read_force=True, read_connectivity=False):
    """ Copies a main Experiment_breif setup class `exp_b` to a new Experiment_brief and then updates values from a given filename
    : q: quantity to plot
    :returns: Experiment_brief, with copied setup data and updated motion info
    """
    # print(t, end = ' ', flush=True)
    # tc_ind = ('tc_%05d' % t)
    # filename = loc+tc_ind+".h5";
    f = h5py.File(filename, "r")

    # copy the setup experiment
    # I suspect this will take a while
    t_exp_b = copy.deepcopy(exp_b)
    # t_exp_b = exp_b

    for name in f:
        if re.match(r'P_[0-9]+', name):
            # get particle id (int, starts from 0)
            pid = int(name[2:])
            P = t_exp_b.PArr[pid]

            if read_CurrPos:
                P.CurrPos = np.array(f[name+'/CurrPos'])
            if read_vel:
                P.vel     = np.array(f[name+'/vel'])
            if read_acc:
                t_exp_b.PArr[pid].acc    = np.array(f[name+'/acc'])
            if read_force:
                P.force   = np.array(f[name+'/force'])
            if read_connectivity:
                P.connectivity   = np.array(f[name+'/Connectivity'])


            ## Quantity to plot: examples
            if (q == 'force_norm'):
                P.q = np.sqrt(np.sum(np.square(P.force), axis=1))
            if (q == 'damage'):
                P_orig = exp_b.PArr[pid]
                N = len(P_orig.pos)

                orig_nbrs = total_neighbors(P_orig.connectivity, N)
                now_nbrs = total_neighbors(P.connectivity, N)


                # make sure not dividing by zero: probably won't happen unless isolated notes are present 
                P.q = (orig_nbrs - now_nbrs)/ orig_nbrs

                # debug
                # if pid==3:
                    # print('now_nbrs=',now_nbrs)
                    # print('orig_nbrs=',orig_nbrs)
                    # print('P.q=',P.q)

                # print(orig_nbrs)
                # print(now_nbrs)
                # print(P.q)
                # print('max:', np.amax(P.q))
                # print('Done')
            elif (q == 'vel_norm'):
                P.q = np.sqrt(np.sum(np.square(P.vel), axis=1)) 
            elif (q == 'vel_x_abs'):
                P.q = np.abs(P.vel[:,0]) #norm
            else:
                pass

    return t_exp_b

#######################################################################
# plotting related random things

class Val(object):
    """docstring for ClassName"""
    def __init__(self, f_sum, wall_v, wall_h, wall_reaction):
        self.f_sum = f_sum
        self.wall_reaction = wall_reaction
        self.wall_v = wall_v
        self.wall_h = wall_h
        
## smoothing
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def read_force_val(t, loc, dim, get_f_sum=False, input_exp_b=None):
    """ Reads the wall reaction force.
    : get_f_sum: if true, reads the total force of all particles, provided input_exp_b is given
    : returns: class Val
    """
    if (dim ==2):
        wall = Wall()
    else:
        wall = Wall3d()
    print(t, end = ' ', flush=True)
    wall_ind = ('wall_%05d' % t)
    wall_filename = loc+wall_ind+".h5";
    update_wallinfo(wall_filename, wall)


    if get_f_sum:
        tc_ind = ('tc_%05d' % t)
        h5_filename = loc+tc_ind+".h5";
        # t_exp_b = populate_current(h5_filename, input_exp_b, q = None, read_CurrPos=False, read_vel=False, read_acc=False, read_force=True)
        t_exp_b = populate_current(h5_filename, input_exp_b, q = None, read_CurrPos=False, read_vel=False, read_acc=True, read_force=False)
        PArr = t_exp_b.PArr
        f_sum = np.zeros((1, dim))
        for i in range(len(PArr)):
            # f_sum += np.sum(PArr[i].vol * PArr[i].force, axis =0)
            f_sum += np.sum(PArr[i].vol * PArr[i].rho * PArr[i].acc, axis =0)
        f_sum = f_sum[0]
    else:
        f_sum = None

    return Val(f_sum=f_sum, wall_v=wall.get_v(), wall_h=wall.get_h(), wall_reaction=wall.reaction)

# def read_reaction_frombulk(t, loc, dim, get_f_sum=False, input_exp_b=None):
    # """ Compute wall reaction force from the bulk and the relative distance from the wall
    # : get_f_sum: if true, reads the total force of all particles, provided input_exp_b is given
    # : returns: class Val
    # """
    # if (dim ==2):
        # wall = Wall()
    # else:
        # wall = Wall3d()
    # print(t, end = ' ', flush=True)
    # wall_ind = ('wall_%05d' % t)
    # wall_filename = loc+wall_ind+".h5";
    # update_wallinfo(wall_filename, wall)

    # tc_ind = ('tc_%05d' % t)
    # h5_filename = loc+tc_ind+".h5";

    # t_exp_b = populate_current(h5_filename, input_exp_b, q = None, read_CurrPos=True, read_vel=False, read_acc=False, read_force=False)
    # PArr = t_exp_b.PArr
    # f_sum = np.zeros((1, dim))

    # for i in range(len(PArr)):
        # # f_sum += np.sum(PArr[i].vol * PArr[i].force, axis =0)
        # # f_sum += np.sum(PArr[i].vol * PArr[i].rho * PArr[i].acc, axis =0)

        # # top wall
        # CurrPos_y = PArr[i].CurrPos[1]
        # dist = np.abs(CurrPos_y - wall.top)
        # C_r = input_exp_b.contact.contact_radius
        # if dist < C_r :
            # input_exp_b.contact.normal_stiffness * (C_r - dist) * PArr[i].vol[



    # f_sum = f_sum[0]

    # return Val(f_sum=f_sum, wall_v=wall.get_v(), wall_h=wall.get_h(), wall_reaction=wall.reaction)


#######################################################################
## bar

class BarVal(object):
    """docstring for BarVal"""
    def __init__(self, x_max, r_avg, r_min):
        self.x_max = x_max
        self.r_avg = r_avg
        self.r_min = r_min
        

def read_bar_val(t, loc, dim, input_exp_b=None, nodes_right_edge=None):
    """ Reads the values related to the stress-strain curve of the bar
    """

    tc_ind = ('tc_%05d' % t)
    h5_filename = loc+tc_ind+".h5";

    t_exp_b = populate_current(h5_filename, input_exp_b, q = None, read_CurrPos=True, read_vel=False, read_acc=False, read_force=True)

    part = t_exp_b.PArr[0]

    # f_sum = np.zeros((1, dim))
    # for i in range(len(PArr)):
        # # f_sum += np.sum(PArr[i].vol * PArr[i].force, axis =0)
        # f_sum += np.sum(PArr[i].vol * PArr[i].rho * PArr[i].acc, axis =0)
    # f_sum = f_sum[0]

    # compute right edge x-val
    x_max = np.amax(part.CurrPos[:,0])

    r_avg = np.mean(part.CurrPos[nodes_right_edge,0])
    r_min = np.amin(part.CurrPos[nodes_right_edge,0])
    # print(r_avg)

    return BarVal(x_max=x_max, r_avg=r_avg, r_min =r_min)
