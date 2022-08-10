import numpy as np

class Material(object):
    """docstring for Material"""
    def __init__(self, delta, rho, snot, cnot, bulk_modulus, E = None, nu = None, Gnot = None, shear_modulus = None, name=None):
        super(Material, self).__init__()
        self.delta  = delta
        self.rho    = rho     
        self.snot   = snot
        self.cnot   = cnot
        self.bulk_modulus = bulk_modulus

        self.E = E
        self.nu = nu
        self.Gnot = Gnot
        self.shear_modulus = shear_modulus

        self.name = name

    # def generate(self, delta, rho_scale=1, K_scale=1, G_scale=1, Gnot_scale=1):

        # if self.name == 'peridem':
            # self.bulk_modulus = 2.0e9 * K_scale
            # self.shear_modulus = 1.33e+09 * G_scale
            # self.rho=1200.0	* rho_scale
            # self.Gnot = 135.0 * Gnot_scale
            # self.nu = 1/3

            # self.E = 9 * self.bulk_modulus * self.shear_modulus / ( 9 * self.bulk_modulus + self.shear_modulus);

            # self.cnot = 24 * self.E /( (1 - self.nu) * np.pi * (self.delta**3) );
            # self.snot = np.sqrt(4 * np.pi * self.Gnot /(9*self.E*self.delta));
        
        # if self.name ==  'sodalime_similar_to_peridem'
            # E = 1e9 * E_scale
            # rho=1200 * rho_scale
            # # nu = 1/4
            # nu = 1/3
            # Gnot = 135 * Gnot_scale

            # cnot = 6*E/( np.pi * (delta**3) * (1 - nu))

            # snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta))

            # bulk_modulus = E/ (3 * ( 1 - 2 *nu)) 
            # # extra
            # shear_modulus = E/ 2/ (1 + nu)



    def print(self):
        """print info
        """
        print('delta: ', self.delta)
        print('rho: ', self.rho)
        print('cnot: ', self.cnot)
        print('snot: ', self.snot)
        print('E: ', self.E)
             

def peridem(delta):
    """Generate material properties and peridynamic constants using delta
    """
    bulk_modulus = 2.0e9
    shear_modulus = 1.33e+09
    rho=1200.0	
    Gnot = 135.0

    #  nu = 0.2278
    nu = (3 * bulk_modulus - 2 * shear_modulus) / ( 2 * ( 3 * bulk_modulus + shear_modulus))
    #  E = 1.23e9
    E = 9 * bulk_modulus * shear_modulus / ( 9 * bulk_modulus + shear_modulus);


    cnot = 24 * E /( (1 - nu) * np.pi * (delta**3) );
    snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta));

    # print('nu = ', nu)
    # print('E = ', E)

    # return Material(delta, rho, snot, cnot, bulk_modulus)
    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )

def sodalime(delta):
    # print('There is some issue with these settings')
    E = 72e9
    rho=2440
    nu = 0.22 
    Gnot = 135

    cnot = 6*E/( np.pi * (delta**3) * (1 - nu))

    snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta))

    bulk_modulus = E/ (3 * ( 1 - 2 *nu)) 

    # extra
    shear_modulus = E/ 2/ (1 + nu)

    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )

def sodalime_similar_to_peridem(delta, E_scale=1, rho_scale=1, Gnot_scale=1):
    # print('There is some issue with these settings')
    E = 1e9 * E_scale
    rho=1200 * rho_scale
    # nu = 1/4
    nu = 1/3
    Gnot = 135 * Gnot_scale

    cnot = 6*E/( np.pi * (delta**3) * (1 - nu))

    snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta))

    bulk_modulus = E/ (3 * ( 1 - 2 *nu)) 
    # extra
    shear_modulus = E/ 2/ (1 + nu)

    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )

def peridem_1d_deformable(delta, rho_scale=1, K_scale=1, G_scale=1, Gnot_scale=1):
    """ Smaller fracture toughness
    """
    bulk_modulus = 2.0e9 * K_scale
    shear_modulus = 1.33e+09 * G_scale
    rho=1200.0	* rho_scale
    Gnot = 135.0 * Gnot_scale

    nu = 1/3
    E = 9 * bulk_modulus * shear_modulus / ( 9 * bulk_modulus + shear_modulus);

    # cnot = 24 * E /( (1 - nu) * np.pi * (delta**3) );
    cnot = 5 * E/ (delta**5)

    # incorrect snot
    snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta));

    # print('nu = ', nu)
    # print('E = ', E)

    # return Material(delta, rho, snot, cnot, bulk_modulus)
    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )


def peridem_deformable(delta, rho_scale=1, K_scale=1, G_scale=1, Gnot_scale=1):
    """ Smaller fracture toughness
    """
    bulk_modulus = 2.0e9 * K_scale
    shear_modulus = 1.33e+09 * G_scale
    rho=1200.0	* rho_scale
    Gnot = 135.0 * Gnot_scale

    nu = 1/3
    E = 9 * bulk_modulus * shear_modulus / ( 9 * bulk_modulus + shear_modulus);

    cnot = 24 * E /( (1 - nu) * np.pi * (delta**3) );
    snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta));

    # print('nu = ', nu)
    # print('E = ', E)

    # return Material(delta, rho, snot, cnot, bulk_modulus)
    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )

def peridem_softer_1(delta):
    """ Smaller fracture toughness
    """
    bulk_modulus = 2.0e9
    shear_modulus = 1.33e+09
    rho=1200.0	
    Gnot = 135.0/20

    #  nu = 0.2278
    nu = (3 * bulk_modulus - 2 * shear_modulus) / ( 2 * ( 3 * bulk_modulus + shear_modulus))
    #  E = 1.23e9
    E = 9 * bulk_modulus * shear_modulus / ( 9 * bulk_modulus + shear_modulus);

    cnot = 24 * E /( (1 - nu) * np.pi * (delta**3) );
    snot = np.sqrt(4 * np.pi * Gnot /(9*E*delta));

    # print('nu = ', nu)
    # print('E = ', E)

    # return Material(delta, rho, snot, cnot, bulk_modulus)
    return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )

def peridem_3d(delta):
  bulk_modulus = 2.e+09
  shear_modulus = 1.33e+09
  rho=1200
  nu = (3 * bulk_modulus - 2 * shear_modulus) / ( 2 * ( 3 * bulk_modulus + shear_modulus))
  E = 9 * bulk_modulus * shear_modulus / ( 9 * bulk_modulus + shear_modulus)

  ## %% 3d: From silling-askari
  cnot = 18 * bulk_modulus / (np.pi * delta**4)
  Gnot = 135
  snot = np.sqrt(5 * Gnot / (9 * bulk_modulus * delta))
  return Material(delta, rho, snot, cnot, bulk_modulus, E = E, nu = nu, Gnot = Gnot, shear_modulus = shear_modulus )


