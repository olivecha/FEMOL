import numpy as np


class OrthotropicMaterial(object):
    """
    OrthotropicMaterial class for storing material proprieties
    OrthotropicMaterial(name=None, Ex=None, Ey=None, Es=None, vx=None, Xt=None, Xc=None, Yt=None, Yc=None,
                 Sc=None, ho=None, rho=None)
    """

    def __init__(self, name=None, Ex=None, Ey=None, Es=None, vx=None, Xt=None, Xc=None, Yt=None, Yc=None,
                 Sc=None, ho=None, rho=None, Gyz=None, Gxz=None, ):
        self.name = name
        self.kind = 'orthotropic'

        # Modulus
        self.Ex = Ex  # [Pa]
        self.Ey = Ey  # [Pa]
        self.Es = Es  # [Pa]

        # defined out of plane shear
        if Gyz and Gxz:
            self.Gyz = Gyz
            self.Gxz = Gxz
            self.out_of_plane_shear = True

        # Estimated with Ey
        else:
            self.Gyz, self.Gxz = self.Ey, self.Es

        # Poisson ratios
        self.vx = vx  # []
        self.vy = vx * (Ey / Ex)  # []
        self.vy = np.min([1, self.vy])

        # Thickness in m
        self.hi = ho  # [m]

        # Density
        self.rho = rho  # [kg/m^3]

        # S matrix
        self.Sxx = 1 / Ex  # [1/Pa]
        self.Sxy = - self.vy / Ey  # [1/Pa]
        self.Syx = - self.vx / Ex  # [1/Pa]
        self.Syy = 1 / Ey  # [1/Pa]
        self.Sss = 1 / Es  # [1/Pa]
        self.S = np.array([[self.Sxx, self.Sxy, 0], [self.Syx, self.Syy, 0], [0, 0, self.Sss]])

        # Q matrix
        self.m = 1 / (1 - vx * self.vy)  # []
        self.Qxx = self.m * Ex  # [Pa]
        self.Qyy = self.m * Ey
        self.Qyx = self.m * vx * Ey
        self.Qxy = self.m * self.vy * Ex
        self.Qss = Es
        self.Q = np.array([[self.Qxx, self.Qxy, 0], [self.Qyx, self.Qyy, 0], [0, 0, self.Qss]])

        # Resistance
        self.Xt = Xt  # [MPa]
        self.Xc = Xc  # [MPa]
        self.Yt = Yt  # [MPa]
        self.Yc = Yc  # [MPa]
        self.Sc = Sc  # [MPa]

    def Q_mat(self, angle):
        """
        Computes the off-axis Q matrix of a material at an angle
        :param angle: Ply angle to compute off axis Q matrix
        :return: Off-axis Q matrix for the material
        """

        # U_i values for the block multiplication
        U_1 = (1 / 8) * (3 * self.Qxx + 3 * self.Qyy + 2 * self.Qxy + 4 * self.Qss)
        U_2 = (1 / 2) * (self.Qxx - self.Qyy)
        U_3 = (1 / 8) * (self.Qxx + self.Qyy - 2 * self.Qxy - 4 * self.Qss)
        U_4 = (1 / 8) * (self.Qxx + self.Qyy + 6 * self.Qxy - 4 * self.Qss)
        U_5 = (1 / 8) * (self.Qxx + self.Qyy - 2 * self.Qxy + 4 * self.Qss)

        # U vector for the vector multiplication
        U_vec = np.array([1, U_2, U_3])

        # Creating the off-axis Q matrix for the ply

        # U matrix for the vector multiplication
        theta = np.radians(angle)
        U_mat = np.array([[U_1, np.cos(2 * theta), np.cos(4 * theta)],
                          [U_1, -np.cos(2 * theta), np.cos(4 * theta)],
                          [U_4, 0, -np.cos(4 * theta)],
                          [U_5, 0, -np.cos(4 * theta)],
                          [0, 0.5 * np.sin(2 * theta), np.sin(4 * theta)],
                          [0, 0.5 * np.sin(2 * theta), -np.sin(4 * theta)]])

        Q = U_mat @ U_vec
        return np.array([[Q[0], Q[2], Q[4]],
                         [Q[2], Q[1], Q[5]],
                         [Q[4], Q[5], Q[3]]])

    def S_mat(self, angle):
        # Compute the U values
        U_1 = (1 / 8) * (3 * self.Sxx + 3 * self.Syy + 2 * self.Sxy + self.Sss)
        U_2 = (1 / 2) * (self.Sxx - self.Syy)
        U_3 = (1 / 8) * (self.Sxx + self.Syy - 2 * self.Sxy - self.Sss)
        U_4 = (1 / 8) * (self.Sxx + self.Syy + 6 * self.Sxy - self.Sss)
        U_5 = (1 / 2) * (self.Sxx + self.Syy - 2 * self.Sxy + self.Sss)

        # U vector
        U_vector = [1, U_2, U_3]

        # Convert angle to radians
        theta = np.radians(angle)
        U_mat = np.array([[U_1, np.cos(2 * theta), np.cos(4 * theta)],
                          [U_1, -np.cos(2 * theta), np.cos(4 * theta)],
                          [U_4, 0, -np.cos(4 * theta)],
                          [U_5, 0, -4 * np.cos(4 * theta)],
                          [0, np.sin(2 * theta), 2 * np.sin(4 * theta)],
                          [0, np.sin(2 * theta), -2 * np.sin(4 * theta)], ])

        S = U_mat @ U_vector

        return np.array([[S[0], S[2], S[4]],
                         [S[2], S[1], S[5]],
                         [S[4], S[5], S[3]]])

    def G_mat(self, angle):
        """
        Projection of the out of plane shear onto the laminate ply angle
        """
        theta = np.radians(angle)
        Gxz = self.Gxz * np.cos(theta) + self.Gyz * np.sin(theta)
        Gyz = self.Gyz * np.cos(theta) + self.Gxz * np.sin(theta)

        return np.array([[Gxz, 0],
                         [0, Gyz]])





class IsotropicMaterial(object):

    def __init__(self, E, mu, rho):
        self.kind = 'isotropic'
        self.E = E
        self.mu = mu
        self.rho = rho

    def plane_tensor(self, t):
        """
        Plane stress 3x3 stifness tensor
        """
        E = self.E
        mu = self.mu
        A = E / (1 - mu ** 2) * np.array([[1, mu, 0],
                                          [mu, 1, 0],
                                          [0, 0, (1 - mu) / 2]])
        return A * t

    def bending_tensor(self, t):
        """
        Plate bending 3x3 stiffness tensor
        """
        E = self.E
        mu = self.mu
        I = t ** 3 / 12

        D = I * E / (1 - mu ** 2) * np.array([[1, mu, 0],
                                              [mu, 1, 0],
                                              [0, 0, (1 - mu) / 2], ])

        return D

    @staticmethod
    def coupled_tensor():
        """
        Coupled tensor for isotropic material
        (always zero)
        """
        return np.zeros((3, 3))

    def shear_tensor(self, t):
        """
        Out of plane shear tensor for an isotropic material
        Reissner-Mindlin plate model
        """
        kappa = 5 / 6  # Shear ratio constant
        G = kappa * t * 0.5 * self.E / (1 + self.mu)  # shear modulus
        C = np.array([[G, 0],
                      [0, G]])
        return C


"""
Orthotropic laminate materials
"""


def general_carbon():
    """
    A function returning the general carbon OrthotropicMaterial class instance
    :return: OrthotropicMaterial(carbon proprieties)
    """
    # Carbon material proprieties
    name = 'general_carbon'
    Ex = 130e9
    Ey = 3.5e9
    Es = 5e9
    vx = 0.28
    ho = 0.00015
    rho = 1550
    return OrthotropicMaterial(name=name, Ex=Ex, Ey=Ey, Es=Es, vx=vx, ho=ho, rho=rho)


def general_flax():
    """
     A function returning the general flax OrthotropicMaterial class instance
    :return: OrthotropicMaterial(flax proprieties)
    """
    # Flax material proprieties
    name = 'general_flax'
    Ex = 19e9
    Ey = 2.9e9
    Es = 2.5e9
    vx = 0.28
    ho = 0.00033
    rho = 1100
    Gyz = Ey
    Gxz = Es
    return OrthotropicMaterial(name=name, Ex=Ex, Ey=Ey, Es=Es, vx=vx, ho=ho, rho=rho, Gyz=Gyz, Gxz=Gxz)


def laminate_resin():
    """
    Orthotropic material instance for the resin used in laminates
    """
    name = "laminate resin"
    Ex = 2.9e9  # Pa
    Ey = 2.9e9  # Pa
    vxy = 0.3
    vyz = 0.3
    vxz = 0.3
    Gxz = 0.5 * Ex / (1 + vxz)  # shear modulus
    Gyz = 0.5 * Ey / (1 + vyz)  # shear modulus

    Es = Ex * (1 - vxy) / 2 / (1 - vxy ** 2)  # Pa
    ho = 0.01  # m
    rho = 1100

    return OrthotropicMaterial(name, Ex, Ey, Es, vxy, ho=ho, rho=rho, Gxz=Gxz, Gyz=Gyz)


def laminate_core(infill_density=1):
    """
    Orthotropic material instance for the core material used in laminates
    Represented as PLA with an infill density value
    """
    name = "laminate core"
    Ex = infill_density*4.8e9  # Pa
    Ey = infill_density*4.8e9  # Pa
    vxy = 0.3
    vyz = 0.3
    vxz = 0.3
    Gxz = 0.5 * Ex / (1 + vxz)  # shear modulus
    Gyz = 0.5 * Ey / (1 + vyz)  # shear modulus

    Es = Ex * (1 - vxy) / 2 / (1 - vxy ** 2)  # Pa
    ho = 0.01  # m
    rho = 1240

    return OrthotropicMaterial(name, Ex, Ey, Es, vxy, ho=ho, rho=rho, Gxz=Gxz, Gyz=Gyz)


def T300_N5208():
    """
     A function returning the T300_N5208 OrthotropicMaterial class instance
    """
    # T300_N5208 material proprieties
    name = "T300_N5208"
    Ex = 181e9  # Pa
    Ey = 10.3e9  # Pa
    Es = 7.17e9  # Pa
    vx = 0.28
    Xt = 1500e6  # Pa
    Xc = 1500e6  # Pa
    Yt = 40e6  # Pa
    Yc = 246e6  # Pa
    Sc = 68e6  # Pa
    ho = 0.000125  # m
    rho = 1600  # kg/m3

    return OrthotropicMaterial(name=name, Ex=Ex, Ey=Ey, Es=Es, vx=vx, Xt=Xt, Xc=Xc,
                               Yt=Yt, Yc=Yc, Sc=Sc, ho=ho, rho=rho)


def Eglass_epoxy():
    """
    Function returning a OrthotropicMaterial class instance for the Eglass epoxy material
    """
    # Eglass_epoxy material proprieties
    name = "Eglass_epoxy"
    Ex = 38.6e9  # Pa
    Ey = 8.27e9  # Pa
    Es = 4.14e9  # Pa
    vx = 0.3
    Xt = 1062e6  # Pa
    Xc = 610e6  # Pa
    Yt = 31e6  # Pa
    Yc = 118e6  # Pa
    Sc = 72e6  # Pa
    ho = 0.000125  # m
    rho = 1800  # kg/m3

    return OrthotropicMaterial(name=name, Ex=Ex, Ey=Ey, Es=Es, vx=vx, Xt=Xt, Xc=Xc,
                               Yt=Yt, Yc=Yc, Sc=Sc, ho=ho, rho=rho)


def Kev49_epoxy():
    """
    Function returning a OrthotropicMaterial class instance for the Kev49 epoxy material
    """
    # Kev49_epoxy material proprieties
    name = "Kev49_epoxy"
    Ex = 76e9  # Pa
    Ey = 5.5e9  # Pa
    Es = 2.3e9  # Pa
    vx = 0.34
    Xt = 1400e6  # Pa
    Xc = 235e6  # Pa
    Yt = 12e6  # Pa
    Yc = 53e6  # Pa
    Sc = 34e6  # Pa
    ho = 0.000125  # m
    rho = 1460  # kg/m3

    return OrthotropicMaterial(name=name, Ex=Ex, Ey=Ey, Es=Es, vx=vx, Xt=Xt, Xc=Xc,
                               Yt=Yt, Yc=Yc, Sc=Sc, ho=ho, rho=rho)


def AS_H3501():
    """
    Function returning a OrthotropicMaterial class instance for the AS H3501 material
    """
    # AS H3501 material proprieties
    name = "AS_H3501"
    Ex = 138e9  # Pa
    Ey = 8.96e9  # Pa
    Es = 7.10e9  # Pa
    vx = 0.3
    Xt = 1447e6  # Pa
    Xc = 1447e6  # Pa
    Yt = 51.7e6  # Pa
    Yc = 206e6  # Pa
    Sc = 93e6  # Pa
    ho = 0.000125  # m
    rho = 1600  # kg/m3

    return OrthotropicMaterial(name=name, Ex=Ex, Ey=Ey, Es=Es, vx=vx, Xt=Xt, Xc=Xc,
                               Yt=Yt, Yc=Yc, Sc=Sc, ho=ho, rho=rho)


def flax_epoxy():
    """
    Function returning a OrthotropicMaterial class instance for the Flax epoxy material
    """
    # Flax epoxy material proprieties
    name = "flax_epoxy"
    Ex = 28.2e9  # Pa
    Ey = 3.31e9  # Pa
    Es = 5.2e9  # Pa
    vx = 0.34
    Xt = 286e6  # Pa
    Xc = 96e6  # Pa
    Yt = 12e6  # Pa
    Yc = 41e6  # Pa
    Sc = 27e6  # Pa
    ho = 0.000303  # m
    rho = 1330  # kg/m3

    return OrthotropicMaterial(name=name, Ex=Ex, Ey=Ey, Es=Es, vx=vx, Xt=Xt, Xc=Xc,
                               Yt=Yt, Yc=Yc, Sc=Sc, ho=ho, rho=rho)


def AS4_PEEK():
    """
    Function returning a OrthotropicMaterial class instance for the AS4 PEEK material
    """
    # Flax epoxy material proprieties
    name = "AS4_PEEK"
    Ex = 134e9  # Pa
    Ey = 8.9e9  # Pa
    Es = 5.10e9  # Pa
    vx = 0.28
    Xt = 2130e6  # Pa
    Xc = 1100e6  # Pa
    Yt = 80e6  # Pa
    Yc = 200e6  # Pa
    Sc = 160e6  # Pa
    ho = 0.000125  # m
    rho = 1600  # kg/m3

    return OrthotropicMaterial(name, Ex, Ey, Es, vx, Xt, Xc,
                               Yt, Yc, Sc, ho, rho)


def abaqus_benchmark():
    """
    Function returning an OrthotropicMaterial instance for the laminate bending benchmark
    Test R0031/1 from NAFEMS publication R0031, “Composites Benchmarks,” February 1995.
    """
    name = "abacus_benchmark"
    Ex = 100e9  # Pa
    Ey = 5e9  # Pa
    Ez = 5e9  # Pa

    vxy = 0.4
    vyz = 0.3
    vxz = 0.3

    Gxy = 3e9  # Pa
    Gxz = 2e9  # Pa
    Gyz = 2e9  # Pa

    ho = 0.0001  # m

    rho = 1000

    return OrthotropicMaterial(name, Ex, Ey, Gxy, vxy, ho=ho, rho=rho, Gxz=Gxz, Gyz=Gyz)


def isotropic_laminate():
    """
    Isotropic Laminate material to test the stiffness tensor formulations
    """
    name = "isotropic laminate"
    Ex = 10920  # Pa
    Ey = 10920  # Pa
    vxy = 0.3
    vyz = 0.3
    vxz = 0.3

    kappa = 5 / 6  # Shear ratio constant
    Gxz = 0.5 * Ex / (1 + vxz)  # shear modulus
    Gyz = 0.5 * Ey / (1 + vyz)  # shear modulus

    Es = Ex * (1 - vxy) / 2 / (1 - vxy ** 2)  # Pa

    ho = 0.01  # m

    rho = 1000

    return OrthotropicMaterial(name, Ex, Ey, Es, vxy, ho=ho, rho=rho, Gxz=Gxz, Gyz=Gyz)


def orthotropic_steel():
    """
    Return an orthotropic material class instances with steel properties
    """
    name = "orthotropic steel"
    E = 190e9
    rho = 7840
    mu = 0.28
    Ex = E  # Pa
    Ey = E  # Pa
    vxy = mu
    vyz = mu
    vxz = mu

    kappa = 5 / 6  # Shear ratio constant
    Gxz = 0.5 * Ex / (1 + vxz)  # shear modulus
    Gyz = 0.5 * Ey / (1 + vyz)  # shear modulus

    Es = Ex * (1 - vxy) / 2 / (1 - vxy ** 2)  # Pa

    ho = 0.001  # m

    return OrthotropicMaterial(name, Ex, Ey, Es, vxy, ho=ho, rho=rho, Gxz=Gxz, Gyz=Gyz)


ALL_LAMINATE_MTR = [general_carbon(),
                    general_flax(),
                    T300_N5208(),
                    Eglass_epoxy(),
                    Kev49_epoxy(),
                    AS_H3501(),
                    flax_epoxy(),
                    AS4_PEEK(),
                    abaqus_benchmark(), ]


def random_laminate_material():
    i = np.random.randint(len(ALL_LAMINATE_MTR))
    return ALL_LAMINATE_MTR[i]


"""
Isotropic materials
"""


def general_isotropic():
    """
    A Function returning a E=1 and v=0.3 rho=1 isotropic material
    :return: IsotropicMaterial(1, 0.3, 1)
    """
    E = 1  # Pa
    mu = 0.3
    rho = 1  # kg/m3
    return IsotropicMaterial(E, mu, rho)


def isotropic_bending_benchmark():
    E = 10920  # Pa
    mu = 0.3
    rho = 1  # kg/m3
    return IsotropicMaterial(E, mu, rho)


def solid_PLA():
    E = 3500
    mu = 0.36
    rho = 1.252
    return IsotropicMaterial(E, mu, rho)


def printed_PLA(infill=1):
    PLA = solid_PLA()
    E = infill*PLA.E
    mu = infill*PLA.mu
    rho = infill*PLA.rho
    return IsotropicMaterial(E, mu, rho)